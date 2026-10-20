# -*- coding: utf-8 -*-
"""训练苹果叶部病害分类模型

用法:
    python tools/train_apple_cls.py                # 云端主模型 (m-cls, 24GB卡)
    python tools/train_apple_cls.py --size s       # 再训一个 s-cls 给本地 4GB 演示用
    python tools/train_apple_cls.py --smoke        # 烟雾测试：3 epochs + 10% 数据
    python tools/train_apple_cls.py --data /root/data/apple_cls   # 云端数据路径

必须显式指定的几个参数（都是踩过的坑）:
  * imgsz —— 分类任务默认 224（models/yolo/classify/train.py），不会跟随检测的 640
  * optimizer/lr0 —— 'auto' 的 lr 按 iterations 跳变（nc=7 时约 7e-4；iter>10000 切 MuSGD 0.01），
                     行为随 epoch 数突变、不可复现，所以钉死 AdamW + 1e-3
  * dropout/erasing —— 默认 erasing=0.4 对"病灶很小"的类太狠，会把唯一的病灶擦掉
  * scale —— 默认 0.5 裁得狠，细粒病害吃亏
"""
import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

# 数据集路径默认值：由 tools/config.py 解析（环境变量 PLANT_DISEASE_ROOT 可覆盖），
# 云端用 --data 显式指定
LOCAL_DATA = str(config.CLS_DATA)

SIZES = {
    # 尺度: (预训练权重, 每卡 batch)
    "n": ("yolo11n-cls.pt", 128),
    "s": ("yolo11s-cls.pt", 64),
    "m": ("yolo11m-cls.pt", 64),
    "l": ("yolo11l-cls.pt", 32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=LOCAL_DATA, help="分类数据集目录")
    ap.add_argument("--size", default="m", choices=sorted(SIZES), help="模型尺度")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=None, help="默认按尺度取 SIZES 里的值")
    ap.add_argument("--name", default=None, help="run 名，默认 apple_cls_<size>")
    ap.add_argument("--smoke", action="store_true",
                    help="烟雾测试：3 epochs + 10%% 数据")   # 注意 %% —— argparse 会对 help 做 % 格式化，单个 % 会让 --help 直接崩
    ap.add_argument("--cache", action="store_true",
                    help="把图片解码后缓存进内存。**默认关**：云实例内存有限，"
                         "实测开着会被内核 OOM killer 杀掉（日志里是'已杀死'）")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader 进程数。默认 4（原为 8，也是内存压力源）")
    ap.add_argument("--resume", action="store_true",
                    help="从该 run 的 last.pt 继续训练。云平台会随机杀进程，"
                         "配合外层 while 循环可以一直续到跑完")
    config.add_device_arg(ap)          # 原为硬编码 device=0，纯 CPU 机器会直接报错
    args = ap.parse_args()

    weights, default_batch = SIZES[args.size]
    batch = args.batch or default_batch
    name = args.name or f"apple_cls_{args.size}"
    if args.smoke:
        args.epochs, batch = 3, max(4, batch // 8)

    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"!! 数据集目录不存在: {data}")
        sys.exit(1)

    # 类数自检：目录名排序序就是类别索引，必须和 class_map.json 逐字一致
    train_dir = data / "train"
    dirs = sorted(p.name for p in train_dir.iterdir() if p.is_dir()) if train_dir.exists() else []
    n_cls = len(dirs)
    print(f"[数据] {data}")
    print(f"[数据] {n_cls} 类: {', '.join(dirs) if dirs else '（空！）'}")
    for split in ("train", "val", "test"):
        d = data / split
        if d.exists():
            print(f"       {split}: {sum(1 for p in d.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})} 张")
    if not dirs:
        print("!! train 下没有类别子目录，先跑 tools/build_apple_cls.py")
        sys.exit(1)

    from ultralytics import YOLO

    # 续训：优先找该 run 的 last.pt。云平台会随机杀进程，靠这个接着跑。
    if args.resume:
        lasts = sorted(config.RUNS.rglob(f"{name}/weights/last.pt"))
        if lasts:
            print(f"[续训] 从 {lasts[0]} 接着跑")
            m = YOLO(str(lasts[0]))
            m.train(resume=True)
            best = sorted(config.RUNS.rglob(f"{name}/weights/best.pt"))
            print(f"[权重位置] {best[0] if best else '（没找到 best.pt）'}")
            return
        print("[续训] 没找到 last.pt，改为从头训")

    m = YOLO(weights)
    print(f"[模型] {weights}   imgsz={args.imgsz}  batch={batch}  epochs={args.epochs}")

    m.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,          # 必须显式给
        batch=batch,
        workers=args.workers,
        cache=("ram" if args.cache else False),
        device=args.device,        # None = 交给 ultralytics 自动选（有 GPU 用 GPU）
        optimizer="AdamW",         # 必须显式给，见文件头说明
        lr0=1e-3,
        cos_lr=True,
        dropout=0.2,               # 防过拟合
        erasing=0.2,               # 别把唯一病灶擦掉
        scale=0.3,                 # 别裁太狠
        flipud=0.5,                # 叶片无固定朝向，白捡一倍数据
        patience=60,               # 30 太短：实测第 25 轮最好、第 55 轮就早停了，没跑够
        seed=0,
        project=str(config.RUNS),   # 绝对路径：不依赖当前工作目录
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # 归档权重到 models/，沿用项目的 *_best.pt 命名惯例
    # 注意：分类任务下 run 目录会被 ultralytics 嵌一层任务名
    # （project=<runs> 实际落在 <runs>/classify/runs/<name>），所以这里用搜索而不是拼路径
    cands = sorted(config.RUNS.rglob(f"{name}/weights/best.pt"))
    if not cands:
        print(f"!! 没找到 best.pt，检查 {config.RUNS} 下的目录结构")
        return
    best = cands[0]
    # 原为 sys.platform == "win32" 判断，因为硬编码的 r"G:\YOLO\models" 在 Linux 上
    # 会被当成**相对路径**（反斜杠是合法文件名字符），真建出名为 "G\YOLO\models" 的垃圾目录。
    # 改用 config.MODELS（绝对路径）后这个平台分支就没有存在必要了。
    dst = config.MODELS / f"apple_cls_{args.size}_best.pt"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dst)
        print(f"[归档] {best} → {dst}")
    except Exception as e:
        print(f"[归档跳过] 写不进 {dst}: {type(e).__name__}")
    print(f"[权重位置] {best}")

    try:
        r = m.val(data=str(data), split="val", imgsz=args.imgsz, batch=batch, device=args.device)
        print(f"[结果] top1={r.top1:.3f}  top5={r.top5:.3f}")
    except Exception as e:
        print(f"[跳过验证] {type(e).__name__}: {str(e)[:120]}")

    print("\n注意：分类的 val() 只给 top1/top5，每类召回要用 tools/val_apple_cls.py 从混淆矩阵算")


if __name__ == "__main__":
    main()