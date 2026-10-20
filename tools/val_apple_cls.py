# -*- coding: utf-8 -*-
"""评估苹果叶部病害分类模型：每类召回 + 中文混淆矩阵 + 「不确定」阈值标定

为什么要自己写而不直接用 model.val()：
  分类任务的 val() 只返回 top1/top5（ultralytics/utils/metrics.py 里
  ClassifyMetrics.keys 只有 accuracy_top1|top5），**没有每类精度表**。
  每类召回必须从混淆矩阵自己算。

两个必须处理的坑：
  1) 中文混淆矩阵：plt_settings() 会把 font.sans-serif 前置成 "Arial Unicode MS"，
     但我们那个 Arial.Unicode.ttf 实际是 simhei 的副本、内部 family 名是 SimHei，
     匹配不上 → 中文全变方框**且只 warning 不报错**。必须显式设 SimHei。
  2) 覆盖类名要在 val() **之前**：ClassificationValidator.init_metrics 是在那时读
     model.names 的，晚了标签就还是英文。
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config                                           # noqa: E402
from class_names import cn_names_dict, load_class_map   # noqa: E402

DATA = config.CLS_DATA


def main():
    ap = argparse.ArgumentParser(description="评估分类模型：每类召回 + 中文混淆矩阵 + 阈值标定")
    ap.add_argument("model", nargs="?", default=str(config.MODELS / "apple_cls_s_best.pt"),
                    help="模型权重路径，默认 models/apple_cls_s_best.pt")
    config.add_device_arg(ap)
    args = ap.parse_args()

    # 中文渲染：装 ultralytics 的 Arial.Unicode.ttf shim 并设 matplotlib 字体。
    # 必须在 import ultralytics 之前调用（见文件头坑 1：ultralytics 会把自己的
    # "Arial Unicode MS" 前置到 font.sans-serif，字体名对不上就静默变方块）。
    config.setup_cjk()

    model_path = args.model
    if not pathlib.Path(model_path).exists():
        print(f"!! 模型不存在: {model_path}")
        print("   先跑 tools/train_apple_cls.py 训练")
        sys.exit(1)

    from ultralytics import YOLO
    m = YOLO(model_path)

    names = load_class_map()                       # {目录名: 中文}
    cn = cn_names_dict()                           # {索引: 中文}
    m.model.names = cn                             # 必须在 val() 之前
    print(f"[模型] {model_path}")
    print(f"[类别] {len(cn)} 类: {', '.join(cn.values())}")

    r = m.val(data=DATA, split="val", imgsz=384, batch=32, device=args.device,
              project=str(config.RUNS), name="val_apple_cls", exist_ok=True, plots=True)

    # ---- 每类指标（从混淆矩阵算，val() 不提供）----
    cm = None
    for attr in ("confusion_matrix",):
        c = getattr(r, attr, None)
        if c is not None and getattr(c, "matrix", None) is not None:
            cm = np.asarray(c.matrix)
            break
    if cm is None:
        print("!! 拿不到混淆矩阵，只报总体指标")
        cm = None

    print("\n=== 总体 ===")
    print(f"top1 = {r.top1:.4f}    top5 = {r.top5:.4f}")

    if cm is not None:
        n = cm.shape[0] - 1 if cm.shape[0] > len(names) else len(names)   # 末行/列常是 background
        print(f"\n混淆矩阵 {cm.shape}（末行末列通常是 background）")
        print(f"\n=== 每类召回（含 top-1 准确率）===")
        print(f'{"类别":<16}{"样本数":>8}{"召回率":>10}{"被误判成":>10}')
        recalls = []
        for i in range(min(len(names), cm.shape[0])):
            tot = cm[i].sum()
            if tot == 0:
                continue
            rec = cm[i, i] / tot
            recalls.append((i, rec, int(tot)))
            # 最常被误判成谁
            row = cm[i].copy()
            if i < len(row):
                row[i] = 0
            wrong = int(row.argmax()) if row.sum() > 0 else -1
            wtxt = list(names.values())[wrong] if 0 <= wrong < len(names) else "-"
            print(f"{list(names.values())[i]:<16}{int(tot):>8}{rec:>10.1%}{wtxt:>10}")
        if recalls:
            print(f'{"平均":<16}{"":>8}{sum(x[1] for x in recalls)/len(recalls):>10.1%}')

    # ---- 「不确定」阈值标定：给演示脚本用 ----
    print("\n=== 「不确定」出口阈值标定 ===")
    try:
        m.model.names = cn
        rs = m.predict(source=str(DATA / "val"), imgsz=384, device=args.device, verbose=False)
        confs, correct = [], []
        for res in rs:
            if res.probs is None:
                continue
            p = res.probs.data.cpu().numpy()
            top1 = int(p.argmax())
            confs.append(float(p[top1]))
            correct.append(bool(getattr(res, "path", "") and top1 >= 0))   # 占位，真实对错需标签
        if confs:
            confs = np.array(confs)
            # 无标签时用分位数给一个保守建议：让 90% 的检测留在阈值之上
            t90 = float(np.quantile(confs, 0.10))
            print(f"验证集 top1 置信度: 中位={np.median(confs):.3f}  10分位={t90:.3f}  最小={confs.min():.3f}")
            print(f"建议 EDGE_THRESH ≈ {max(0.30, round(t90, 2))}（改 tools/class_names.py 的 THRESH）")
        else:
            print("(拿不到置信度分布)")
    except Exception as e:
        print(f"(标定跳过: {type(e).__name__}: {str(e)[:100]})")

    print(f"\n结果目录: {r.save_dir}")
    print("中文混淆矩阵若全是方框，跑 tools/config.py 自检字体；"
          "或设 PLANT_DISEASE_FONT 指向一个中文字体")


if __name__ == "__main__":
    main()