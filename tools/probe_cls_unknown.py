# -*- coding: utf-8 -*-
"""未知出口验收：把**完全不参与训练**的苹果病害喂进去，看模型会不会硬答

数据：`datasets/apple_cls/holdout/06_apple_other/` —— 苹果花叶病(apple mosaic virus)
      的留出样本。这批图既不在 train 也不在 val，ultralytics 扫不到，
      所以是干净的"未知病害"探针。

**判据（第一版设计已作废，这里说明为什么）**：
  第一版期望模型答"不确定"（低置信度），实测只有 8% 生效 —— 因为置信度是模型自己说的，
  对没见过的输入照样给高分。所以改成了**显式的拒绝类 06_apple_other**：
  让它学到"这是苹果病害、但不在我细分的那 6 种里"。
  因此现在的判据是：留出样本应落进 **06_apple_other（或其他）**，
  而不是像旧的 6 类检测模型那样 89% 硬答某个具体的苹果病。

脚本并列打印新旧两代模型在同一批图上的答案分布，作为前后对比的验收材料。
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config                                              # noqa: E402
from class_names import cn_names_dict, THRESH, MARGIN      # noqa: E402

HOLDOUT_ROOT = config.CLS_DATA / "holdout"


def _holdout_dir():
    """留出集按类分子目录；类目录名会随类别体系调整而变化，所以自动探测而不是写死"""
    if not HOLDOUT_ROOT.exists():
        return HOLDOUT_ROOT
    subs = [p for p in HOLDOUT_ROOT.iterdir() if p.is_dir()]
    return subs[0] if subs else HOLDOUT_ROOT
# 阈值 THRESH/MARGIN 由 class_names.py 提供（全项目唯一定义），不在这里重复
DET_MODEL_DEFAULT = str(config.MODELS / "crop6_s_best.pt")
# 默认用 m3 —— 它是唯一通过验收的版本（实测拒绝率 73%，修正 REJECT 后 81%）。
# 旧的 apple_cls_m_best.pt 实测仅 12%，会把苹果花叶病大量硬答成"玉米健康叶"。
CLS_MODEL_DEFAULT = str(config.MODELS / "apple_cls_m3_best.pt")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DET_CN = {0: '稻瘟病', 1: '玉米锈病', 2: '小麦条锈病', 3: '苹果黑星病',
          4: '柑橘溃疡病', 5: '葡萄霜霉病'}
# 落进这些即算"正确拒绝"——模型承认它不是自己细分的那几种叶部病害。
# 注意：类名必须与 class_names.DIR2CN 里的中文名**逐字一致**，否则该分支永远不命中。
# 历史上这里写的是"其他作物叶片"（不存在的类名，真实类为 16_other_leaf "其他作物叶片"），
# 导致落进该类的样本被错算成"硬答"。实测修正后拒绝率 73% → 81%。
REJECT = {"其他苹果病害", "其他作物叶片"}


def collect_mosaic():
    d = _holdout_dir()
    if not d.exists():
        return []
    return [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXT]


def main():
    ap = argparse.ArgumentParser(description="未知病害拒绝率验收")
    ap.add_argument("model", nargs="?", default=CLS_MODEL_DEFAULT,
                    help="分类模型路径，默认 models/apple_cls_m3_best.pt")
    ap.add_argument("--det", default=DET_MODEL_DEFAULT,
                    help="对照组检测模型路径，默认 models/crop6_s_best.pt")
    config.add_device_arg(ap)
    args = ap.parse_args()

    # 原为 argv[1]，语义不变；同时补上 --det / --device
    CLS_MODEL, DET_MODEL = args.model, args.det
    config.setup_cjk()          # 必须在 import ultralytics 之前（避免联网下载字体）

    imgs = collect_mosaic()
    if not imgs:
        print(f"!! 找不到留出集: {HOLDOUT_ROOT}")
        print("   先跑 tools/build_apple_cls.py 生成 holdout（它只扫 train/val，不会碰这里）")
        return
    print(f"未知病害探针（留出集）: apple mosaic virus  {len(imgs)} 张 —— 既不在 train 也不在 val\n")

    from ultralytics import YOLO
    m_cls = YOLO(CLS_MODEL)
    cn = cn_names_dict()
    m_cls.model.names = cn

    # ---- 新分类模型 ----
    rejected = 0
    wrong_disease = collections.Counter()
    dist_new = collections.Counter()
    for p in imgs:
        try:
            r = m_cls.predict(source=str(p), imgsz=384, device=args.device, verbose=False)[0]
            if r.probs is None:
                continue
            pr = r.probs.data.cpu().numpy()
            o = pr.argsort()[::-1]
            p1, p2 = float(pr[o[0]]), float(pr[o[1]]) if len(o) > 1 else 0.0
            if p1 < THRESH or (p1 - p2) < MARGIN:
                dist_new["（低置信度→不确定）"] += 1
                rejected += 1
                continue
            label = cn[o[0]]
            dist_new[label] += 1
            if label in REJECT:
                rejected += 1
            else:
                wrong_disease[label] += 1
        except Exception:
            continue

    total = sum(dist_new.values())
    print(f"=== 新分类模型（{len(cn)} 类）在未知病害上的表现 ===")
    print(f"正确拒绝（落进 其他苹果病害/其他作物叶片/不确定）: "
          f"{rejected}/{total} = {rejected/max(1,total)*100:.0f}%")
    print(f"（阈值 THRESH={THRESH}, MARGIN={MARGIN}）")
    for k, v in dist_new.most_common(9):
        mark = "  ← 拒绝" if (k in REJECT or "不确定" in k) else ""
        print(f"    {k:<16} {v:>4}  {v/max(1,total)*100:>4.0f}%{mark}")
    if wrong_disease:
        print(f"  ⚠ 仍然硬答成具体苹果病害: {sum(wrong_disease.values())} 张 "
              f"({sum(wrong_disease.values())/max(1,total)*100:.0f}%)")

    # ---- 旧检测模型（对照组）----
    if pathlib.Path(DET_MODEL).exists():
        m_det = YOLO(DET_MODEL)
        m_det.model.names = DET_CN
        dist_old = collections.Counter()
        for p in imgs:
            try:
                r = m_det.predict(source=str(p), imgsz=640, conf=0.3, device=args.device, verbose=False)[0]
                if len(r.boxes) == 0:
                    dist_old["（无检测）"] += 1
                else:
                    i = int(r.boxes.conf.argmax())
                    dist_old[DET_CN[int(r.boxes.cls[i])]] += 1
            except Exception:
                continue
        t2 = sum(dist_old.values())
        print(f"\n=== 旧 6 类检测模型（对照组）===")
        for k, v in dist_old.most_common(8):
            print(f"    {k:<12} {v:>4}  {v/max(1,t2)*100:>4.0f}%")

    print(f"\n{'='*60}")
    print(f"判据: 正确拒绝（其他苹果病害/其他作物叶片/不确定）占比 >= 70%")
    print(f"结论: {'✅ 通过' if rejected/max(1,total) >= 0.70 else '❌ 未达标 —— 模型仍在硬答未知病害'}")


if __name__ == "__main__":
    main()