# -*- coding: utf-8 -*-
"""图像质量巡检：找出数据集里的坏图（破坏性增广残留 / 商品图 / 纯色块）

用法:
    python tools/qa_image_quality.py <目录>          # 目录下每个子目录当一个类
    python tools/qa_image_quality.py <目录> --flat   # 目录下的图整体当一类

三把尺子（都是实测踩出来的）:
  1) 唯一颜色数  —— 正常照片几万种；被 Roboflow 破坏性增广搞成 6x6 色块后只剩几十种
  2) 块状度      —— 缩到 1/8 再放大回来的误差很小，说明原图信息量极低（本来就糊）
  3) 白边占比    —— 四周近乎纯白 = 商品图/白底 Stock 图（PlantDoc 大量此类）
"""
import argparse
import pathlib
import statistics
import sys

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_N = 200          # 每类最多采样多少张，够统计就行


def metrics(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    a = np.asarray(im, dtype=np.uint8)
    # 唯一颜色数（**尺寸归一化**：NEAREST 压到最长边 <=128 的网格再数，
    # 不插值、不改像素值，否则小图会因采样像素少被误判成坏图）
    s = min(1.0, 128 / max(w, h))
    small = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST) if s < 1.0 else im
    uniq = len(np.unique(np.asarray(small).reshape(-1, 3), axis=0))
    # 块状度：缩到 1/8 再放大，与原图的平均绝对差（越低越像像素块）
    tiny = im.resize((max(1, w // 8), max(1, h // 8)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    ta = np.asarray(tiny, dtype=np.float32)
    blocky = float(np.abs(a.astype(np.float32) - ta).mean())
    # 白边占比
    b = max(2, int(min(h, w) * 0.06))
    edge = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    white = float((edge.min(axis=1) >= 235).mean())
    return uniq, blocky, white


def report(name, files):
    if not files:
        print(f"{name:<26}  (无图)")
        return
    u, bl, wh = [], [], []
    bad = []
    for p in files[:MAX_N]:
        try:
            a, b, c = metrics(p)
        except Exception:
            continue
        u.append(a)
        bl.append(b)
        wh.append(c)
        if a < 3000:                      # 唯一色(归一化后)过低 → 基本是坏图
            bad.append(p.name)
    n = len(u)
    if not n:
        return
    print(f"{name:<26} n={n:>4}  唯一色中位={int(statistics.median(u)):>6}  "
          f"块状度中位={statistics.median(bl):>5.1f}  "
          f"白边>60%={sum(1 for x in wh if x > 0.6)/n*100:>4.0f}%  "
          f"坏图(色<3000)={len(bad)/n*100:>4.0f}%")
    if bad:
        print(f"{'':<26} 坏图示例: {', '.join(bad[:3])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--flat", action="store_true", help="把目录下的图整体当一类")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    if not root.exists():
        print("目录不存在:", root)
        sys.exit(1)

    if args.flat:
        files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXT]
        report(root.name, files)
        return

    subs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subs:
        files = [p for p in sorted(root.glob("*")) if p.suffix.lower() in IMG_EXT]
        report(root.name, files)
        return
    for d in subs:
        files = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
        report(d.name, files)


if __name__ == "__main__":
    main()