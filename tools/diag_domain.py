# -*- coding: utf-8 -*-
"""诊断：量化"翻拍屏幕"这一层退化对准确率的影响

做法：
  1. 从验证集抽 N 张图，先用原图跑一遍，只保留【模型答对】的那些（作为基线）
  2. 把这些正确图做"翻拍屏幕"式退化（降分辨率回放 / 摩尔纹 / 失焦 / 偏冷偏灰 / JPEG）
  3. 再跑一遍，看有多少张从"答对"变成"答错"或"什么都没框出来"
  → 翻转率 = 退化导致的准确率损失

这样能把「模型能力问题」和「拍摄环节问题」分开。
"""
import argparse
import pathlib
import random
import shutil
import sys
import numpy as np
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

CROP6 = config.DATA / 'crop6'
VAL = CROP6 / 'images' / 'val'
LBL = CROP6 / 'labels' / 'val'
OUT = config.TMP / 'domdiag'
N = 250
CONF = 0.4
CLASSES_CN = {0: '稻瘟病', 1: '玉米锈病', 2: '小麦条锈病',
              3: '苹果黑星病', 4: '柑橘溃疡病', 5: '葡萄霜霉病'}


def gt_class(img):
    """取该图标签里的多数类作为真值"""
    lp = LBL / (img.stem + '.txt')
    if not lp.exists():
        return None
    cs = [int(l.split()[0]) for l in lp.read_text().splitlines() if len(l.split()) >= 5]
    if not cs:
        return None
    return max(set(cs), key=cs.count)


def degrade(im):
    """模拟：摄像头拍显示器"""
    h, w = im.shape[:2]
    # 1) 先降到摄像头分辨率再放大回来 —— 细节损失
    s = 640 / max(h, w)
    small = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # 2) 摩尔纹（屏幕像素栅格与传感器干涉）
    yy, xx = np.mgrid[0:h, 0:w]
    moire = 7.0 * np.sin(2 * np.pi * xx / 3.0) * np.sin(2 * np.pi * yy / 3.0)
    out = np.clip(out.astype(np.float32) + moire[..., None], 0, 255)
    # 3) 失焦模糊
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    # 4) 对比度下降 + 整体提亮 + 屏幕偏冷（BGR 的第 0 通道是蓝）
    out = out * 0.82 + 30
    out[..., 0] *= 1.07
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 5) JPEG 压缩
    ok, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 45])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description="诊断：量化『翻拍屏幕』这一层退化对准确率的影响")
    config.add_device_arg(ap)
    args = ap.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / 'orig').mkdir(parents=True)
    (OUT / 'degr').mkdir(parents=True)

    config.setup_cjk()

    # 抽样本：每个类均衡一点，否则苹果黑星病会被淹没
    by_cls = {}
    for img in sorted(VAL.glob('*')):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        g = gt_class(img)
        if g is not None:
            by_cls.setdefault(g, []).append(img)
    rng = random.Random(0)
    sample = []
    per = max(1, N // max(1, len(by_cls)))
    for g, imgs in sorted(by_cls.items()):
        rng.shuffle(imgs)
        sample += imgs[:per]
    print(f'样本 {len(sample)} 张，分布: ' +
          ', '.join(f'{CLASSES_CN[g]}={len([1 for i in sample if gt_class(i)==g])}' for g in sorted(by_cls)))

    from ultralytics import YOLO
    m = YOLO(str(config.MODELS / 'crop6_s_best.pt'))

    pairs = []  # (原图路径, 退化图路径, 真值类)
    for img in sample:
        im = cv2.imread(str(img))
        if im is None:
            continue
        d = OUT / 'orig' / img.name
        e = OUT / 'degr' / img.name
        cv2.imwrite(str(d), im)
        cv2.imwrite(str(e), degrade(im))
        pairs.append((d, e, gt_class(img)))

    def top_pred(path):
        r = m.predict(source=str(path), conf=CONF, device=args.device, imgsz=640, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        return int(r.boxes.cls[i])

    print('\n跑原图 ...')
    base = [top_pred(d) for d, _, _ in pairs]
    print('跑退化图 ...')
    deg = [top_pred(e) for _, e, _ in pairs]

    gts = [g for _, _, g in pairs]
    correct_base = [i for i in range(len(pairs)) if base[i] == gts[i]]
    n_base_ok = len(correct_base)
    still = sum(1 for i in correct_base if deg[i] == pairs[i][2])
    lost = n_base_ok - still
    nodet = sum(1 for i in correct_base if deg[i] is None)

    print('\n' + '=' * 56)
    print(f'原图上答对        : {n_base_ok:>4} / {len(pairs)}  ({n_base_ok/len(pairs)*100:.1f}%)')
    print(f'退化后仍答对      : {still:>4} / {n_base_ok}  ({still/max(1,n_base_ok)*100:.1f}%)')
    print(f'退化后答错/漏检   : {lost:>4}  其中完全没框出 {nodet}')
    print(f'→ 翻拍屏幕造成的翻转率: {lost/max(1,n_base_ok)*100:.1f}%')
    print('=' * 56)

    # 看看答错的都跑哪去了
    flips = {}
    for i in correct_base:
        if deg[i] != pairs[i][2]:
            k = f'{CLASSES_CN[pairs[i][2]]} → {CLASSES_CN[deg[i]] if deg[i] is not None else "（无检测）"}'
            flips[k] = flips.get(k, 0) + 1
    if flips:
        print('\n翻转明细:')
        for k, v in sorted(flips.items(), key=lambda x: -x[1]):
            print(f'  {k:<28} {v:>3} 次')
    print(f'\n退化图样例存于: {OUT / "degr"}')


if __name__ == '__main__':
    main()