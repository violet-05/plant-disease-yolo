# -*- coding: utf-8 -*-
"""诊断：把训练时没见过的苹果叶部病害喂给 6 类模型，看它答成什么

数据用 apple_catherine（当初因"全是整图框"被排除，没进训练集）。
其中 Apple Scab / scab 两个类是我们的 6 类之一 → 作为对照组。
其余 8 类（alternaria 斑点落叶病、marssonina 褐斑病…）模型从没见过。
"""
import argparse
import pathlib
import random
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

RAW = config.RAW / 'rf' / 'apple_catherine'
SRC_CLASSES = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'alternaria', 'anthracnose',
               'marssonina', 'scab', 'sootyblotch', 'valsacanker', 'whiterot']
OURS = {0: '稻瘟病', 1: '玉米锈病', 2: '小麦条锈病', 3: '苹果黑星病', 4: '柑橘溃疡病', 5: '葡萄霜霉病'}
PER = 60          # 每类抽多少张
CONF = 0.4


def main():
    ap = argparse.ArgumentParser(description="诊断：把训练时没见过的苹果叶部病害喂给 6 类模型，看它答成什么")
    config.add_device_arg(ap)
    args = ap.parse_args()

    config.setup_cjk()

    # 收集每类的图（valid + test，取标签里多数类）
    pool = {}
    for split in ('valid', 'test'):
        d = RAW / split
        for img in sorted((d / 'images').glob('*')):
            if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            lp = d / 'labels' / (img.stem + '.txt')
            if not lp.exists():
                continue
            cs = [int(l.split()[0]) for l in lp.read_text(errors='ignore').splitlines() if len(l.split()) >= 5]
            if not cs:
                continue
            pool.setdefault(max(set(cs), key=cs.count), []).append(img)

    rng = random.Random(0)
    from ultralytics import YOLO
    m = YOLO(str(config.MODELS / 'crop6_s_best.pt'))

    print(f'{"apple_catherine 里的真实类别":<34} {"张数":>4}   模型答成什么（Top-1 分布）')
    print('-' * 100)
    for cid, name in enumerate(SRC_CLASSES):
        imgs = pool.get(cid, [])
        if not imgs:
            continue
        rng.shuffle(imgs)
        imgs = imgs[:PER]
        cnt = Counter()
        for img in imgs:
            r = m.predict(source=str(img), conf=CONF, device=args.device, imgsz=640, verbose=False)[0]
            if len(r.boxes) == 0:
                cnt['（无检测）'] += 1
            else:
                cnt[OURS[int(r.boxes.cls[int(r.boxes.conf.argmax())])]] += 1
        top = '  '.join(f'{k}×{v}' for k, v in cnt.most_common(4))
        tag = '  ★我们的类' if cid in (0, 6) else ''
        print(f'{name:<34} {len(imgs):>4}   {top}{tag}')
    print('-' * 100)
    print('★  = 对照组：这两类本来就在训练集里，模型答对是应该的')


if __name__ == '__main__':
    main()