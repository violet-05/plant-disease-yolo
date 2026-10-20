# -*- coding: utf-8 -*-
"""诊断单张图：打印所有候选框的原始分数 + 不同 conf 阈值下的存活情况
用法: python tools/diag_image.py <图片路径>
      （省略路径时使用 demo_pics 里的样例图）
"""
import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

CLASSES_CN = {
    0: '稻瘟病', 1: '玉米锈病', 2: '小麦条锈病',
    3: '苹果黑星病', 4: '柑橘溃疡病', 5: '葡萄霜霉病',
}


def main():
    ap = argparse.ArgumentParser(description="诊断单张图：打印所有候选框的原始分数 + 不同 conf 阈值下的存活情况")
    ap.add_argument("path", nargs="?",
                    default=str(config.ROOT / 'demo_pics' / '4苹果黑星病_1.jpg'),
                    help="待诊断的图片路径（省略则用 demo_pics 下的样例图）")
    config.add_device_arg(ap)
    args = ap.parse_args()

    src = args.path
    config.setup_cjk()

    from ultralytics import YOLO
    m = YOLO(str(config.MODELS / 'crop6_s_best.pt'))
    m.model.names = CLASSES_CN

    # conf 压到 0.01，把所有候选都捞出来
    r = m.predict(source=src, conf=0.01, device=args.device, imgsz=640, verbose=False)[0]

    print(f'\n图片: {src}   尺寸: {r.orig_shape}\n')
    print(f'{"#":>3}  {"类别":<12} {"分数":>6}   {"框(x1,y1,x2,y2)":<28} {"框面积占比":>8}')
    print('-' * 70)
    idx = []
    for i, b in enumerate(r.boxes):
        c = int(b.cls)
        s = float(b.conf)
        x1, y1, x2, y2 = [round(v) for v in b.xyxy[0].tolist()]
        H, W = r.orig_shape
        area = (x2 - x1) * (y2 - y1) / (H * W) * 100
        idx.append((s, c))
        flag = '' if s >= 0.3 else '  <- 当前 conf=0.3 会滤掉'
        print(f'{i:>3}  {CLASSES_CN[c]:<12} {s:>6.3f}   {str((x1,y1,x2,y2)):<28} {area:>7.1f}%{flag}')

    print('\n=== 不同 conf 阈值下剩几个框 ===')
    for t in (0.25, 0.3, 0.4, 0.5, 0.6, 0.7):
        keep = [c for s, c in idx if s >= t]
        detail = ' '.join(f'{CLASSES_CN[c]}' for c in keep) or '（全滤掉，画面干净）'
        print(f'  conf>={t:<4} 剩 {len(keep):>2} 框   {detail}')


if __name__ == '__main__':
    main()