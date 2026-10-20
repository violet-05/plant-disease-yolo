# -*- coding: utf-8 -*-
"""本地验证 crop6_s best.pt：每类精度表 + 混淆矩阵（Windows 多进程需 __main__ 保护）"""
import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="本地验证 6 类检测模型：每类精度表 + 混淆矩阵")
    config.add_device_arg(ap)
    args = ap.parse_args()

    # 字体预置到 ultralytics 配置目录（跳过联网下载）
    config.setup_cjk()

    from ultralytics import YOLO
    m = YOLO(str(config.ROOT / 'best.pt'))
    r = m.val(data=str(config.DATA / 'crop6' / 'data.yaml'), split='val', device=args.device, batch=8,
              workers=8, project=str(config.RUNS), name='val_s', exist_ok=True)

    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
    print('\n=== 每类精度 ===')
    for i, n in names.items():
        print(f'{n:22} P={r.box.p[i]:.3f}  R={r.box.r[i]:.3f}  mAP50={r.box.ap50[i]:.3f}  mAP50-95={r.box.ap[i]:.3f}')
    print(f'{"ALL":22} P={r.box.mp:.3f}  R={r.box.mr:.3f}  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}')
    print('结果目录:', r.save_dir)


if __name__ == '__main__':
    main()