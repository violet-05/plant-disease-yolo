# -*- coding: utf-8 -*-
"""本地摄像头实时检测演示：对着病害图片拍 → 实时框出 6 类病虫害（中文标签，窗口内按 q 退出）

注意：这是**旧一代 6 类检测模型**的演示。它能框出病斑位置，但那 6 个类名对苹果叶不可信
（实测遇到没见过的叶斑病 78%/89% 硬答"苹果黑星病"）。要看带「不确定」出口的分类模型，
用 tools/webcam_dual_demo.py。
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

# 类别英文名 → 中文名，顺序必须和训练时的 data.yaml 一致
CLASSES_CN = {
    0: '稻瘟病',
    1: '玉米锈病',
    2: '小麦条锈病',
    3: '苹果黑星病',
    4: '柑橘溃疡病',
    5: '葡萄霜霉病',
}


def main():
    ap = argparse.ArgumentParser(description="旧 6 类检测模型的摄像头演示")
    config.add_cam_arg(ap)
    config.add_device_arg(ap)
    args = ap.parse_args()

    # 中文标签会触发 ultralytics 走 PIL 绘制，并去找 Arial.Unicode.ttf；
    # 找不到就会联网下载，所以这里先用系统中文字体顶上。
    config.setup_cjk()

    import cv2
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('摄像头打开失败：没找到摄像头，或被其他程序占用（QQ/微信/钉钉的视频通话都会占用摄像头）')
        return
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    print(f'摄像头正常: {int(w)}x{int(h)}，启动实时检测窗口，按 q 退出...')

    from ultralytics import YOLO
    model_path = config.MODELS / 'crop6_s_best.pt'
    if not pathlib.Path(model_path).exists():
        print(f"!! 检测模型不存在: {model_path}")
        print("   见 README 的模型权重下载说明，或自行训练")
        return
    m = YOLO(str(model_path))
    m.model.names = CLASSES_CN  # names 是只读属性，只能改底层模型上的那份
    m.predict(source=args.cam, show=True, device=args.device, conf=0.3, imgsz=640)


if __name__ == '__main__':
    main()