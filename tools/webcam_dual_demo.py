# -*- coding: utf-8 -*-
"""双轨摄像头演示：检测画框定位病斑 + 分类说出病名（中文，含「不确定」出口）

按 q 退出，按 t 打印帧率。

三个必须遵守的点（都是踩出来的）：
  1) **分类输入必须是正方形 ROI** —— 分类的预处理是 Resize(短边) + CenterCrop
     （ultralytics/data/augment.py），非方形输入的左右/上下会被直接裁掉，
     整帧 640x480 送进去等于把叶子裁没了。所以要先裁正方形。
  2) **检测结果只画框、不画它的类名** —— crop6 检测模型那 6 个类名对苹果叶是错的
     （诊断实测：遇到没见过的苹果叶斑病 78%/89% 硬答"苹果黑星病"）。
     框表示"这里有病斑"，病名一律交给分类模型。
  3) **中文必须走 ultralytics 的 Annotator**，不能用 cv2.putText（必变方框）。
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config                                                   # noqa: E402
from class_names import cn_names_dict, THRESH, MARGIN   # noqa: E402

DET_MODEL = str(config.MODELS / "crop6_s_best.pt")   # 检测：定位病斑（类名不可信，只用它的框）
# 分类：说病名。用 m 尺度（最大数据版）——实测在 GTX 1650 上 imgsz=384 约
# 36ms/帧，配检测的 26ms 能到 ~16 FPS，演示够用。想要更高帧率再换 s 尺度。
CLS_MODEL = str(config.MODELS / "apple_cls_m3_best.pt")
CLS_IMGSZ = 384
DET_IMGSZ = 640
DET_CONF = 0.30
ROI_MARGIN = 0.15          # 检测框并集往外扩多少

# 「不确定」出口的 THRESH/MARGIN 由 class_names.py 提供（全项目唯一定义），不再本地重复

CJK = "中文"               # 只要是非 ASCII，Annotator 就会走 PIL + Unicode 字体


def square_roi(frame, boxes, margin=ROI_MARGIN):
    """把检测框的并集扩一点、裁成正方形 —— 分类预处理会 CenterCrop，非方形会被裁边"""
    h, w = frame.shape[:2]
    if boxes is None or len(boxes) == 0:
        s = min(h, w)
        return frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    xy = boxes.xyxy.cpu().numpy()
    x1, y1 = xy[:, 0].min(), xy[:, 1].min()
    x2, y2 = xy[:, 2].max(), xy[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    side = max(side, 64)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, x1 + side))
    y2 = int(min(h, y1 + side))
    # 贴边时补足正方形
    x2 = int(min(w, x1 + max(1, x2 - x1)))
    y2 = int(min(h, y1 + max(1, y2 - y1)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        s = min(h, w)
        return frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return frame[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser(description="双轨演示：检测圈病斑 + 分类说病名")
    config.add_cam_arg(ap)
    config.add_device_arg(ap)
    args = ap.parse_args()

    # 中文渲染 shim，必须在 import ultralytics 之前
    config.setup_cjk()
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator

    if not pathlib.Path(CLS_MODEL).exists():
        print(f"!! 分类模型不存在: {CLS_MODEL}")
        print("   先跑 tools/train_apple_cls.py --size s 训练，再执行演示")
        return
    if not pathlib.Path(DET_MODEL).exists():
        print(f"!! 检测模型不存在: {DET_MODEL}")
        return

    cn = cn_names_dict()                  # {索引: 中文}
    print(f"[分类模型] {CLS_MODEL}  {len(cn)} 类: {', '.join(cn.values())}")
    m_det = YOLO(DET_MODEL)
    m_cls = YOLO(CLS_MODEL)
    m_cls.model.names = cn                # 分类模型上 names 一样是只读属性，改底层

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("摄像头打开失败：没找到摄像头，或被其他程序占用（QQ/微信/钉钉视频通话都会占用）")
        return
    print("启动中…… 窗口内按 q 退出，按 t 打印帧率")

    fps_t, fps_n, fps = time.time(), 0, 0.0
    cls_text, cls_color, top3 = "对准叶片……", (200, 200, 200), []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读不到画面，退出")
            break

        r_det = m_det.predict(frame, imgsz=DET_IMGSZ, conf=DET_CONF, device=args.device, verbose=False)[0]
        roi = square_roi(frame, r_det.boxes)
        try:
            r_cls = m_cls.predict(roi, imgsz=CLS_IMGSZ, device=args.device, verbose=False)[0]
            if r_cls.probs is not None:
                probs = r_cls.probs.data.cpu().numpy()
                order = probs.argsort()[::-1]
                top3 = [(i, float(probs[i])) for i in order[:3]]
                i1, p1 = top3[0]
                p2 = top3[1][1] if len(top3) > 1 else 0.0
                if p1 < THRESH or (p1 - p2) < MARGIN:
                    cls_text, cls_color = "不确定（可能是未见过的病害）", (0, 200, 255)
                else:
                    cls_text, cls_color = f"{cn[i1]}  {p1:.0%}", (0, 255, 0)
        except Exception as e:
            cls_text, cls_color = f"分类失败: {type(e).__name__}", (0, 0, 255)

        # ---- 渲染：中文必须走 Annotator（cv2.putText 会变方框）----
        ann = Annotator(frame, line_width=2, example=CJK)
        for b in r_det.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            ann.box_label([x1, y1, x2, y2], "")            # 只画框，不画检测模型自己的类名
        h, w = frame.shape[:2]
        ann.rectangle([0, 0, w, 30], fill=(0, 0, 0))
        # 两个坑：参数名是 txt_color 不是 color；xy 必须是 **list**
        # （text() 的 PIL 分支会在循环里对 xy 原地做 xy[1] += h，传 tuple 会
        #  报 "'tuple' object does not support item assignment"）
        ann.text([8, 8], cls_text, txt_color=cls_color)
        y = h - 12
        for i, p in top3[1:]:
            ann.text([8, y], f"{cn[i]}  {p:.0%}", txt_color=(180, 180, 180))
            y -= 22
        # Annotator.result() 返回的数组是**只读**的，后面 cv2.putText 要写进去会报
# "img marked as output argument, but provided NumPy array marked as readonly"，所以要拷贝
        frame = np.copy(ann.result())

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n / (time.time() - fps_t)
            fps_t, fps_n = time.time(), 0
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO 双轨：检测 + 分类（q 退出 / t 打印帧率）", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("t"):
            print(f"当前帧率 {fps:.1f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()