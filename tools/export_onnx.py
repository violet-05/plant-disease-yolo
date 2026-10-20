# -*- coding: utf-8 -*-
"""整理模型文件并导出 ONNX（部署到 Orin/TensorRT 的前置）"""
import pathlib, shutil, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402


def main():
    models = config.MODELS
    models.mkdir(exist_ok=True)
    src = config.ROOT / 'best.pt'
    dst = models / 'crop6_s_best.pt'
    if not dst.exists():
        shutil.copy(src, dst)
    print('模型已归档:', dst)

    from ultralytics import YOLO
    m = YOLO(str(dst))
    out = m.export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False)
    print('ONNX 导出完成:', out)


if __name__ == '__main__':
    main()