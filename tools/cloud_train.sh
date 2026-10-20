#!/usr/bin/env bash
# 云端（阿里云 DSW / 任意 Linux GPU 机器）训练苹果叶部病害分类模型
#
# 用法：把 apple_cls.zip、train_apple_cls.py、config.py 和本脚本传到同一目录，然后在 Terminal 里：
#     bash cloud_train.sh              # 烟雾测试 + 正式训练 m 和 s
#     bash cloud_train.sh smoke        # 只跑烟雾测试
#
# 注意：train_apple_cls.py 依赖同目录的 config.py（import config），两者缺一不可。
#
# 需要联网下载 ImageNet 预训练权重 yolo11m-cls.pt / yolo11s-cls.pt
set -e

DATA_ZIP="apple_cls.zip"
WS="$(pwd)"

echo "==================== 环境检查 ===================="
echo "[1/6] GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || {
    echo "!! 没检测到 GPU，检查实例规格"; exit 1; }

echo
echo "[2/6] Python / ultralytics:"
python -c "import sys; print('  python', sys.version.split()[0])" 2>/dev/null || { echo "!! 没有 python"; exit 1; }
# 版本必须钉死：云端和本地的 ultralytics 不一致会导致 best.pt 无法在本地加载
python -c "import ultralytics; print('  ultralytics', ultralytics.__version__)" 2>/dev/null \
  || { echo "  未安装，正在安装 ultralytics==8.4.173 ..."; pip install -q ultralytics==8.4.173; }
python - <<'EOF'
import ultralytics, sys
v = ultralytics.__version__
if v != "8.4.173":
    print(f"  !! 当前 {v}，本地用的是 8.4.173 —— 版本不一致可能导致权重在本地加载失败")
    print(f"     建议: pip install -U ultralytics==8.4.173")
else:
    print("  OK: 版本与本地一致")
EOF

echo
echo "[3/6] 解压数据集:"
if [ ! -d "apple_cls/train" ]; then
    if [ -f "$DATA_ZIP" ]; then
        unzip -q -o "$DATA_ZIP"
        echo "  已解压 $DATA_ZIP"
    else
        echo "!! 找不到 $DATA_ZIP，请先把它上传到 $WS"; exit 1
    fi
else
    echo "  apple_cls/ 已存在，跳过"
fi
echo "  类别目录:"
ls apple_cls/train | sed 's/^/    /'
NPY=$(find apple_cls/train -name '*.jpg' | wc -l)
echo "  train 图片数: $NPY"

echo
echo "[4/6] 训练脚本:"
if [ ! -f "train_apple_cls.py" ]; then
    echo "  !! 找不到 train_apple_cls.py，请和 apple_cls.zip 一起上传"; exit 1
fi
if [ ! -f "config.py" ]; then
    echo "  !! 找不到 config.py，train_apple_cls.py 需要它（import config），请一并上传"; exit 1
fi
echo "  OK"

echo
echo "==================== 开始训练 ===================="
echo "[5/6] 烟雾测试（3 epochs + 10% 数据，约 2 分钟）"
echo "      日志里必须出现 'found N images in 7 classes'；见 'requires 7 classes, not K' 就是某类缺目录"
python train_apple_cls.py --data "$WS/apple_cls" --size s --smoke --name smoke_test

if [ "$1" = "smoke" ]; then
    echo "烟雾测试完成（--smoke 模式，就此结束）"
    exit 0
fi

echo
echo "[6/6] 正式训练"
echo "--- m 尺度（主模型，精度上限高）---"
python train_apple_cls.py --data "$WS/apple_cls" --size m --name apple_cls_m

echo
echo "--- s 尺度（给本地 4GB 显卡跑演示用）---"
python train_apple_cls.py --data "$WS/apple_cls" --size s --name apple_cls_s

echo
echo "==================== 完成 ===================="
echo "下载这两个文件到本地仓库的 models/ 目录："
ls -lh runs/apple_cls_m/weights/best.pt runs/apple_cls_s/weights/best.pt 2>/dev/null || true
echo "    runs/apple_cls_m/weights/best.pt  →  models/apple_cls_m_best.pt"
echo "    runs/apple_cls_s/weights/best.pt  →  models/apple_cls_s_best.pt"
echo
echo "中文混淆矩阵在 runs/apple_cls_*/ 下（云端画的是英文标签，本地用 val_apple_cls.py 出中文版）"