#!/usr/bin/env bash
# 云端训练 + 自动续训
#
# 为什么需要它：阿里云 DSW 会随机杀训练进程（实测死在 9 / 53 / 123 轮），
# 每次都从头跑就永远跑不完。ultralytics 每轮都会存 last.pt，
# 所以被中断后用 --resume 接着跑，反复重试直到真正跑完。
#
# 用法:
#   bash cloud_train_resume.sh              # m 尺度
#   bash cloud_train_resume.sh s            # s 尺度
#   bash cloud_train_resume.sh m apple_cls_m3 /root/apple_cls
#
# 建议用 nohup 挂后台:
#   nohup bash cloud_train_resume.sh > /dev/null 2>&1 &
#
# 注意：train_apple_cls.py 依赖同目录的 config.py（import config），两者都要在 cwd 下。

SIZE="${1:-m}"
NAME="${2:-apple_cls_${SIZE}_r}"
DATA="${3:-/root/apple_cls}"
LOG="train_${NAME}.log"
MAX_TRY=40

echo "尺度=$SIZE  名字=$NAME  数据=$DATA  日志=$LOG" | tee -a "$LOG"

# 环境自检 —— 必须放在循环外面。
# 踩过的坑：新实例没装 ultralytics，脚本在循环里秒崩秒重试，
# 刷出 28 次"尝试"却一次训练都没进去，看起来像"平台随机杀进程"，其实是依赖缺失。
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "!! 没装 ultralytics，正在安装 ..." | tee -a "$LOG"
    pip install -q ultralytics==8.4.173 || {
        echo "!! 安装失败，请手动: pip install ultralytics==8.4.173" | tee -a "$LOG"; exit 1; }
fi
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "!! torch 不可用或没有 CUDA，训练会退化到 CPU。先检查环境再重跑。" | tee -a "$LOG"
    exit 1
fi
python -c "import ultralytics,torch; print('环境 OK: ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" | tee -a "$LOG"
if [ ! -f "train_apple_cls.py" ]; then
    echo "!! 当前目录找不到 train_apple_cls.py，请把它和 config.py 一起放到 cwd" | tee -a "$LOG"; exit 1
fi
if [ ! -f "config.py" ]; then
    echo "!! 当前目录找不到 config.py，train_apple_cls.py 需要它（import config），请一并上传" | tee -a "$LOG"; exit 1
fi
if [ ! -d "$DATA/train" ]; then
    echo "!! 数据目录不对: $DATA/train 不存在" | tee -a "$LOG"; exit 1
fi

for i in $(seq 1 $MAX_TRY); do
    echo "===== 第 $i 次尝试  $(date '+%m-%d %H:%M:%S') =====" >> "$LOG"

    if [ "$i" -eq 1 ]; then
        python train_apple_cls.py --data "$DATA" --size "$SIZE" --name "$NAME" >> "$LOG" 2>&1
    else
        python train_apple_cls.py --data "$DATA" --size "$SIZE" --name "$NAME" --resume >> "$LOG" 2>&1
    fi

    # 跑完的判据：出现收尾行
    if tr '\r' '\n' < "$LOG" | grep -qE "epochs completed|EarlyStopping"; then
        echo "===== 训练完成  $(date '+%m-%d %H:%M:%S') =====" | tee -a "$LOG"
        echo
        echo "最后结果:"
        tr '\r' '\n' < "$LOG" | grep -E "^\s+all\s" | tail -3
        ls -lh runs/classify/runs/"$NAME"/weights/best.pt 2>/dev/null
        exit 0
    fi

    echo "===== 被中断，5 秒后续训 =====" >> "$LOG"
    sleep 5
done

echo "===== 试了 $MAX_TRY 次仍未跑完，放弃 =====" | tee -a "$LOG"
exit 1