# plant-disease-yolo

[English](README.md) | **简体中文**

用 YOLO 做**作物叶部病害分类**：从公开数据集清洗到训练、评估、摄像头实时演示的完整流水线。

> Crop leaf disease classification with YOLO — data cleaning, training, evaluation and a live webcam demo.

支持 **17 类**：苹果 7 类 + 玉米 4 类 + 柑橘 4 类 + 稻瘟病 + 其他作物叶片。

---

## 这个项目解决什么问题

起因很具体：拿摄像头拍一片标着「叶斑病」的叶子，模型却信誓旦旦地答「苹果黑星病」。

诊断后发现根因不是模型太弱，而是**目标检测器没有「我不认识」这个出口**——它必须从给定的 N 类里挑一个最像的。三个对照实验：

| 诊断项 | 结果 | 结论 |
|---|---|---|
| 翻拍屏幕造成的画质损失 | 翻转率仅 7.6% | 不是主因 |
| 已知 6 类的认病能力 | Top-1 91.5% | 其实不差 |
| **遇到没见过的病害** | **78%/89% 硬答成「苹果黑星病」** | **病根** |

所以本项目做了两件事：

1. **把任务从「检测」改成「整叶分类」**——分类器更容易学，也更匹配「对着叶子拍 → 这是什么病」的用法
2. **加入显式的拒绝类**（`其他苹果病害` / `其他作物叶片`）——让模型能说「这是苹果病害，但不在我认识的几种里」，而不是硬猜

效果：

| | 旧的检测模型 | 本项目的分类模型 |
|---|---|---|
| 遇到训练集外的病害 | **32% 硬答「苹果黑星病」** | **81~88% 正确拒绝** |
| 能否说「这不是苹果叶」 | 不能 | 能 |
| 能否说「这是健康叶」 | 不能 | 能 |

---

## 快速开始

```bash
pip install ultralytics==8.4.173    # 版本要钉死，否则权重加载会出问题
```

### 1. 准备数据

数据集不入库（体积大、且各有第三方许可）。各来源见 `tools/download_roboflow.py` 与 `tools/harvest_apple.py`，放到 `datasets/raw/` 下。

### 2. 巡检数据质量

**这一步别跳。** 公开数据集的质量问题比想象中严重（见下文「数据集质量陷阱」），先跑一遍：

```bash
python tools/qa_image_quality.py datasets/raw/<某个源>/
```

### 3. 构建数据集

```bash
python tools/build_apple_cls.py
```

会输出 `datasets/apple_cls/{train,val,holdout}/`，并打包成 zip 便于上传云端。

### 4. 训练

```bash
python tools/train_apple_cls.py --size m                 # 本地
bash tools/cloud_train_resume.sh m <run名> <数据目录>     # 云端（带断点续训）
```

### 5. 评估

```bash
python tools/val_apple_cls.py models/<你的模型>.pt    # 每类召回 + 中文混淆矩阵
python tools/probe_cls_unknown.py models/<你的模型>.pt # 未知病害拒绝率验收
```

### 6. 摄像头演示

```bash
python tools/webcam_dual_demo.py
```

双轨显示：检测模型圈出病斑位置，分类模型说出病名，低置信度时显示「不确定」。

---

## 目录结构

```
tools/
├── search_roboflow_apple.py   # 在 Roboflow Universe 搜数据集（自动甄别果实/叶片）
├── harvest_apple.py           # 批量下载（按项目类型自动选导出格式）
├── qa_image_quality.py        # 图像质量巡检：坏图 / 商品图 / 果实
├── build_apple_cls.py         # 数据集构建：去增广 / 防泄漏 / 配平 / 质检 / 打包
├── class_names.py             # 类别映射（顺序即索引，单一来源）
├── train_apple_cls.py         # 训练
├── val_apple_cls.py           # 每类召回 + 中文混淆矩阵 + 阈值标定
├── probe_cls_unknown.py       # 未知病害拒绝率验收
├── webcam_dual_demo.py        # 双轨摄像头演示
├── cloud_train_resume.sh      # 云端训练（环境自检 + 断点续训）
└── diag_*.py                  # 诊断脚本（本项目的结论都出自这三个）
```

---

## 数据集质量陷阱

**这是本项目最值得看的部分。** 公开的植物病害数据集坑非常多，我们几乎每一个都踩过：

### 1. 标称图片数是假的

`apple_catherine` 号称 26412 张，去掉 `.rf.<hash>` 增广副本后**只有 1586 张唯一原图**。所有基于图片数的判断都会失准。

### 2. 官方划分本身有泄漏

同一个原图的增广副本被分到 train 和 valid 两边（实测 train∩valid = 351 张原图）。**沿用官方划分，val 会虚高 10~20 个百分点**。必须按原图分组重划。

我们实测过代价：同一批数据，按来源划分时 top1 = **51.4%**，按图片随机划分时 = **98.8%**——后者是假的。

### 3. 增广可能是破坏性的

`apple_catherine` 的 `Apple Scab` / `Black Rot` / `Cedar Apple Rust` 三类被施加了**像素化增广**，整类作废（阈值收紧后，250 张原图只剩 1 张可用）。

判据：把图用 NEAREST 压到 ≤128 网格再数唯一色数（尺寸归一化，否则小图会被误杀）。实测真照片 ≥5000 种，坏图 ≤1500 种。

### 4. 「苹果病害」数据集多半在拍果实

- `apple_catherine` 的 10 个类里，一半是**果实病和枝干病**，不是叶部
- `apple-1` 里有个类叫 `Marssonia blotch`，名字和我们褐斑病几乎一样，但**全是果实**
- `citrus_research` 有 45~58% 是白底橙子

**必须肉眼验收，不能只看类名。** 我们每轮构建后都把每类抽 8~9 张拼成一张图看一遍——这是唯一能抓住「类名合错」「果实混叶片」的手段。

部分可自动化的判据：绿像素占比。柑橘「健康」类中位仅 0.07（果实），「黄龙病」0.46（叶片），阈值 0.12 能滤掉大部分果实。

### 5. 网络爬取的数据集噪声大

PlantDoc 的 `Apple leaf` 类有 60% 是带水印的白底商品图；但它的病害类（Apple Scab Leaf / Apple rust leaf / Corn 系列）质量可用。

### 6. 病灶裁剪 ≠ 整叶

`plantseg` 的病灶框裁剪中位只有 43px，且构图与整叶完全不符，做分类不可用（79% 的裁剪小于 96px）。

---

## 已知短板

诚实记录，不粉饰：

- **`苹果黑星病` 最弱**（同域召回 54.5%）——只有 98 张训练图，是全表最薄的类
- **类别越多，拒绝能力越弱**：8 类时未知病害正确拒绝率 88%，扩到 17 类后降到 81%。选项多了，「硬猜一个」的诱惑就大（用 `tools/probe_cls_unknown.py` 在 26 张苹果花叶病留出集上测得）
- **同域 val 的数字不可信**：仓库里 0.936 那个成绩含划分泄漏。**诚实数字是 8 类版本的跨来源 51.4%**
- **柑橘类混入约 20% 果实**（绿色过滤只挡住了一部分）
- 演示脚本依赖摄像头，只能本地跑

---

## 环境

- ultralytics **8.4.173**（版本必须与训练时一致，否则 `best.pt` 可能加载失败）
- PyTorch（GPU 版）
- 本地演示在 GTX 1650 4GB 上可跑（检测 imgsz=640 + 分类 imgsz=384 约 16 FPS）

## 许可

代码采用 [MIT 许可证](LICENSE)。

**数据集不在本仓库内**，各来源有其各自的许可（多为 CC BY 4.0），使用时请遵守原数据集的要求。

模型权重见配套仓库：[plant-disease-yolo-models](https://github.com/violet-05/plant-disease-yolo-models)