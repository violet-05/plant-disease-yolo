# -*- coding: utf-8 -*-
"""构建苹果叶部病害分类数据集（Phase 1，本方案核心）

把多个异构来源统一成 ultralytics 分类格式：
    apple_cls/{train,val,test}/<目录名>/*.jpg

四条关键规则（都是实测踩出来的）：
  1) **按原图分组重划划分**——apple_catherine 的官方 split 有泄漏
     （train∩valid=351 张原图重复），沿用会让验证准确率虚高
  2) **val/test 每个原图只留 1 个变体**——否则"图数"是假的，指标没意义
  3) **构建期配平**——ultralytics 分类任务没有类别权重（loss 是裸 cross_entropy）
  4) 中文名只出现在显示层，目录名一律 ASCII + 数字前缀（见 class_names.py 的说明）
"""
import csv
import hashlib
import json
import pathlib
import random
import shutil
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config                                               # noqa: E402
from class_names import CLS_DIRS, DIR2CN, write_class_map   # noqa: E402

# 路径统一由 tools/config.py 解析（环境变量 PLANT_DISEASE_ROOT 可覆盖）
RAW = config.RAW
OUT = config.CLS_DATA
SAMPLES = config.SAMPLES
REPORT = config.DATA / "report_build_cls.txt"
PACK = config.TMP / "apple_cls_pack"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 512            # 统一缩到长边 ≤512，压低体积便于上传云端
JPEG_Q = 85

# 配平参数：每类原图上限、每类目标样本数、单原图最多取几个变体
CAP_ORIG = 300
TARGET = 600
MAX_VAR = 6

# 图像质量阈值：唯一颜色数低于此值视为坏图。
# 实测分布（按原图，子采样 160x160 计唯一色）：
# 判据见 quality_ok()：先在 NEAREST 归一化到 <=128 的网格上数唯一色。
#   毁掉的 Apple Scab / Black Rot / Cedar Apple Rust —— 991 / 1436 / 1026
#   真照片（含下载的小图）                        —— 5168 ~ 10188
# 3000 落在两者之间的空白区，余量充足。
MIN_UNIQUE_COLORS = 3000
# 柑橘专用：绿色占比低于此值判为果实照片（实测「健康」类中位仅 0.07）
CITRUS_MIN_GREEN = 0.12
# 白边占比高于此值视为白底商品图（PlantDoc 大量此类）
MAX_WHITE_BORDER = 0.6

# 是否启用 plantseg 病灶裁剪。实测 79% 的裁剪 <96px（中位仅 43px），
# 且是紧贴病灶的局部图，与"整叶"分类的尺度不符 —— 默认关闭。
USE_PLANTSEG = False

# 划分模式：
#   "group"  —— 按**原图**分组划分（所有来源都尽量进训练集）。**最终交付模型用这个**：
#               数据全用上，val 只做早停判断。
#   "source" —— 按**来源**整体留出。测的是"迁移到全新数据集"的能力，比演示场景苛刻得多，
#               而且会从训练集里挖走一整个来源（实测 alternaria 从 350 张掉到 126 张）。
#               诊断用，已测得诚实泛化 top1 = 51.4%，不需要每次重跑。
SPLIT_MODE = "group"

VAL_RATIO, TEST_RATIO = 0.10, 0.10
VAL_CAP, TEST_CAP = 60, 60     # val/test 每个类最多留多少张（跨来源留出时可能一整源都进来）
SEED = 0

# 类内「按来源限幅」。07_apple_other 混了两类东西：
#   catherine 的果实/树皮病（~424 张原图）—— 教"不是那 6 种病"
#   plantseg 的 apple mosaic virus（118 张）—— 教"未知的叶部病害"
# 演示是对着**叶子**拍的，所以必须压住果实/树皮那部分，
# 否则 mosaic 只占 22%、会被淹没，"未知叶部病害 → 其他"学不出来。
SRC_CAP = {
    "06_apple_other": {"catherine": 150},
}

# 来源「按比例切分」覆盖。默认 split_by_source 会把整个来源留作 val，
# 但对 07_apple_other 这行不通：mosaic virus 是**唯一**一类"未知叶部病害"样本，
# 整源留作 val 就意味着训练集里一张都没有，"未知叶部病害 → 其他"这个出口根本学不出来。
# 所以按比例切：多数进 train 让它学到，少数留 val。
SRC_SPLIT = {
    "06_apple_other": {"plantseg-mosaic": 0.25},
}

# 真·留出集：完全不参与训练、也不进 val，单独写到 <root>/holdout/。
# ultralytics 只扫 train/val/test，不会碰 holdout，所以这批图是干净的"未知病害"探针。
# 没有它的话，mosaic 进了训练集，probe_cls_unknown.py 就变成"考训练过的题"，验收无效。
HOLDOUT = {
    "06_apple_other": {"plantseg-mosaic": 0.25},
}

# ---- 各来源的类名映射 ----

# apple_catherine —— 只取经质检确认干净的两个类。
#
# 被排除的类及原因（都做过肉眼 + 定量验收）：
#   'Apple Scab' / 'Black Rot' / 'Cedar Apple Rust'
#       —— 被 Roboflow 施加了**破坏性增广**（像素化）。归一化唯一色数实测：
#          Cedar Apple Rust 991 / Apple Scab 1026 / Black Rot 1436，
#          而真照片 5168~10188，两者不重叠。放宽阈值也救不回来：
#          阈值 10000 时 250 张原图只剩 1 张。
#          **这三个类现在都有干净的下载源（见 DOWNLOAD_MAP），不需要抢救。**
#   'anthracnose' / 'whiterot' / 'sootyblotch' —— 全是**果实**病害
#   'valsacanker'                              —— 全是**树皮**病害
#   （与"叶部"任务不符）
#
#   'scab' —— 图像质量确实干净（98% 通过质检），但**肉眼验收确认 100% 是果实**
#             （苹果果皮上的黑星病斑）。按"严格叶部"方针剔除。
#
# 保留的两个类：实测 89%~99% 原图干净，且肉眼确认是**叶片**。
CATHERINE_MAP = {
    "alternaria": "04_apple_alternaria",
    "marssonina": "05_apple_marssonina",
    # 下面四类图**本身是干净的**（质检 84%~98% 通过），只是内容为果实/树皮。
    # 不丢，改归入「其他苹果病害」——给模型一个显式的拒绝出口：
    # 学到「这些是苹果病害，但不在我细分的那 6 种叶部病害里」。
    # 这比靠置信度阈值事后拒答可靠得多（实测阈值拒答对未知病害只有 8% 生效）。
    "anthracnose": "06_apple_other",    # 果实炭疽
    "sootyblotch": "06_apple_other",    # 果实煤污
    "whiterot": "06_apple_other",       # 果实白腐
    "valsacanker": "06_apple_other",    # 枝干溃疡
}

# plantdoc：它的 'Apple leaf'（健康叶）实测 60% 是带水印的白底商品图，
# 剩下的是果园果实照，**不可用**，故不映射；健康叶改由新下载数据提供。
PLANTDOC_APPLE = {
    "Apple rust leaf": "03_apple_cedar_rust",
    "Apple Scab Leaf": "01_apple_scab",
    # PlantDoc 的玉米叶类正好对应我们的玉米三个病，并进对应类。
    # 价值在于 PlantDoc 是**叶片特写**，而 corn_aziz 是田间整株照 —— 混着用能缓解构图单一。
    "Corn leaf blight": "08_corn_blight",
    "Corn rust leaf": "09_corn_rust",
    "Corn Gray leaf spot": "10_corn_gray_spot",
}
PLANTDOC_OTHER_LEAF = {
    "Blueberry leaf", "Peach leaf", "Raspberry leaf", "Strawberry leaf",
    "Tomato leaf", "Bell_pepper leaf", "Soyabean leaf", "Cherry leaf", "grape leaf",
    "Tomato leaf yellow virus", "Tomato Septoria leaf spot",
    "Potato leaf early blight", "Tomato mold leaf", "Tomato leaf bacterial spot",
    "Squash Powdery mildew leaf", "Bell_pepper leaf spot", "Potato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf late blight", "Tomato Early blight leaf",
    "grape leaf black rot",
    # 'Potato leaf'(3 张) 和 'Tomato two spotted spider mites leaf'(2 张) 样本太少，已剔除
}

# ---- 新扩展的三个作物源 ----
# 玉米 corn_aziz：框面积 87~100%，接近整图；质检 0% 坏图，实测干净
CORN_MAP = {
    "health": "07_corn_healthy",
    "blight": "08_corn_blight",
    "common_rust": "09_corn_rust",
    "gray_spot": "10_corn_gray_spot",
}
# 柑橘 citrus_research：只有 train 一个 split；框面积 4.9~57%，是真实的病斑框；
# 质检 9% 坏图（会被质量过滤挡掉）
CITRUS_MAP = {
    "Healthy": "11_citrus_healthy",
    "Canker": "12_citrus_canker",
    "Blackspot": "13_citrus_blackspot",
    "Greening": "14_citrus_greening",
}
# 稻瘟 bingban：单类，标签是多边形分割，图片是**白底上的水稻叶片标本**（1024x1024 切片）。
# 白边占比 88% —— 是标本留白，不是商品图，所以这个源要**关掉白边过滤**，
# 并且裁掉白边让它更接近其它叶片照（见 WHITE_MARGIN_CROP 的用法）。
_BINGBAN = RAW / "bingban" / "87_bingban_tiles_v3"
BINGBAN_ROOTS = [_BINGBAN / s for s in ("train", "val", "test")]

# 新下载数据（datasets/raw/apple_new/*）的类名映射。
# 每个名字都经过拼图肉眼验收，确认为**叶片**才映射。
# 特别注意排除项：
#   apple-1 的 'Marssonia blotch' / 'Anthracnose' 和 'apple' —— 验收确认全是**果实**
#   （名称与我们的褐斑病极像，但绝不能映射，否则整类污染）
DOWNLOAD_MAP = {
    "Apple-Cedar-Rust": "03_apple_cedar_rust",
    "cedar_apple_rust": "03_apple_cedar_rust",
    "Apple_black_rot": "02_apple_black_rot",
    "black_rot": "02_apple_black_rot",
    "apple_scab": "01_apple_scab",
    "Apple_healthy_leaves": "00_apple_healthy",
    "healthy_apple": "00_apple_healthy",
    "healthy": "00_apple_healthy",
    "Apple-Healthy": "00_apple_healthy",
    "Healthy leaf": "00_apple_healthy",
    "alternaria": "04_apple_alternaria",
    "Rust": "03_apple_cedar_rust",      # 苹果叶数据集里的 Rust 即苹果锈病
    # apple-leaf-w23si 的 Brown spot：肉眼确认是"黄化叶片 + 深褐色斑块"，
    # 与 catherine 的 marssonina（褐斑病）样本形态完全一致，故归入褐斑病。
    "Brown spot": "05_apple_marssonina",
    # 故意不映射：
    #   'Apple-Srab'（apple-scab-detection 数据集，注意作者把 Scab 拼成了 Srab）
    #       —— 肉眼验收确认**100% 是果实/果园照**，按"严格叶部"方针剔除
}

# plantseg：只取苹果病灶框裁图（apple mosaic virus 故意不取，留作「未知」验收探针）
PLANTSEG_APPLE = {
    "apple scab": "01_apple_scab",
    "apple rust": "03_apple_cedar_rust",
    "apple black rot": "02_apple_black_rot",
}

records = []   # (cls_dir, group_key, path, src, crop)  crop=None 表示整图


# ---------- 通用 ----------

def _add(cls_dir, group, path, src, crop=None):
    # group 统一带上来源前缀：划分时要能按「来源」整体留出，
    # 否则同一拍摄批次的图会同时出现在 train 和 val，val 成绩就是假的
    # （实测：这样出来的 val top1=98.8%，但拿没见过的病害一试就露馅）
    records.append((cls_dir, f"{src}|{group}", str(path), src, crop))


def _stem_group(name):
    """apple_catherine 的文件名形如 100_JPG.rf.<hash>.jpg，去掉 .rf.<hash> 即原图 key"""
    return name.split(".rf.")[0]


def load_yolo_names(root):
    """读 roboflow data.yaml 的 names，返回 {类号: 类名}"""
    import yaml
    for p in list(root.glob("*.yaml")) + list(root.rglob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
            ns = d.get("names") if d else None
            if isinstance(ns, dict):
                return {int(k): str(v).strip() for k, v in ns.items()}
            if isinstance(ns, list):
                return {i: str(v).strip() for i, v in enumerate(ns)}
        except Exception:
            continue
    return {}


def parse_label(lp):
    """YOLO label → [(cls, cx, cy, w, h)]，兼容多边形（分割）标注"""
    out = []
    for line in lp.read_text(errors="ignore").splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        v = [float(x) for x in t[1:]]
        if len(v) == 4:
            out.append((c, *v))
        else:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


# ---------- 各来源 ----------

def load_catherine():
    root = RAW / "rf" / "apple_catherine"
    if not root.exists():
        print("  [跳过] apple_catherine 不存在")
        return
    names = load_yolo_names(root)
    n0 = len(records)
    for split in ("train", "valid", "test"):
        d = root / split
        if not (d / "images").exists():
            continue
        for img in sorted((d / "images").glob("*")):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lp = d / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            boxes = parse_label(lp)
            if not boxes:
                continue
            cs = [b[0] for b in boxes]
            cid = max(set(cs), key=cs.count)
            cls = CATHERINE_MAP.get(names.get(cid, ""))
            if cls:
                _add(cls, _stem_group(img.name), img, "catherine")
    print(f"  apple_catherine: +{len(records)-n0} 张（含增广变体）")


def load_plantdoc():
    root = RAW / "plantdoc"
    if not root.exists():
        print("  [跳过] plantdoc 不存在")
        return
    n0 = len(records)
    for csvf, sub in [("train_labels.csv", "TRAIN"), ("test_labels.csv", "TEST")]:
        p = root / csvf
        if not p.exists():
            continue
        # 必须用 csv 模块：文件名里有带逗号的引号形式，str.split(',') 会错位
        per = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            k = (r.get("class") or "").strip()
            if k not in PLANTDOC_APPLE and k not in PLANTDOC_OTHER_LEAF:
                continue
            per.setdefault((r.get("filename") or "").strip(), []).append(k)
        for fn, ks in per.items():
            if not fn:
                continue
            # 文件名可能带 ?查询串；%20 是磁盘上的字面量，不要 urldecode
            cand = root / sub / fn.split("?")[0]
            if not cand.exists():
                continue
            k = max(set(ks), key=ks.count)
            cls = PLANTDOC_APPLE.get(k) or "16_other_leaf"
            _add(cls, "pd:" + fn.split("?")[0], cand, "plantdoc")
    print(f"  plantdoc: +{len(records)-n0} 张")


def load_plantseg_mosaic():
    """只取 plantseg 的 apple mosaic virus（苹果花叶病）→ 其他苹果病害。

    这是**唯一**一类"我们没细分、且是叶部"的苹果病害数据，是「其他」出口的关键样本。
    故意**不裁框**、用整图：plantseg 的裁剪中位只有 43~192px 太小，
    而"其他"类要学的是整叶形态，不是病斑特写。
    """
    root = RAW / "rf" / "plantseg"
    if not root.exists():
        print("  [跳过] plantseg 不存在")
        return
    names = load_yolo_names(root)
    cid = next((i for i, n in names.items() if "mosaic" in n), None)
    if cid is None:
        print("  [跳过] plantseg 里找不到 apple mosaic virus")
        return
    n0 = len(records)
    for split in ("train", "valid", "test"):
        d = root / split
        if not (d / "images").exists():
            continue
        for lp in sorted((d / "labels").glob("*.txt")):
            cs = [int(float(t.split()[0])) for t in lp.read_text(errors="ignore").splitlines() if len(t.split()) >= 5]
            if not cs or cid not in cs:
                continue
            img = None
            for e in IMG_EXT:
                if (d / "images" / (lp.stem + e)).exists():
                    img = d / "images" / (lp.stem + e)
                    break
            if img:
                _add("06_apple_other", f"ps-mosaic:{lp.stem}", img, "plantseg-mosaic")
    print(f"  plantseg mosaic(整图): +{len(records)-n0} 张 → 其他苹果病害")


def load_plantseg():
    if not USE_PLANTSEG:
        print("  [跳过] plantseg 病灶裁剪（USE_PLANTSEG=False：裁剪中位仅 43px，与整叶尺度不符）")
        return
    root = RAW / "rf" / "plantseg"
    if not root.exists():
        print("  [跳过] plantseg 不存在")
        return
    names = load_yolo_names(root)
    want = {cid: PLANTSEG_APPLE[nm] for cid, nm in names.items() if nm in PLANTSEG_APPLE}
    n0 = len(records)
    for split in ("train", "valid", "test"):
        d = root / split
        if not (d / "images").exists():
            continue
        for lp in sorted((d / "labels").glob("*.txt")):
            img = d / "images" / (lp.stem + ".jpg")
            if not img.exists():
                for e in IMG_EXT:
                    if (d / "images" / (lp.stem + e)).exists():
                        img = d / "images" / (lp.stem + e)
                        break
            if not img.exists():
                continue
            for i, (c, cx, cy, w, h) in enumerate(parse_label(lp)):
                if c in want:
                    # 每张裁出来的图都是独立样本（无增广变体），group 带序号
                    _add(want[c], f"ps:{lp.stem}#{i}", img, "plantseg",
                         crop=(cx, cy, w, h))
    print(f"  plantseg(裁框): +{len(records)-n0} 个病灶裁剪")


def load_downloaded():
    """datasets/raw/apple_new/* ：自动识别『分类目录树』与『检测 images/labels』两种结构"""
    root = RAW / "apple_new"
    if not root.exists():
        print("  [跳过] apple_new 不存在（Phase 0 还没下载）")
        return
    n0 = len(records)
    for proj in sorted(p for p in root.iterdir() if p.is_dir()):
        # 找展开后的真实根（zip 里常多套一层）
        base = proj
        for _ in range(2):
            subs = [p for p in base.iterdir() if p.is_dir()]
            if len(subs) == 1 and not any(
                    (base / s).exists() for s in ("train", "valid", "test", "val")):
                base = subs[0]
            else:
                break

        names = load_yolo_names(base)
        mapped = {cid: DOWNLOAD_MAP.get(nm) for cid, nm in names.items()}
        mapped = {k: v for k, v in mapped.items() if v}

        # A. 分类目录树： <split>/<类名>/*.jpg
        # 必须排除 images/labels —— 检测格式的 train/images 父目录也叫 train，
        # 不排除就会把检测数据集误判成分类结构，然后 continue 掉，整份数据被静默跳过。
        cls_dirs = [p for p in base.rglob("*")
                    if p.is_dir() and p.parent.name in ("train", "valid", "test", "val")
                    and p.name not in ("images", "labels")]
        if cls_dirs:
            for cd in cls_dirs:
                leaf = cd.name.strip()
                tgt = DOWNLOAD_MAP.get(leaf)
                if not tgt and leaf in PLANTDOC_OTHER_LEAF:
                    tgt = "16_other_leaf"
                if not tgt:
                    continue
                for img in cd.glob("*"):
                    if img.suffix.lower() in IMG_EXT:
                        _add(tgt, f"{proj.name}:{img.name}", img, proj.name)
            continue

        # B. 检测结构： <split>/{images,labels}
        for split in ("train", "valid", "test", "val"):
            d = base / split
            if not (d / "images").exists():
                continue
            for img in sorted((d / "images").glob("*")):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lp = d / "labels" / (img.stem + ".txt")
                if not lp.exists():
                    continue
                boxes = parse_label(lp)
                if not boxes:
                    continue
                cs = [b[0] for b in boxes]
                cid = max(set(cs), key=cs.count)
                tgt = mapped.get(cid)
                if tgt:
                    _add(tgt, f"{proj.name}:{_stem_group(img.name)}", img, proj.name)
    print(f"  apple_new(下载): +{len(records)-n0} 张")


def _load_det_source(root, cmap, tag, splits=("train", "valid", "test")):
    """通用：读取 roboflow 检测格式的源，按每张图的多数类归入目标类"""
    if not root.exists():
        print(f"  [跳过] {tag} 不存在")
        return
    names = load_yolo_names(root)
    want = {cid: cmap[nm] for cid, nm in names.items() if nm in cmap}
    if not want:
        print(f"  [跳过] {tag}: 类名对不上，检查 cmap")
        return
    n0 = len(records)
    for split in splits:
        d = root / split
        if not (d / "images").exists():
            continue
        for img in sorted((d / "images").glob("*")):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lp = d / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            cs = [c for c, *_ in parse_label(lp)]
            if not cs:
                continue
            tgt = want.get(max(set(cs), key=cs.count))
            if tgt:
                _add(tgt, f"{tag}:{_stem_group(img.name)}", img, tag)
    print(f"  {tag}: +{len(records)-n0} 张")


def load_corn():
    _load_det_source(RAW / "rf" / "corn_aziz", CORN_MAP, "corn_aziz")


def load_citrus():
    # 注意：这个源**只有 train**，没有 valid/test
    _load_det_source(RAW / "rf" / "citrus_research", CITRUS_MAP, "citrus_research",
                     splits=("train",))


def load_bingban():
    """稻瘟病：白底叶片标本切片，单类"""
    n0 = len(records)
    for r in BINGBAN_ROOTS:
        if not (r / "images").exists():
            continue
        for img in sorted((r / "images").glob("*")):
            if img.suffix.lower() not in IMG_EXT:
                continue
            _add("15_rice_blast", f"bingban:{img.stem}", img, "bingban",
                 crop="WHITE")   # crop="WHITE" 表示裁掉白边，见 crop_white_margin
    print(f"  bingban(稻瘟): +{len(records)-n0} 张")


# ---------- 划分 / 配平 / 落盘 ----------

def crop_square(im, box, margin=0.30):
    """按 YOLO 归一化框裁一个带 margin 的正方形区域"""
    W, H = im.size
    cx, cy, w, h = box
    side = max(w * W, h * H) * (1 + margin)
    side = max(side, 32)
    x1 = max(0, int(cx * W - side / 2))
    y1 = max(0, int(cy * H - side / 2))
    x2 = min(W, int(x1 + side))
    y2 = min(H, int(y1 + side))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return im.crop((x1, y1, x2, y2))


def crop_white_margin(im, thr=238, pad=0.03):
    """裁掉白底留白（稻瘟 bingban 的标本切片用）

    实测 bingban 有 88% 白边——是标本留白不是商品图，直接删掉整类就废了。
    裁到非白内容的包围盒，能让它更接近其它叶片照片的构图。
    """
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w, _ = a.shape
    mask = a.min(axis=2) < thr                    # 非白像素
    ys, xs = np.where(mask)
    if len(xs) < 50:                              # 几乎是纯白，裁不出来
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x2 - x1) * pad) + 2, int((y2 - y1) * pad) + 2
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return im.crop((x1, y1, x2, y2))


def green_frac(im):
    """绿色像素占比 —— 用来把**果实**照片从柑橘源里筛掉

    citrus_research 里「健康」类中位只有 0.07（多半是白底橙子），
    「黄龙病」0.46（基本都是叶片）。白边过滤救不了（柑橘的叶片照也是白底），
    但颜色分得开：果实是橙色，叶片是绿色。
    """
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((g > r * 1.06) & (g > b * 1.06) & (g > 50)).mean())


def quality_ok(im, check_white=True, min_green=0.0):
    """坏图判定：唯一颜色数过低（被破坏性增广像素化）或四周纯白（白底商品图）

    唯一色数必须**按尺寸归一化**后再比：先用 NEAREST 把最长边压到 <=128 的网格
    （不插值、不改像素值），再数唯一色。否则小图的采样像素天然少、唯一色数上不去，
    会被误判成坏图——实测下载来的一批干净小图（中位 5168~7089）就是这么被误杀的。

    归一化后的实测分布：
        毁掉的 Apple Scab / Black Rot / Cedar Apple Rust —— 991 / 1436 / 1026
        真照片（含下载的小图）                        —— 5168 ~ 10188
    两者不重叠，3000 这条线余量充足。
    """
    im = im.convert("RGB")
    w, h = im.size
    if check_white and MAX_WHITE_BORDER < 1.0:
        b = max(2, int(min(h, w) * 0.06))
        full = np.asarray(im, dtype=np.uint8)
        edge = np.concatenate([full[:b].reshape(-1, 3), full[-b:].reshape(-1, 3),
                               full[:, :b].reshape(-1, 3), full[:, -b:].reshape(-1, 3)])
        if float((edge.min(axis=1) >= 235).mean()) > MAX_WHITE_BORDER:
            return False
    if min_green > 0 and green_frac(im) < min_green:
        return False
    s = min(1.0, 128 / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
    a = np.asarray(im)
    return len(np.unique(a.reshape(-1, 3), axis=0)) >= MIN_UNIQUE_COLORS


def save_img(im, dst):
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=JPEG_Q)


def split_groups(groups, rng):
    """按**原图**随机划分（所有来源都会同时出现在 train 和 val 里）。

    数据利用率最高，但 val 会有同域泄漏、数字偏高 —— 所以只用来做早停判断，
    不用来当泛化能力的证据。泛化能力请看 SPLIT_MODE="source" 那一轮的结果。
    """
    gs = sorted(groups)
    rng.shuffle(gs)
    n_val = min(max(1, int(len(gs) * VAL_RATIO)), VAL_CAP)
    if len(gs) < 6:
        n_val = 1
    return gs[n_val:], gs[:n_val], [], True


def split_by_source(groups, rng):
    """按**来源**整体留出 val/test —— 拿到诚实的泛化数字。

    为什么必须这样：同一来源（同一数据集/同一拍摄批次）的图，背景、光线、相机都一致。
    若按图片随机划分，模型只要认出"这块灰布""这个白底"就能答对，val 会虚高到离谱
    （实测 98.8%，但用训练时没见过的病害一测，只有 8% 落进"不确定"）。
    让整个来源进 val，训练时完全没见过那批图的风格，数字才有意义。

    每类做法：
      * ≥2 个来源 → 最大的进 train，次大的进 val，第三大的进 test，其余全进 train
      * 只有 1 个来源 → 退化成按原图分组划分（拿不到跨域信息，报告里会标出来）
    """
    by_src = {}
    for g in groups:
        src = g.split("|", 1)[0]
        by_src.setdefault(src, []).append(g)

    if len(by_src) < 2:
        gs = sorted(groups)
        rng.shuffle(gs)
        n_val = min(max(1, int(len(gs) * VAL_RATIO)), VAL_CAP)
        if len(gs) < 6:
            n_val = 1
        # 同样不建 test：若只有这一个类有 test 目录，ultralytics 训练末的自动验证会报
        # "found N images in 1 classes (requires 8 classes, not 1)"
        return gs[n_val:], gs[:n_val], [], True

    ranked = sorted(by_src.items(), key=lambda kv: -len(kv[1]))
    train = list(ranked[0][1])
    val = list(ranked[1][1])
    # 第三个及以后的来源**全部并入 train**，**不建 test 集**。
    # 原因：不是每个类都凑得出三个来源，test/ 下就会只有部分类目录，
    # ultralytics 训练结束会自动验证 test，类数对不上直接报
    # "found N images in 4 classes (requires 8 classes, not 4)"。
    # 我们关心的是 val（跨来源泛化），test 没有额外价值，索性不建。
    for _, gs in ranked[2:]:
        train += gs
    return train, val, [], False


def main():
    rng = random.Random(SEED)
    print("[1/5] 解析各来源 ...")
    load_catherine()
    load_plantdoc()
    load_plantseg()
    load_plantseg_mosaic()
    load_downloaded()
    load_corn()
    load_citrus()
    load_bingban()
    print(f"  合计 {len(records)} 条原始记录")

    print("\n[2/5] 按原图分组（去增广变体）...")
    by_cls = {}
    for cls, group, path, src, crop in records:
        by_cls.setdefault(cls, {}).setdefault(group, []).append((path, src, crop))
    for cls in by_cls:
        for g in by_cls[cls]:
            by_cls[cls][g].sort(key=lambda x: x[0])   # 确定性

    # 泄漏自检：一个 group 只能属于一个类
    seen = {}
    for cls, gs in by_cls.items():
        for g in gs:
            if g in seen and seen[g] != cls:
                print(f"  !! {g} 同时映射到 {seen[g]} 和 {cls}，保留前者")
            seen.setdefault(g, cls)

    print("\n[3/5] 分层划分 + 配平 + 落盘 ...")
    for sub in ("train", "val", "test", "holdout"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    # ultralytics 的扫描缓存也要清，否则会拿着上一次的旧文件列表
    for cache in OUT.glob("*.cache"):
        cache.unlink()

    report = ["=== 分类数据集构建报告 ===", "",
              f"来源记录 {len(records)} 条 → 去增广后按原图分组", ""]
    stat = {}
    dropped = {}
    holdout_groups = []
    for cls in CLS_DIRS:
        gs = by_cls.get(cls, {})
        if not gs:
            print(f"  !! {cls} 没有任何样本 ← 空类")
            report.append(f"[{cls}] {DIR2CN[cls]:<12} 0 张  ← 空类！缺少数据源")
            continue
        # 先按来源整体划分，再限幅（顺序很重要：先限幅会把来源切碎）
        splitter = split_by_source if SPLIT_MODE == "source" else split_groups
        gtr, gva, gte, fallback = splitter(sorted(gs), rng)
        # 来源按比例切分（只对 source 模式有意义，见 SRC_SPLIT 的说明）
        for src_name, frac in (SRC_SPLIT.get(cls, {}) if SPLIT_MODE == "source" else {}).items():
            moved = [g for g in gva if g.split("|", 1)[0] == src_name]
            if moved:
                rng.shuffle(moved)
                nv = max(1, int(len(moved) * frac))
                gva = [g for g in gva if g.split("|", 1)[0] != src_name] + moved[:nv]
                gtr = gtr + moved[nv:]

        # 真·留出集（见 HOLDOUT 的说明）：从 train/val 里摘出来，单独落盘
        for src_name, frac in HOLDOUT.get(cls, {}).items():
            pool = [g for g in (gtr + gva) if g.split("|", 1)[0] == src_name]
            if pool:
                rng.shuffle(pool)
                nh = max(1, int(len(pool) * frac))
                hold = set(pool[:nh])
                gtr = [g for g in gtr if g not in hold]
                gva = [g for g in gva if g not in hold]
                holdout_groups.append((cls, [g for g in pool[:nh]]))

        # 类内按来源限幅（见 SRC_CAP 的说明）
        for src_name, cap in SRC_CAP.get(cls, {}).items():
            same = [g for g in gtr if g.split("|", 1)[0] == src_name]
            rng.shuffle(same)
            gtr = [g for g in gtr if g.split("|", 1)[0] != src_name] + same[:cap]
        rng.shuffle(gtr)
        gtr = gtr[:CAP_ORIG]
        gva, gte = gva[:VAL_CAP], gte[:TEST_CAP]
        k = max(1, min(MAX_VAR, round(TARGET / max(1, len(gtr)))))

        # 断言：同一原图的变体绝不跨 split
        assert not (set(gtr) & set(gva)), f"{cls} train/val 泄漏"
        assert not (set(gtr) & set(gte)), f"{cls} train/test 泄漏"

        counts = {"train": 0, "val": 0, "test": 0}
        for split, glist, take in (("train", gtr, k), ("val", gva, 1), ("test", gte, 1)):
            for g in glist:
                got = 0
                # 扫遍该原图的所有变体，**取够 take 张通过质检的为止**。
                # 不能只取前 take 张再质检——同一原图的变体里混着好图与像素化的坏图，
                # 若前几张恰好全坏，这个原图就白丢了（实测坏图率 38%~70%）。
                for vi, (path, src, crop) in enumerate(gs[g]):
                    if got >= take:
                        break
                    try:
                        im = Image.open(path)
                        im.load()
                    except Exception:
                        continue
                    white_src = (crop == "WHITE")
                    if white_src:
                        im = crop_white_margin(im)      # 稻瘟标本：裁掉白边
                        if im is None:
                            dropped[cls] = dropped.get(cls, 0) + 1
                            continue
                    elif crop is not None:
                        im = crop_square(im, crop)
                        if im is None:
                            continue
                    if not quality_ok(im, check_white=not white_src,
                                      min_green=CITRUS_MIN_GREEN if src == "citrus_research" else 0.0):
                        dropped[cls] = dropped.get(cls, 0) + 1
                        continue
                    name = (f"{src}_{pathlib.Path(path).stem}"
                            + (f"_c{vi}" if crop else f"_v{vi}") + ".jpg")
                    try:
                        save_img(im, OUT / split / cls / name)
                        counts[split] += 1
                        got += 1
                    except Exception:
                        continue
        n_orig = len(gs)
        stat[cls] = (n_orig, len(gtr), k, counts)
        nd = dropped.get(cls, 0)
        srcs = sorted({g.split("|", 1)[0] for g in gs})
        tag = ""
        if SPLIT_MODE == "source" and fallback:
            tag = "  ⚠ 只有一个来源，val 非跨域"
        report.append(f"[{cls}] {DIR2CN[cls]:<12} 原图 {n_orig:>4}（{len(srcs)} 个来源） → train {counts['train']:>4} "
                      f"/ val {counts['val']:>3} / test {counts['test']:>3}  ×{k}变体"
                      + (f"   质检剔除 {nd}" if nd else "") + tag)
        report.append(f"{'':<14}   来源: {', '.join(srcs)}")
        print(f"  {cls:<22} 原图{n_orig:>5}（{len(srcs)}源） → train {counts['train']:>4}  val {counts['val']:>3}  "
              f"test {counts['test']:>3}" + (f"   坏图剔除 {nd}" if nd else "") + tag)

    # 落盘真·留出集（不参与训练，专供 probe_cls_unknown.py 验收）
    n_hold = 0
    for cls, glist in holdout_groups:
        for g in glist:
            for vi, (path, src, crop) in enumerate(by_cls[cls][g]):
                try:
                    im = Image.open(path)
                    im.load()
                except Exception:
                    continue
                white_src = (crop == "WHITE")
                if white_src:
                    im = crop_white_margin(im)
                    if im is None:
                        continue
                elif crop is not None:
                    im = crop_square(im, crop)
                    if im is None:
                        continue
                if not quality_ok(im, check_white=not white_src,
                                      min_green=CITRUS_MIN_GREEN if src == "citrus_research" else 0.0):
                    continue
                save_img(im, OUT / "holdout" / cls / f"{src}_{pathlib.Path(path).stem}_h{vi}.jpg")
                n_hold += 1
    if n_hold:
        print(f"  [留出集] {n_hold} 张 → {OUT / 'holdout'}（不参与训练，供未知病害验收）")
        report += ["", f"留出集(holdout) {n_hold} 张，不参与训练 —— 供 probe_cls_unknown.py 使用"]

    print("\n[4/5] 写 class_map.json + 报告 ...")
    dirs = sorted(p.name for p in (OUT / "train").iterdir() if p.is_dir())
    missing = [d for d in CLS_DIRS if d not in dirs]
    if missing:
        print(f"  !! 以下类没有数据，将从 class_map 中移除（否则训练会因类数不匹配报错）: {missing}")
    cm = write_class_map(only=set(dirs))
    total = sum(sum(c.values()) for _, _, _, c in stat.values())
    report += ["", f"总计 {total} 张", f"class_map: {cm}", ""]
    report.append("train 下实际目录顺序: " + ", ".join(dirs))
    report.append("与 CLS_DIRS 一致: " + str(dirs == CLS_DIRS))
    if missing:
        report.append(f"缺失的类（无数据源）: {', '.join(missing)}")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[-6:]))

    print("\n[5/5] 画样例图供肉眼验收 ...")
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rng2 = random.Random(SEED)
    for cls in CLS_DIRS:
        d = OUT / "train" / cls
        imgs = sorted(d.glob("*.jpg")) if d.exists() else []
        if not imgs:
            continue
        for k2 in range(min(2, len(imgs))):
            pick = imgs[rng2.randrange(len(imgs))]
            try:
                im = Image.open(pick).convert("RGB").resize((384, 384))
            except Exception:
                continue
            dr = ImageDraw.Draw(im)
            # 中文字体跨平台探测；探测不到时退回 PIL 默认位图字体
            # ——该字体没有 CJK 字形，类名会显示成方块，但不会中断构建
            _font_file = config.font_path()
            try:
                f = (ImageFont.truetype(str(_font_file), 26) if _font_file
                     else ImageFont.load_default())
            except Exception:
                f = ImageFont.load_default()
            dr.rectangle([0, 0, 384, 38], fill=(0, 0, 0))
            dr.text((8, 6), DIR2CN[cls], fill=(0, 255, 128), font=f)
            im.save(SAMPLES / f"{cls}_{k2}.png")
    print(f"  样例图 → {SAMPLES}   （务必肉眼过一遍，这是唯一能抓住类名合错的手段）")

    pack()
    print("\n完成。下一步：把 zip 传云端训练")


def pack():
    """打包成 zip 供上传云端（含 class_map.json，保证云端类序一致）"""
    import zipfile
    PACK.mkdir(parents=True, exist_ok=True)
    zpath = PACK / "apple_cls.zip"
    if zpath.exists():
        zpath.unlink()
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
                n += 1
    size = zpath.stat().st_size / 1e6
    print(f"\n[打包] {n} 个文件 → {zpath}  ({size:.0f}MB)")
    print(f"       上传: scp \"{zpath}\" root@<云主机>:/root/data/  然后解压到 /root/data/apple_cls")


if __name__ == "__main__":
    main()