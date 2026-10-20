# -*- coding: utf-8 -*-
"""作物叶部病害 17 类的「目录名 ↔ 中文显示名」映射（单一来源）

为什么目录名要带数字前缀：
  ultralytics 分类任务的类索引 = `train/` 下子目录名的**排序序**
  （见 ultralytics/data/utils.py 的 check_cls_dataset）。
  数字前缀保证「排序序 == 期望序」，中文映射天然对齐；
  若用中文目录名，排序结果与映射表一旦错位，标签会整体张冠李戴**且不报错**。

旧代码里 CLASSES_CN 在 4 个脚本中各自重复定义（webcam_demo / diag_image /
diag_domain / diag_unknown），那是 6 类检测模型的映射。本模块是分类任务的新来源。

数据集目录由 tools/config.py 统一解析（环境变量 PLANT_DISEASE_ROOT 可覆盖），
本文件不再硬编码任何绝对路径。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

CLS_ROOT = config.CLS_DATA
CLASS_MAP_JSON = CLS_ROOT / "class_map.json"

# 顺序即索引，不要随意调整。
# 2026-10-07 收敛为 7 类：质检发现 catherine 的 anthracnose(炭疽)/sootyblotch(煤污)/
# whiterot(白腐) 全是**果实**病害、valsacanker(溃疡) 是**枝干**病害，与"叶部"任务不符，已剔除。
CLS_DIRS = [
    # ---- 苹果（细分到病害）----
    "00_apple_healthy",       # 苹果健康叶
    "01_apple_scab",          # 苹果黑星病
    "02_apple_black_rot",     # 苹果黑腐病
    "03_apple_cedar_rust",    # 苹果锈病
    "04_apple_alternaria",    # 苹果斑点落叶病
    "05_apple_marssonina",    # 苹果褐斑病
    "06_apple_other",         # 其他苹果病害 —— 拒绝出口
    # ---- 玉米（corn_aziz + PlantDoc 的玉米叶类）----
    "07_corn_healthy",
    "08_corn_blight",         # 玉米大斑病
    "09_corn_rust",           # 玉米锈病
    "10_corn_gray_spot",      # 玉米灰斑病
    # ---- 柑橘（citrus_research）----
    "11_citrus_healthy",
    "12_citrus_canker",       # 柑橘溃疡病
    "13_citrus_blackspot",    # 柑橘黑星病
    "14_citrus_greening",     # 柑橘黄龙病
    # ---- 水稻（bingban）----
    "15_rice_blast",          # 稻瘟病
    # ---- 兜底 ----
    "16_other_leaf",          # 其他作物叶片（番茄/桃/葡萄/蓝莓… 未细分）
]

DIR2CN = {
    "00_apple_healthy": "苹果健康叶",
    "01_apple_scab": "苹果黑星病",
    "02_apple_black_rot": "苹果黑腐病",
    "03_apple_cedar_rust": "苹果锈病",
    "04_apple_alternaria": "苹果斑点落叶病",
    "05_apple_marssonina": "苹果褐斑病",
    "06_apple_other": "其他苹果病害",
    "07_corn_healthy": "玉米健康叶",
    "08_corn_blight": "玉米大斑病",
    "09_corn_rust": "玉米锈病",
    "10_corn_gray_spot": "玉米灰斑病",
    "11_citrus_healthy": "柑橘健康叶",
    "12_citrus_canker": "柑橘溃疡病",
    "13_citrus_blackspot": "柑橘黑星病",
    "14_citrus_greening": "柑橘黄龙病",
    "15_rice_blast": "稻瘟病",
    "16_other_leaf": "其他作物叶片",
}

# 演示与验收脚本「不确定」出口的阈值 —— **全项目唯一定义**，其它脚本从这里 import。
#
# 为什么是 0.50/0.12：这是验收脚本（probe_cls_unknown.py）与双轨演示
# （webcam_dual_demo.py）实际在用的值，README 里「73~88% 正确拒绝」就是由它测得的。
# 改这个值会改变对外公布的数字，需要重跑验收。
# 历史上本文件曾是 0.55/0.15、与那两个脚本不一致，已按实际行为收敛。
THRESH = 0.50
MARGIN = 0.12


def write_class_map(only=None):
    """构建脚本调用，落盘给训练/评估/演示共用

    only: 只写这些目录名（按 CLS_DIRS 顺序过滤）。用于构建时某类没数据的情况——
          空类必须从 class_map 里去掉，否则 ultralytics 训练会因类数不匹配报错。
    """
    keys = [d for d in CLS_DIRS if (only is None or d in only)]
    CLASS_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLASS_MAP_JSON.write_text(
        json.dumps({k: DIR2CN[k] for k in keys}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return CLASS_MAP_JSON


def load_class_map():
    """返回 {目录名: 中文名}，保持文件里的顺序（顺序即类别索引）"""
    if CLASS_MAP_JSON.exists():
        try:
            return dict(json.loads(CLASS_MAP_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass
    return dict(DIR2CN)


def cn_names_dict():
    """给 `model.model.names = ...` 用的 {索引: 中文名}，从 class_map 读，与训练时的类序一致"""
    return {i: v for i, v in enumerate(load_class_map().values())}


if __name__ == "__main__":
    m = load_class_map()
    print(f"class_map.json 现有 {len(m)} 类:")
    for i, (d, cn) in enumerate(m.items()):
        print(f"{i:>2}  {d:<24} {cn}")