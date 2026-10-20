# -*- coding: utf-8 -*-
"""搜 Roboflow Universe 上的苹果病害数据集，输出候选清单供人工挑选

流程：多关键词搜索 → 翻页 → 按 url 去重 → 算苹果相关度 → 打印表格 + 存 JSON
关键：结果里的 type 字段标明 classification / object-detection。
      **classification 类型是整图标签，正好是我们要的分类数据**，优先选。
"""
import json
import pathlib
import sys
import time

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

API = "https://api.roboflow.com"
OUT_JSON = config.RAW / "apple_candidates.json"

# 搜索关键词：前几个打苹果总面积，后面几个专门捞我们缺的稀少类
QUERIES = [
    "apple leaf disease", "apple scab", "apple leaf spot", "apple disease",
    "apple alternaria", "apple marssonina", "apple anthracnose", "apple rust",
    "apple black rot", "apple leaf", "malus leaf disease",
]
PAGES = 3          # 每个关键词翻几页（每页 12 条）

# 我们 11 类体系里关心的病害词（命中就给相关度加分）
DISEASE_HITS = {
    "alternaria": 6, "斑点落叶": 6,
    "marssonina": 6, "褐斑": 6,
    "anthracnose": 5, "炭疽": 5,
    "leaf spot": 4, "spot": 2, "叶斑": 4,
    "scab": 4, "黑星": 4,
    "rust": 3, "锈": 3,
    "black rot": 4, "黑腐": 4,
    "sooty": 3, "煤污": 3,
    "canker": 2, "溃疡": 2, "valsa": 3,
    "white rot": 2, "白腐": 2,
    "mosaic": 1, "healthy": 2, "健康": 2,
}

# 果实/品种类数据集的特征词——出现即剔除（我们要的是叶片，不是苹果果实分级）
FRUIT_WORDS = (
    "rotten", "fresh", "green apple", "red apple", "fruit", "bruise", "maturity",
    "ripen", "cultivar", "braeburn", "crimson", "golden type", "granny", "fuji",
    "apple logo", "cracking", "quality", "defect",
)
# 必须命中其一，否则是纯"苹果识别"数据集
LEAF_WORDS = (
    "scab", "rust", "blotch", "spot", "mosaic", "anthracnose", "alternaria",
    "marssonina", "powdery", "leaf", "leaves", "canker", "mildew", "mold",
    "disease", "pest", "fungus", "healthy",
)
# 无效/占位类名
JUNK_CLASSES = {
    "objects", "apple", "apples", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "100", "25", "50", "75", "37", "233", "apple-4feu", "unlabeled", "disease",
}


def search(q, page, key):
    r = requests.get(f"{API}/universe/search",
                     params={"q": q, "api_key": key, "page": page, "page_size": 20},
                     timeout=45)
    r.raise_for_status()
    return r.json().get("results") or []


def parse_url(url):
    """https://universe.roboflow.com/<ws>/<proj> → (ws, proj)"""
    try:
        parts = url.rstrip("/").split("/")
        return parts[-2], parts[-1]
    except Exception:
        return None, None


def relevance(name, classes):
    """苹果叶片病害相关度。必须提到苹果，且不能是果实/品种类，再叠加病害词命中分"""
    text = (name + " " + " ".join(classes)).lower()
    if not any(k in text for k in ("apple", "malus")):
        return -1
    # 果实/品种类数据集对我们没用，直接剔除
    if any(f in text for f in FRUIT_WORDS):
        return -1
    # 必须沾点"叶"或"病"的边，否则是纯苹果识别数据集
    if not any(k in text for k in LEAF_WORDS):
        return -1
    score = 10
    for k, w in DISEASE_HITS.items():
        if k in text:
            score += w
    return score


def clean_classes(classes):
    """去掉占位/无效类名（数字、分隔线、objects 之类）"""
    out = []
    for c in classes:
        s = str(c).strip()
        if not s or s.startswith("=") or s.lower() in JUNK_CLASSES:
            continue
        out.append(s)
    return out


def main():
    key = config.roboflow_key()
    found = {}
    for q in QUERIES:
        for page in range(1, PAGES + 1):
            try:
                rs = search(q, page, key)
            except Exception as e:
                print(f"[失败] {q} p{page}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                break
            if not rs:
                break
            for it in rs:
                url = it.get("url")
                if not url or url in found:
                    continue
                ws, proj = parse_url(url)
                if not ws or not proj:
                    continue
                cls = clean_classes(it.get("classes") or [])
                found[url] = {
                    "name": it.get("name", ""),
                    "url": url,
                    "workspace": ws,
                    "project": proj,
                    "projectId": it.get("projectId"),
                    "type": it.get("type", "?"),
                    "classes": cls,
                    "images": it.get("images", 0),
                    "stars": it.get("stars", 0),
                    "downloads": it.get("downloads", 0),
                    "latestVersion": it.get("latestVersion"),
                    "relevance": relevance(it.get("name", ""), cls),
                }
            time.sleep(0.4)      # 别把 API 打疼
        print(f"  [搜索完] {q}  累计 {len(found)} 条", flush=True)

    cands = [c for c in found.values() if c["relevance"] >= 0]
    # 排序：分类类型优先（整图标签，直接可用）> 相关度 > 图像数
    cands.sort(key=lambda c: (c["type"] != "classification", -c["relevance"], -c["images"]))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== 苹果叶片病害候选 {len(cands)} 个（已存 {OUT_JSON}）===")
    print(f'{"类型":<16}{"名称":<30}{"图像":>6}{"星":>4}  {"类别":<52}workspace/project')
    print("-" * 150)
    for c in cands:
        t = "★分类" if c["type"] == "classification" else c["type"]
        cls = ",".join(c["classes"])[:50]
        wp = f'{c["workspace"]}/{c["project"]}'
        print(f'{t:<16}{c["name"][:29]:<30}{c["images"]:>6}{c["stars"]:>4}  {cls:<52}{wp}')
    print("-" * 150)
    print("★分类 = 整图标签，拿来就能当分类数据用（优先）")
    print(f"名单见 {OUT_JSON}，harvest_apple.py 按名单下载")


if __name__ == "__main__":
    main()