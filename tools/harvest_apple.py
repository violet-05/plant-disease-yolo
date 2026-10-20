# -*- coding: utf-8 -*-
"""下载补数据用的苹果病害数据集（Phase 0）

流程：项目信息 → 取最新版本 → 按项目类型选导出格式 → 轮询拿链接 → 下载 zip → 解压
已下载的自动跳过，可安全重跑。

**关键：导出格式随项目类型而变**
  - multilabel-classification / multiclass-classification → `folder`（yolov8 会报 invalid format）
  - object-detection / instance-segmentation 等          → `yolov8`

下载到 datasets/raw/apple_new/<项目名>/，不改动已有的 raw/rf/ 源。
"""
import json
import pathlib
import sys
import time
import zipfile

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

API = "https://api.roboflow.com"
OUT = config.RAW / "apple_new"
CANDS = config.RAW / "apple_candidates.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}

# 人工挑选的清单（(workspace, project, 补哪个类, 备注)）
TARGETS = [
    ("capstone-7yk2c", "apple-1", "炭疽病 + 褐斑病", "稀少类主力，全网只搜到这一个褐斑病来源"),
    ("anhtay", "apple-lotkh", "炭疽病", "补充炭疽病"),
    ("deep-learning-ux05o", "apple-leaf-pfjeu", "斑点落叶病", "alternaria"),
    ("tan-workspace", "apple-disease-detection-qegjx", "斑点落叶病", "alternaria"),
    ("moobashara-jawed-nkyzf", "apple-disease-uyswd", "健康叶 + 黑腐病", "健康叶是全表最弱类"),
    ("research-niikp", "apple-scab-detection", "健康叶 + 黑星病", "含 Apple-Healthy"),
    ("jiyad", "apple-leaf-w23si", "健康叶 + 褐斑/锈", "整图标签项目"),
    ("alias-workspace-99qul", "apple-disease-bgyrs", "健康叶 + 锈病", ""),
    ("reiaslites-project", "apple-leaf-disease-classification-r2zpc",
     "健康 + 黑星 + 黑腐 + 锈", "整图标签，类别正好对应 00/01/02/03"),
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def proj_type(ws, proj):
    """从搜索结果缓存里取项目类型（项目级 API 不返回 type）"""
    if not CANDS.exists():
        return None
    try:
        for c in json.loads(CANDS.read_text(encoding="utf-8")):
            if c.get("workspace") == ws and c.get("project") == proj:
                return c.get("type")
    except Exception:
        pass
    return None


def latest_version(ws, proj, key):
    d = requests.get(f"{API}/{ws}/{proj}", params={"api_key": key}, timeout=60).json()
    vs = d.get("versions") or []
    if isinstance(vs, dict):
        vs = [dict(v, id=(v.get("id") if isinstance(v, dict) else None) or k) for k, v in vs.items()]
    if not vs:
        return None, None

    def vnum(e):
        s = str(e.get("id") or e.get("version") or "")
        try:
            return int(s.rsplit("/", 1)[-1])
        except Exception:
            return -1

    e = sorted(vs, key=vnum)[-1]
    return e.get("id"), e.get("images")


def export_link(vid, ptype, key):
    """按项目类型选格式；分类项目用 folder，检测项目用 yolov8"""
    fmt = "folder" if ptype and "classification" in ptype else "yolov8"
    for _ in range(60):
        j = requests.get(f"{API}/{vid}/{fmt}", params={"api_key": key}, timeout=60).json()
        err = j.get("error")
        if isinstance(err, str) and "invalid format" in err:
            # 类型判断错了，从报错里读可用格式，优先 folder
            fmt = "folder" if "folder" in err else err.rsplit(":", 1)[-1].split(",")[0].strip()
            print(f"    [格式切换] 改用 {fmt}", flush=True)
            continue
        ex = j.get("export")
        if isinstance(ex, dict) and ex.get("link"):
            return ex["link"]
        if isinstance(j.get("link"), str):
            return j["link"]
        time.sleep(5)
    return None


def detect_layout(d):
    """展开后看目录长什么样，判断是分类目录树还是检测 images/labels"""
    kinds = []
    for sub in sorted(p for p in d.rglob("*") if p.is_dir()):
        rel = sub.relative_to(d)
        if any(x in ("images", "labels") for x in rel.parts):
            kinds.append(("检测 images/labels", str(rel)))
        elif rel.parts and rel.parts[0] in ("train", "valid", "test", "val") and len(rel.parts) == 2:
            kinds.append(("分类 <split>/<类名>", str(rel)))
    seen, out = set(), []
    for k in kinds:
        if k[0] not in seen:
            seen.add(k[0])
            out.append(k)
    return out


def count_imgs(d):
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def main():
    key = config.roboflow_key()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"目标 {len(TARGETS)} 个数据集 → {OUT}\n")
    for ws, proj, covers, note in TARGETS:
        dest = OUT / proj
        if dest.exists() and any(dest.rglob("*")):
            print(f"[跳过] {proj}：已存在 {count_imgs(dest)} 张图  ({covers})", flush=True)
            continue
        try:
            vid, nimg = latest_version(ws, proj, key)
            if not vid:
                raise RuntimeError("取不到版本")
            ptype = proj_type(ws, proj) or "?"
            print(f"[开始] {proj}  v={str(vid).rsplit('/',1)[-1]}  自报 {nimg} 张  "
                  f"类型={ptype}  补={covers}  {note}", flush=True)

            link = export_link(vid, ptype, key)
            if not link:
                raise RuntimeError("导出生成超时")

            tmp = OUT / (proj + ".zip")
            with requests.get(link, headers=UA, timeout=3600, stream=True) as r:
                r.raise_for_status()
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        size += len(chunk)
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(tmp).extractall(dest)
            tmp.unlink()

            layout = detect_layout(dest)
            n = count_imgs(dest)
            print(f"[完成] {proj}: {size/1e6:.0f}MB  解压后 {n} 张图", flush=True)
            for k, path in layout:
                print(f"        结构: {k}  e.g. {path}", flush=True)
        except Exception as e:
            print(f"[失败] {proj}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    print("\n=== 全部结束 ===")
    print("下一步：用 tools/inspect_datasets.py 的思路巡检这些新数据集的类别分布")


if __name__ == "__main__":
    main()