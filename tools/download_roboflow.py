# -*- coding: utf-8 -*-
"""批量下载 Roboflow Universe 数据集
流程：项目信息 → 版本列表(取最新) → 触发/轮询 yolov8 导出 → 下载 zip → 解压
走 api.roboflow.com，大陆可达；已下载的项目自动跳过（可安全重跑）"""
import pathlib, sys, time, zipfile
import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

OUT = config.RAW / "rf"
API = "https://api.roboflow.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}

PROJECTS = [
    ("iqra-national-university-hayatabad-peshawar", "plantseg-disease-detection", "plantseg", "6类底座(病斑级)"),
    ("nzzzzzz", "-z8ngy-fidm8", "wheat_stripe_big", "小麦条锈病(大批量)"),
    ("iot-lab", "wheat-disease-zcx0w", "wheat_patch", "小麦锈病(病斑级)"),
    ("aziz-jda2k", "corn-leaf-disease-hgosu", "corn_aziz", "玉米锈病/灰斑/叶枯"),
    ("catherine-jebet-zfz8s", "apple-diseases-detection-1po19", "apple_catherine", "苹果黑星病"),
    ("citrus-research", "citrus-disease-detection-v2", "citrus_research", "柑橘溃疡病"),
    ("grape-leaf-disease-detection", "ld-yvref", "grape_ld", "葡萄霜霉病(小)"),
    ("majorproject-p2ceq", "rice_diseases-tdebh", "rice_tdebh", "水稻病害(18类)"),
]


def get_versions(ws, proj, key):
    d = requests.get(f"{API}/{ws}/{proj}", params={"api_key": key}, timeout=60).json()
    vs = d.get("versions") or []
    if isinstance(vs, dict):
        vs = [dict(v, id=(v.get("id") if isinstance(v, dict) else None) or k) for k, v in vs.items()]

    def vnum(e):
        s = str(e.get("id") or e.get("version") or "")
        try:
            return int(s.rsplit("/", 1)[-1])
        except Exception:
            return -1

    return sorted(vs, key=vnum)


def main():
    key = config.roboflow_key()
    OUT.mkdir(parents=True, exist_ok=True)
    for ws, proj, name, covers in PROJECTS:
        dest = OUT / name
        if dest.exists() and any(dest.rglob("*.yaml")):
            print(f"[跳过] {name}: 已存在", flush=True)
            continue
        try:
            vs = get_versions(ws, proj, key)
            if not vs:
                raise RuntimeError("取不到版本")
            e = vs[-1]
            vid = e.get("id")
            ver = str(vid).rsplit("/", 1)[-1]
            print(f"[开始] {name}: v{ver}, {e.get('images')}张图 ({covers})", flush=True)

            link = None
            for _ in range(60):
                j = requests.get(f"{API}/{vid}/yolov8", params={"api_key": key}, timeout=60).json()
                ex = j.get("export")
                if isinstance(ex, dict) and ex.get("link"):
                    link = ex["link"]; break
                if isinstance(j.get("link"), str):
                    link = j["link"]; break
                time.sleep(5)
            if not link:
                raise RuntimeError("导出生成超时")

            tmp_zip = OUT / (name + ".zip")
            with requests.get(link, headers=UA, timeout=3600, stream=True) as r:
                r.raise_for_status()
                size = 0
                with open(tmp_zip, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk); size += len(chunk)
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(tmp_zip).extractall(dest)
            tmp_zip.unlink()
            print(f"[完成] {name} (v{ver}, {size/1e6:.0f}MB)", flush=True)
        except Exception as ex:
            print(f"[失败] {name}: {type(ex).__name__}: {str(ex)[:150]}", flush=True)

    print("=== 全部结束 ===")


if __name__ == "__main__":
    main()