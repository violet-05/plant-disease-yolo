# -*- coding: utf-8 -*-
"""把所有原始数据集合并 → 标准 6 类 YOLO 数据集
输出: <项目根>/datasets/crop6/{images,labels}/{train,val} + data.yaml
规则:
  - 多边形(分割)标注自动转检测框(min/max)
  - 框面积过滤: 丢弃 <0.05% 噪点框 和 >99% 的整图框(catherine 那类)
  - 只保留映射到 6 类的框; 映射后无框的图整张丢弃
  - 划分: 各源 train→train, valid/test→val; 无 val 的源取 10% 训练图做 val
"""
import csv, pathlib, random, shutil, statistics, sys
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

RAW = config.RAW
RF = RAW / "rf"
OUT = config.DATA / "crop6"
SAMPLES = config.DATA / "crop6_samples"
CLASSES = ["rice_blast", "corn_rust", "wheat_stripe_rust", "apple_scab", "citrus_canker", "grape_downy_mildew"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AREA_MIN, AREA_MAX = 0.0005, 0.99

records = []   # (src, split, img_path, [(cls,cx,cy,w,h)])
stats = {}     # (src, cls) -> [imgs, boxes]


def parse_label_file(lp):
    out = []
    for line in lp.read_text(errors="ignore").splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        v = [float(x) for x in t[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
        else:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        out.append((c, cx, cy, w, h))
    return out


def clean(boxes):
    """裁剪到图内 + 面积过滤"""
    out = []
    for c, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)
        w2, h2 = x2 - x1, y2 - y1
        if w2 <= 0 or h2 <= 0 or not (AREA_MIN <= w2 * h2 <= AREA_MAX):
            continue
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, w2, h2))
    return out


def add(src, split, img, boxes):
    boxes = clean(boxes)
    if not boxes:
        return
    records.append((src, split, img, boxes))
    for c, *_ in boxes:
        s = stats.setdefault((src, c), [0, 0])
        s[1] += 1
    for c in {b[0] for b in boxes}:
        stats[(src, c)][0] += 1


def rf_names(root):
    for p in list(root.glob("*.yaml")) + list(root.rglob("data.yaml")):
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


def collect_rf(key, root, name_map, slugs=None):
    """slugs: {目标类id: 文件名前缀}；给定后只接受『文件名前缀所指的类 == 标签映射后的类』的框（双信号一致性过滤）"""
    names = rf_names(root)
    src2tgt = {cid: name_map[nm] for cid, nm in names.items() if nm in name_map}
    if not src2tgt:
        print(f"!! {key}: 没有类名匹配成功，检查映射"); return
    slug2cls = {v: k for k, v in (slugs or {}).items()}
    for sd, split in [("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")]:
        d = root / sd
        if not d.exists():
            continue
        for img in (d / "images").glob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            allow = None
            if slug2cls:
                fname = img.stem.split(".rf.")[0].lower()
                for s, c in slug2cls.items():
                    if fname.startswith(s):
                        allow = c
                        break
                if allow is None:
                    continue
            lp = d / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            add(key, split, img, [(src2tgt[c], cx, cy, w, h) for c, cx, cy, w, h in parse_label_file(lp)
                                  if c in src2tgt and (allow is None or src2tgt[c] == allow)])


def collect_plantdoc():
    root = RAW / "plantdoc"
    nm = {"Corn rust leaf": 1, "Apple Scab Leaf": 3}
    for csvf, split, sub in [("train_labels.csv", "train", "TRAIN"), ("test_labels.csv", "val", "TEST")]:
        p = root / csvf
        if not p.exists():
            continue
        per = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["class"] not in nm:
                continue
            img = root / sub / r["filename"]
            if not img.exists():
                continue
            W, H = float(r["width"]), float(r["height"])
            if W <= 0 or H <= 0:
                continue
            x1, y1, x2, y2 = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
            per.setdefault(img, []).append((nm[r["class"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        for img, boxes in per.items():
            add("plantdoc", split, img, boxes)


def collect_bingban():
    root = RAW / "bingban" / "87_bingban_tiles_v3"
    for sd in ("train", "val", "test"):
        d = root / sd
        if not d.exists():
            continue
        split = "train" if sd == "train" else "val"
        for img in (d / "images").glob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lp = d / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            add("bingban", split, img, [(0, cx, cy, w, h) for _, cx, cy, w, h in parse_label_file(lp)])


RF_SOURCES = [
    # (输出前缀, 实际目录名, 类名映射, 文件名前缀过滤)
    ("plantseg", "plantseg", {"rice blast": 0, "corn rust": 1, "wheat stripe rust": 2, "apple scab": 3, "citrus canker": 4, "grape downy mildew": 5},
     {0: "rice_blast", 1: "corn_rust", 2: "wheat_stripe_rust", 3: "apple_scab", 4: "citrus_canker", 5: "grape_downy_mildew"}),
    ("wheatbig", "wheat_stripe_big", {"wheat stripe rust": 2}, {2: "wheat_stripe_rust"}),
    ("wheatpatch", "wheat_patch", {"yellow_rust_patch": 2}, None),
    ("aziz", "corn_aziz", {"common_rust": 1}, None),
    ("tdebh", "rice_tdebh", {"Rice_Blast": 0}, None),
    ("citrus", "citrus_research", {"Canker": 4}, None),
    ("grape", "grape_ld", {"downy mildew": 5}, None),
]
# 注: apple_catherine 被排除——它的框全是整图(100%面积)，等价于分类标签，非检测标注
#     plantseg/wheatbig 使用双信号一致性过滤（该镜像的文件名可信、类号有串味）

def main():
    print("[1/3] 收集并解析所有源 ...")
    for key, dirname, nm, slugs in RF_SOURCES:
        collect_rf(key, RF / dirname, nm, slugs)
        print(f"  {key} 完成")
    collect_plantdoc(); print("  plantdoc 完成")
    collect_bingban(); print("  bingban 完成")

    print(f"[2/3] 写出 {len(records)} 张图 → {OUT} ...")
    shutil.rmtree(OUT, ignore_errors=True)
    rng = random.Random(0)
    for i, (src, split, img, boxes) in enumerate(records):
        if src == "citrus" and split == "train" and rng.random() < 0.1:   # 该源无val，抽10%充当
            split = "val"
        stem = f"{src}_{img.stem}"
        di = OUT / "images" / split; dl = OUT / "labels" / split
        di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, di / (stem + img.suffix.lower()))
        (dl / (stem + ".txt")).write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(records)}")

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")

    print("[3/3] 画样例图供肉眼检查 ...")
    SAMPLES.mkdir(parents=True, exist_ok=True)
    by_cls = {c: [] for c in range(6)}
    for src, split, img, boxes in records:
        for c, *_ in boxes:
            if len(by_cls[c]) < 2:
                by_cls[c].append((src, split, img, boxes))
    for c, items in by_cls.items():
        for k, (src, split, img, boxes) in enumerate(items):
            im = Image.open(img).convert("RGB").resize((448, 448))
            dr = ImageDraw.Draw(im)
            for bc, cx, cy, w, h in boxes:
                if bc != c: continue
                dr.rectangle([(cx - w / 2) * 448, (cy - h / 2) * 448, (cx + w / 2) * 448, (cy + h / 2) * 448], outline=(255, 0, 0), width=3)
            im.save(SAMPLES / f"{c}_{CLASSES[c]}_s{k}_{src}.png")

    rep = ["=== 合并报告 ===", "", f"总图片: {len(records)}", ""]
    for c, name in enumerate(CLASSES):
        imgs = sum(1 for _, _, _, b in records if any(x[0] == c for x in b))
        boxes = sum(1 for _, _, _, b in records for x in b if x[0] == c)
        rep.append(f"[{c}] {name:<20} {imgs:>6} 图 / {boxes:>7} 框")
        for (src, sc) in sorted(k for k in stats if k[1] == c):
            rep.append(f"      - {src:<12} {stats[(src, sc)][0]:>6} 图 / {stats[(src, sc)][1]:>7} 框")
    for split in ("train", "val"):
        n = sum(1 for _, s, _, _ in records if s == split)
        rep.append(f"{split}: {n} 图")
    (config.RAW / "report_merge.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()