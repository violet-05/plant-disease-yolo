# -*- coding: utf-8 -*-
"""抽取各源目标类样例并画框，供肉眼 QA（用原始源数据，验证每个源的类号语义）"""
import pathlib, sys
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

SAMPLES = config.DATA / "crop6_samples"
RAW = config.RAW
PICKS = [
    # (样例名, 源目录, 该源 yaml 里目标类的原始类号集合)
    ("A_bingban_rice", RAW / "bingban" / "87_bingban_tiles_v3" / "train", {0}),
    ("B_aziz_corn", RAW / "rf" / "corn_aziz" / "train", {1}),
    ("C_wheatpatch_wheat", RAW / "rf" / "wheat_patch" / "train", {2}),
    ("D_wheatbig_wheat", RAW / "rf" / "wheat_stripe_big" / "train", {3}),
    ("E_tdebh_rice", RAW / "rf" / "rice_tdebh" / "train", {2}),
    ("F_citrus_canker", RAW / "rf" / "citrus_research" / "train", {1}),
    ("G_grape_ld", RAW / "rf" / "grape_ld" / "train", {1}),
    ("H_plantseg_wheat110", RAW / "rf" / "plantseg" / "train", {110}),
]


def parse(lp):
    boxes = []
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
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((c, cx, cy, w, h))
    return boxes


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, root, want in PICKS:
        for img in sorted((root / "images").glob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lp = root / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            boxes = [b for b in parse(lp) if b[0] in want]
            if not boxes:
                continue
            im = Image.open(img).convert("RGB").resize((448, 448))
            dr = ImageDraw.Draw(im)
            for c, cx, cy, w, h in boxes:
                dr.rectangle([(cx - w / 2) * 448, (cy - h / 2) * 448, (cx + w / 2) * 448, (cy + h / 2) * 448], outline=(255, 0, 0), width=2)
            out = SAMPLES / f"extra_{name}.png"
            im.save(out)
            print("saved", out.name, "| boxes:", len(boxes), "| 源文件:", img.stem[:44])
            break


if __name__ == '__main__':
    main()