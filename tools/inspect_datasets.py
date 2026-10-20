# -*- coding: utf-8 -*-
"""体检所有原始数据集：结构 / 类别清单 / 每类框数与框中位面积"""
import pathlib, statistics, sys
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402

RAW = config.RAW
RF = RAW / "rf"
REPORT = config.DATA / "report_inspect.txt"

lines = []
def out(s=""):
    print(s, flush=True)
    lines.append(s)

def read_names(root):
    for p in list(root.glob("*.yaml")) + list(root.rglob("data.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
            if d and d.get("names"):
                return d["names"], p
        except Exception:
            continue
    return None, None

def survey(root: pathlib.Path):
    names, yp = read_names(root)
    out(f"\n=== {root.name} ===  (yaml: {yp.name if yp else '未找到'})")
    if isinstance(names, dict):
        out(f"类别数: {len(names)}")
    elif isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
        out(f"类别数: {len(names)}")
    else:
        out("!! 读不到类别名"); return
    box_cnt, areas = {}, {}
    total_img = 0
    for split in ("train", "valid", "val", "test"):
        d = root / split
        if not d.exists():
            continue
        imgs = [p for p in (d / "images").glob("*") if p.is_file()]
        lbls = list((d / "labels").glob("*.txt"))
        total_img += len(imgs)
        out(f"  {split}: {len(imgs)} 图 / {len(lbls)} 标签")
        for lp in lbls:
            for line in lp.read_text(errors="ignore").splitlines():
                t = line.split()
                if len(t) >= 5:
                    c = int(float(t[0]))
                    box_cnt[c] = box_cnt.get(c, 0) + 1
                    if len(t) == 5:  # 检测格式，可直接算面积
                        areas.setdefault(c, []).append(float(t[3]) * float(t[4]))
    out(f"  合计图片: {total_img}")
    out("  类别明细(按框数排序):")
    for c, n in sorted(box_cnt.items(), key=lambda x: -x[1]):
        med = statistics.median(areas[c]) if areas.get(c) else None
        ms = f"框中位面积 {med:.2%}" if med is not None else "分割格式"
        out(f"    [{c:>3}] {names.get(c, '?'):<38} {n:>6} 框 | {ms}")

def main():
    for root in sorted(RF.iterdir()):
        if root.is_dir():
            try:
                survey(root)
            except Exception as e:
                out(f"\n=== {root.name} === 体检失败: {type(e).__name__}: {e}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已存: {REPORT}")


if __name__ == "__main__":
    main()