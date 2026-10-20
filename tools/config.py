# -*- coding: utf-8 -*-
"""项目级配置：路径、跨平台中文渲染、凭据 —— 所有脚本的唯一来源

为什么需要这个模块：
  原先 20 个脚本各自硬编码作者本机的绝对路径（``G:\\YOLO\\...``）和 Windows 专属
  字体（``C:\\Windows\\Fonts\\simhei.ttf``），别人 clone 下来一条命令都跑不通。
  这里把「项目在哪、字体在哪、密钥在哪」收敛成一处。

路径怎么定：
  1) 环境变量 ``PLANT_DISEASE_ROOT`` 优先 —— 数据集放在别处时用
  2) 否则由本文件位置上溯（``tools/`` 的父目录）—— 零配置开箱可用

所有路径都由 ``ROOT`` 派生，脚本里不再出现绝对路径。
"""
import functools
import os
import pathlib
import shutil
import sys

# Windows 控制台默认按 GBK 编码输出，脚本里大量中文会乱码或直接 UnicodeEncodeError。
# 尽量切到 UTF-8；不支持 reconfigure 的旧环境忽略即可。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------ 路径

ROOT = pathlib.Path(
    os.environ.get("PLANT_DISEASE_ROOT") or pathlib.Path(__file__).resolve().parent.parent
).expanduser().resolve()

DATA = ROOT / "datasets"
RAW = DATA / "raw"
CLS_DATA = DATA / "apple_cls"          # class_names.py 的 CLS_ROOT 取此值
SAMPLES = DATA / "apple_cls_samples"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
TMP = ROOT / ".tmp"

# ------------------------------------------------------------------ ultralytics 配置目录
#
# 必须在 import ultralytics **之前**设好。原因见 ultralytics/utils/__init__.py:1021
#   USER_CONFIG_DIR = get_user_config_dir()      # 模块导入时就固化，之后再设无效
# 它优先读 YOLO_CONFIG_DIR，否则落到平台默认目录（Windows 是
# ~/AppData/Roaming/Ultralytics，Linux 是 ~/.config/Ultralytics）。
#
# 不设这个变量的后果有两个，都不明显：
#   1) 字体拷到项目内也没用 —— ultralytics 压根不去那儿找，于是它**联网下载**
#      Arial.Unicode.ttf（各脚本注释里「避免联网下载」说的就是这件事）；
#   2) 在用户主目录（C 盘）创建配置文件。
# 用 setdefault：外部已显式指定则尊重外部。
CONFIG_DIR = ROOT / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))
ULTRA_CFG = CONFIG_DIR / "Ultralytics"

# ------------------------------------------------------------------ 跨平台中文渲染

# 按平台给出候选字体，取第一个存在的。
# 原先硬编码 simhei.ttf，且多数脚本的 shutil.copy 没有 try/except，非 Windows 直接崩。
_FONT_CANDIDATES = {
    "win32": [
        r"C:\Windows\Fonts\simhei.ttf",      # 黑体，原项目用的就是它
        r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
        r"C:\Windows\Fonts\simsun.ttc",      # 宋体
    ],
    "darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ],
    "linux": [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ],
}

# matplotlib 的字体族候选列表。
#
# 必须给**列表**而不是单个 "SimHei"：ultralytics 的 plt_settings() 会把自己的
# "Arial Unicode MS" 前置到该列表（utils/__init__.py:425），而拷进去的那份字体
# 内部 family 名是它自己的真实名字（SimHei / PingFang SC / WenQuanYi Zen Hei…），
# 匹配不上 "Arial Unicode MS" 就**静默变成方块 —— 只 warning 不报错**。
# 把各平台真实 family 名都列上，无论拷进去的是哪一个都能命中。
CJK_FAMILIES = [
    "SimHei",                 # Windows
    "Microsoft YaHei",        # Windows
    "PingFang SC",            # macOS
    "Heiti SC",               # macOS
    "WenQuanYi Zen Hei",      # Linux
    "Noto Sans CJK SC",       # Linux
    "Arial Unicode MS",       # 兜底
]


@functools.lru_cache(maxsize=1)
def font_path():
    """返回一个含 CJK 字形的字体文件路径，找不到返回 None。

    可用环境变量 ``PLANT_DISEASE_FONT`` 显式指定。
    结果缓存 —— 探测要遍历候选路径，调用点可能在循环里。
    """
    override = os.environ.get("PLANT_DISEASE_FONT")
    if override:
        p = pathlib.Path(override).expanduser()
        if p.exists():
            return p
        warn(f"PLANT_DISEASE_FONT 指向的字体不存在: {p}，改用自动探测")

    # 未知平台退回 Linux 的候选（多为 FreeBSD 等共享 fontconfig 路径的系统）
    for cand in _FONT_CANDIDATES.get(sys.platform, _FONT_CANDIDATES["linux"]):
        p = pathlib.Path(cand)
        if p.exists():
            return p
    return None


def setup_ultralytics_font():
    """把 CJK 字体装到 ultralytics 的配置目录，返回该目录。

    ultralytics 对非 ASCII 文本会用 ``Arial.Unicode.ttf``（utils/plotting.py:334
    的 ``check_font``），缺失时会**联网下载**。把系统自带的 CJK 字体顶到该文件名，
    即可离线渲染中文。

    失败只警告不抛异常 —— 字体缺失不应该让评估脚本整体跑不起来。
    """
    try:
        ULTRA_CFG.mkdir(parents=True, exist_ok=True)
        dst = ULTRA_CFG / "Arial.Unicode.ttf"
        if not dst.exists():
            src = font_path()
            if src is None:
                warn("未找到含中文的字体，图表中文可能显示为方块。"
                     "可设 PLANT_DISEASE_FONT 指向一个 .ttf/.ttc 解决。"
                     "Debian/Ubuntu 可装 fonts-wqy-zenhei 或 fonts-noto-cjk。")
                return ULTRA_CFG
            shutil.copy(src, dst)
            print(f"[字体] {src} → {dst}")
        return ULTRA_CFG
    except Exception as e:                                    # 只警告，不中断
        warn(f"安装中文字体失败（图表中文可能为方块）: {e}")
        return ULTRA_CFG


def font_family():
    """返回探测到的字体的**真实 family 名**（如 SimHei / PingFang SC），拿不到返回 None。

    为什么要动态取而不是写死：ultralytics 的 ``plt_settings()`` 会把自己的
    "Arial Unicode MS" 前置到 ``font.sans-serif``。若拷进去的是 simhei，其内部
    family 名是 "SimHei"，匹配不上就会**静默把中文画成方块（只 warning 不报错）**。
    动态取真实名字再放进列表，就不依赖猜。

    另外：若使用者没禁用联网、ultralytics 下载了正版 Arial Unicode，
    该字体 family 名恰为 "Arial Unicode MS"，与前置项直接命中，同样正确。
    """
    src = font_path()
    if src is None:
        return None
    try:
        from matplotlib.font_manager import FontProperties
        return FontProperties(fname=str(src)).get_name()
    except Exception:
        return None


def setup_cjk():
    """脚本里唯一需要调用的中文字体入口：装 shim + 配 matplotlib。

    **必须在 import ultralytics / matplotlib 绘图之前调用。**

    返回 ultralytics 配置目录。
    """
    cfg = setup_ultralytics_font()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam = font_family()
    # 探测到的真实 family 排最前，历史上各平台的名字与 "Arial Unicode MS" 依次兜底
    plt.rcParams["font.sans-serif"] = ([fam] if fam else []) + CJK_FAMILIES
    plt.rcParams["axes.unicode_minus"] = False
    return cfg


def ultralytics_config_dir():
    """返回 ultralytics 实际使用的配置目录（用于诊断输出）。"""
    try:
        from ultralytics.utils import USER_CONFIG_DIR
        return pathlib.Path(USER_CONFIG_DIR)
    except Exception:
        return ULTRA_CFG


# ------------------------------------------------------------------ 凭据


def roboflow_key():
    """取 Roboflow API key：环境变量优先，其次项目内 .secrets/roboflow_key.txt。

    原先三个脚本在**模块 import 期**就硬编码绝对路径读文件，别人机器上仅仅
    import 就 FileNotFoundError。这里改为按需调用，并给出可操作的错误信息。
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key

    f = ROOT / ".secrets" / "roboflow_key.txt"
    if f.exists():
        key = f.read_text(encoding="utf-8").strip()
        if key:
            return key

    raise SystemExit(
        "未找到 Roboflow API key。二选一：\n"
        f"  1) 写进 {f}（只放一行 key 本身）\n"
        "  2) 设环境变量 ROBOFLOW_API_KEY\n"
        "key 在 https://app.roboflow.com/settings/api 获取。"
    )


# ------------------------------------------------------------------ 命令行公共参数


def add_device_arg(ap):
    """加 --device。默认 None = 交给 ultralytics 自动选（有 GPU 用 GPU，没有用 CPU）。

    原先各处硬编码 device=0，在只有 CPU 的机器上会直接报错。
    """
    ap.add_argument("--device", default=None,
                    help="推理设备，如 0 / cpu；默认自动选择")
    return ap


def add_cam_arg(ap):
    """加 --cam，摄像头索引。原先硬编码为 0。"""
    ap.add_argument("--cam", type=int, default=0, help="摄像头索引，默认 0")
    return ap


# ------------------------------------------------------------------ 杂项


def warn(msg):
    print(f"[警告] {msg}", file=sys.stderr)


if __name__ == "__main__":
    # 自检：排查环境问题先跑这个。不需要数据集。
    print(f"平台         = {sys.platform}")
    print(f"ROOT         = {ROOT}")
    print(f"CLS_DATA     = {CLS_DATA}  ({'存在' if CLS_DATA.exists() else '不存在'})")
    print(f"MODELS       = {MODELS}  ({'存在' if MODELS.exists() else '不存在'})")
    print(f"RUNS         = {RUNS}")
    print(f"CONFIG_DIR   = {CONFIG_DIR}")
    print(f"字体文件      = {font_path() or '（未找到，中文图表会变方块）'}")
    print(f"字体 family   = {font_family() or '（未知）'}")
    print(f"ultra 配置目录 = {ultralytics_config_dir()}")

    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    secret = ROOT / ".secrets" / "roboflow_key.txt"
    print(f"Roboflow key = {'环境变量' if key else (str(secret) if secret.exists() else '未配置')}")