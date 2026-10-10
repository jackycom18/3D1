"""文件工具：data 目录约定、自动编号、取最后文件。

供主程序与各模块复用：统一在选定工作目录下创建 data/ 数据文件夹，
内部按文件类型分 9 个子文件夹；同名文件按“前缀+序号+扩展名”自动编号。
"""

import os
import re

# data 下 9 个子文件夹名，按文件类型分类（与核心算法输出文件一一对应）
DATA_SUBDIRS = (
    "neican",         # 鱼眼标定结果（内参）calib_resultN.npz
    "centroids",      # 四点交点像素坐标 centroidsN.npz
    "xy",             # 平移台 xy 移动记录 xyN.npz
    "初始四点坐标",    # 初始四点现实坐标 初始四点坐标N.npz
    "ang_move",       # 位移台夹角 ang_moveN.npz
    "H",              # 单应矩阵 HN.npz
    "points",         # 激光线像素坐标 pointsN.npz
    "z",              # 深度值 zN.npz
    "pointcloud",     # 重建点云 pointcloudN.ply
)


def ensure_data_dirs(work_dir):
    """功能：在 work_dir 下创建 data/ 及其 9 个类型子文件夹，返回 data 目录路径。
    输入：work_dir：str，工作目录（照片文件夹的父目录）
    输出：str，data 目录的绝对路径
    """
    data_dir = os.path.join(work_dir, "data")
    for sub in DATA_SUBDIRS:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
    return data_dir


def _scan_named(dir_path, base, ext):
    """功能：扫描目录里形如 base+序号+ext 的文件，返回最大序号与其文件路径。
    输入：dir_path：str，目录；base：str，文件名前缀；ext：str，扩展名（含点，如 ".npz"）
    输出：tuple(int, str 或 None)，(最大序号, 序号最大的文件路径)；无匹配时为 (-1, None)
    """
    if not os.path.isdir(dir_path):
        return -1, None
    pat = re.compile(re.escape(base) + r"(\d+)" + re.escape(ext) + r"$")
    max_n = -1                      # 当前扫描到的最大序号
    best = None                     # 序号最大文件的完整路径
    for name in os.listdir(dir_path):
        m = pat.match(name)
        if not m:
            continue
        n = int(m.group(1))
        if n > max_n:
            max_n = n
            best = os.path.join(dir_path, name)
    return max_n, best


def next_path(dir_path, base, ext):
    """功能：返回 dir_path 下 base+序号+ext 的下一个可用文件路径（序号自动 +1）。
    输入：dir_path：str，目录（不存在会自动创建）；base：str，文件名前缀；ext：str，扩展名
    输出：str，新文件完整路径，如 base1.npz
    """
    os.makedirs(dir_path, exist_ok=True)
    max_n, _ = _scan_named(dir_path, base, ext)
    start = max(max_n + 1, 1)   # 首个序号从 1 开始；已有文件则取最大序号 +1
    return os.path.join(dir_path, f"{base}{start}{ext}")


def latest_file(dir_path, base, ext):
    """功能：返回 dir_path 下 base+序号+ext 中序号最大的文件路径（多个数据取最后一个）。
    输入：dir_path：str，目录；base：str，文件名前缀；ext：str，扩展名
    输出：str 或 None，序号最大的文件完整路径；无匹配时返回 None
    """
    _, best = _scan_named(dir_path, base, ext)
    return best
