"""结构光三维重建：像素坐标 → 现实 XY → 配 z → ply 点云。

流程：读 images_rebuild 的 png → 复用 图像处理_重建 的处理流程得到 cluster 骨架
      → 取 cluster 全部前景点(像素坐标) → H 单应矩阵映射到现实 XY(mm)
      → 按图片顺序配 z 坐标 → 合并三维点 → 写 ASCII ply
运行：python 重建.py
"""

import glob
import importlib.util
import os
import re

import cv2
import numpy as np

# ---------- 路径（按需手改） ----------
IMAGE_DIR = r"C:\python\3D_project_3\takephoto\images1"                  # 输入图片文件夹
PROC_PATH = r"C:\python\3D_project_3\核心代码_标定2\图像处理_重建.py"  # 复用处理流程的模块
H_NPZ = r"C:\python\3D_project_3\核心代码_标定2\H.npz"            # 像素->现实 单应矩阵
Z_NPZ = r"C:\python\3D_project_3\核心代码_标定2\data\z\z.npz"            # z 坐标一维列表
OUTPUT_PLY = r"C:\python\3D_project_3\核心代码_标定2\pointcloud.ply"   # 输出点云


def _load_module(path, name):
    """功能：动态加载中文文件名的 python 模块，复用其函数与参数。
    输入：path：str，模块文件路径；name：str，注册的模块名
    输出：module 对象；文件不存在返回 None
    """
    if not os.path.isfile(path):
        print(f"找不到模块：{path}")
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_z(z_npz):
    """功能：从 z.npz 读取一维 z 坐标列表，优先取字段 z，否则取第一个字段。
    输入：z_npz：str，z.npz 路径
    输出：(N,) float64，z 坐标数组
    """
    data = np.load(z_npz)
    key = "z" if "z" in data.files else data.files[0]
    return np.asarray(data[key], np.float64).ravel()


def _pixels_to_xy(pix, H):
    """功能：用单应矩阵 H 把像素坐标映射到现实平面 XY。
    输入：pix：(N,2) float，(x=列, y=行) 像素坐标；H：(3,3) 单应矩阵
    输出：(N,2) float，(X, Y) 现实坐标
    """
    n = len(pix)
    uv = np.column_stack((pix[:, 0], pix[:, 1], np.ones(n)))  # 齐次 (u, v, 1)
    p = H @ uv.T                                              # (3, N)
    return np.column_stack((p[0] / p[2], p[1] / p[2]))


def _write_ply(pts, path):
    """功能：把 (N,3) 点云写为 ASCII ply 文件。
    输入：pts：(N,3) float；path：str，输出路径
    输出：无
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def reconstruct(image_dir=IMAGE_DIR, h_npz=H_NPZ, z_npz=Z_NPZ, calib_npz=None,
                out_ply=OUTPUT_PLY, proc_path=PROC_PATH):
    """功能：逐图提取骨架前景点→H 映射现实 XY→配 z→合并三维点→写 ASCII ply。
    输入：image_dir：str，重建图片文件夹；h_npz：str，像素->现实单应矩阵路径；
          z_npz：str，z 坐标一维列表路径；calib_npz：str 或 None，鱼眼标定结果(K/D)路径，
          None 时回退到处理模块内置的 CALIB_NPZ；out_ply：str，输出 ply 路径；proc_path：str，复用处理模块路径
    输出：str 或 None，out_ply 输出路径；模块/图片/z 缺失或未生成任何三维点时返回 None
    """
    proc = _load_module(proc_path, "image_proc_rebuild")
    if proc is None:
        return None
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))  # 按文件名数字自然排序，与 z 列表一一对应
    if not files:
        print(f"文件夹内没有 png：{image_dir}")
        return None
    zs = _load_z(z_npz)
    if len(zs) != len(files):
        print(f"z 坐标数量({len(zs)})与图片数量({len(files)})不一致")
        return None

    if calib_npz is not None:
        setattr(proc, "CALIB_NPZ", calib_npz)              # 指定时覆盖，image_processing 内部用它加载 K/D
    H = np.load(h_npz)["H"]                                   # 像素->现实 单应矩阵

    clouds = []                                               # 累积每张图的三维点 (N,3)

    cv2.imwrite = lambda *a, **k: True                        # 屏蔽中间图写盘，避免写死路径文件
    points_list = proc.image_processing(image_dir)            # 一次处理整个文件夹，返回每张图的点集 list
    if len(points_list) != len(files):
        print(f"image_processing 返回点集数({len(points_list)})与图片数({len(files)})不一致")
        return None

    for z, fp, points in zip(zs, files, points_list):
        if points is None or len(points) == 0:
            print(f"未提取到骨架点：{os.path.basename(fp)}")
            continue
        pix = np.asarray(points, np.float64)[:, ::-1]         # [y, x] -> (x=列, y=行)
        xy = _pixels_to_xy(pix, H)                            # 现实 XY(mm)
        xyz = np.column_stack((xy, np.full(len(xy), float(z))))  # 补 z 坐标
      
        clouds.append(xyz)
        print(f"{os.path.basename(fp)}: {len(xyz)} 点, z={z}")

    if not clouds:
        print("未生成任何三维点，未保存点云")
        return None
    pts = np.vstack(clouds)
    _write_ply(pts, out_ply)
    print(f"已保存 {len(pts)} 个三维点到 {out_ply}")
    return out_ply


def main():
    """功能：脚本入口，用默认路径常量跑一次重建流程。
    输入：无
    输出：无
    """
    reconstruct()


if __name__ == "__main__":
    main()
