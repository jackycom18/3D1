"""结构光三维重建（Steger 升级版）：像素坐标 → 现实 XY → 配 z → ply 点云与可视化图像。

流程：
1. 批量处理 images1 中的 22 张 png，调用 图像处理_重建1.py (Steger 亚像素算法)
2. 利用 H 单应矩阵将亚像素坐标映射到现实平面 XY(mm)
3. 对应位移台真实深度 z(mm)，合成 (X, Y, Z) 三维点云
4. 导出 pointcloud_steger.ply 和 pointcloud.ply
5. 生成多视角三维点云渲染图像 pointcloud_3d_view.png

运行：py -3.8 重建1.py
"""

import glob
import importlib.util
import os
import re
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 路径（按需手改） ----------
IMAGE_DIR = r"C:\python\3D_project_3\takephoto\images1"                  # 重建输入图片文件夹
PROC_PATH = r"C:\python\3D_project_3\核心代码_标定2\图像处理_重建1.py"  # Steger 处理模块
H_NPZ = r"C:\python\3D_project_3\核心代码_标定2\H.npz"                  # 像素->现实 单应矩阵
Z_NPZ = r"C:\python\3D_project_3\核心代码_标定2\data\z\z.npz"          # z 坐标一维列表
OUTPUT_PLY = r"C:\python\3D_project_3\核心代码_标定2\pointcloud_steger.ply"   # 输出 Steger 点云
STANDARD_PLY = r"C:\python\3D_project_3\核心代码_标定2\pointcloud.ply"        # 覆盖标准点云
DEBUG_DIR = r"C:\python\3D_project_3\_debug"                           # 输出可视化渲染图目录


def _load_module(path, name):
    """动态加载 python 模块。"""
    if not os.path.isfile(path):
        print(f"找不到模块：{path}")
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_z(z_npz, n_files):
    """读取 z 坐标序列，缺失则自动回退为等间距步进序列。"""
    if os.path.isfile(z_npz):
        data = np.load(z_npz)
        key = "z" if "z" in data.files else data.files[0]
        zs = np.asarray(data[key], np.float64).ravel()
        if len(zs) == n_files:
            return zs
    print(f"提示：未找到匹配的 {z_npz}，使用默认等步进 z=0, 1, 2... {n_files-1} mm")
    return np.arange(n_files, dtype=np.float64)


def _pixels_to_xy(pix, H):
    """用单应矩阵 H 把像素坐标映射到现实平面 XY(mm)。"""
    n = len(pix)
    uv = np.column_stack((pix[:, 0], pix[:, 1], np.ones(n)))
    p = H @ uv.T
    return np.column_stack((p[0] / p[2], p[1] / p[2]))


def _write_ply(pts, path):
    """写 ASCII PLY 点云文件。"""
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


def render_pointcloud_images(pts, out_png):
    """把三维点云绘制为多视角工程图（3D 轴测图、俯视图 XY、正视图 XZ、侧视图 YZ）。"""
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.suptitle("3D Laser Point Cloud Reconstruction (Steger Subpixel)", fontsize=16, fontweight='bold', y=0.98)

    # 1. 3D 轴测图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    p1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=0.8, alpha=0.8)
    ax1.set_title("3D Isometric View", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    fig.colorbar(p1, ax=ax1, shrink=0.6, label="Depth Z (mm)")

    # 2. 俯视图 XY (显示截面轮廓度与几何尺寸)
    ax2 = fig.add_subplot(2, 2, 2)
    p2 = ax2.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis', s=0.8, alpha=0.8)
    ax2.set_title("Top View (XY Cross-Section)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. 正视图 XZ (显示立壁垂直度与平整度)
    ax3 = fig.add_subplot(2, 2, 3)
    p3 = ax3.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], cmap='plasma', s=0.8, alpha=0.8)
    ax3.set_title("Front View (XZ Wall Profile)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. 侧视图 YZ (显示另一侧立壁垂直度)
    ax4 = fig.add_subplot(2, 2, 4)
    p4 = ax4.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap='plasma', s=0.8, alpha=0.8)
    ax4.set_title("Side View (YZ Wall Profile)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"点云多视角渲染图已保存至：{out_png}")


def reconstruct(image_dir=IMAGE_DIR, h_npz=H_NPZ, z_npz=Z_NPZ,
                out_ply=OUTPUT_PLY, proc_path=PROC_PATH):
    """主重建流程。"""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    proc = _load_module(proc_path, "image_proc_rebuild1")
    if proc is None:
        return None

    def _file_index(f):
        m = re.search(r'\d+', os.path.basename(f))
        return int(m.group()) if m else 0

    files = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=_file_index)
    if not files:
        print(f"文件夹内没有 png：{image_dir}")
        return None

    zs = _load_z(z_npz, len(files))
    H = np.load(h_npz)["H"]

    print(f"\n==================================================")
    print(f"正在进行 3D 点云重建 (采用 Steger 亚像素算法)...")
    print(f"输入图像目录: {image_dir} (共 {len(files)} 帧)")
    print(f"单应矩阵 H: {h_npz}")
    print(f"==================================================")

    t0_all = time.time()
    # 逐图执行 Steger 提取
    points_list = proc.image_processing(image_dir, method="steger")

    clouds = []
    for z, fp, points in zip(zs, files, points_list):
        if points is None or len(points) == 0:
            print(f"未提取到点：{os.path.basename(fp)}")
            continue
        pix = np.asarray(points, np.float64)[:, ::-1]  # [y, x] -> (u=列, v=行)
        xy = _pixels_to_xy(pix, H)                     # 现实平面 XY(mm)
        xyz = np.column_stack((xy, np.full(len(xy), float(z))))
        clouds.append(xyz)

    if not clouds:
        print("未生成任何有效三维点")
        return None

    pts = np.vstack(clouds)
    t_total = time.time() - t0_all

    # 保存 PLY 点云文件
    _write_ply(pts, out_ply)
    _write_ply(pts, STANDARD_PLY)
    print(f"\n三维点云构建完成！")
    print(f"总点数: {len(pts)} 个三维点")
    print(f"总耗时: {t_total:.2f} 秒 (平均每帧 {(t_total/len(files))*1000:.1f} ms)")
    print(f"已导出 PLY 文件: {out_ply}")
    print(f"已同步更新: {STANDARD_PLY}")

    # 绘制高精度多视角可视化图像
    render_out = os.path.join(DEBUG_DIR, "pointcloud_3d_view.png")
    render_pointcloud_images(pts, render_out)

    return out_ply


if __name__ == "__main__":
    reconstruct()
