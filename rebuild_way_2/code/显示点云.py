"""点云显示程序：读取点云文件（.ply / .npz）并用 Open3D 交互式窗口显示。

功能：加载点云 → 按 z 深度着色 → 打印坐标范围 → 打开交互窗口。
      窗口操作：左键拖动旋转，滚轮缩放，Shift+左键平移，R 键复位视角。
用法：
    py -3.8 显示点云.py                 # 用默认路径（rebuild_way_2/data/pointcloud/pointcloud_steger.ply）
    py -3.8 显示点云.py <文件路径>       # 指定 .ply 或 .npz 点云文件
"""

import os
import sys
import numpy as np
import open3d as o3d
_HERE = os.path.dirname(os.path.abspath(__file__))                     # 本文件所在目录
DEFAULT_PLY = os.path.join(os.path.dirname(_HERE), "data", "pointcloud", "pointcloud_steger.ply")  # 默认点云文件路径


def load_pointcloud(path):
    """功能：从 ply 或 npz 文件读取点云坐标为 (N,3) 数组。
    输入：path：str，点云文件路径（.ply 或 .npz）
    输出：(N,3) float64 点云数组；文件缺失或格式不支持返回 None
    """
    if not os.path.isfile(path):
        print(f"找不到点云文件：{path}")
        return None
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        pts = np.asarray(o3d.io.read_point_cloud(path).points, np.float64)  # Open3D 直接读 ply
    elif ext == ".npz":
        data = np.load(path)
        key = next((k for k in ("points", "xyz", "vertices") if k in data), None)  # 优先按常见字段名
        if key is None:                                                    # 字段名不符时按 (N,3) 形状兜底
            key = next((k for k in data.files
                        if np.asarray(data[k]).ndim == 2 and np.asarray(data[k]).shape[1] == 3), None)
        if key is None:
            print("npz 中找不到 (N,3) 点云字段")
            return None
        pts = np.asarray(data[key], np.float64).reshape(-1, 3)
    else:
        print(f"不支持的文件类型：{ext}（仅支持 .ply / .npz）")
        return None

    return pts


def show(pts, title="点云"):
    """功能：把点云按 z 深度着色并用 Open3D 交互窗口显示，同时打印坐标范围。
    输入：pts：(N,3) float 点云；title：str，窗口标题
    输出：无
    """
    if pts is None or len(pts) == 0:
        print("点云为空，无法显示")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    z = pts[:, 2]                                              # z 深度，用于颜色渐变
    t = (z - z.min()) / (z.max() - z.min()) if z.max() > z.min() else np.zeros_like(z)
    pcd.colors = o3d.utility.Vector3dVector(                   # 近处偏红、远处偏蓝，直观区分深度
        np.column_stack((t, 0.3 * np.ones_like(t), 1.0 - t)))

    print(f"点数：{len(pts)}")
    print(f"X 范围：{pts[:, 0].min():.3f} ~ {pts[:, 0].max():.3f} mm")
    print(f"Y 范围：{pts[:, 1].min():.3f} ~ {pts[:, 1].max():.3f} mm")
    print(f"Z 范围：{pts[:, 2].min():.3f} ~ {pts[:, 2].max():.3f} mm")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)  # 原点坐标轴，作尺度参考
    o3d.visualization.draw_geometries([pcd, axes], window_name=title)


def main():
    """功能：程序入口，读命令行指定的点云文件（缺省用默认路径）并显示。
    输入：无（命令行可选参数：点云文件路径）
    输出：无
    """
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLY
    pts = load_pointcloud(path)
    if pts is not None:
        show(pts, title=os.path.basename(path))


if __name__ == "__main__":
    main()
