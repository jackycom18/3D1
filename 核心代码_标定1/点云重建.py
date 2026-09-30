"""点云重建：用 H 单应矩阵把像素坐标映射到光平面 XY，拼上深度 z 得到三维点云，写 PLY。

流程：
1. 读 H.npz 单应矩阵 H、calib_result.npz 内参 K/D
2. 读像素坐标 npz (N,2) 与 z 深度 npz (N,)
3. cv2.fisheye.undistortPoints 把像素坐标去畸变到 z=1 虚拟平面
4. 虚拟平面坐标乘 H 得到光平面现实 XY(mm)
5. XY 拼 z 成 (N,3) 点云，写 ASCII PLY
运行：py -3.8 点云重建.py
"""

import numpy as np
import cv2

# ---------- 可调参数（路径按实际修改） ----------
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"    # 鱼眼标定结果（含 K/D）
H_NPZ = r"C:\python\3D_project_3\H.npz"                    # 单应矩阵（第 4 步输出）
POINTS_NPZ = r"C:\python\3D_project_3\points.npz"          # 激光线像素坐标 (N,2)
Z_NPZ = r"C:\python\3D_project_3\z.npz"                    # 每个点的深度 z (N,)，单位 mm
OUT_PLY = r"C:\python\3D_project_3\pointcloud.ply"         # 点云保存路径


def _load_array(npz_path, names, ndim, ncol=None):
    """功能：按候选字段名读 npz 里某个数组，读不到再按形状兜底匹配。
    输入：npz_path：str，npz 路径；names：tuple，候选字段名；ndim：int，期望维度；ncol：int 或 None，第二维列数
    输出：np.ndarray float64；找不到返回 None
    """
    data = np.load(npz_path)
    for name in names:                                       # 优先按字段名精确匹配
        if name in data:
            return np.asarray(data[name], np.float64)
    for name in data.files:                                  # 字段名对不上时按形状兜底
        arr = np.asarray(data[name], np.float64)
        if arr.ndim == ndim and (ncol is None or (arr.ndim >= 2 and arr.shape[1] == ncol)):
            return arr
    return None


def _pixels_to_plane(pixels, K, D):
    """功能：把像素坐标去畸变投影到 z=1 虚拟平面。
    输入：pixels：(N,2) float，像素坐标(uv)；K：(3,3) float，内参；D：(4,1) float，畸变系数
    输出：(N,2) float，虚拟平面归一化坐标 (x,y)
    """
    pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)


def _plane_to_xy(plane, H):
    """功能：虚拟平面坐标乘单应矩阵 H，得到光平面现实 XY(mm)。
    输入：plane：(N,2) float，虚拟平面坐标；H：(3,3) float，单应矩阵
    输出：(N,2) float，光平面 XY(mm)
    """
    ones = np.hstack([plane, np.ones((len(plane), 1))])       # 齐次坐标 [x, y, 1]
    proj = ones @ H.T
    return proj[:, :2] / proj[:, 2:3]                         # 齐次归一化


def _write_ply(path, xyz):
    """功能：把 (N,3) 点云写为 ASCII PLY 文件。
    输入：path：str，输出 ply 路径；xyz：(N,3) float，点云坐标(mm)
    输出：None
    """
    header = ("ply\n"
              "format ascii 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\n"
              "property float y\n"
              "property float z\n"
              "end_header\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, xyz, fmt="%.6f")


def reconstruct(calib_npz=CALIB_NPZ, h_npz=H_NPZ, points_npz=POINTS_NPZ,
                z_npz=Z_NPZ, out_ply=OUT_PLY):
    """功能：读内参/单应/像素坐标/深度，重建 (N,3) 点云并写 PLY。
    输入：calib_npz：str，内参 npz；h_npz：str，单应 npz；points_npz：str，像素坐标 npz；
          z_npz：str，深度 npz；out_ply：str，ply 保存路径
    输出：(N,3) float 点云(mm)；数据缺失或点数不匹配返回 None
    """
    cal = np.load(calib_npz)
    K = cal["K"].astype(np.float64)                          # 内参矩阵
    D = cal["D"].astype(np.float64)                          # 畸变系数
    H = np.load(h_npz)["H"].astype(np.float64)               # 单应矩阵

    pixels = _load_array(points_npz, ("points", "uv", "px", "pixels"), 2, 2)
    if pixels is None:
        print("像素坐标 npz 里找不到 (N,2) 数组，请确认字段名")
        return None
    pixels = pixels.reshape(-1, 2)

    z = _load_array(z_npz, ("z", "depth", "d", "depths"), 1)
    if z is None:                                            # 也兼容 (N,1) 二维存法
        z = _load_array(z_npz, ("z", "depth", "d", "depths"), 2, 1)
    if z is None:
        print("z 坐标 npz 里找不到一维数组，请确认字段名")
        return None
    z = z.ravel()

    if len(pixels) != len(z):                                # 点与 z 必须逐点一一对应
        print(f"像素坐标点数 {len(pixels)} 与 z 点数 {len(z)} 不一致，无法一一对应")
        return None

    plane = _pixels_to_plane(pixels, K, D)                   # 去畸变到虚拟平面
    xy = _plane_to_xy(plane, H)                              # 光平面 XY(mm)
    xyz = np.column_stack([xy, z])                           # (N,3) 点云

    _write_ply(out_ply, xyz)
    return xyz


if __name__ == "__main__":
    xyz = reconstruct()
    if xyz is not None:
        print(f"点云重建完成：{len(xyz)} 个点，已保存到 {OUT_PLY}")
