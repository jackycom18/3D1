"""标定第 2 步：用鱼眼内参把图像交点去畸变到 z=1 虚拟平面，反解两个方向角并重建光平面四点现实坐标。

流程：
1. 读 calib_result.npz 内参 K/D，读 centroids.npz 第一组 4 个交点像素坐标
2. cv2.fisheye.undistortPoints 去畸变到 z=1 虚拟平面，后续所有点计算都用虚拟平面坐标
3. 求左上角两条相邻边的夹角 ang、左上角到两相邻点距离比 k2
4. 解方程组 k1·(sin ang1 / sin ang2) = k2 与 cos ang = cos ang1 · cos ang2，得 ang1、ang2
5. 用第一组点的 ang1、ang2、ang 重建光平面四点现实坐标(左上为原点、左上→右上为 x 正半轴)，单位 mm，存 npz
运行：py -3.8 标定第二步.py
"""

import cv2
import numpy as np

# ---------- 可调参数（数值按需手改） ----------
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"    # 鱼眼标定结果（第 1 步输出）
CENTROIDS_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 多组四点交点像素坐标
OUT_NPZ = r"C:\python\3D_project_3\初始四点坐标.npz"       # 重建四点现实坐标保存路径
K1 = 1.0        # 镂空区域长宽比（正方形=1）
L_MM = 10.0     # 镂空区域现实长度(mm)，按实际修改
W_MM = 10.0     # 镂空区域现实宽度(mm)，按实际修改


def _undistort_to_plane(points_px, K, D):
    """功能：把 4 个交点像素坐标去畸变投影到 z=1 虚拟平面。
    输入：points_px：(4,2) float，像素坐标(uv)；K：(3,3) float，内参矩阵；D：(4,1) float，畸变系数
    输出：(4,2) float，z=1 虚拟平面上的理想归一化坐标 (x,y)
    """
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)   # undistortPoints 要求 (N,1,2)
    norm = cv2.fisheye.undistortPoints(pts, K, D)               # 理想针孔归一化坐标，即 z=1 平面
    return norm.reshape(-1, 2)


def _corner_geometry(plane):
    """功能：在虚拟平面上求左上角两邻边夹角 ang 与左上角到两相邻点的距离比 k2。
    输入：plane：(4,2) float，虚拟平面坐标，按 左上→右上→右下→左下 排序
    输出：(ang, k2)：ang 为左上角两条相邻边的夹角(弧度,0~π)，k2 = |左上→右上| / |左上→左下|
    """
    p0, p1, _, p3 = plane                    # 左上、右上、左下；右下不参与
    v1 = p1 - p0                             # 左上→右上
    v3 = p3 - p0                             # 左上→左下
    n1, n3 = np.linalg.norm(v1), np.linalg.norm(v3)   # 两邻边长度
    ang = float(np.arccos(np.clip(np.dot(v1, v3) / (n1 * n3), -1.0, 1.0)))  # 两邻边夹角
    k2 = float(n1 / n3)                      # 距离比
    return ang, k2


def _solve_angles(ang, k2, k1):
    """功能：解方程组求 ang1、ang2。
    输入：ang：float，左上角两邻边夹角(弧度)；k2：float，距离比；k1：float，长宽比
    输出：(ang1, ang2) 弧度，均取 (0, π) 内有效解；无有效解返回 (None, None)
    """
    r = k2 / k1                              # sin(ang1) = r·sin(ang2)
    C = float(np.cos(ang))                   # cos(ang1)·cos(ang2) = C
    a = r * r
    b = -(1.0 + r * r)
    c = 1.0 - C * C
    disc = b * b - 4.0 * a * c               # 二次方程判别式
    if disc < 0:
        return None, None
    sq = np.sqrt(disc)
    for u in ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)):   # u = sin²(ang2) 两个候选根
        if not (0.0 < u <= 1.0):
            continue
        s2 = np.sqrt(u)                      # sin(ang2)，ang2 取 (0,π) 内正值
        s1 = r * s2                          # sin(ang1)
        if s1 > 1.0:
            continue
        c2 = np.sqrt(1.0 - u)                # cos(ang2) 取正（ang2 锐角）
        c1 = C / c2 if c2 != 0 else 0.0      # 由 cos(ang1)·cos(ang2)=C 反推 cos(ang1)
        if abs(c1 * c1 - (1.0 - s1 * s1)) > 1e-9:  # 校验 c1² + s1² = 1
            continue
        ang1 = float(np.arctan2(s1, c1))
        ang2 = float(np.arctan2(s2, c2))
        if ang1 <= 0.0 or ang2 <= 0.0:
            continue
        return ang1, ang2
    return None, None


def _rebuild_points(ang1, ang2, ang, L_mm, W_mm):
    """功能：用第一组点的角度与镂空现实长宽重建光平面上四点现实坐标(左上为原点)。
    输入：ang1、ang2：float，两个方向角(弧度)；ang：float，左上角两邻边夹角(弧度)；
          L_mm、W_mm：float，镂空区域现实长、宽(mm)
    输出：(4,2) float，四点坐标(mm)，顺序 左上→右上→右下→左下
    """
    d1 = L_mm / np.sin(ang1)                 # 左上→右上现实距离(mm)，x 正半轴
    d2 = W_mm / np.sin(ang2)                 # 左上→左下现实距离(mm)
    ox, oy = d2 * np.cos(ang), d2 * np.sin(ang)   # 左下相对左上的偏移量
    return np.array([
        [0.0, 0.0],                          # 左上
        [d1, 0.0],                           # 右上
        [d1 + ox, oy],                       # 右下 = 右上 + 偏移
        [ox, oy],                            # 左下
    ], np.float64)


def calibrate_step2(calib_npz=CALIB_NPZ, centroids_npz=CENTROIDS_NPZ,
                    k1=K1, L_mm=L_MM, W_mm=W_MM, out_npz=OUT_NPZ):
    """功能：读内参与第一组交点，去畸变到虚拟平面，解方向角并重建四点现实坐标(mm)存 npz。
    输入：calib_npz：str，内参 npz；centroids_npz：str，交点 npz；k1：float，长宽比；
          L_mm、W_mm：float，镂空现实长宽(mm)；out_npz：str，结果保存路径
    输出：(ang1, ang2) 弧度；无有效解时返回 (None, None)
    """
    cal = np.load(calib_npz)
    K = cal["K"].astype(np.float64)          # 内参矩阵
    D = cal["D"].astype(np.float64)          # 畸变系数
    data = np.load(centroids_npz)
    if "points" in data:
        first = np.asarray(data["points"], np.float64).reshape(-1, 4, 2)[0]
    else:                                    # 旧版多个 imgX 键：取第一个键的第一组
        key = sorted(data.files)[0]
        first = np.asarray(data[key], np.float64).reshape(-1, 4, 2)[0]

    plane = _undistort_to_plane(first, K, D)          # 虚拟平面四点
    ang, k2 = _corner_geometry(plane)                 # 夹角与距离比
    ang1, ang2 = _solve_angles(ang, k2, k1)           # 解两个方向角
    if ang1 is None:
        return None, None
    pts4 = _rebuild_points(ang1, ang2, ang, L_mm, W_mm)   # 光平面四点现实坐标(mm)
    np.savez_compressed(out_npz, points=pts4, unit="mm")
    return ang1, ang2


if __name__ == "__main__":
    ang1, ang2 = calibrate_step2()
    if ang1 is None:
        print("无有效解，请检查交点顺序或长宽比 k1")
    else:
        print(f"ang1 = {np.degrees(ang1):.4f}°, ang2 = {np.degrees(ang2):.4f}°")
