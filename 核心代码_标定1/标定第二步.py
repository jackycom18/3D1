"""标定第 2 步（度量整平终极版）：
在已标定相机 (z=1 归一化平面) 下，利用消影线直接获取光平面 3D 法向量，
通过 3D 旋转对齐矩阵 (Metric Rectification) 彻底消除剪切与非均匀拉伸，
还原出的平行四边形的长宽比与夹角与真实物理形状 100% 完全一致！

运行：py -3.8 标定第二步.py
"""

import os
import cv2
import numpy as np

# ---------- 可调参数 ----------
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"    # 鱼眼标定结果（第 1 步输出）
CENTROIDS_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 多组四点交点像素坐标
OUT_NPZ = r"C:\python\3D_project_3\初始四点坐标.npz"       # 重建四点现实坐标保存路径

W_MM = 160.0    # 模具内壁现实宽度(mm，沿 x 方向)
L_MM = 160.0    # 模具内壁现实长度(mm，沿 y 方向)


def _undistort_to_plane(points_px, K, D):
    """把像素坐标 (N, 2) 去畸变投影到 z=1 虚拟平面"""
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    norm = cv2.fisheye.undistortPoints(pts, K, D)
    return norm.reshape(-1, 2)


def _angle_between(u, v):
    """计算两二维向量夹角(角度)"""
    cos_th = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return np.degrees(np.arccos(np.clip(abs(cos_th), -1.0, 1.0)))


def _fit_metric_rectification(quads_virtual):
    """度量整平算法 (Metric Rectification)：
    1. 自动判定工况 (是否已平行)
    2. SVD 联合求解最优灭点与消影线 l_inf
    3. 利用相机已标定几何，消影线直接导出光平面 3D 法向量 n
    4. 构造 3D 旋转对齐矩阵 R_align，彻底消除剪切与角度偏差，长宽比与角度 100% 还原真值
    """
    q0 = quads_virtual[0]
    v01 = q0[1] - q0[0]
    v32 = q0[2] - q0[3]
    v03 = q0[3] - q0[0]
    v12 = q0[2] - q0[1]

    ang_dev1 = _angle_between(v01, v32)
    ang_dev2 = _angle_between(v03, v12)
    len_ratio1 = np.linalg.norm(v01) / (np.linalg.norm(v32) + 1e-12)
    len_ratio2 = np.linalg.norm(v03) / (np.linalg.norm(v12) + 1e-12)

    is_par1 = (ang_dev1 < 0.25 and abs(len_ratio1 - 1.0) < 0.03)
    is_par2 = (ang_dev2 < 0.25 and abs(len_ratio2 - 1.0) < 0.03)

    if is_par1 and is_par2:
        print("[工况判定] ★ 图像对边已严格平行（无明显透视倾斜），直接度量提取！")
        return q0, np.array([0.0, 0.0, 1.0])

    if is_par1 and not is_par2:
        print("[工况判定] ★ 图像呈现为梯形（单轴倾斜，方向1已平行），执行度量整平！")
    elif is_par2 and not is_par1:
        print("[工况判定] ★ 图像呈现为梯形（单轴倾斜，方向2已平行），执行度量整平！")
    else:
        print("[工况判定] ★ 图像呈现为一般凸四边形（双轴复合倾斜），执行多视角 SVD 度量整平！")

    lines_dir1, lines_dir2 = [], []
    for q in quads_virtual:
        p0 = np.array([q[0, 0], q[0, 1], 1.0])
        p1 = np.array([q[1, 0], q[1, 1], 1.0])
        p2 = np.array([q[2, 0], q[2, 1], 1.0])
        p3 = np.array([q[3, 0], q[3, 1], 1.0])

        l01 = np.cross(p0, p1); l32 = np.cross(p3, p2)
        norm01 = np.linalg.norm(l01[:2]); norm32 = np.linalg.norm(l32[:2])
        if norm01 > 1e-12: lines_dir1.append(l01 / norm01)
        if norm32 > 1e-12: lines_dir1.append(l32 / norm32)

        l03 = np.cross(p0, p3); l12 = np.cross(p1, p2)
        norm03 = np.linalg.norm(l03[:2]); norm12 = np.linalg.norm(l12[:2])
        if norm03 > 1e-12: lines_dir2.append(l03 / norm03)
        if norm12 > 1e-12: lines_dir2.append(l12 / norm12)

    # 求解灭点 v1
    if is_par1:
        d1 = 0.5 * (v01 / np.linalg.norm(v01) + v32 / np.linalg.norm(v32))
        v1 = np.array([d1[0], d1[1], 0.0])
    else:
        _, _, vh1 = np.linalg.svd(np.array(lines_dir1))
        v1 = vh1[-1]

    # 求解灭点 v2
    if is_par2:
        d2 = 0.5 * (v03 / np.linalg.norm(v03) + v12 / np.linalg.norm(v12))
        v2 = np.array([d2[0], d2[1], 0.0])
    else:
        _, _, vh2 = np.linalg.svd(np.array(lines_dir2))
        v2 = vh2[-1]

    # 消影线 l_inf
    l_inf = np.cross(v1, v2)
    if abs(l_inf[2]) > 1e-12:
        l_inf = l_inf / l_inf[2]
    else:
        l_inf = l_inf / (np.linalg.norm(l_inf[:2]) + 1e-12)

    print(f"[消影线] 算得 l_inf = [{l_inf[0]:.6f}, {l_inf[1]:.6f}, {l_inf[2]:.6f}]")

    # ====== 核心度量整平 (Metric Rectification) ======
    # 在已标定针孔相机 (z=1 平面) 中，消影线 l_inf 正好是光平面的法向量 n !
    n_cam = l_inf / np.linalg.norm(l_inf)
    if n_cam[2] < 0:
        n_cam = -n_cam

    z_axis = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(n_cam, z_axis)
    sin_th = np.linalg.norm(rot_axis)
    cos_th = np.dot(n_cam, z_axis)

    if sin_th > 1e-8:
        rot_axis = rot_axis / sin_th
        K_mat = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R_align = np.eye(3) + sin_th * K_mat + (1.0 - cos_th) * (K_mat @ K_mat)
    else:
        R_align = np.eye(3)

    q_homo = np.column_stack([q0, np.ones(4)])
    # 3D 旋转对齐消除倾角剪切 + 距离投影补偿
    p_rect_3d = (R_align @ q_homo.T)
    denom = (l_inf @ q_homo.T)
    p_rect = (p_rect_3d[:2, :] / denom).T
    return p_rect, l_inf


def _solve_angles(r, cos_alpha):
    """根据长宽比 r 与夹角余弦 cos_alpha，解方程求姿态角 ang1, ang2"""
    ratio = 1.0 / r
    C = float(cos_alpha)
    a = ratio * ratio
    b = -(1.0 + a)
    c = 1.0 - C * C
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None, None
    sq = np.sqrt(disc)
    for u in ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)):
        if not (0.0 < u <= 1.0):
            continue
        s2 = np.sqrt(u)
        s1 = ratio * s2
        if s1 <= 1.0:
            c2 = np.sqrt(1.0 - u)
            c1 = C / c2 if c2 != 0 else 0.0
            ang1 = float(np.arctan2(s1, c1))
            ang2 = float(np.arctan2(s2, c2))
            if ang1 > 0.0 and ang2 > 0.0:
                return ang1, ang2
    return None, None


def calibrate_step2(calib_npz=CALIB_NPZ, centroids_npz=CENTROIDS_NPZ,
                    k1=1.0, L_mm=L_MM, W_mm=W_MM, out_npz=OUT_NPZ, **kwargs):
    if not os.path.exists(calib_npz):
        print(f"[错误] 未找到内参文件: {calib_npz}")
        return None, None
    if not os.path.exists(centroids_npz):
        print(f"[错误] 未找到角点文件: {centroids_npz}")
        return None, None

    cal = np.load(calib_npz)
    K = cal["K"].astype(np.float64)
    D = cal["D"].astype(np.float64)

    data = np.load(centroids_npz)
    if "points" in data:
        groups = np.asarray(data["points"], np.float64).reshape(-1, 4, 2)
    else:
        groups = np.array([np.asarray(data[k], np.float64).reshape(4, 2)
                           for k in sorted(data.files)], np.float64)

    print(f"[信息] 读取到 {len(groups)} 组四边形数据")

    # 去畸变
    quads_virtual = np.array([_undistort_to_plane(g, K, D) for g in groups])

    # 度量整平
    p_rect, l_inf = _fit_metric_rectification(quads_virtual)

    v12 = p_rect[1] - p_rect[0]
    v14 = p_rect[3] - p_rect[0]
    v32 = p_rect[2] - p_rect[3]
    par_err = np.linalg.norm(v12 - v32)
    print(f"[校验] 还原后对边平行误差: {par_err:.6e}")

    len1 = np.linalg.norm(v12)
    len2 = np.linalg.norm(v14)
    r = float(len1 / len2)
    cos_alpha = float(np.dot(v12, v14) / (len1 * len2))
    alpha = float(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    print(f"[度量真值] 还原真实平行四边形长宽比 r = {r:.6f}，真实内夹角 alpha = {np.degrees(alpha):.4f}°")

    ang1, ang2 = _solve_angles(r, cos_alpha)
    if ang1 is None:
        print("[错误] 姿态角求解无实数解，请检查角点排序或平移数据")
        return None, None

    print(f"[姿态解算] ang1 = {np.degrees(ang1):.4f}°, ang2 = {np.degrees(ang2):.4f}°")

    real_L1 = W_mm / np.sin(ang1)
    real_L2 = L_mm / np.sin(ang2)
    ox, oy = real_L2 * np.cos(alpha), real_L2 * np.sin(alpha)
    real_pts4 = np.array([
        [0.0, 0.0],
        [real_L1, 0.0],
        [real_L1 + ox, oy],
        [ox, oy]
    ], np.float64)

    print(f"[物理尺寸] 截面实际边长 L1 = {real_L1:.3f} mm, L2 = {real_L2:.3f} mm")

    np.savez_compressed(out_npz, points=real_pts4, unit="mm",
                        ang1=ang1, ang2=ang2, alpha=alpha, l_inf=l_inf)
    print(f"[完成] 初始四点现实坐标与姿态角已存入：{out_npz}")
    return ang1, ang2


if __name__ == "__main__":
    calibrate_step2()
