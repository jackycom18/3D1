"""标定第 4 步：用真实移动后的现实坐标与对应虚拟平面坐标，求解光平面单应矩阵 H。

流程：
1. 读 records.npz 的平移台 xy 移动量 (M,2)，按角度修正得到光平面上的真实位移 real_move：
   real_move_x = x / cos(ang1 + ang_move)，real_move_y = y / cos(ang2)
   （注：磨具与位移台无系统误差时 ang_move = 0.0）
2. 读 初始四点坐标.npz 的四点现实坐标作为起点，逐步累加 real_move 得到全部 real_points（含初始）
3. 读 centroids.npz 每组四点像素坐标，用内参去畸变到 z=1 虚拟平面，得到 plane_pts
4. real_points 与 plane_pts 一一对应，用 cv2.findHomography 求解 H（虚拟平面 -> 光平面现实坐标），存 npz
运行：python 标定第四步.py
"""

import os
import numpy as np
import cv2

# ---------- 可调参数（路径按实际修改） ----------
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"    # 鱼眼标定结果（第 1 步输出，含 K/D）
CENTROIDS_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 多组四点交点像素坐标
INIT_NPZ = r"C:\python\3D_project_3\初始四点坐标.npz"      # 初始四点现实坐标（第 2 步输出，含 ang1/ang2）
ANG_MOVE_NPZ = r"C:\python\3D_project_3\ang_move.npz"      # 位移台夹角（可选，无则默认为 0.0）
RECORDS_NPZ = r"C:\python\3D_project_3\records.npz"        # 平移台 xy 移动记录（平移台软件输出）
OUT_NPZ = r"C:\python\3D_project_3\H.npz"                  # 单应矩阵保存路径

_COS_EPS = 1e-6        # cos 下限保护，低于此值视为除零风险


def _natural_key(name):
    """功能：从键名中提取数字作自然排序 key，避免 img2 排在 img10 之后。"""
    digits = "".join(ch for ch in name if ch.isdigit())
    return (digits and int(digits)) or 0


def _undistort_to_plane(points_px, K, D):
    """功能：把一组四点像素坐标去畸变投影到 z=1 虚拟平面。"""
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    norm = cv2.fisheye.undistortPoints(pts, K, D)
    return norm.reshape(-1, 2)


def _read_moves(records_npz):
    """功能：读平移台 xy 移动记录，返回 (N,2) 纯移动量。"""
    data = np.load(records_npz)
    if "dx" in data and "dy" in data:
        moves = np.column_stack([np.asarray(data["dx"], np.float64).ravel(),
                                 np.asarray(data["dy"], np.float64).ravel()])
    else:
        cand = [np.asarray(data[k], np.float64) for k in data.files
                if np.asarray(data[k]).ndim == 2 and np.asarray(data[k]).shape[1] == 2]
        if not cand:
            return None
        moves = cand[0].reshape(-1, 2)
    return moves


def _build_real_points(init_pts, moves, ang1, ang2, ang_move=0.0):
    """功能：用初始四点坐标逐次累加修正后的真实位移，得到全部现实坐标（含初始）。
    输入：init_pts：(4,2) float，初始四点现实坐标；moves：(M,2) float，xy 移动量；
          ang1、ang2：float，方向角(弧度)；ang_move：float，位移台夹角(弧度，默认 0.0)
    输出：(M+1,4,2) float，第 0 组为初始坐标，之后每组在上组基础上平移 real_move
    """
    cos_x = np.cos(ang1 + ang_move)                          # x 方向投影到光平面的余弦
    cos_y = np.cos(ang2)                                     # y 方向投影到光平面的余弦
    if abs(cos_x) < _COS_EPS or abs(cos_y) < _COS_EPS:
        raise ValueError("cos 值接近 0，real_move 除零风险，请检查 ang1/ang2/ang_move")
    real = [np.asarray(init_pts, np.float64)]                # 第 0 组：初始坐标
    cur = real[0].copy()
    for x, y in moves:
        rx = x / cos_x                                       # x 移动在光平面上的真实位移
        ry = y / cos_y                                       # y 移动在光平面上的真实位移
        cur = cur + np.array([rx, ry])                       # 整体平移
        real.append(cur.copy())
    return np.array(real)


def calibrate_step4(calib_npz=CALIB_NPZ, centroids_npz=CENTROIDS_NPZ,
                    init_npz=INIT_NPZ, ang_move_npz=ANG_MOVE_NPZ,
                    records_npz=RECORDS_NPZ, out_npz=OUT_NPZ, ang_move=None):
    """功能：读内参、交点、初始坐标、夹角与移动记录，求解并保存单应矩阵 H。
    若未提供或未找到 ang_move_npz，则自动按无系统误差假设 ang_move = 0.0 计算。
    """
    cal = np.load(calib_npz)
    K = cal["K"].astype(np.float64)
    D = cal["D"].astype(np.float64)

    init = np.load(init_npz)
    init_pts = np.asarray(init["points"], np.float64).reshape(4, 2)
    ang1 = float(init["ang1"])
    ang2 = float(init["ang2"])

    if ang_move is None:
        if ang_move_npz and os.path.exists(ang_move_npz):
            ang_move = float(np.load(ang_move_npz)["ang_move"])
            print(f"[信息] 读取到 ang_move = {np.degrees(ang_move):.4f}°")
        else:
            ang_move = 0.0
            print("[信息] 磨具与平移台无系统误差假设成立，ang_move 设为 0.0°")

    moves = _read_moves(records_npz)
    if moves is None:
        print("[错误] records.npz 里找不到 (N,2) 移动量或 dx/dy 字段")
        return None, None
    real_points = _build_real_points(init_pts, moves, ang1, ang2, ang_move)

    data = np.load(centroids_npz)
    if "points" in data:
        groups = np.asarray(data["points"], np.float64).reshape(-1, 4, 2)
    else:
        groups = np.array([np.asarray(data[k], np.float64).reshape(4, 2)
                           for k in sorted(data.files, key=_natural_key)], np.float64)

    if groups.shape[0] != real_points.shape[0]:
        print(f"[警告] 交点组数 {groups.shape[0]} 与真实坐标组数 {real_points.shape[0]} 不一致，"
              f"请确认 records 移动次数与拍照张数是否对应")
        # 兼容处理：截断到较小长度
        min_n = min(groups.shape[0], real_points.shape[0])
        groups = groups[:min_n]
        real_points = real_points[:min_n]

    plane_pts = np.vstack([_undistort_to_plane(groups[i], K, D)
                           for i in range(groups.shape[0])])
    real_flat = real_points.reshape(-1, 2)

    H, mask = cv2.findHomography(plane_pts, real_flat, method=0)
    if H is None:
        print("[错误] findHomography 失败，请检查点对应关系")
        return None, None

    # 重投影误差评估
    ones = np.hstack([plane_pts, np.ones((len(plane_pts), 1))])
    proj = ones @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum((proj - real_flat) ** 2, axis=1))))

    np.savez_compressed(out_npz, H=H, rms=rms, n_points=len(real_flat))
    print(f"[完成] 单应矩阵 H 已计算并保存至 {out_npz}")
    print(f"       重投影误差 rms = {rms:.4f} mm")
    return H, rms


if __name__ == "__main__":
    calibrate_step4()
