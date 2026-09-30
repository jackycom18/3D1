"""鱼眼镜头标定：用棋盘格照片标定相机内参 K 与畸变系数 D，结果存 npz。

流程：遍历 calib_images 里的图片 → findChessboardCorners 找角点 → cornerSubPix 精化 →
      cv2.fisheye.calibrate 标定 → 保存 K/D/rms/image_size 到 npz
运行：python 鱼眼标定.py
"""

import glob
import os

import cv2
import numpy as np

# ---------- 可调参数 ----------
CALIB_DIR = r"C:\python\3D_project_3\calib_images"    # 标定棋盘格照片文件夹
OUT_NPZ = r"C:\python\3D_project_3\calib_result.npz"   # 标定结果保存路径
BOARD_W = 9                  # 棋盘格内角点列数
BOARD_H = 7                  # 棋盘格内角点行数
SQUARE_MM = 13.0          # 单个格子的物理边长(mm)
MAX_SIDE = 1500              # 角点检测前长边缩放上限(px)，超过则等比缩到该值

_CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)  # 角点精化终止条件
_CRIT_CALIB = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)   # 标定迭代终止条件


def _find_corners(gray, board):
    """功能：在灰度图上找棋盘格内角点并做亚像素精化；先缩放长边到 MAX_SIDE，标准检测失败再用 SB。
    输入：gray：np.ndarray，uint8 灰度图；board：(cols, rows)，棋盘格内角点数
    输出：np.ndarray，(N, 1, 2) float32 角点像素坐标，已还原到原图尺度；找不到时返回 None
    """
    h, w = gray.shape[:2]
    scale = 1.0                                            # 缩放系数，1.0 表示未缩放
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, board, flags)
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(small, board, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None

    cv2.cornerSubPix(small, corners, (11, 11), (-1, -1), _CRIT_SUBPIX)
    if scale != 1.0:
        corners = corners / scale
    return corners


def calibrate(path=CALIB_DIR, out_npz=OUT_NPZ):
    """功能：批量读取棋盘格照片，标定鱼眼镜头内参 K 与畸变系数 D 并存 npz。
    输入：path：str，标定图片文件夹；out_npz：str，结果 npz 保存路径
    输出：tuple(K, D) 或 None：K 为 (3,3) 内参矩阵，D 为 (4,1) 畸变系数；有效图片不足 3 张返回 None
    """
    files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
    board = (BOARD_W, BOARD_H)
    objp = np.zeros((BOARD_W * BOARD_H, 1, 3), np.float32)              # 棋盘格世界坐标模板，shape(N,1,3) 才是 CV_32FC3
    objp[:, 0, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM  # (x,y)，z=0，单位 mm
    obj_points = []                     # 每张有效图的世界坐标角点
    img_points = []                     # 每张有效图的像素坐标角点
    image_size = None
    for file_path in files:
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"跳过无法读取的图片：{file_path}")
            continue
        corners = _find_corners(gray, board)
        if corners is None:
            print(f"未找到角点：{file_path}")
            continue
        image_size = gray.shape[::-1]   # (宽, 高)
        obj_points.append(objp)
        img_points.append(corners)
    if len(obj_points) < 3:
        print(f"有效标定图片 {len(obj_points)} 张，不足 3 张，无法标定")
        return None
    K = np.eye(3, dtype=np.float64)         # 内参矩阵初值（标定时会被覆盖）
    D = np.zeros((4, 1), dtype=np.float64)  # 畸变系数初值 k1~k4
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_points, img_points, image_size, K, D, None, None, flags, _CRIT_CALIB)
    np.savez_compressed(out_npz, K=K, D=D, rms=rms, image_size=image_size)
    print(f"标定完成：{len(obj_points)} 张图，重投影误差 rms={rms:.4f}px")
    print(f"K=\n{K}\nD={D.ravel()}")
    return K, D


if __name__ == "__main__":
    calibrate()
