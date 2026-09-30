"""鱼眼镜头标定：用棋盘格照片标定相机内参 K 与畸变系数 D，结果存 npz。

流程：
  1. 自适应鲁棒角点提取：候选 ROI（绿色方块聚类 / 绿色热力图 / 全图）× 多种预处理 × SB/传统检测器
  2. 角点方向标准化，统一为 (BOARD_W, BOARD_H) 左上角起点的顺序
  3. cv2.fisheye.calibrate 标定
  4. 保存 K/D/rms/image_size 到 npz
运行：python 鱼眼标定.py
"""

import glob
import os

import cv2
import numpy as np

# ---------- 可调参数 ----------
CALIB_DIR = r"C:\python\3D_project_3\calib_images\images"    # 标定棋盘格照片文件夹
OUT_NPZ = r"C:\python\3D_project_3\calib_result.npz"   # 标定结果保存路径
BOARD_W = 9                  # 棋盘格内角点列数
BOARD_H = 7                  # 棋盘格内角点行数
SQUARE_MM = 13.0          # 单个格子的物理边长(mm)

_CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)  # 角点精化终止条件
_CRIT_CALIB = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)   # 标定迭代终止条件


def _cv_imread(file_path):
    """功能：读取图片，支持包含中文的路径。
    输入：file_path：str，图片路径
    输出：np.ndarray，BGR 彩色图；读取失败返回 None
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _roi_by_square_clustering(img):
    """功能：通过绿色方块的空间聚类定位标定板高精度 ROI。
    输入：img：np.ndarray，BGR 彩色图
    输出：tuple(x, y, w, h)；找不到足够绿色方块时返回 None
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_ch = lab[:, :, 1]                 # Lab 的 a 通道，绿色偏负
    L_ch = lab[:, :, 0]                 # Lab 亮度通道
    green_mask = (a_ch < 118) & (L_ch > 40) & (L_ch < 220)
    contours, _ = cv2.findContours((green_mask * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_sqs = []                      # 符合面积/长宽比的绿色方块，存 (cx, cy, x, y, w, h)
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 15000:
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) / (min(w, h) + 1e-5) < 3.2:
                valid_sqs.append((x + w / 2, y + h / 2, x, y, w, h))
    if len(valid_sqs) < 4:
        return None
    pts = np.array([[s[0], s[1]] for s in valid_sqs])   # 方块中心点集
    clusters, visited = [], [False] * len(pts)
    dist_thresh = 260                   # 相邻方块中心距离阈值(px)
    for i in range(len(pts)):
        if visited[i]:
            continue
        cluster, queue = [i], [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            dists = np.linalg.norm(pts - pts[curr], axis=1)
            for n in np.where((dists < dist_thresh) & (~np.array(visited)))[0]:
                visited[n] = True
                cluster.append(n)
                queue.append(n)
        clusters.append(cluster)
    best = max(clusters, key=len)       # 取方块最多的簇
    if len(best) < 4:
        return None
    boxes = [valid_sqs[i] for i in best]
    min_x = min(b[2] for b in boxes)
    min_y = min(b[3] for b in boxes)
    max_x = max(b[2] + b[4] for b in boxes)
    max_y = max(b[3] + b[5] for b in boxes)
    pad = 120
    x0 = max(0, min_x - pad)
    y0 = max(0, min_y - pad)
    x1 = min(img.shape[1], max_x + pad)
    y1 = min(img.shape[0], max_y + pad)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _roi_by_heatmap(img, win_size=1200):
    """功能：基于 Lab a 通道绿色密度热力图定位标定板中心 ROI。
    输入：img：np.ndarray，BGR 彩色图；win_size：int，正方形窗口边长(px)
    输出：tuple(x, y, w, h)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_ch = lab[:, :, 1].astype(float)
    L_ch = lab[:, :, 0]
    score = np.maximum(0, 125.0 - a_ch)     # 绿色得分
    score[L_ch < 40] = 0
    small = cv2.resize(score, (img.shape[1] // 4, img.shape[0] // 4))
    blur = cv2.GaussianBlur(small, (71, 71), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blur)
    cx, cy = max_loc[0] * 4, max_loc[1] * 4
    x0 = max(0, min(img.shape[1] - win_size, cx - win_size // 2))
    y0 = max(0, min(img.shape[0] - win_size, cy - win_size // 2))
    return int(x0), int(y0), win_size, win_size


def _detect_corners(img, board):
    """功能：自适应鲁棒提取鱼眼图像中的棋盘格内角点。
    做法：多个候选 ROI × 多种预处理（CLAHE/合成通道/灰度）× 优先 SB 检测器、备选传统检测器，逐个尝试直到成功。
    输入：img：np.ndarray，BGR 彩色图；board：(cols, rows)，棋盘格内角点数
    输出：tuple(ok, corners, pat, method)，ok 为 bool；corners 为 (N,2) float32 原图坐标；找不到时 (False, None, None, None)
    """
    H, W = img.shape[:2]
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    pattern_sizes = [board, (board[1], board[0])]       # 正反两种朝向

    rois = [('Cluster', _roi_by_square_clustering(img)),
            ('Heatmap1200', _roi_by_heatmap(img, 1200)),
            ('Heatmap1400', _roi_by_heatmap(img, 1400)),
            ('Full', (0, 0, W, H))]

    for roi_name, roi in rois:
        if roi is None:
            continue
        rx, ry, rw, rh = roi
        crop = img[ry:ry + rh, rx:rx + rw]
        c_a = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)[:, :, 1]   # a 通道
        c_r = crop[:, :, 2]                                     # 红色通道
        c_g = crop[:, :, 1]                                     # 绿色通道
        c_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gr_diff = cv2.normalize(c_g.astype(float) - c_r.astype(float), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        synth = 255 - gr_diff                                    # 合成通道，突出绿色

        preps = [('a_clahe2', clahe2.apply(c_a)),
                 ('a_clahe4', clahe4.apply(c_a)),
                 ('synth_clahe', clahe2.apply(synth)),
                 ('r_clahe', clahe2.apply(c_r)),
                 ('r', c_r),
                 ('gray', c_gray),
                 ('gray_clahe', clahe2.apply(c_gray))]

        for prep_name, prep_im in preps:
            for pat in pattern_sizes:
                ok, corners = cv2.findChessboardCornersSB(
                    prep_im, pat, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                if ok:
                    corners = corners.reshape(-1, 2)
                    corners[:, 0] += rx
                    corners[:, 1] += ry
                    return True, corners, pat, f'{roi_name}+SB+{prep_name}'
                ok, corners = cv2.findChessboardCorners(
                    prep_im, pat, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    cv2.cornerSubPix(prep_im, corners, (11, 11), (-1, -1), _CRIT_SUBPIX)
                    corners = corners.reshape(-1, 2)
                    corners[:, 0] += rx
                    corners[:, 1] += ry
                    return True, corners, pat, f'{roi_name}+Trad+{prep_name}'
    return False, None, None, None


def _standardize_corners(corners, pat, target_pat):
    """功能：统一角点顺序与方向，使每张图的角点都从标定板左上角起、按先行后列排列。
    输入：corners：(N,2) float32 角点坐标；pat：(w, h) 实际检测到的角点数；target_pat：(cols, rows) 目标角点数
    输出：(N,2) float32，规范化后的角点坐标
    """
    w, h = pat
    grid = corners.reshape(h, w, 2)                 # (rows, cols, 2)
    if (w, h) == (target_pat[1], target_pat[0]):    # 检测到转置朝向，转回来
        grid = np.transpose(grid, (1, 0, 2))
        w, h = target_pat
    if grid[0, 0, 1] > grid[-1, -1, 1]:             # 首角点在下方，纵向翻转
        grid = grid[::-1, :, :]
    if grid[0, 0, 0] > grid[0, -1, 0]:              # 首角点在右侧，横向翻转
        grid = grid[:, ::-1, :]
    return grid.reshape(-1, 2)


def _calibrate_once(obj_points, img_points, image_size):
    """功能：用给定的世界点集与像素点集做一次鱼眼标定。
    输入：obj_points：list，每张图的世界坐标角点；img_points：list，每张图的像素坐标角点；
          image_size：(宽, 高)
    输出：tuple(K, D, rms)，K 为 (3,3) 内参，D 为 (4,1) 畸变系数，rms 为重投影误差(px)
    """
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    # OpenCV 版本兼容：CALIB_* 常量在 5.0 位于顶层 cv2，旧版位于 cv2.fisheye，都取不到时用官方固定值兜底
    rec = getattr(cv2, 'CALIB_RECOMPUTE_EXTRINSIC', None) or getattr(cv2.fisheye, 'CALIB_RECOMPUTE_EXTRINSIC', None) or 8388608
    fix = getattr(cv2, 'CALIB_FIX_SKEW', None) or getattr(cv2.fisheye, 'CALIB_FIX_SKEW', None) or 33554432
    flags = rec + fix
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_points, img_points, image_size, K, D, None, None, flags, _CRIT_CALIB)
    return K, D, rms


def calibrate(path=CALIB_DIR, out_npz=OUT_NPZ):
    """功能：批量读取棋盘格照片，鲁棒提取角点后标定鱼眼镜头内参 K 与畸变系数 D 并存 npz。
    输入：path：str，标定图片文件夹；out_npz：str，结果 npz 保存路径
    输出：tuple(K, D) 或 None：K 为 (3,3) 内参矩阵，D 为 (4,1) 畸变系数；有效图片不足 3 张返回 None
    """
    files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
    board = (BOARD_W, BOARD_H)
    objp = np.zeros((1, BOARD_W * BOARD_H, 3), np.float64)              # 棋盘格世界坐标模板，shape(1,N,3)
    objp[0, :, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM  # (x,y)，z=0，单位 mm

    obj_points = []                     # 每张有效图的世界坐标角点
    img_points = []                     # 每张有效图的像素坐标角点
    image_size = None

    for idx, file_path in enumerate(files, 1):
        img = _cv_imread(file_path)                                     # 读彩色图（绿色 ROI 需 BGR）
        if img is None:
            print(f"[{idx}/{len(files)}] {os.path.basename(file_path)}：读取失败")
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])                   # (宽, 高)
        ok, corners, pat, method = _detect_corners(img, board)
        if not ok:
            print(f"[{idx}/{len(files)}] {os.path.basename(file_path)}：未检测到角点")
            continue
        corners = _standardize_corners(corners, pat, board)
        obj_points.append(objp)
        img_points.append(corners.reshape(1, -1, 2).astype(np.float64))
        print(f"[{idx}/{len(files)}] {os.path.basename(file_path)}：检测成功，{len(corners)} 个角点（{method}）")

    if len(obj_points) < 3:
        print(f"有效标定图片 {len(obj_points)} 张，不足 3 张，无法标定")
        return None

    # ---------- 最终标定 ----------
    K, D, rms = _calibrate_once(obj_points, img_points, image_size)
    np.savez_compressed(out_npz, K=K, D=D, rms=rms, image_size=image_size)
    print(f"最终标定完成：{len(obj_points)} 张图，重投影误差 rms={rms:.4f}px")
    print(f"K=\n{K}\nD={D.ravel()}")
    return K, D


if __name__ == "__main__":
    calibrate()
