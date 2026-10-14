"""鱼眼镜头标定：用棋盘格照片标定相机内参 K 与畸变系数 D，结果存 npz。

流程：
  1. 自适应鲁棒角点提取：候选 ROI（绿色方块聚类 / 绿色热力图 / 全图）× 多种预处理 × SB/传统检测器
     - 单张图片硬性限时 30 秒，超时直接判定提取失败；
     - 提取成功的角点用红色实心圆点标注并保存到 C:\\python\\3D_project_3\\debug 文件夹中；
  2. 角点方向标准化（保证右手坐标系手性，杜绝非法单轴镜像反射）；
  3. 第一轮初标定与真实重投影误差显式计算（基于 cv2.fisheye.projectPoints）；
  4. 离群图像自适应筛选（剔除误差超过平均值指定倍数或异常过大的低质量照片）；
  5. 第二轮高精度精标定与最终内参/畸变系数输出；
  6. 详细日志写入 C:\\python\\3D_project_3\\rebuild_way_2\\code\\log 文件夹，详细记录提取失败及剔除照片明细。

运行：python 鱼眼标定.py
"""

import glob
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# 确保 Windows 终端中文字符正常输出
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 可调参数 ----------
CALIB_DIR = r"C:\python\3D_project_3\calib_images\images"                     # 标定棋盘格照片文件夹
OUT_NPZ = r"C:\python\3D_project_3\rebuild_way_2\data\neican\calib_result.npz"    # 标定结果保存路径
DEBUG_DIR = r"C:\python\3D_project_3\debug"                                  # 角点提取可视化保存路径
LOG_DIR = r"C:\python\3D_project_3\rebuild_way_2\code\log"                   # 标定与提取失败日志目录

BOARD_W = 9                  # 棋盘格内角点列数
BOARD_H = 7                  # 棋盘格内角点行数
SQUARE_MM = 13.0          # 单个格子的物理边长(mm)

MAX_TIME_PER_IMAGE = 30.0    # 单张照片最大处理时间(秒)，超时则判定提取失败
# 离群图像统计学过滤参数（正态残差准则：上限 = 平均误差 + k * 误差标准差）
K_SIGMA_THRESH = 2.5         # 离群倍数 k (正态分布 2.5σ 覆盖率 98.8%，超出 average + 2.5 * std 的判定为离群图像)

_CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)  # 角点精化终止条件
_CRIT_CALIB = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)  # 标定迭代终止条件（100次确保充分收敛）


def _cv_imread(file_path):
    """功能：读取图片，支持包含中文的路径。
    输入：file_path：str，图片路径
    输出：np.ndarray，BGR 彩色图；读取失败返回 None
    """
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _cv_imwrite(file_path, img):
    """功能：保存图片，支持包含中文的路径。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ext = os.path.splitext(file_path)[1]
    if not ext:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(file_path)
    return ok


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
    if np.max(score) < 1.0:
        return None
    small = cv2.resize(score, (img.shape[1] // 4, img.shape[0] // 4))
    blur = cv2.GaussianBlur(small, (71, 71), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blur)
    cx, cy = max_loc[0] * 4, max_loc[1] * 4
    x0 = max(0, min(img.shape[1] - win_size, cx - win_size // 2))
    y0 = max(0, min(img.shape[0] - win_size, cy - win_size // 2))
    return int(x0), int(y0), win_size, win_size


def _detect_corners(img, board, timeout_sec=MAX_TIME_PER_IMAGE):
    """功能：自适应鲁棒提取鱼眼图像中的棋盘格内角点（带硬性单张超时保护）。
    做法：多个候选 ROI × 多种预处理（CLAHE/合成通道/灰度）× 优先 SB 检测器、备选传统检测器。
    输入：img：np.ndarray，BGR 彩色图；board：(cols, rows)，棋盘格内角点数；timeout_sec：最大允许耗时(秒)
    输出：tuple(ok, corners, pat, method, elapsed)
    """
    start_t = time.time()
    H, W = img.shape[:2]
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    pattern_sizes = [board, (board[1], board[0])]       # 正反两种朝向

    rois = [('Cluster', _roi_by_square_clustering(img)),
            ('Heatmap1200', _roi_by_heatmap(img, 1200)),
            ('Heatmap1400', _roi_by_heatmap(img, 1400)),
            ('Full', (0, 0, W, H))]

    for roi_name, roi in rois:
        # 单张超时检查
        if time.time() - start_t > timeout_sec:
            return False, None, None, f'超时(超过{timeout_sec:.0f}s)', time.time() - start_t

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
                 ('gray_clahe', clahe2.apply(c_gray)),
                 ('r_clahe', clahe2.apply(c_r)),
                 ('synth_clahe', clahe2.apply(synth)),
                 ('a_clahe4', clahe4.apply(c_a)),
                 ('gray', c_gray),
                 ('r', c_r)]

        for prep_name, prep_im in preps:
            # 单张超时检查
            if time.time() - start_t > timeout_sec:
                return False, None, None, f'超时(超过{timeout_sec:.0f}s)', time.time() - start_t

            for pat in pattern_sizes:
                # 策略 A：SB 检测器（优先精准与快速）
                try:
                    ok, corners = cv2.findChessboardCornersSB(
                        prep_im, pat, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
                    if ok and corners is not None:
                        corners = corners.reshape(-1, 2)
                        corners[:, 0] += rx
                        corners[:, 1] += ry
                        return True, corners, pat, f'{roi_name}+SB+{prep_name}', time.time() - start_t
                except Exception:
                    pass

                # 策略 B：传统检测器
                try:
                    ok, corners = cv2.findChessboardCorners(
                        prep_im, pat, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                    if ok and corners is not None:
                        cv2.cornerSubPix(prep_im, corners, (11, 11), (-1, -1), _CRIT_SUBPIX)
                        corners = corners.reshape(-1, 2)
                        corners[:, 0] += rx
                        corners[:, 1] += ry
                        return True, corners, pat, f'{roi_name}+Trad+{prep_name}', time.time() - start_t
                except Exception:
                    pass

    return False, None, None, '未检测到规则角点', time.time() - start_t


def _standardize_corners(corners, pat, target_pat):
    """功能：确保角点维度与目标规格一致。
    若检测到的是转置规格 (H, W)，则进行保持手性的 2D 刚体旋转变换，杜绝非法镜像反射。
    输入：corners：(N,2) float32 角点坐标；pat：(w, h) 实际检测到的角点数；target_pat：(cols, rows) 目标角点数
    输出：(N,2) float32，规范化后的角点坐标
    """
    w, h = pat
    if (w, h) == target_pat:
        return corners.reshape(-1, 2)
    elif (w, h) == (target_pat[1], target_pat[0]):
        # 转置朝向：保持右手性（det = +1 的二维刚体旋转变换，转置 + 翻转列）
        grid = corners.reshape(h, w, 2)
        grid_rot = np.transpose(grid, (1, 0, 2))[:, ::-1, :]
        return grid_rot.reshape(-1, 2)
    return corners.reshape(-1, 2)


def _calibrate_core(obj_points, img_points, image_size, filenames, K_init=None, D_init=None):
    """功能：执行一次鱼眼内参标定并显式计算严谨的每张图重投影物理误差。
    输入：obj_points：list；img_points：list；image_size：(宽, 高)；filenames：list；
          K_init, D_init：可选初值
    输出：tuple(K, D, overall_rms, per_image_records)
    """
    if K_init is not None:
        K = K_init.copy()
        D = D_init.copy() if D_init is not None else np.zeros((4, 1), dtype=np.float64)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RECOMPUTE_EXTRINSIC + cv2.CALIB_FIX_SKEW
    else:
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = cv2.CALIB_RECOMPUTE_EXTRINSIC + cv2.CALIB_FIX_SKEW

    rms_solver, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, image_size, K, D, None, None, flags, _CRIT_CALIB)

    # 显式计算真实物理重投影残差 (cv2.fisheye.projectPoints)
    total_sq_err = 0.0
    total_pts = 0
    per_image_records = []

    for i in range(len(obj_points)):
        pts_proj, _ = cv2.fisheye.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        pts_proj = pts_proj.reshape(-1, 2)
        pts_real = img_points[i].reshape(-1, 2)
        diff = pts_real - pts_proj
        sq_err = np.sum(diff ** 2)
        img_rms = np.sqrt(sq_err / len(pts_real))
        img_mean = np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))

        name = filenames[i] if i < len(filenames) else f"Image_{i+1}"
        per_image_records.append({
            'name': name,
            'rms': float(img_rms),
            'mean': float(img_mean),
            'obj_points': obj_points[i],
            'img_points': img_points[i]
        })
        total_sq_err += sq_err
        total_pts += len(pts_real)

    overall_rms = np.sqrt(total_sq_err / total_pts)
    return K, D, overall_rms, per_image_records


def calibrate(
    path=CALIB_DIR,
    out_npz=OUT_NPZ,
    debug_dir=DEBUG_DIR,
    log_dir=LOG_DIR,
    k_sigma=K_SIGMA_THRESH
):
    """功能：批量读取棋盘格照片，两阶段自适应标定鱼眼镜头内参 K 与畸变系数 D，剔除离群废图并输出日志。
    输入：
        path: 标定图片文件夹
        out_npz: 标定结果 npz 保存路径
        debug_dir: 角点提取标记图输出目录
        log_dir: 标定日志输出目录
        k_sigma: 离群图像统计残差倍数（单张误差超过 average + k * std 则判定为离群剔除）
    输出：
        tuple(K, D, rms)
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"calib_log_{timestamp_str}.log")
    latest_log_path = os.path.join(log_dir, "calibration_latest.log")

    log_lines = []

    def log(msg, to_console=True):
        log_lines.append(msg)
        if to_console:
            print(msg, flush=True)

    log("=" * 60)
    log(f"鱼眼镜头标定任务开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"标定图像目录: {path}")
    log(f"结果保存路径: {out_npz}")
    log(f"单张超时限制: {MAX_TIME_PER_IMAGE:.1f} 秒")
    log(f"离群剔除准则: average + {k_sigma:.1f} × std (正态残差统计界限)")
    log("=" * 60)

    files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
    board = (BOARD_W, BOARD_H)
    objp = np.zeros((1, BOARD_W * BOARD_H, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    detected_records = []
    failed_records = []
    image_size = None

    log(f"\n[阶段 1] 开始读取图像并提取棋盘格角点 (共 {len(files)} 张):")

    for idx, file_path in enumerate(files, 1):
        file_name = os.path.basename(file_path)
        img = _cv_imread(file_path)
        if img is None:
            log(f"  [{idx:02d}/{len(files)}] {file_name}：读取失败")
            failed_records.append((file_name, "文件读取失败 (None)", 0.0))
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        ok, corners, pat, method, elapsed = _detect_corners(img, board, timeout_sec=MAX_TIME_PER_IMAGE)

        if not ok:
            log(f"  [{idx:02d}/{len(files)}] {file_name}：[提取失败] 原因: {method} (耗时: {elapsed:.2f}s)")
            failed_records.append((file_name, method, elapsed))
            continue

        corners = _standardize_corners(corners, pat, board)
        detected_records.append({
            'name': file_name,
            'corners': corners,
            'method': method,
            'elapsed': elapsed,
            'img': img
        })
        log(f"  [{idx:02d}/{len(files)}] {file_name}：[提取成功] 模式: {method} (耗时: {elapsed:.2f}s, 角点: {len(corners)})")

        # 标注红色实心圆点并保存到 debug 目录
        if debug_dir:
            vis_img = img.copy()
            for pt in corners:
                cx, cy = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(vis_img, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_img, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)
            base_name = os.path.splitext(file_name)[0]
            out_debug_img = os.path.join(debug_dir, f"{base_name}_corners.png")
            _cv_imwrite(out_debug_img, vis_img)

    log(f"\n角点提取汇总: 成功 {len(detected_records)} 张，失败 {len(failed_records)} 张")

    if len(detected_records) < 3:
        log("\n[错误] 有效标定图像不足 3 张，标定终止！")
        _write_log_files(log_lines, log_file_path, latest_log_path)
        return None

    # ---------- 第一轮：初步标定与误差统计 ----------
    log("\n[阶段 2] 执行第一轮初步内参标定并统计各图像重投影误差:")
    obj_pts_r1 = [objp for _ in detected_records]
    img_pts_r1 = [r['corners'].reshape(1, -1, 2).astype(np.float64) for r in detected_records]
    names_r1 = [r['name'] for r in detected_records]

    K_r1, D_r1, rms_r1, records_r1 = _calibrate_core(
        obj_pts_r1, img_pts_r1, image_size, names_r1
    )

    all_rms_list = np.array([r['rms'] for r in records_r1])
    mean_err_r1 = float(np.mean(all_rms_list))
    std_err_r1 = float(np.std(all_rms_list, ddof=1)) if len(all_rms_list) > 1 else 0.0

    log(f"第一轮综合 RMS 重投影误差: {rms_r1:.4f} px (单张 RMS 均值 μ: {mean_err_r1:.4f} px, 标准差 σ: {std_err_r1:.4f} px)")
    log("第一轮各图像重投影误差一览:")
    for r in records_r1:
        log(f"  - {r['name']}: RMS = {r['rms']:.4f} px, Mean = {r['mean']:.4f} px")

    # ---------- 离群图像判定与过滤 ----------
    # 统计区间准则：[average - k*残差, average + k*残差]
    # 在重投影误差评估中，误差 >= 0 且越小越优；下界为超高精度样本，不予剔除；
    # 统计上界 stat_thresh = mean + k * std 作为超差低质图像剔除门槛。
    upper_stat_thresh = mean_err_r1 + k_sigma * std_err_r1
    lower_stat_thresh = max(0.0, mean_err_r1 - k_sigma * std_err_r1)
    effective_thresh = upper_stat_thresh

    log(f"\n[阶段 3] 离群图像质量评估 (正态残差控制区间 [μ - {k_sigma}σ, μ + {k_sigma}σ] -> [{lower_stat_thresh:.4f}, {upper_stat_thresh:.4f}] px):")
    log(f"  * 纯统计判定剔除阈值 (μ + {k_sigma}*σ): {effective_thresh:.4f} px")

    outliers = [r for r in records_r1 if r['rms'] > effective_thresh]
    clean_records = [r for r in records_r1 if r['rms'] <= effective_thresh]

    if outliers:
        log(f"检测到 {len(outliers)} 张低质量/大误差离群图像，将予以剔除，不带入最终内参计算:")
        for o in outliers:
            log(f"  * [剔除] {o['name']}：RMS 误差高达 {o['rms']:.4f} px (超出阈值 {effective_thresh:.4f} px)")
    else:
        log("所有有效图像重投影误差均在容许范围内，无需剔除。")

    # ---------- 第二轮：精化标定 ----------
    if outliers and len(clean_records) >= 3:
        log(f"\n[阶段 4] 执行第二轮精标定 (使用剩余 {len(clean_records)} 张高质量图像，内参初值引导):")
        obj_pts_r2 = [r['obj_points'] for r in clean_records]
        img_pts_r2 = [r['img_points'] for r in clean_records]
        names_r2 = [r['name'] for r in clean_records]

        K_final, D_final, rms_final, records_r2 = _calibrate_core(
            obj_pts_r2, img_pts_r2, image_size, names_r2, K_init=K_r1, D_init=D_r1
        )
        used_records = records_r2
    else:
        log("\n[阶段 4] 使用第一轮标定结果作为最终内参。")
        K_final, D_final, rms_final = K_r1, D_r1, rms_r1
        used_records = records_r1

    # ---------- 保存最终结果 ----------
    np.savez_compressed(
        out_npz,
        K=K_final,
        D=D_final,
        rms=rms_final,
        image_size=image_size,
        used_images=np.array([r['name'] for r in used_records]),
        excluded_images=np.array([o['name'] for o in outliers])
    )

    log("\n" + "=" * 60)
    log("标定任务全部顺利完成！")
    log(f"最终纳入内参计算的图像数: {len(used_records)} 张")
    log(f"全局综合真实重投影误差 RMS: {rms_final:.4f} px")
    log(f"相机内参矩阵 K:\n{K_final}")
    log(f"畸变系数向量 D:\n{D_final.ravel()}")
    log(f"标定参数文件已保存至: {out_npz}")
    if debug_dir:
        log(f"角点标注图已保存至: {debug_dir}")
    log("=" * 60)

    # 专门输出失败与剔除汇总表
    log("\n##################### 异常与失败图像汇总表 #####################")
    if failed_records:
        log("【角点提取失败的图像】:")
        for name, reason, elap in failed_records:
            log(f"  - 图像: {name:<12} | 失败原因: {reason:<20} | 耗时: {elap:.2f}s")
    else:
        log("【角点提取失败的图像】: 无 (全部图像角点均提取成功)")

    if outliers:
        log("\n【因误差过大被剔除的低质量图像】:")
        for o in outliers:
            log(f"  - 图像: {o['name']:<12} | RMS误差: {o['rms']:.4f} px | 超标量: +{o['rms']-effective_thresh:.4f} px")
    else:
        log("\n【因误差过大被剔除的低质量图像】: 无")
    log("################################################################\n")

    _write_log_files(log_lines, log_file_path, latest_log_path)
    print(f"\n详细标定日志已生成: {log_file_path}", flush=True)

    return K_final, D_final, rms_final


def _write_log_files(log_lines, log_file_path, latest_log_path):
    """将日志行写入独立文件及 latest 文件。"""
    content = "\n".join(log_lines) + "\n"
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        with open(latest_log_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[警告] 写入日志文件失败: {e}", flush=True)


if __name__ == "__main__":
    calibrate()
