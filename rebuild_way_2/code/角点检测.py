"""单张图像相机角点检测模块（兼容白绿与黑白标定板，适配鱼眼镜头）

功能：
  1. 支持检测单张鱼眼相机拍摄的棋盘格图像；
  2. 颜色兼容：
     - 绿白标定板：自适应通过绿色聚类/热力图 ROI + Lab a通道/R通道/差分通道提取；
     - 黑白标定板：自适应通过全图/灰度 CLAHE 提取；
  3. 超时保护：设定预计检测时限，若耗时超过时限立即返回 fail，杜绝长时间卡死；
  4. 可视化：检测成功后将角点绘制为红色圆点，保存到 C:\\python\\3D_project_3\\debug 目录。
"""

import os
import sys
import time
import cv2
import numpy as np

# 确保 Windows 终端输出中文字符不乱码
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 默认参数配置 ----------
DEFAULT_IMAGE_PATH = r"C:\python\3D_project_3\calib_images\images\img10.png"
DEBUG_OUTPUT_DIR = r"C:\python\3D_project_3\debug"
DEFAULT_BOARD_SIZE = (9, 7)    # 棋盘格内角点数量 (cols, rows)
DEFAULT_TIMEOUT_SEC = 10.0     # 预计最大检测时间(秒)，超时则立即返回 fail

_CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)


def cv_imread(file_path):
    """支持包含中文路径的图片读取。"""
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(file_path, img):
    """支持包含中文路径的图片保存。"""
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    ext = os.path.splitext(file_path)[1]
    if not ext:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(file_path)
    return ok


def _roi_by_square_clustering(img):
    """通过绿色方块的空间聚类定位标定板高精度 ROI（绿白标定板专属加速）。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_ch = lab[:, :, 1]                 # Lab a 通道，绿色偏负
    L_ch = lab[:, :, 0]                 # 亮度通道
    green_mask = (a_ch < 118) & (L_ch > 40) & (L_ch < 220)
    contours, _ = cv2.findContours((green_mask * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_sqs = []
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 15000:
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) / (min(w, h) + 1e-5) < 3.2:
                valid_sqs.append((x + w / 2, y + h / 2, x, y, w, h))
    if len(valid_sqs) < 4:
        return None
    pts = np.array([[s[0], s[1]] for s in valid_sqs])
    clusters, visited = [], [False] * len(pts)
    dist_thresh = 260
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
    best = max(clusters, key=len)
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
    """基于 Lab a 通道绿色密度热力图定位标定板中心 ROI（绿白备选）。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_ch = lab[:, :, 1].astype(float)
    L_ch = lab[:, :, 0]
    score = np.maximum(0, 125.0 - a_ch)
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


def detect_corners_single_image(
    image_path=DEFAULT_IMAGE_PATH,
    board_size=DEFAULT_BOARD_SIZE,
    timeout_sec=DEFAULT_TIMEOUT_SEC,
    output_dir=DEBUG_OUTPUT_DIR
):
    """检测单张图像中的棋盘格内角点（带硬性超时截断与自适应 ROI/多通道）。"""
    start_time = time.time()
    print(f"[角点检测] 开始处理: {os.path.basename(image_path)}", flush=True)
    print(f"[角点检测] 目标规格: {board_size}, 超时限制: {timeout_sec:.1f}s", flush=True)

    img = cv_imread(image_path)
    if img is None:
        print(f"[角点检测][FAIL] 无法读取图像: {image_path}", flush=True)
        return False, None, None

    H, W = img.shape[:2]
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    pattern_sizes = [board_size, (board_size[1], board_size[0])]

    # 1. 候选 ROI 列表：优先绿色方块聚类与热力图，最后全图（覆盖黑白标定板）
    rois = []
    cluster_roi = _roi_by_square_clustering(img)
    if cluster_roi is not None:
        rois.append(('Cluster', cluster_roi))

    heat_roi = _roi_by_heatmap(img, 1200)
    if heat_roi is not None:
        rois.append(('Heatmap1200', heat_roi))

    # 全图 ROI（黑白标定板或聚类未命中时的全图兜底）
    rois.append(('Full', (0, 0, W, H)))

    corners_found = None
    matched_method = None
    matched_pat = None

    for roi_name, roi in rois:
        # 超时保护判断
        if time.time() - start_time > timeout_sec:
            elapsed = time.time() - start_time
            print(f"[角点检测][FAIL] 耗时 {elapsed:.2f}s 超过预计时限 {timeout_sec:.1f}s，终止检测！", flush=True)
            return False, None, None

        if roi is None:
            continue
        rx, ry, rw, rh = roi
        crop = img[ry:ry + rh, rx:rx + rw]

        # 准备通道：
        # 对于绿白标定板：Lab a 通道与 R 通道对比度最高
        # 对于黑白标定板：灰度 Gray 通道对比度最高
        c_a = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)[:, :, 1]
        c_r = crop[:, :, 2]
        c_g = crop[:, :, 1]
        c_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gr_diff = cv2.normalize(c_g.astype(float) - c_r.astype(float), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        synth = 255 - gr_diff

        preps = [
            ('a_clahe2', clahe2.apply(c_a)),      # 绿白标定板首选
            ('gray_clahe', clahe2.apply(c_gray)),  # 黑白标定板首选
            ('r_clahe', clahe2.apply(c_r)),        # 绿白高反差
            ('gray', c_gray),                      # 标准黑白
            ('synth_clahe', clahe2.apply(synth)),  # 绿红差分
            ('a_clahe4', clahe4.apply(c_a)),      # 强对比 a 通道
            ('r', c_r)
        ]

        for prep_name, prep_im in preps:
            # 超时保护判断
            if time.time() - start_time > timeout_sec:
                elapsed = time.time() - start_time
                print(f"[角点检测][FAIL] 耗时 {elapsed:.2f}s 超过预计时限 {timeout_sec:.1f}s，终止检测！", flush=True)
                return False, None, None

            for pat in pattern_sizes:
                # 策略 A：优先 SB 检测器（快速精准模式）
                try:
                    ok, corners = cv2.findChessboardCornersSB(
                        prep_im, pat, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
                    )
                    if ok and corners is not None:
                        corners = corners.reshape(-1, 2)
                        corners[:, 0] += rx
                        corners[:, 1] += ry
                        corners_found = corners
                        matched_method = f"{roi_name}+SB+{prep_name}"
                        matched_pat = pat
                        break
                except Exception:
                    pass

                # 策略 B：传统检测器 + 亚像素精化
                try:
                    ok, corners = cv2.findChessboardCorners(
                        prep_im, pat, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                    if ok and corners is not None:
                        cv2.cornerSubPix(prep_im, corners, (11, 11), (-1, -1), _CRIT_SUBPIX)
                        corners = corners.reshape(-1, 2)
                        corners[:, 0] += rx
                        corners[:, 1] += ry
                        corners_found = corners
                        matched_method = f"{roi_name}+Trad+{prep_name}"
                        matched_pat = pat
                        break
                except Exception:
                    pass

            if corners_found is not None:
                break

        if corners_found is not None:
            break

    elapsed_total = time.time() - start_time

    # 检测失败
    if corners_found is None:
        print(f"[角点检测][FAIL] 未检测到角点 (总耗时: {elapsed_total:.2f}s)", flush=True)
        return False, None, None

    # 检测成功：绘制红色圆点并保存到 debug 文件夹
    print(f"[角点检测][SUCCESS] 检测成功! 耗时: {elapsed_total:.2f}s, 模式: {matched_method}, 提取角点数: {len(corners_found)}", flush=True)

    vis_img = img.copy()
    for pt in corners_found:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        # 红色实心圆点标注
        cv2.circle(vis_img, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
        # 白色细圈边框增强高对比清晰度
        cv2.circle(vis_img, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_corners.png")
    cv_imwrite(out_path, vis_img)
    print(f"[角点检测] 标注结果已保存至: {out_path}", flush=True)

    return True, corners_found, out_path


if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIMEOUT_SEC

    ok, corners, save_path = detect_corners_single_image(
        image_path=target_img,
        timeout_sec=timeout
    )
    if ok:
        print(f"[完成] 检测到 {len(corners)} 个角点，结果文件: {save_path}")
    else:
        print("[完成] 检测失败")