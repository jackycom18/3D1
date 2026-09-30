"""测试：读取标定板图片，提取棋盘格内角点并显示/保存结果图。

运行：
    py -3.8 测试_角点提取.py
只需修改下面的 IMAGE_PATH 变量即可。
默认内角点数为 9x7，如果标定板不是 9x7，修改 BOARD_W / BOARD_H。
如果不知道内角点数量，把 AUTO_DETECT_BOARD 设为 True，脚本会自动试。
"""

import os
import time

import cv2
import numpy as np

# ==================== 只改这里 ====================
IMAGE_PATH = r"C:\python\3D_project_3\calib_images\img55.png"
BOARD_W = 9                 # 棋盘格内角点列数
BOARD_H = 7                 # 棋盘格内角点行数
AUTO_DETECT_BOARD = False   # True：自动尝试常见内角点尺寸，忽略上面的 W/H
SHOW_WINDOW = False         # True：弹窗显示结果（需要按键关闭）；False：只保存文件
MAX_SIDE = 1500             # 检测前把长边缩到这个值以内，加速 SB 检测
# =================================================

OUT_SUFFIX = "_角点结果.png"
_CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)


# ---------------------- 中文路径读写 ----------------------
def imread_unicode(path):
    """功能：读取图片，兼容中文/非 ASCII 路径。

    输入：
        path：str，图片路径
    输出：
        np.ndarray 或 None，BGR 图像
    """
    if not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """功能：保存图片，兼容中文/非 ASCII 路径。

    输入：
        path：str，保存路径
        img：np.ndarray，BGR 图像
    输出：
        bool，是否保存成功
    """
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ---------------------- 角点检测 ----------------------
def _try_find(gray, board, use_sb=False):
    """功能：在灰度图上尝试检测一次棋盘格角点（不缩放、不精化）。

    输入：
        gray：np.ndarray，灰度图
        board：tuple，(cols, rows)
        use_sb：bool，True 用 SB 检测，False 用标准检测
    输出：
        (ok, corners)，ok 为 bool，corners 为 (N,1,2) float32 或 None
    """
    if use_sb:
        return cv2.findChessboardCornersSB(gray, board, cv2.CALIB_CB_NORMALIZE_IMAGE)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, board, flags)


def auto_detect_board(gray, w_range=range(5, 16), h_range=range(4, 13)):
    """功能：自动尝试常见的内角点尺寸，返回第一个能检测到的 (cols, rows)。

    输入：
        gray：np.ndarray，灰度图
        w_range：可迭代，列数候选
        h_range：可迭代，行数候选
    输出：
        tuple 或 None，(cols, rows)
    """
    for w in w_range:
        for h in h_range:
            ok, _ = _try_find(gray, (w, h), use_sb=True)
            if ok:
                return (w, h)
    return None


def _find_corners(gray, board):
    """功能：在灰度图上找棋盘格内角点；先标准检测，失败再用 SB，最后亚像素精化。

    输入：
        gray：np.ndarray，uint8 灰度图
        board：tuple，(cols, rows)
    输出：
        np.ndarray，(N,1,2) float32，坐标已还原到原图尺度；找不到返回 None
    """
    h, w = gray.shape[:2]

    # 大图先缩放，SB 检测在大图上非常慢
    scale = 1.0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    print(f"[信息] 原图尺寸：{w}x{h}，缩放系数：{scale:.4f}，检测尺寸：{small.shape[1]}x{small.shape[0]}")

    t0 = time.time()
    ok, corners = _try_find(small, board, use_sb=False)
    print(f"[信息] 标准检测：{ok}，耗时 {time.time() - t0:.2f}s")

    if not ok:
        t0 = time.time()
        ok, corners = _try_find(small, board, use_sb=True)
        print(f"[信息] SB 检测：{ok}，耗时 {time.time() - t0:.2f}s")

    if not ok:
        return None

    # 亚像素精化在缩放图上做，再还原坐标
    t0 = time.time()
    cv2.cornerSubPix(small, corners, (11, 11), (-1, -1), _CRIT_SUBPIX)
    print(f"[信息] 亚像素精化完成，耗时 {time.time() - t0:.2f}s")

    if scale != 1.0:
        corners = corners / scale
    return corners


# ---------------------- 主流程 ----------------------
def extract_corners(image_path, board=(BOARD_W, BOARD_H), auto_board=AUTO_DETECT_BOARD):
    """功能：读取一张标定板图片，提取内角点，在原图上画出并保存结果图。

    输入：
        image_path：str，图片路径
        board：tuple，(cols, rows)，内角点数量
        auto_board：bool，True 时自动探测内角点尺寸
    输出：
        np.ndarray，(N,1,2) float32 角点坐标；失败返回 None
    """
    print(f"[信息] 读取图片：{image_path}")
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"图片不存在或无法读取：{image_path}")
    print(f"[信息] 读取成功，尺寸：{img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自动探测尺寸
    if auto_board:
        print("[信息] 开始自动探测内角点尺寸 ...")
        t0 = time.time()
        # 探测也在缩放图上做，加速
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / float(max(h, w))
            probe = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            probe = gray
        found = auto_detect_board(probe)
        print(f"[信息] 自动探测结束，耗时 {time.time() - t0:.2f}s，结果：{found}")
        if found is None:
            print("[失败] 自动探测未找到任何常见尺寸，请检查图片质量。")
            return None
        board = found
        print(f"[信息] 使用内角点尺寸：{board[0]}x{board[1]}")

    print(f"[信息] 使用内角点尺寸：{board[0]}x{board[1]}")
    t0 = time.time()
    corners = _find_corners(gray, board)
    print(f"[信息] 角点检测总耗时 {time.time() - t0:.2f}s")

    if corners is None:
        print(f"[失败] 未找到 {board[0]}x{board[1]} 个内角点，请检查：\n"
              f"        1) BOARD_W/BOARD_H 是否为内角点数量（不是方格数）\n"
              f"        2) 图片是否完整拍到整个棋盘格\n"
              f"        3) 图片是否过曝/过暗/模糊\n"
              f"        4) 可把 AUTO_DETECT_BOARD 设为 True 让脚本自动试尺寸")
        return None

    # 画结果
    result = img.copy()
    cv2.drawChessboardCorners(result, board, corners, True)
    for point in corners.reshape(-1, 2):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.putText(result, f"{len(corners)} corners", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    base, _ = os.path.splitext(image_path)
    out_path = base + OUT_SUFFIX
    ok = imwrite_unicode(out_path, result)
    if ok:
        print(f"[成功] 找到 {len(corners)} 个角点，结果图已保存：{out_path}")
    else:
        print(f"[警告] 结果图保存失败：{out_path}")

    if SHOW_WINDOW:
        cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
        cv2.imshow("corners", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corners


if __name__ == "__main__":
    extract_corners(IMAGE_PATH, (BOARD_W, BOARD_H), auto_board=AUTO_DETECT_BOARD)