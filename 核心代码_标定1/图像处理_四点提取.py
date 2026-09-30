"""结构光图像处理：从多张 png 中提取每段光线主干上的拐点。

流程：读图 → 灰度 → 自适应二值化 → 细化 → 最长路径去毛刺(只留主干) → 按拟合直线方向变化挑拐点 → npz 保存
运行：python 图像处理.py
"""

from audioop import reverse

import cv2
import numpy as np
import glob
import os
from collections import deque

from numpy.ma import masked_invalid

# ---------- 可调参数（数值按需手改） ----------

IMAGE_DIR = r"C:\python\3D_project_3\takephoto\images"           # 输入图片文件夹
OUTPUT_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 质点结果保存路径
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
MIN_AREA = 500                 # 最小连通域面积(px)，小于此值视作噪声丢弃
SPUR_MIN_AREA = 100            # 骨架连通域最小面积(px)，小于此值整段丢弃
CURV_K = 30                 # 直线拟合步长(px)：前后各取 k 个像素各拟合一条直线，越大越抗骨架抖动
MIN_DIST = 400              # 拐点最小间距(px)：选中的任意两个拐点欧氏距离至少这么大，避免多个点挤在同一拐角
kernel = np.array([3,3])
AREA = [                       # 要去掉的区域多边形顶点(x,y)，按顺序首尾相连，边不能交叉
    [1200,800],                # 左上
    [2400,600],                # 右上
    [2300,1900],               # 右下
    [1200,1900],               # 左下
]
_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 8 邻域行列偏移


def _remove_small(binary, min_area):
    """功能：去掉面积小于 min_area 的前景连通域，只保留大块。
    输入：binary：np.ndarray，uint8，0/255 二值图；min_area：int，最小连通域面积(px)
    输出：np.ndarray，uint8，0/255，面积小于 min_area 的连通域被清 0
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= min_area
    keep[0] = False                                     # 背景不保留
    return np.where(keep[labels], 255, 0).astype(np.uint8)

def _thin(binary):
    """功能：用 Zhang-Suen 算法把二值图细化成 1 像素宽的骨架。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255 骨架图（与输入同尺寸）
    """
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return np.zeros_like(binary)
    y0, x0 = ys.min(), xs.min()                                    # 前景包围盒，缩小计算量
    y1, x1 = ys.max() + 1, xs.max() + 1
    img = (binary[y0:y1, x0:x1] > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]     # 3x3 邻域顺时针编号
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            nbr = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9            # 8 邻域前景数
            turns = sum(((ring[i] == 0) & (ring[i+1] == 1)).astype(np.uint8) for i in range(8))  # 0→1 跳变数
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (nbr >= 2) & (nbr <= 6) & (turns == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
    out = np.zeros_like(binary)
    out[y0:y1, x0:x1] = img * 255
    return out


def _bfs(mask, start):
    """功能：在掩膜内做 8 邻域广度优先搜索，得到 start 到各前景像素的最短步数。
    输入：mask：np.ndarray，bool，前景掩膜；start：(行, 列)，起点坐标
    输出：(dist, prev)：与 mask 同尺寸的两个 int32 数组，dist 为步数(-1 表示不可达)，
          prev 为前驱像素的线性索引(-1 表示无前驱)
    """
    h, w = mask.shape
    flat = mask.ravel()
    dist = np.full(h * w, -1, np.int32)
    prev = np.full(h * w, -1, np.int32)
    begin = start[0] * w + start[1]                     # 起点的线性索引
    dist[begin] = 0
    queue = deque([begin])
    while queue:
        cur = queue.popleft()
        r, c = divmod(cur, w)
        step = dist[cur] + 1
        for dr, dc in _NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nxt = nr * w + nc
                if flat[nxt] and dist[nxt] < 0:
                    dist[nxt] = step
                    prev[nxt] = cur
                    queue.append(nxt)
    return dist.reshape(h, w), prev


def _longest_path(mask, endpoints):
    """功能：在一个骨架连通域内，求连接两个端点且经过像素最多的一条路径（双次 BFS 求图直径）。
    输入：mask：np.ndarray，bool，单个连通域的骨架掩膜（包围盒裁剪后）；
          endpoints：(N, 2) int，该连通域内的端点坐标(行, 列)
    输出：(P, 2) int32，路径像素按顺序排列的坐标；端点少于 2 个时返回 None
    """
    if len(endpoints) < 2:
        return None
    dist, _ = _bfs(mask, tuple(endpoints[0]))
    far = dist[endpoints[:, 0], endpoints[:, 1]].argmax()      # 距首个端点最远的端点作起点
    dist, prev = _bfs(mask, tuple(endpoints[far]))
    tail = int(np.argmax(dist))                                # 距起点最远的前景像素作终点
    path = []
    while tail != -1:
        path.append(divmod(tail, mask.shape[1]))
        tail = prev[tail]
    return np.array(path[::-1], np.int32)


def _prune_and_simplify(skeleton):
    """功能：逐段处理骨架，只保留主干——每个连通域内取“连接两端点、经过像素最多”的那一条路径，毛刺分支自然被丢掉。
    输入：skeleton：np.ndarray，uint8，0/255 骨架图
    输出：pruned：np.ndarray，uint8，0/255、与输入同尺寸，去毛刺后的单像素宽主干
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), connectivity=8)
    conv = cv2.filter2D((skeleton > 0).astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8))
    endpoints_all = (skeleton > 0) & (conv == 2)               # 含自身共 2 个前景像素 = 只有 1 个邻居
    pruned = np.zeros_like(skeleton)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < SPUR_MIN_AREA:
            continue
        sub = labels[y:y+h, x:x+w] == idx                      # 该连通域（已裁剪到包围盒）掩膜
        path = _longest_path(sub, np.argwhere(endpoints_all[y:y+h, x:x+w]))
        if path is None:                                       # 端点不足两个（闭环）：无毛刺可去，整段保留
            path = np.argwhere(sub)
        keep = np.zeros_like(sub)
        keep[path[:, 0], path[:, 1]] = True
        pruned[y:y+h, x:x+w][keep] = 255
    return pruned


def _seg_angle(seg):
    """功能：对一批等长点列做最小二乘（主轴）直线拟合，给出各条拟合直线的方向角——相当于斜率，但竖直线不发散。
    输入：seg：np.ndarray，(M, k, 2) float，M 条直线、每条 k 个 (行, 列) 像素点
    输出：np.ndarray，(M,) float，各直线方向角(弧度)，范围 (-π/2, π/2]
    """
    d = seg - seg.mean(axis=1, keepdims=True)                  # 每段先减自身质心，再算协方差才有意义
    rr = (d[..., 0] * d[..., 0]).sum(1)                        # 协方差 C11 = ΣΔ行²
    cc = (d[..., 1] * d[..., 1]).sum(1)                        # 协方差 C22 = ΣΔ列²
    rc = (d[..., 0] * d[..., 1]).sum(1)                        # 协方差 C12 = ΣΔ行Δ列
    return 0.5 * np.arctan2(2.0 * rc, rr - cc)                 # 2x2 对称阵的主轴角，就是拟合直线的方向


def _slope_points(pruned, k=CURV_K, topn=4, min_dist=MIN_DIST):
    """功能：沿主干给每个前景像素取前后各 k 个像素分别拟合成一条直线，算出每个像素处的方向变化量，
          再跨所有连通域按方向变化量从大到小贪心挑点，保证选中的任意两个拐点至少相距 min_dist。
    输入：pruned：np.ndarray，uint8，0/255、单像素宽主干；k：int，前后各取 k 个像素做直线拟合；
          topn：int，最多输出的拐点数；min_dist：float，拐点之间最小欧氏距离(px)
    输出：tuple，元素为 (x, y) 拐点元组，按方向变化量从大到小排列；候选像素不足 topn 个就全输出
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats((pruned > 0).astype(np.uint8), connectivity=8)
    conv = cv2.filter2D((pruned > 0).astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8))
    endpoints_all = (pruned > 0) & (conv == 2)                 # 只有 1 个邻居的像素 = 主干两端
    all_turn = []                                              # 各连通域候选像素的方向变化量，最后拼成一个大数组
    all_xy = []                                                # 对应的整图 (x, y) 坐标，与 all_turn 一一对应
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area <= 2 * k:
            continue
        sub = labels[y:y+h, x:x+w] == idx                      # 该连通域（已裁剪到包围盒）掩膜
        path = _longest_path(sub, np.argwhere(endpoints_all[y:y+h, x:x+w]))
        if path is None:                                       # 闭环（无端点）：拿域内全部像素当端点候选取一条贯穿路径
            path = _longest_path(sub, np.argwhere(sub))
        if path is None or len(path) <= 2 * k:
            continue
        mid = path[k:len(path)-k]                              # 待判定的像素，局部 (行, 列)；天然只留前后各够 k 格的，不足 k 格的已落在切片外
        win = np.arange(len(mid))[:, None] + np.arange(k)      # (M, k) 窗口下标：前段 path[i:i+k]、后段 path[i+k:i+2k]
        ang_a = _seg_angle(path[win].astype(np.float64))       # 前 k 个点拟合直线的方向角
        ang_b = _seg_angle(path[win + k].astype(np.float64))   # 后 k 个点拟合直线的方向角
        diff = np.abs(ang_b - ang_a)
        turn = np.degrees(np.minimum(diff, np.pi - diff))      # 直线方向无正负之分，折到 0~90°
        all_turn.append(turn)
        all_xy.append(np.column_stack((mid[:, 1] + x, mid[:, 0] + y)))  # 局部(行,列)转整图(x,y)
    if not all_turn:
        return ()
    turn = np.concatenate(all_turn)                            # 所有候选像素的方向变化量
    xy = np.concatenate(all_xy)                                # 对应整图坐标 (N, 2)
    order = np.argsort(turn)[::-1]                             # 方向变化量从大到小的候选下标
    picked = []                                                # 已选中的候选下标
    for i in order:
        p = xy[i]                                              # 当前候选的 (x, y)
        if picked and np.any(np.sum((xy[picked] - p) ** 2, axis=1) < min_dist ** 2):
            continue                                           # 与任一已选点不足 min_dist，放弃该候选
        picked.append(i)
        if len(picked) == topn:
            break
    return tuple((int(xy[i, 0]), int(xy[i, 1])) for i in picked)


def _order_points(points):
    """功能：把四个拐点按 左上→右上→右下→左下 排序。
    输入：points：可迭代的 (x, y) 像素坐标，应为 4 个点
    输出：list，[(x, y), ...] 按左上/右上/右下/左下顺序；点数不足 4 个时按原顺序返回
    """
    pts = list(points)
    if len(pts) != 4:
        return pts
    arr = np.asarray(pts, np.float64)                    # (4,2) 像素坐标
    dist = np.sum(arr ** 2, axis=1)                      # 到图像左上原点(0,0)的平方距离 sqrt(u²+v²)
    tl = int(np.argmin(dist))                            # 左上：离原点最近
    br = int(np.argmax(dist))                            # 右下：离原点最远
    rest = [i for i in range(4) if i not in (tl, br)]    # 剩下两个顶点的下标
    i0, i1 = rest
    tr, bl = (i0, i1) if arr[i0, 0] >= arr[i1, 0] else (i1, i0)  # x 较大的是右上，另一个是左下
    order = [tl, tr, br, bl]
    return [(int(arr[i, 0]), int(arr[i, 1])) for i in order]


def _draw_points(points, shape, radius=5):
    """功能：把拐点画在纯黑背景上（白色实心圆），用来肉眼核对位置。
    输入：points：可迭代的 (x, y) 拐点元组；shape：(h, w) 或 (h, w, c)，决定画布尺寸；
          radius：int，白点半径(px)
    输出：np.ndarray，uint8，黑底白点图，尺寸为 shape[:2]
    """
    canvas = np.zeros(shape[:2], np.uint8)                     # 纯黑画布
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), radius, 255, -1)   # 半径 5 的实心白圆
    return canvas


def _devide_area(img, area):
    """功能：划分有效成像区域，把 area 多边形覆盖住的像素清 0，也就是去掉这块区域。
    输入：img：np.ndarray，单通道或三通道图；area：[(x, y), ...]，多边形顶点，按顺序首尾相连
    输出：np.ndarray，与 img 同尺寸同通道，多边形内部为 0
    """
    h, w = img.shape[:2]                                        # 单通道是 (h,w)，三通道是 (h,w,3)
    pts = np.asarray(area, np.int32).reshape(-1, 1, 2)          # fillPoly 要求 int32 的 (N,1,2) 顶点
    mask = np.full((h, w), 255, np.uint8)                       # 先全保留，uint32 会让 fillPoly 直接报错
    cv2.fillPoly(mask, [pts], 0)                                # 再把多边形内部涂 0，等于去掉这块
    return cv2.bitwise_and(img, img, mask=mask)                 # 只留下多边形以外的像素


def image_processing(path=IMAGE_DIR, out_npz=OUTPUT_NPZ):
    """功能：批量处理文件夹内的 png，提取每张图的四个拐点，按图片顺序存成 npz。
    输入：path：str，图片文件夹路径；out_npz：str，结果 npz 保存路径
    输出：list，每个元素是该图四个拐点的 ((x, y), ...) 二维元组
    """
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    result = []
    for file_path in files:
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"跳过无法读取的图片：{file_path}")
            continue
        gray = _devide_area(gray,AREA)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = _remove_small(binary,MIN_AREA)
        skeleton = _thin(binary)
        pruned = _prune_and_simplify(skeleton)
        points = _order_points(_slope_points(pruned))
        result.append(tuple(points))
        print(f"{os.path.basename(file_path)}：拐点 {len(points)} 个")
    np.savez_compressed(out_npz, points=result)
    return result


if __name__ == "__main__":
    image_processing()
