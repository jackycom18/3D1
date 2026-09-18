"""结构光图像处理：从多张 png 中提取每段光线的质点。

流程：读图 → 灰度 → 顶帽(保留细长结构光) → 自适应二值化 → 细化 → 最长路径去毛刺
      → approxPolyDP 简化并填充原线与简化线之间的空白 → 连通域质心 → npz 保存
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

IMAGE_DIR = r"C:\python\3D_project_3\images"           # 输入图片文件夹
OUTPUT_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 质点结果保存路径
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
MIN_AREA = 500                 # 最小连通域面积(px)，小于此值视作噪声丢弃
SPUR_MIN_AREA = 100            # 骨架连通域最小面积(px)，小于此值整段丢弃
APPROX_EPSILON = 3.0           # approxPolyDP 简化容差(px)，越大折线顶点越少
kernel = np.array([3,3])
AREA = [                       # 要去掉的区域多边形顶点(x,y)，按顺序首尾相连，边不能交叉
    [1200,800],                # 左上
    [2400,600],                # 右上
    [2300,1900],               # 右下
    [1200,1900],               # 左下
]
_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 8 邻域行列偏移


def _remove_small(binary, min_area, drop_rank=2):
    """
    drop_rank: 丢弃面积第几大的前景连通域。1=最大，2=第二大，以此类推。
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= min_area
    keep[0] = False                                     # 背景不保留
    if num > 1:
        fg_labels = np.arange(1, num)                   # 前景连通域的真实标签 [1, 2, 3, ...]
        fg_area = area[1:]                              # 对应的面积
        order = fg_area.argsort()                       # 升序
        target_label = fg_labels[order[-drop_rank]]     # 第 drop_rank 大的真实标签
        keep[target_label] = False
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
    """功能：逐段处理骨架——先按最长路径去掉毛刺，再用 approxPolyDP 简化，并把原线与简化线之间的空白填成 255。
    输入：skeleton：np.ndarray，uint8，0/255 骨架图
    输出：(pruned, filled)：去毛刺后的骨架图与填充后的图，均 uint8、0/255、与输入同尺寸
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), 8)
    conv = cv2.filter2D((skeleton > 0).astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8))
    endpoints_all = (skeleton > 0) & (conv == 2)               # 含自身共 2 个前景像素 = 只有 1 个邻居
    pruned = np.zeros_like(skeleton)
    filled = np.zeros_like(skeleton)
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
        approx = cv2.approxPolyDP(path.reshape(-1, 1, 2).astype(np.int32), APPROX_EPSILON, False)
        if len(approx) < 2:
            continue
        poly = np.vstack([path, approx.reshape(-1, 2)[::-1]]).astype(np.int32)   # 沿原线去、沿简化线回的封闭多边形
        patch = np.zeros_like(sub, np.uint8)
        cv2.fillPoly(patch, [poly], 255)                       # 把两线之间的空白填充为 255
        filled[y:y+h, x:x+w] = np.maximum(filled[y:y+h, x:x+w], patch)
    return pruned, filled


def _centroids(binary, min_area):
    """功能：求二值图各连通域的质心。
    输入：binary：np.ndarray，uint8，0/255 二值图；min_area：int，最小面积(px)
    输出：(N, 2) float32 数组，每行是一个质心的 (x, y) 坐标；无有效连通域时为空数组
    """
    num, _, stats, cents = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    keep = stats[1:, cv2.CC_STAT_AREA] >= min_area             # 按面积过滤噪点连通域
    return cents[1:][keep].astype(np.float32)


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
    """功能：批量处理文件夹内的 png，提取每张图的质点并以“文件名 → 质点数组”存成 npz。
    输入：path：str，图片文件夹路径；out_npz：str，结果 npz 保存路径
    输出：dict，键为图片名(不含扩展名)，值为该图 (N, 2) float32 质点数组
    """
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    result = {}
    for file_path in files:
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"跳过无法读取的图片：{file_path}")
            continue
        blur = cv2.GaussianBlur(gray,(5,5),0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = _remove_small(binary, MIN_AREA)
        binary = _devide_area(binary,AREA)
        cv2.namedWindow("binary",cv2.WINDOW_NORMAL)
        cv2.imshow("binary",binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        skeleton = _thin(binary)
        cv2.imwrite(r"_debug/skeleton.png",skeleton)
        pruned, filled = _prune_and_simplify(skeleton)
        cv2.imwrite("_debug/pruned.png",pruned)
        centroids = _centroids(filled, MIN_AREA)
        result[os.path.splitext(os.path.basename(file_path))[0]] = centroids
        print(f"{os.path.basename(file_path)}：质点 {len(centroids)} 个")
    np.savez_compressed(out_npz, **result)
    return result


if __name__ == "__main__":
    image_processing()
