"""结构光图像处理：从多张 png 中提取每段光线的质点。

流程：读图 → 灰度 → 顶帽(保留细长结构光) → 自适应二值化 → 细化 → 最长路径去毛刺
      → approxPolyDP 简化并填充原线与简化线之间的空白 → 连通域质心 → npz 保存
运行：python 图像处理.py
"""

import cv2
import numpy as np
import glob
import os
from collections import deque

# ---------- 可调参数（数值按需手改） ----------

IMAGE_DIR = r"C:\python\3D_project_3\images"           # 输入图片文件夹
OUTPUT_NPZ = r"C:\python\3D_project_3\points.npz"  
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"  # 鱼眼标定结果(K/D)路径
BALANCE = 0.0                   # 去畸变保留比例：0 完整保留(黑边多)，1 尽量填满(会裁掉部分内容)
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
MIN_AREA = 300                 # 最小连通域面积(px)，小于此值视作噪声丢弃
APPROX_EPSILON = 3.0           # approxPolyDP 简化容差(px)，越大折线顶点越少
kernel = np.array([3,3])
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
        x, y, w, h, _ = stats[idx]                             # 连通域包围盒
        sub = labels[y:y+h, x:x+w] == idx                      # 该连通域（已裁剪到包围盒）掩膜
        path = _longest_path(sub, np.argwhere(endpoints_all[y:y+h, x:x+w]))
        if path is None:                                       # 端点不足两个（闭环）：无毛刺可去，整段保留
            path = np.argwhere(sub)
        keep = np.zeros_like(sub)
        keep[path[:, 0], path[:, 1]] = True
        pruned[y:y+h, x:x+w][keep] = 255
    return pruned


def _largest_cluster(pruned, max_dist=200):
    """功能：把 pruned 中值为 255 的点按像素距离聚类，返回只含最大聚类的二值图。
    输入：pruned：np.ndarray，uint8，0/255 图；max_dist：int，同一聚类的最大像素距离(px)
    输出：np.ndarray，uint8，与 pruned 同尺寸，仅最大聚类的点为 255，其余为 0，可直接 imshow
    """
    ys, xs = np.nonzero(pruned)                           # 有效点(值=255)的行、列坐标
    n = len(xs)
    if n == 0:
        return np.zeros_like(pruned)

    parent = np.arange(n)                                 # 并查集：每个点所属的父节点
    size = np.ones(n, np.int32)                           # 并查集：每棵树大小，用于按秩合并

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]                 # 路径减半压缩
            i = parent[i]
        return i

    cell = max_dist                                       # 网格边长=阈值，距离≤阈值的两点必在相邻格内
    cx, cy = xs // cell, ys // cell
    buckets = {}
    for i in range(n):
        buckets.setdefault((cx[i], cy[i]), []).append(i)

    d2 = max_dist * max_dist                              # 距离平方阈值，比较时省去开方
    for i in range(n):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx[i] + dx, cy[i] + dy), ()):
                    if j <= i:
                        continue
                    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= d2:
                        ri, rj = _find(i), _find(j)
                        if ri != rj:
                            if size[ri] < size[rj]:
                                ri, rj = rj, ri
                            parent[rj] = ri
                            size[ri] += size[rj]

    roots = np.array([_find(i) for i in range(n)])        # 压缩后每个点的最终根
    root, count = np.unique(roots, return_counts=True)
    best = root[np.argmax(count)]                         # 点数最多的类的根
    keep = roots == best
    mask = np.zeros_like(pruned)                          # 同尺寸全 0 图，只点亮最大聚类
    mask[ys[keep], xs[keep]] = 255
    return mask


def _connect_endpoints(cluster):
    """功能：把 cluster 中的所有端点，各自连到离自己最近、尚未配对、且不在同一连通域内的端点，用直线补齐。
    输入：cluster：np.ndarray，uint8，0/255 二值图（骨架）
    输出：np.ndarray，uint8，与 cluster 同尺寸，在骨架基础上画了直线，可直接 imshow
    """
    fg = (cluster > 0).astype(np.uint8)                       # 前景像素
    conv = cv2.filter2D(fg, cv2.CV_16U, np.ones((3, 3), np.uint8))  # 每个点 3x3 邻域前景计数
    ys, xs = np.nonzero(fg & (conv == 2))                     # 端点：邻域内只含自身 + 1 个邻居
    out = cluster.copy()
    n = len(xs)
    if n < 2:
        return out                                            # 端点不足两个，无需连接
    pts = np.column_stack((xs, ys)).astype(np.int32)          # (N,2) 每行 (x, y)，供 cv2.line 使用
    _, labels = cv2.connectedComponents(fg, connectivity=8)   # 给前景打连通域标签，0 为背景
    lab = labels[ys, xs]                                      # 每个端点所属的连通域标签
    used = np.zeros(n, bool)                                  # 标记该端点是否已配对，避免重复连线
    for i in range(n):
        if used[i]:
            continue
        d = ((pts[:, 0] - pts[i, 0]) ** 2 + (pts[:, 1] - pts[i, 1]) ** 2).astype(np.float64)  # i 到各端点的距离平方
        d[used] = np.inf
        d[i] = np.inf
        d[lab == lab[i]] = np.inf                             # 不与同一连通域内的端点相连
        j = int(np.argmin(d))                                 # 最近的、尚未配对且异域的端点
        if d[j] == np.inf:
            continue                                          # 剩余端点都已配对或同域不可连
        cv2.line(out, (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])), 255, 1)    # 画 1 像素宽的直线连接
        used[i] = used[j] = True
    return out


def _ordered_paths(result):
    """功能：把二值骨架图拆成若干有序路径段，供 k-cosine 角点检测使用。
    输入：result：np.ndarray，uint8，0/255 骨架图（可能含端点、交叉点与闭合环）
    输出：list，每项为 (seg, closed)；seg 是 (P,2) int32 的 (x,y) 有序点列，closed 为 True 表示闭合环
    """
    fg = (result > 0).astype(np.uint8)
    h, w = result.shape
    conv = cv2.filter2D(fg, cv2.CV_16U, np.ones((3, 3), np.uint8))  # 每个点 3x3 邻域前景计数
    num, labels, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    def _to_xy(seg):
        return np.array(seg, np.int32)[:, ::-1]           # (y,x) 有序点列转成 (x,y)

    paths = []
    for idx in range(1, num):                             # idx=0 是背景
        ys, xs = np.nonzero(labels == idx)
        n = len(xs)
        if n == 0:
            continue
        coords = np.column_stack((ys, xs))                # (n,2) 每行 (y,x)，行/列方便直接索引图像
        pos = {(int(ys[i]), int(xs[i])): i for i in range(n)}  # (y,x)->局部索引
        deg = conv[ys, xs] - 1                            # 邻居数 = 邻域前景数减自身
        nodes = np.nonzero(deg != 2)[0]                   # 度!=2 的点作节点（端点/交叉点）
        visited = np.zeros(n, bool)                       # 普通点是否已归属某段；节点不标记，可被多段共享
        edges_done = set()                                # 已添加的节点-节点无向边，避免重复

        if nodes.size == 0:                               # 纯环：所有点度都为 2
            order = [coords[0]]
            prev, cur = -1, 0
            while True:
                cy, cx = coords[cur]
                nxt = -1
                for dy, dx in _NEIGHBORS8:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == idx:
                        j = pos[(int(ny), int(nx))]
                        if j != prev:
                            nxt = j
                            break
                if nxt == 0:                              # 绕回起点，闭环完成
                    break
                order.append(coords[nxt])
                prev, cur = cur, nxt
            paths.append((_to_xy(order), True))
            continue

        for ni in nodes:
            ny0, nx0 = coords[ni]
            for dy, dx in _NEIGHBORS8:
                ty, tx = ny0 + dy, nx0 + dx
                if not (0 <= ty < h and 0 <= tx < w and labels[ty, tx] == idx):
                    continue
                j = pos[(int(ty), int(tx))]
                if deg[j] == 2:                           # 邻居是普通点：沿度=2 路径走到下一个节点
                    if visited[j]:
                        continue
                    seg = [coords[ni]]
                    prev, cur = ni, j
                    visited[j] = True
                    seg.append(coords[cur])
                    while deg[cur] == 2:
                        cy, cx = coords[cur]
                        nxt = -1
                        for dy2, dx2 in _NEIGHBORS8:
                            ny2, nx2 = cy + dy2, cx + dx2
                            if 0 <= ny2 < h and 0 <= nx2 < w and labels[ny2, nx2] == idx:
                                k = pos[(int(ny2), int(nx2))]
                                if k != prev:
                                    nxt = k
                                    break
                        if nxt < 0:
                            break
                        seg.append(coords[nxt])
                        prev, cur = cur, nxt
                        if deg[cur] == 2:
                            visited[cur] = True
                    paths.append((_to_xy(seg), False))
                else:                                     # 邻居也是节点：直接一条两点的边
                    key = (min(ni, j), max(ni, j))
                    if key in edges_done:
                        continue
                    edges_done.add(key)
                    paths.append((_to_xy([coords[ni], coords[j]]), False))
    return paths


def _fit_direction(seg):
    """功能：用总最小二乘（PCA）求点集的主方向角，规避垂直线斜率无穷的问题。
    输入：seg：np.ndarray，(N,2) 的 (x,y) 点集
    输出：float，主方向角（弧度，范围 [-π/2, π/2)）
    """
    x = seg[:, 0].astype(np.float64)
    y = seg[:, 1].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    sxx = (x * x).sum()                                   # 点集相对均值的协方差分量
    syy = (y * y).sum()
    sxy = (x * y).sum()
    return 0.5 * np.arctan2(2.0 * sxy, sxx - syy)


def _extract_corners(result, k=30, angle_deg=20, min_dist=30):
    """功能：用 k-cosine 提取 result 骨架上的角点（曲率/拐弯剧烈的点）。
    输入：result：np.ndarray，uint8，0/255 骨架图；
          k：int，前后各取的点数；angle_deg：float，夹角阈值(度)；min_dist：float，角点间最小距离(px)
    输出：(N,2) float32 数组，每行是角点的 (x,y)；无角点时返回 (0,2) 空数组
    """
    angle_rad = np.deg2rad(angle_deg)
    cands = []                                            # 候选角点，每项 (x, y, 夹角响应)
    for seg, closed in _ordered_paths(result):
        p = len(seg)
        if p < 2 * k + 1:
            continue                                      # 段太短，前后凑不满 k 个点，跳过
        for i in range(p):
            if closed:                                    # 闭合环用环形索引，首尾相连
                front = seg[np.arange(i - k, i + 1) % p]  # 前 k 个点 + 当前点
                back = seg[np.arange(i, i + k + 1) % p]   # 当前点 + 后 k 个点
            else:
                if i < k or i > p - 1 - k:                # 开放路径两端不足 k 个点，跳过
                    continue
                front = seg[i - k:i + 1]
                back = seg[i:i + k + 1]
            d = abs(_fit_direction(front) - _fit_direction(back))
            d = min(d, np.pi - d)                         # 两条直线夹角取较小的锐角
            if d > angle_rad:
                cands.append((float(seg[i, 0]), float(seg[i, 1]), d))
    if not cands:
        return np.empty((0, 2), np.float32)
    cands = np.array(cands)                               # (M,3) 列 0=x 列 1=y 列 2=响应
    order = np.argsort(-cands[:, 2])                      # 按夹角响应从大到小排序
    picked = []
    for i in order:
        x, y = cands[i, 0], cands[i, 1]
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist * min_dist for px, py in picked):
            picked.append((x, y))                         # 贪心：与已选角点距离≥min_dist 才保留
    return np.array(picked, np.float32).reshape(-1, 2)


def _undistort_fisheye(img, K, D, balance=0.0):
    """功能：对单通道灰度图做鱼眼去畸变并裁剪黑边，完整保留广角内容。
    输入：img：np.ndarray，uint8 单通道灰度图；K：(3,3) float64 内参；D：(4,1) 畸变系数；
          balance：float，0 完整保留(黑边多)，1 尽量填满(会裁掉部分内容)
    输出：np.ndarray，uint8，去畸变并裁掉黑边后的灰度图，可直接用于后续处理
    """
    h, w = img.shape[:2]                                        # 高、宽，Size 需按 (宽,高) 传入
    R = np.eye(3, dtype=np.float64)                             # 单位旋转，不额外旋转图像
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R, balance=balance)                       # 缩小焦距后的新内参，广角内容完整装入画布
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, Knew, (w, h), cv2.CV_32FC1)                    # 重映射表，float 坐标便于后续判有效区
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 黑边区域填 0
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map1, map2, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 全白图过一遍，得到有效区掩膜
    ys, xs = np.nonzero(mask)                                   # 有效像素坐标，黑边处 mask 为 0
    if ys.size == 0:                                            # 理论不触发，防御性返回
        return undist
    y0, y1 = ys.min(), ys.max() + 1                             # 有效区域包围盒
    x0, x1 = xs.min(), xs.max() + 1
    return undist[y0:y1, x0:x1]


def image_processing(path=IMAGE_DIR):
    """功能：批量处理文件夹内的 png，每张生成去毛刺后的骨架图 _debug/pruned.png。
    输入：path：str，图片文件夹路径
    输出：无
    """
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    calib = np.load(CALIB_NPZ)                              # 只读加载标定结果，避免每张图重复 IO
    K = calib['K']                                          # (3,3) 内参矩阵
    D = calib['D']                                          # (4,1) 畸变系数 k1~k4
    all_corners = []                                        # 累积所有图的角点，最后统一保存到 npz
    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"跳过无法读取的图片：{file_path}")
            continue
        img = img[:, :, 2]                                  # 取红通道得到单通道灰度图
        img = _undistort_fisheye(img, K, D, BALANCE)        # 鱼眼去畸变并裁黑边
        blur = cv2.GaussianBlur(img,(5,5),0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = _remove_small(binary, MIN_AREA)
        skeleton = _thin(binary)
        pruned = _prune_and_simplify(skeleton)
        cluster = _largest_cluster(pruned, 100)
        result = _connect_endpoints(cluster)
        corners = _extract_corners(result)
        if len(corners):
            all_corners.append(corners)                         # 收集本图角点，稍后合并保存
        print(f"角点数量：{len(corners)}")
    if all_corners:                                          # 所有图的角点合并成 (N,2) 后写入 points.npz
        pts = np.vstack(all_corners).astype(np.float64)
        np.savez_compressed(OUTPUT_NPZ, points=pts)          # 字段名 points，点云重建 _load_array 可直接读
        print(f"已保存 {len(pts)} 个角点到 {OUTPUT_NPZ}")
    else:
        print("未提取到角点，未保存 points.npz")
if __name__ == "__main__":
    image_processing()
