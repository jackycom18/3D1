"""结构光图像处理（Steger 升级版）：从多张 png 中提取结构光骨架并用 Steger 算法提取高精度亚像素中心。

流程：读图 → 取红通道 → 鱼眼去畸变 → 高斯滤波/自适应二值化 → 快速细化 
      → 拓扑闭环/开放判断 → 最长路径/主环剪枝 → Steger 算法提取无偏亚像素中心点 → 输出点集
运行：python 图像处理_重建1.py
"""

import cv2
import numpy as np
import glob
import os
import re
import time
from skimage.morphology import skeletonize

# ---------- 可调参数（数值按需手改） ----------

IMAGE_DIR = r"C:\python\3D_project_3\takephoto\images"           # 输入图片文件夹
CALIB_NPZ = r"C:\python\3D_project_3\calib_result1.npz"         # 鱼眼标定结果(K/D)路径
BALANCE = 0.0                   # 去畸变保留比例：0 完整保留(黑边多)，1 尽量填满(会裁掉部分内容)
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
kernel = np.ones((3, 3), np.uint8)  # 开运算核(3x3)，去掉细小噪声
DILATE_SIZE = 5                # 膨胀窗口大小(px)
_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 8 邻域行列偏移
DEBUG_DIR = r"C:\python\3D_project_3\_debug"                     # 调试输出文件夹
MIN_LOOP_LEN = 700             # 闭合环长度阈值(px)
STEGER_SIGMA = 2.5             # Steger 算法高斯核尺度（约等于光条半宽的 1/sqrt(3) ~ 1倍）


def _remove_small(binary):
    """功能：只保留面积最大的前景连通域，其余连通域全部清 0，用于去噪。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255，仅保留面积最大的连通域
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    if stats.shape[0] <= 1:
        return np.zeros_like(binary)
    area = stats[:, cv2.CC_STAT_AREA]
    best = int(np.argmax(area[1:])) + 1
    return np.where(labels == best, 255, 0).astype(np.uint8)


def _fast_thin(binary):
    """功能：用 C/Cython 加速的 Zhang-Suen 算法提取 1 像素宽骨架（比纯 Python 快 50~100 倍）。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255 骨架图
    """
    if not binary.any():
        return np.zeros_like(binary)
    return (skeletonize(binary > 0, method='zhang') * 255).astype(np.uint8)


def _degree(skel):
    """功能：计算骨架每个前景点的度数(8 邻域前景邻居数，不含自身)。"""
    h, w = skel.shape
    conv = cv2.filter2D(skel, cv2.CV_16U, np.ones((3, 3), np.uint8))
    deg = np.zeros((h, w), np.int16)
    m = skel > 0
    deg[m] = conv[m] - 1
    return deg


def _build_graph(skel):
    """功能：把骨架压缩成图——节点为分支点(度>=3)与端点(度==1)，边为中间点(度==2)构成的路径。"""
    h, w = skel.shape
    deg = _degree(skel)
    node_mask = (deg >= 3) | (deg == 1)
    node_id = {}
    nodes = []
    branch_idx = []
    for (y, x) in np.argwhere(node_mask):
        node_id[(int(y), int(x))] = len(nodes)
        nodes.append((int(y), int(x)))
        if deg[y, x] >= 3:
            branch_idx.append(len(nodes) - 1)

    edge_map = {}
    for i, (ny, nx) in enumerate(nodes):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ty, tx = ny + dy, nx + dx
                if not (0 <= ty < h and 0 <= tx < w) or not skel[ty, tx]:
                    continue
                if (ty, tx) in node_id:
                    j = node_id[(ty, tx)]
                    key = (min(i, j), max(i, j))
                    edge_map.setdefault(key, []).append((1, []))
                    continue
                path = []
                cy, cx, prev = ty, tx, (ny, nx)
                ok = False
                while True:
                    if (cy, cx) in node_id:
                        ok = True
                        break
                    if deg[cy, cx] != 2:
                        break
                    path.append((cy, cx))
                    nxt = None
                    for ddy in (-1, 0, 1):
                        for ddx in (-1, 0, 1):
                            if ddy == 0 and ddx == 0:
                                continue
                            t2y, t2x = cy + ddy, cx + ddx
                            if 0 <= t2y < h and 0 <= t2x < w and skel[t2y, t2x] and (t2y, t2x) != prev:
                                nxt = (t2y, t2x)
                                break
                        if nxt is not None:
                            break
                    if nxt is None:
                        break
                    prev = (cy, cx)
                    cy, cx = nxt
                if ok:
                    j = node_id[(cy, cx)]
                    key = (min(i, j), max(i, j))
                    edge_map.setdefault(key, []).append((len(path) + 1, path))

    adj = {i: [] for i in range(len(nodes))}
    for (a, b), segs in edge_map.items():
        segs.sort(key=lambda s: s[0])
        length, pixels = segs[0]
        adj[a].append((b, length, pixels))
        adj[b].append((a, length, pixels))
    return nodes, adj, branch_idx


def _find_cycle(nodes, adj, start, min_len):
    """功能：在压缩图上找长于 min_len 的闭合环。"""
    visited = {start}
    parent = {start: -1}
    stack = [(start, -1, 0, 0)]

    while stack:
        cur, par, cur_cum, idx = stack[-1]
        if idx >= len(adj[cur]):
            stack.pop()
            continue
        stack[-1] = (cur, par, cur_cum, idx + 1)
        n2, w2, _ = adj[cur][idx]
        if n2 == par:
            continue
        if n2 == start:
            if cur_cum + w2 > min_len:
                chain = []
                u = cur
                while u != n2:
                    chain.append(u)
                    u = parent[u]
                chain.append(n2)
                chain.reverse()
                pixels = []
                for k in range(len(chain)):
                    a = chain[k]
                    b = chain[(k + 1) % len(chain)]
                    pixels.append(nodes[a])
                    for (nb, _w, mid) in adj[a]:
                        if nb == b:
                            pixels.extend(mid)
                            break
                return pixels
            continue
        if n2 in visited:
            continue
        visited.add(n2)
        parent[n2] = cur
        stack.append((n2, cur, cur_cum + w2, 0))
    return None


def _prune_and_simplify(skeleton, min_len=MIN_LOOP_LEN):
    """功能：去毛刺提取主闭合环。"""
    skel = (skeleton > 0).astype(np.uint8)
    if not skel.any():
        return skeleton
    deg = _degree(skel)
    if not (deg >= 3).any():
        return skeleton
    nodes, adj, branch_idx = _build_graph(skel)
    for i in branch_idx:
        cycle = _find_cycle(nodes, adj, i, min_len)
        if cycle is not None:
            out = np.zeros_like(skel)
            for (py, px) in cycle:
                out[py, px] = 1
            return (out * 255).astype(np.uint8)
    return skeleton


def _longest_path(skeleton):
    """功能：提取最长路径主干（适用于开放曲线）。"""
    skel = (skeleton > 0).astype(np.uint8)
    if not skel.any():
        return skeleton
    deg = _degree(skel)
    if not (deg >= 3).any():
        return skeleton
    nodes, adj, _ = _build_graph(skel)

    def _farthest(src):
        dist = {src: 0}
        parent = {src: -1}
        stack = [src]
        while stack:
            u = stack.pop()
            for (v, w, _) in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    stack.append(v)
        far = max(dist, key=lambda k: dist[k])
        return far, dist[far], parent

    u, _, _ = _farthest(0)
    v, _, parent = _farthest(u)
    chain = [v]
    while chain[-1] != u:
        chain.append(parent[chain[-1]])
    chain.reverse()

    out = np.zeros_like(skel)
    for k in range(len(chain) - 1):
        a, b = chain[k], chain[k + 1]
        out[nodes[a][0], nodes[a][1]] = 1
        for (nb, _w, mid) in adj[a]:
            if nb == b:
                for (py, px) in mid:
                    out[py, px] = 1
                break
    out[nodes[chain[-1]][0], nodes[chain[-1]][1]] = 1
    return (out * 255).astype(np.uint8)


def _ordered_paths(result):
    """功能：把二值骨架图拆成若干有序路径段。"""
    fg = (result > 0).astype(np.uint8)
    h, w = result.shape
    conv = cv2.filter2D(fg, cv2.CV_16U, np.ones((3, 3), np.uint8))
    num, labels, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    def _to_xy(seg):
        return np.array(seg, np.int32)[:, ::-1]

    paths = []
    for idx in range(1, num):
        ys, xs = np.nonzero(labels == idx)
        n = len(xs)
        if n == 0:
            continue
        coords = np.column_stack((ys, xs))
        pos = {(int(ys[i]), int(xs[i])): i for i in range(n)}
        deg = conv[ys, xs] - 1
        nodes = np.nonzero(deg != 2)[0]
        visited = np.zeros(n, bool)
        edges_done = set()

        if nodes.size == 0:
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
                if nxt == 0:
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
                if deg[j] == 2:
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
                else:
                    key = (min(ni, j), max(ni, j))
                    if key in edges_done:
                        continue
                    edges_done.add(key)
                    paths.append((_to_xy([coords[ni], coords[j]]), False))
    return paths


def _steger_subpixel(pruned, gray, sigma=STEGER_SIGMA):
    """功能：用 Steger 算法（Hessian 矩阵二阶高斯导数）在 pruned 骨架邻域提取激光条纹中心亚像素点。
    输入：pruned：np.ndarray，uint8，0/255 骨架图；gray：np.ndarray，uint8 灰度底图（去畸变后）；
          sigma：float，高斯核尺度（约等于光条半宽的 1/sqrt(3) ~ 1倍，推荐 2.0 ~ 3.0）
    输出：(M,2) float64，每行 [y, x] 亚像素坐标，按有序路径排列
    """
    ys, xs = np.nonzero(pruned)
    if ys.size == 0:
        return np.empty((0, 2), np.float64)

    # 1. 裁剪 ROI 包围盒以加速卷积
    pad = int(round(4 * sigma)) + 5
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    roi_gray = gray[y0:y1, x0:x1].astype(np.float64)

    # 2. 构造 1D 高斯平滑核与一、二阶导数核
    radius = int(round(3 * sigma))
    x_axis = np.arange(-radius, radius + 1, dtype=np.float64)
    g0 = np.exp(-0.5 * (x_axis / sigma) ** 2)
    g0 /= g0.sum()
    g1 = -x_axis / (sigma ** 2) * g0
    g2 = (x_axis ** 2 - sigma ** 2) / (sigma ** 4) * g0

    # 3. 二维可分离高斯卷积得到一阶、二阶偏导数图
    Ix = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g1, g0)
    Iy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g0, g1)
    Ixx = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g2, g0)
    Iyy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g0, g2)
    Ixy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g1, g1)

    # 4. 沿骨架有序路径逐点求取解析亚像素中心
    pts = []
    for seg, closed in _ordered_paths(pruned):
        for (cx, cy) in seg:
            ly, lx = cy - y0, cx - x0
            ix = Ix[ly, lx]
            iy = Iy[ly, lx]
            ixx = Ixx[ly, lx]
            iyy = Iyy[ly, lx]
            ixy = Ixy[ly, lx]

            # Hessian 特征值分解
            trace = ixx + iyy
            diff = ixx - iyy
            disc = np.sqrt(diff * diff + 4.0 * ixy * ixy)
            lambda1 = 0.5 * (trace - disc)  # 最负特征值（对应亮线山脊峰度）

            # 过滤非凸脊线或曲率过小的平缓区
            if lambda1 >= -0.05:
                continue

            # 对应最大特征值的特征向量（严格法线方向）
            if abs(ixy) > 1e-7:
                nx = ixy
                ny = lambda1 - ixx
            else:
                nx, ny = (1.0, 0.0) if abs(ixx) >= abs(iyy) else (0.0, 1.0)
            norm = np.hypot(nx, ny)
            if norm < 1e-7:
                continue
            nx /= norm
            ny /= norm

            # 沿法线一阶导数为 0 的亚像素偏移量 t
            t = -(ix * nx + iy * ny) / lambda1
            
            # 偏离在合理邻域内且局部灰度达到底噪阈值
            if abs(t) <= 2.5 and roi_gray[ly, lx] >= 20:
                pts.append([cy + t * ny, cx + t * nx])

    if not pts:
        return np.empty((0, 2), np.float64)
    return np.array(pts, np.float64)


def _undistort_fisheye(img, K, D, balance=0.0):
    """功能：对单通道灰度图做鱼眼去畸变并裁剪黑边。"""
    h, w = img.shape[:2]
    R = np.eye(3, dtype=np.float64)
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R, balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, Knew, (w, h), cv2.CV_32FC1)
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map1, map2, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return undist
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return undist[y0:y1, x0:x1]


def image_processing(path=IMAGE_DIR, method="steger"):
    """功能：批量处理文件夹内的 png，返回每张图的亚像素骨架点集。
    输入：path：str，图片文件夹路径；method：str，"steger" 或 "centroid"
    输出：list，每项是 (N,2) float 的 [y, x] 亚像素点集
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    def _file_index(f):
        m = re.search(r'\d+', os.path.basename(f))
        return int(m.group()) if m else 0

    files = sorted(glob.glob(os.path.join(path, "*.png")), key=_file_index)
    calib = np.load(CALIB_NPZ)
    K = calib['K']
    D = calib['D']
    points = []

    print(f"开始批量处理图像，算法模式：{method.upper()}，共 {len(files)} 张图片...")
    for idx, file_path in enumerate(files):
        t0 = time.time()
        img = cv2.imread(file_path)
        if img is None:
            print(f"跳过无法读取的图片：{file_path}")
            points.append(None)
            continue

        # 1. 取红通道并做鱼眼去畸变
        gray = img[:, :, 2]
        undist = _undistort_fisheye(gray, K, D, BALANCE)
        blur = cv2.GaussianBlur(undist, (5, 5), 0)

        # 2. 自适应二值化与连通域去噪
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = _remove_small(binary)

        # 3. C 级极速细化（从 2000ms 降至 30ms）
        skeleton = _fast_thin(binary)

        # 4. 拓扑孔洞判定：直接统计背景连通域数（消除二次细化，耗时从 2000ms 降至 1ms）
        num_labels, _ = cv2.connectedComponents((binary == 0).astype(np.uint8))
        is_closed = (num_labels >= 3)  # 背景数>=2（外背景+内封闭孔洞）即为闭合环

        # 5. 拓扑去毛刺
        if is_closed:
            pruned = _prune_and_simplify(skeleton)
        else:
            pruned = _longest_path(skeleton)

        # 6. 提取亚像素中心（Steger 算法）
        pts = _steger_subpixel(pruned, undist, sigma=STEGER_SIGMA)
        points.append(pts)

        t_elapsed = (time.time() - t0) * 1000
        print(f"[{idx+1}/{len(files)}] {os.path.basename(file_path)}: 提取 {len(pts)} 亚像素点, 耗时 {t_elapsed:.1f}ms")

    return points


if __name__ == "__main__":
    image_processing()
