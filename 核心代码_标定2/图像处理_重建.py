"""结构光图像处理：从多张 png 中提取结构光骨架并去毛刺。

流程：读图 → 取红通道 → 鱼眼去畸变 → 高斯模糊 → 自适应二值化 → 开运算去噪
      → 去小连通域 → 细化 → 最长路径去毛刺 → 最大聚类 → 输出中间图
运行：python 图像处理_重建.py
"""

import cv2
import numpy as np
import glob
import os
import re

# ---------- 可调参数（数值按需手改） ----------

IMAGE_DIR = r"C:\python\3D_project_3\takephoto\images"           # 输入图片文件夹

CALIB_NPZ = r"C:\python\3D_project_3\calib_result1.npz"  # 鱼眼标定结果(K/D)路径
BALANCE = 0.0                   # 去畸变保留比例：0 完整保留(黑边多)，1 尽量填满(会裁掉部分内容)
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
kernel = np.ones((3, 3), np.uint8)  # 开运算核(3x3)，去掉细小噪声
DILATE_SIZE = 5             # 膨胀窗口大小(px，奇数)：只用膨胀图判断骨架闭合/开放，后续处理仍用未膨胀图
_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 8 邻域行列偏移
DEBUG_DIR = r"C:\python\3D_project_3\_debug"              # 调试输出文件夹：每张剪枝图按原文件名依次保存
MIN_LOOP_LEN = 700              # 闭合环长度阈值(px)：环长超过它才判定为真实结构并保留，否则换下一个分支点


def _remove_small(binary):
    """功能：只保留面积最大的前景连通域，其余连通域全部清 0，用于去噪。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255，仅保留面积最大的连通域；无前景时返回全 0
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    if stats.shape[0] <= 1:                               # 只有背景、无前景，直接返回全 0
        return np.zeros_like(binary)
    area = stats[:, cv2.CC_STAT_AREA]                     # 各连通域面积（idx0 是背景）
    best = int(np.argmax(area[1:])) + 1                   # 面积最大的前景连通域标签，跳过背景
    keep = np.zeros_like(area, bool)
    keep[best] = True
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


def _degree(skel):
    """功能：计算骨架每个前景点的度数(8 邻域前景邻居数，不含自身)。
    输入：skel：np.ndarray，uint8，0/1 骨架图
    输出：np.ndarray，int16，与输入同尺寸，前景点处为度数，背景为 0
    """
    h, w = skel.shape
    conv = cv2.filter2D(skel, cv2.CV_16U, np.ones((3, 3), np.uint8))
    deg = np.zeros((h, w), np.int16)
    m = skel > 0
    deg[m] = conv[m] - 1
    return deg


def _build_graph(skel):
    """功能：把骨架压缩成图——节点为分支点(度>=3)与端点(度==1)，边为两节点间由中间点(度==2)构成的路径；
    两节点间多条并行边只保留最短一条(语义同去重)。节点数量远小于像素数，后续找环在图上进行以提速。
    输入：skel：np.ndarray，uint8，0/1 骨架图
    输出：nodes：list[(y,x)] 节点坐标
          adj：dict int -> list[(nbr, length, pixels)]，length 为该边像素数(含两端节点)，pixels 为中间点坐标列表
          branch_idx：list[int]，分支点(度>=3)的节点编号，作为找环起点
    """
    h, w = skel.shape
    deg = _degree(skel)
    node_mask = (deg >= 3) | (deg == 1)                     # 分支点或端点作为节点
    node_id = {}
    nodes = []
    branch_idx = []
    for (y, x) in np.argwhere(node_mask):
        node_id[(int(y), int(x))] = len(nodes)
        nodes.append((int(y), int(x)))
        if deg[y, x] >= 3:
            branch_idx.append(len(nodes) - 1)

    edge_map = {}                                           # key=(a,b)排序 -> [(length, pixels)]
    for i, (ny, nx) in enumerate(nodes):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ty, tx = ny + dy, nx + dx
                if not (0 <= ty < h and 0 <= tx < w) or not skel[ty, tx]:
                    continue
                if (ty, tx) in node_id:                     # 直接相邻节点：长度 1 的边，无中间点
                    j = node_id[(ty, tx)]
                    key = (min(i, j), max(i, j))
                    edge_map.setdefault(key, []).append((1, []))
                    continue
                path = []                                   # 沿中间点(deg==2)追踪到下一节点
                cy, cx, prev = ty, tx, (ny, nx)
                ok = False
                while True:
                    if (cy, cx) in node_id:
                        ok = True
                        break
                    if deg[cy, cx] != 2:                    # 端点或异常，本段不成边
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
                    edge_map.setdefault(key, []).append((len(path) + 1, path))   # 边像素数 = 中间点数+1

    adj = {i: [] for i in range(len(nodes))}
    for (a, b), segs in edge_map.items():
        segs.sort(key=lambda s: s[0])                       # 并行边按长度升序，只保留最短
        length, pixels = segs[0]
        adj[a].append((b, length, pixels))
        adj[b].append((a, length, pixels))
    return nodes, adj, branch_idx


def _find_cycle(nodes, adj, start, min_len):
    """功能：在压缩后的骨架图上，从分支点节点 start 出发沿边走到自身，返回第一个环长>min_len 的闭合环像素序列，无则返回 None。
    输入：nodes：list[(y,x)] 节点坐标
          adj：dict int -> list[(nbr, length, pixels)]，由 _build_graph 生成
          start：int，起始节点编号(分支点)
          min_len：int，环长阈值(px)
    输出：list[(y,x)] 闭环像素序列，或 None
    """
    visited = {start}                                       # 已访问节点集合（全局标记，DFS 中不回退）
    parent = {start: -1}                                    # DFS 树父节点，用于回溯环路径
    stack = [(start, -1, 0, 0)]                             # 迭代栈：(cur, par, cur_cum, 下一个待处理邻居下标)，用显式栈代替递归避免深图栈溢出

    while stack:
        cur, par, cur_cum, idx = stack[-1]
        if idx >= len(adj[cur]):                            # 当前节点邻居遍历完，回溯
            stack.pop()
            continue
        stack[-1] = (cur, par, cur_cum, idx + 1)            # 推进邻居下标
        n2, w2, _ = adj[cur][idx]
        if n2 == par:                                       # 树边反向，跳过
            continue
        if n2 == start:                                     # 回到起点 start：构成经过 start 的环
            if cur_cum + w2 > min_len:                      # 环长=树路径 start..cur + 回边
                chain = []
                u = cur
                while u != n2:                              # 沿父链回溯出树路径 start..cur
                    chain.append(u)
                    u = parent[u]
                chain.append(n2)                            # 补上 start
                chain.reverse()
                pixels = []
                for k in range(len(chain)):                 # 还原环像素：节点 + 边中间点
                    a = chain[k]
                    b = chain[(k + 1) % len(chain)]
                    pixels.append(nodes[a])
                    for (nb, _w, mid) in adj[a]:
                        if nb == b:
                            pixels.extend(mid)
                            break
                return pixels
            continue
        if n2 in visited:                                   # 回边到非根祖先：环不经过 start，跳过
            continue
        visited.add(n2)
        parent[n2] = cur
        stack.append((n2, cur, cur_cum + w2, 0))            # 进入子节点
    return None


def _prune_and_simplify(skeleton, min_len=MIN_LOOP_LEN):
    """功能：去毛刺——①找所有分支点；②两分支点间多条线段只保留最短一条；③无分支点直接返回原图；
    ④依次以每个分支点为起点沿骨架走回自身找闭合环，环长>min_len 则只保留该环、删除其余，否则换下一个分支点。
    输入：skeleton：np.ndarray，uint8，0/255 单像素细化骨架
          min_len：int，闭合环长度阈值(默认取顶部 MIN_LOOP_LEN)
    输出：np.ndarray，uint8，0/255、与输入同尺寸，只保留满足阈值的大环骨架
    """
    skel = (skeleton > 0).astype(np.uint8)
    if not skel.any():
        return skeleton
    deg = _degree(skel)
    if not (deg >= 3).any():                                # 无分支点，不处理
        return skeleton
    nodes, adj, branch_idx = _build_graph(skel)             # 建图 + 并行边去重(保留最短)
    for i in branch_idx:                                    # 依次以每个分支点为起点找环
        cycle = _find_cycle(nodes, adj, i, min_len)         # 返回第一个环长>min_len 的环
        if cycle is not None:
            out = np.zeros_like(skel)
            for (py, px) in cycle:
                out[py, px] = 1
            return (out * 255).astype(np.uint8)
    return skeleton                                         # 无满足阈值的大环，原样返回


def _longest_path(skeleton):
    """功能：取细化骨架中的最长无分支路径作为主干，删除其余分支/毛刺。适用于开放曲线(有端点、无闭合环)。
    输入：skeleton：np.ndarray，uint8，0/255 单像素细化骨架
    输出：np.ndarray，uint8，0/255、与输入同尺寸，只保留最长路径骨架
    """
    skel = (skeleton > 0).astype(np.uint8)
    if not skel.any():
        return skeleton
    deg = _degree(skel)
    if not (deg >= 3).any():                        # 无分支点：本身已是单条路径(或闭合环)，原样返回
        return skeleton
    nodes, adj, _ = _build_graph(skel)              # 建图：节点=端点/分支点，边权=像素长度

    def _farthest(src):
        """从 src 出发沿图走到最远节点，返回(最远节点, 距离, 父指针)。开放骨架无环即树，DFS 正确。"""
        dist = {src: 0}
        parent = {src: -1}
        stack = [src]
        while stack:
            u = stack.pop()
            for (v, w, _) in adj[u]:
                if v not in dist:                   # 树结构：只走未访问节点，避免回走
                    dist[v] = dist[u] + w
                    parent[v] = u
                    stack.append(v)
        far = max(dist, key=lambda k: dist[k])
        return far, dist[far], parent

    u, _, _ = _farthest(0)                          # 任取起点找最远端点 u
    v, _, parent = _farthest(u)                     # 从 u 再找最远端点 v，u-v 即树直径(最长路径)
    chain = [v]
    while chain[-1] != u:                           # 沿父链回溯出 u->v 的节点序列
        chain.append(parent[chain[-1]])
    chain.reverse()

    out = np.zeros_like(skel)
    for k in range(len(chain) - 1):                 # 还原最长路径像素：节点 + 边中间点
        a, b = chain[k], chain[k + 1]
        out[nodes[a][0], nodes[a][1]] = 1
        for (nb, _w, mid) in adj[a]:
            if nb == b:
                for (py, px) in mid:
                    out[py, px] = 1
                break
    out[nodes[chain[-1]][0], nodes[chain[-1]][1]] = 1   # 补上最后一个端点
    return (out * 255).astype(np.uint8)


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


def _gray_centroid_subpixel(pruned, blur, k=4, half=15):
    """功能：对骨架 pruned 上每个前景点，沿其法线方向采样 blur 灰度，用灰度差分定位光条两边界，
    在边界间做灰度质心，得到亚像素中心坐标。
    输入：pruned：np.ndarray，uint8，0/255 骨架图；blur：np.ndarray，uint8 灰度图（与 pruned 同尺寸）；
          k：int，差分跨度(px)；half：int，法线采样半径(px)
    输出：(M,2) float64，每行 [y,x] 亚像素坐标；无有效点时返回 (0,2) 空数组
    """
    h, w = blur.shape
    pts = []                                              # 累积亚像素坐标 [y,x]
    ts = np.arange(-half, half + 1)                       # 法线采样偏移 t，中心下标 half
    for seg, closed in _ordered_paths(pruned):            # 复用有序路径，拿到前后相邻像素
        p = len(seg)
        if p < 5:                                         # 段太短凑不满前后各 2 点
            continue
        for i in range(p):
            if closed:                                    # 闭合环：环形索引取前后各 2 点 + 当前点
                win = seg[np.arange(i - 2, i + 3) % p].astype(np.float64)
            else:                                         # 开放段：两端凑不满 2 点则跳过
                if i < 2 or i > p - 3:
                    continue
                win = seg[i - 2:i + 3].astype(np.float64)
            theta = _fit_direction(win)                   # 5 点窗口的主方向角
            nx, ny = -np.sin(theta), np.cos(theta)        # 法线单位向量
            cx, cy = win[2]                               # 当前中心点 (x,y)
            xs = np.clip(np.round(cx + ts * nx), 0, w - 1).astype(int)  # 沿法线 ±half 的列坐标
            ys = np.clip(np.round(cy + ts * ny), 0, h - 1).astype(int)  # 沿法线 ±half 的行坐标
            g = blur[ys, xs].astype(np.float64)           # (2*half+1,) 采样灰度
            # 左边界：从 -half 向中心(+t)扫描，差分后续 k 点，取 |差分求和| 最大处
            t_left, best = None, -1.0
            for t in range(-half, -k + 1):
                s = t + half
                v = abs(sum(g[s] - g[s + j] for j in range(1, k + 1)))
                if v > best:
                    best, t_left = v, t
            # 右边界：从 +half 向中心(-t)扫描，差分前序 k 点，取 |差分求和| 最大处
            t_right, best = None, -1.0
            for t in range(half, k - 1, -1):
                s = t + half
                v = abs(sum(g[s] - g[s - j] for j in range(1, k + 1)))
                if v > best:
                    best, t_right = v, t
            if t_left is None or t_right is None or t_left >= t_right:
                continue
            m = np.arange(t_left, t_right + 1) + half     # 边界间采样下标
            gv = g[m]                                     # 边界间灰度
            total = gv.sum()
            if total == 0:                                # 全黑无有效灰度，跳过
                continue
            t_center = float((gv * np.arange(t_left, t_right + 1)).sum() / total)  # 一维灰度质心
            pts.append([cy + t_center * ny, cx + t_center * nx])
    if not pts:
        return np.empty((0, 2), np.float64)
    return np.array(pts, np.float64)


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
    """功能：批量处理文件夹内的 png，返回每张图的最大聚类骨架点集。
    输入：path：str，图片文件夹路径
    输出：list，长度与 png 数量一致；每项是 (N,2) int 的 [y, x] 点集，读取失败的图为 None
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)                   # 确保调试输出文件夹存在
    def _file_index(f):                                     # 从文件名提取数字作排序键；无数字时返回 0 兜底，避免 re.search 返回 None 报错
        m = re.search(r'\d+', os.path.basename(f))
        return int(m.group()) if m else 0
    files = sorted(glob.glob(os.path.join(path, "*.png")), key=_file_index)  # 按文件名数字自然排序，避免 img10 排在 img2 前
    calib = np.load(CALIB_NPZ)                              # 只读加载标定结果，避免每张图重复 IO
    K = calib['K']                                          # (3,3) 内参矩阵
    D = calib['D']  
    points = []                                             # 每张图一个点集，与 files 顺序一致
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_SIZE, DILATE_SIZE))  # 膨胀结构元，循环外算一次
    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"跳过无法读取的图片：{file_path}")
            points.append(None)                             # 占位，保持与 files/z 对齐
            continue
        img = img[:, :, 2]                                  # 取红通道得到单通道灰度图
        img = _undistort_fisheye(img, K, D, BALANCE)        # 鱼眼去畸变并裁黑边
        blur = cv2.GaussianBlur(img,(5,5),0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = _remove_small(binary)
        skeleton = _thin(binary)                            # 未膨胀图细化，后续处理都用它
        cv2.imwrite(os.path.join(DEBUG_DIR, os.path.basename(file_path)), skeleton)

        # 只用膨胀图判断骨架闭合/开放（补小缺口后判断更准），后续处理仍用未膨胀 skeleton
        dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
        d_deg = _degree(_thin(dilated))
        is_closed = (int((d_deg == 1).sum()) == 0)          # 膨胀后无端点=闭合环；有端点=开放曲线

        if is_closed:
            pruned = _prune_and_simplify(skeleton)          # 闭合环：去毛刺找主环
        else:
            pruned = _longest_path(skeleton)                # 开放曲线：取最长无分支路径
        cv2.imwrite(os.path.join(DEBUG_DIR, os.path.basename(file_path)), pruned)  # 每张剪枝图按原文件名依次存到 _debug
        points.append(_gray_centroid_subpixel(pruned, blur))  # 每张图 append 一次亚像素点，与 files 顺序一致
    return points

if __name__ == "__main__":
    image_processing()
