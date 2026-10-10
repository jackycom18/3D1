"""结构光图像处理：从多张 png 中提取每段光线主干上的拐点。

流程：读图 → 灰度 → 自适应二值化 → 细化 → 最长路径去毛刺(只留主干) → 按拟合直线方向变化挑拐点 → npz 保存
运行：python 图像处理.py
"""

import cv2
import numpy as np
import glob
import os
import re

# ---------- 可调参数（数值按需手改） ----------

IMAGE_DIR = r"C:\python\3D_project_3\images\images"           # 输入图片文件夹
CALIB_NPZ = r"C:\python\3D_project_3\calib_result1.npz"  # 鱼眼标定结果(K/D)保存路径
BALANCE = 0.0                # 去畸变保留比例：0 完整保留(黑边多)，1 尽量填满
ADAPTIVE_BLOCK = 101           # 自适应二值化窗口(px，奇数)，量级取光条间距
ADAPTIVE_C = -15               # 自适应二值化常数，阈值=局部均值-C，负值更严格
SPUR_MIN_AREA = 100            # 骨架连通域最小面积(px)，小于此值整段丢弃
MIN_DIST = 50              # 拐点最小间距(px)：合并过近顶点时，两个顶点欧氏距离小于它视为同一角
kernel = np.array([3,3])
_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 8 邻域行列偏移
MIN_LOOP_LEN = 700              # 闭合环长度阈值(px)：环长超过它才判定为真实结构并保留，否则换下一个分支点
STEGER_SIGMA = 2.5              # Steger 高斯核尺度，用于直边亚像素脊线提取


def _remove_small(binary):
    """功能：只保留面积最大的那个前景连通域，其余全部清 0。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255，只保留最大连通域
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].copy()
    area[0] = -1                                        # 背景面积置负，不参与最大值比较
    keep = int(np.argmax(area))                         # 面积最大的连通域标签
    return np.where(labels == keep, 255, 0).astype(np.uint8)

def _thin(binary):
    """功能：用 Zhang-Suen 算法把二值图细化成 1 像素宽的骨架。
    输入：binary：np.ndarray，uint8，0/255 二值图
    输出：np.ndarray，uint8，0/255 骨架图（与输入同尺寸）
    """
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return np.zeros_like(binary)
    try:
        from skimage.morphology import skeletonize
        return (skeletonize(binary > 0, method='zhang') * 255).astype(np.uint8)
    except Exception:
        pass
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
                    edge_map.setdefault(key, []).append((1, [], i))
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
                    edge_map.setdefault(key, []).append((len(path) + 1, path, i))   # 记录原始走向起点 i

    adj = {i: [] for i in range(len(nodes))}
    for (a, b), segs in edge_map.items():
        segs.sort(key=lambda s: s[0])                       # 并行边按长度升序，只保留最短
        length, pixels, origin = segs[0]
        # 严格保持边的方向性：反向走时必须将中间点倒序 pixels[::-1]，杜绝空间折返跳跃
        if origin == a:
            adj[a].append((b, length, pixels))
            adj[b].append((a, length, pixels[::-1]))
        else:
            adj[a].append((b, length, pixels[::-1]))
            adj[b].append((a, length, pixels))
    return nodes, adj, branch_idx


def _clean_to_single_pixel_cycle(cycle_pts):
    """功能：消除闭合环上的对角冗余短路点，使环在栅格图像上处处度数严格为 2，成为真正的单像素环。
    输入：cycle_pts：(P, 2) int32 闭环点序列
    输出：(M, 2) int32 严格单像素宽的闭环点序列
    """
    if cycle_pts is None or len(cycle_pts) < 4:
        return cycle_pts
    pts = [tuple(p) for p in cycle_pts]
    changed = True
    while changed:
        changed = False
        new_pts = []
        n = len(pts)
        skip = set()
        for i in range(n):
            if (i - 1) % n in skip:
                new_pts.append(pts[i])
                continue
            p_prev = pts[(i - 1) % n]
            p_cur = pts[i]
            p_next = pts[(i + 1) % n]
            # 若前驱与后继在 8 邻域已直接相连，说明当前点属于对角冗余短路转折点，可安全剔除
            if max(abs(p_prev[0] - p_next[0]), abs(p_prev[1] - p_next[1])) <= 1:
                skip.add(i)
                changed = True
            else:
                new_pts.append(p_cur)
        pts = new_pts
    return np.array(pts, np.int32)


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


def _extract_main_cycle(skeleton, min_len=MIN_LOOP_LEN):
    """功能：从细化骨架中提取主闭合环，返回环上按行走顺序排列的像素序列。
    有分支点时，依次以每个分支点为起点在压缩图上找环长>min_len 的第一个环；无分支点时沿环直接走一圈。
    输入：skeleton：np.ndarray，uint8，0/255 单像素细化骨架
          min_len：int，闭合环长度阈值(默认取顶部 MIN_LOOP_LEN)
    输出：(P, 2) int32，环上像素按行走顺序排列(行, 列)；找不到满足阈值的大环时返回 None
    """
    skel = (skeleton > 0).astype(np.uint8)
    if not skel.any():
        return None
    deg = _degree(skel)
    if not (deg >= 3).any():                                # 无分支点：本身已是纯闭合环
        tr = _trace_cycle(skel > 0)
        return _clean_to_single_pixel_cycle(tr) if tr is not None else None
    nodes, adj, branch_idx = _build_graph(skel)             # 建图 + 并行边去重(保留最短)
    for i in branch_idx:                                    # 依次以每个分支点为起点找环
        cycle = _find_cycle(nodes, adj, i, min_len)         # 返回第一个环长>min_len 的环
        if cycle is not None:
            return _clean_to_single_pixel_cycle(np.array(cycle, np.int32))
    return None                                             # 无满足阈值的大环


def _trace_cycle(mask):
    """功能：在单个闭合环骨架（单像素宽、每点度数=2、无端点）上，任取一点作起点，沿 8 邻域按统一方向走一圈，返回有序点列。
    输入：mask：np.ndarray，bool，单个闭合环骨架掩膜（已裁剪到包围盒）
    输出：(P, 2) int32，环上像素按行走顺序排列（行, 列），长度=环像素数；无法走成闭合环时返回 None
    """
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return None
    start = tuple(pts[0])                                    # 任取一点作起点
    cycle = [start]
    prev = None
    cur = start
    while True:
        nxt = None
        for dr, dc in _NEIGHBORS8:                           # 找 cur 的 8 邻域前景点（跳过刚来的 prev）
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (prev is None or (nr, nc) != prev):
                nxt = (nr, nc)
                break
        if nxt is None:                                      # 走不通，说明不是闭合环
            return None
        if nxt == start:                                     # 回到起点，闭环完成
            break
        cycle.append(nxt)
        prev, cur = cur, nxt
    return np.array(cycle, np.int32)


def _corner_points_from_cycle(cycle, gray=None, min_dist=MIN_DIST, trim_ratio=0.15, sigma=STEGER_SIGMA):
    """功能：用直线边缘拟合求解析交点法提取 4 拐点，彻底消灭细化的倒角内缩效应。
          1. 粗定位：利用 approxPolyDP 将闭合环点序列粗略分割为 4 条直边；
          2. 剔除过渡区：每条边丢弃两头靠近拐角的过渡点（默认 15%），只保留中间平直段；
          3. 数据源升级：在直边上计算纯 Steger 亚像素脊线点（过滤非凸、漂移或暗点，不合入灰度质心）；
          4. 鲁棒拟合：对 4 条边的高精度亚像素点分别拟合解析直线方程；
          5. 解析求交：联立相邻两直线求解精确交点，向外还原真实几何顶点。
    输入：cycle：(P, 2) int32，环上像素(行, 列)；gray：去畸变灰度原图（提供时启动纯 Steger 亚像素）；
          min_dist：顶点最小间距(px)；trim_ratio：float，每条边两头剔除的比例（默认 15%）；
          sigma：float，Steger 高斯核尺度
    输出：tuple，元素为 (x, y) 拐点坐标（float 亚像素）；无法稳定得到 4 顶点时安全回退
    """
    if cycle is None or len(cycle) < 40:
        return ()

    pts_xy = cycle[:, ::-1].astype(np.float64)  # (P, 2) 的 [x, y]
    peri = cv2.arcLength(pts_xy.astype(np.float32), True)

    # 1. 尝试用 approxPolyDP 粗略找到 4 个顶点，作为 4 条直边的分界点
    approx = None
    for ratio in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08):
        app = cv2.approxPolyDP(pts_xy.astype(np.float32), ratio * peri, True).reshape(-1, 2)
        if len(app) == 4:
            approx = app
            break

    # 2. 如果能够成功粗分成 4 条边，则执行高精度直线拟合求解析交点
    if approx is not None:
        indices = sorted([int(np.argmin(np.sum((pts_xy - apt) ** 2, axis=1))) for apt in approx])

        # 若提供了灰度图，预先计算高斯一阶与二阶偏导数图
        Ix, Iy, Ixx, Iyy, Ixy, roi_gray = None, None, None, None, None, None
        if gray is not None:
            radius = int(round(3 * sigma))
            x_axis = np.arange(-radius, radius + 1, dtype=np.float64)
            g0 = np.exp(-0.5 * (x_axis / sigma) ** 2)
            g0 /= g0.sum()
            g1 = -x_axis / (sigma ** 2) * g0
            g2 = (x_axis ** 2 - sigma ** 2) / (sigma ** 4) * g0

            roi_gray = gray.astype(np.float64)
            Ix = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g1, g0)
            Iy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g0, g1)
            Ixx = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g2, g0)
            Iyy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g0, g2)
            Ixy = cv2.sepFilter2D(roi_gray, cv2.CV_64F, g1, g1)

        lines = []
        valid_lines = True
        h, w = (gray.shape[:2]) if gray is not None else (0, 0)

        for k in range(4):
            i_start = indices[k]
            i_end = indices[(k + 1) % 4]
            if i_end > i_start:
                edge = pts_xy[i_start:i_end]
            else:
                edge = np.vstack([pts_xy[i_start:], pts_xy[:i_end]])

            n_pts = len(edge)
            if n_pts < 15:
                valid_lines = False
                break

            trim = int(n_pts * trim_ratio)
            straight_segment = edge[trim : n_pts - trim]
            if len(straight_segment) < 10:
                straight_segment = edge

            pts_to_fit = straight_segment
            # 纯 Steger 亚像素提取（不使用灰度质心回退，不符合条件的点直接丢弃）
            if gray is not None:
                steger_pts = []
                for pt in straight_segment:
                    cx, cy = int(round(pt[0])), int(round(pt[1]))
                    if not (0 <= cy < h and 0 <= cx < w):
                        continue
                    ix = Ix[cy, cx]
                    iy = Iy[cy, cx]
                    ixx = Ixx[cy, cx]
                    iyy = Iyy[cy, cx]
                    ixy = Ixy[cy, cx]

                    trace = ixx + iyy
                    diff = ixx - iyy
                    disc = np.sqrt(diff * diff + 4.0 * ixy * ixy)
                    lambda1 = 0.5 * (trace - disc)
                    if lambda1 >= -0.05:  # 必须是凸脊线山脊点
                        continue

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

                    t = -(ix * nx + iy * ny) / lambda1
                    if abs(t) <= 2.5 and roi_gray[cy, cx] >= 20:
                        steger_pts.append([cx + t * nx, cy + t * ny])

                if len(steger_pts) >= 15:
                    pts_to_fit = np.array(steger_pts, dtype=np.float64)

            # M-estimator 鲁棒直线拟合 (vx, vy, x0, y0)
            line = cv2.fitLine(pts_to_fit.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            lines.append(line)

        if valid_lines and len(lines) == 4:
            corners = []
            for k in range(4):
                vx1, vy1, x1, y1 = lines[k]
                vx2, vy2, x2, y2 = lines[(k + 1) % 4]

                # 直线方程：-vy * x + vx * y = -vy * x0 + vx * y0
                A = np.array([[-vy1, vx1], [-vy2, vx2]], dtype=np.float64)
                B = np.array([-vy1 * x1 + vx1 * y1, -vy2 * x2 + vx2 * y2], dtype=np.float64)

                if abs(np.linalg.det(A)) < 1e-5:
                    corners = []
                    break
                pt = np.linalg.solve(A, B)
                corners.append((float(pt[0]), float(pt[1])))

            if len(corners) == 4:
                return tuple(corners)

    # 3. 安全回退：最小外接矩形兜底
    rect = cv2.minAreaRect(pts_xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    return tuple((float(x), float(y)) for x, y in box)


def _order_points(points):
    """功能：把四个拐点按 左上→右上→右下→左下 排序。
    输入：points：可迭代的 (x, y) 像素坐标，应为 4 个点
    输出：list，[(x, y), ...] 按左上/右上/右下/左下顺序；点数不足 4 个时按原顺序返回
    """
    pts = list(points)
    if len(pts) != 4:
        return pts
    arr = np.asarray(pts, np.float64)                    # (4,2) 像素坐标
    c = arr.mean(axis=0)                                 # 质心：旋转/平移不变的参考点，替代原图像原点(0,0)
    s = (arr[:, 0] - c[0]) + (arr[:, 1] - c[1])          # 相对质心的 x+y：图像坐标下左上极小、右下极大
    tl = int(np.argmin(s))                               # 左上：相对质心最偏左上
    br = int(np.argmax(s))                               # 右下：相对质心最偏右下
    rest = [i for i in range(4) if i not in (tl, br)]    # 剩下两个顶点的下标
    i0, i1 = rest
    tr, bl = (i0, i1) if arr[i0, 0] >= arr[i1, 0] else (i1, i0)  # x 较大的是右上，另一个是左下
    order = [tl, tr, br, bl]
    return [(float(arr[i, 0]), float(arr[i, 1])) for i in order]


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


def _prepare_undistort(K, D, size, balance=0.0):
    """功能：预先计算鱼眼去畸变的 remap 表与有效区包围盒，供多张同尺寸图复用，避免每张图重复算 map。
    输入：K：(3,3) float64 内参；D：(4,1) 畸变系数；size：(w, h) 图像宽高；balance：float 去畸变保留比例
    输出：tuple(map1, map2, y0, y1, x0, x1)，map1/map2 为重映射表，y0..x1 为有效像素区包围盒
    """
    w, h = size                                                # Size 需按 (宽,高) 传入
    R = np.eye(3, dtype=np.float64)                            # 单位旋转，不额外旋转图像
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R, balance=balance)                      # 缩小焦距后的新内参，广角内容完整装入画布
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, Knew, (w, h), cv2.CV_32FC1)                   # 重映射表，float 坐标便于后续判有效区
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map1, map2, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 全白图过一遍，得到有效区掩膜
    ys, xs = np.nonzero(mask)                                  # 有效像素坐标，黑边处 mask 为 0
    if ys.size == 0:                                           # 理论不触发，防御性返回全图包围盒
        return map1, map2, 0, h, 0, w
    y0, y1 = ys.min(), ys.max() + 1                            # 有效区域包围盒
    x0, x1 = xs.min(), xs.max() + 1
    return map1, map2, y0, y1, x0, x1


def _undistort_fisheye(img, K, D, balance=0.0):
    """功能：对单通道灰度图做鱼眼去畸变并裁剪黑边，完整保留广角内容（单图便捷封装，内部复用 _prepare_undistort）。
    输入：img：np.ndarray，uint8 单通道灰度图；K：(3,3) float64 内参；D：(4,1) 畸变系数；
          balance：float，0 完整保留(黑边多)，1 尽量填满(会裁掉部分内容)
    输出：np.ndarray，uint8，去畸变并裁掉黑边后的灰度图，可直接用于后续处理
    """
    h, w = img.shape[:2]                                       # 高、宽，Size 需按 (宽,高) 传入
    map1, map2, y0, y1, x0, x1 = _prepare_undistort(K, D, (w, h), balance)
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 黑边区域填 0
    return undist[y0:y1, x0:x1]


def image_processing(path=IMAGE_DIR, calib_npz=CALIB_NPZ):
    """功能：批量处理文件夹内的 png，先做鱼眼去畸变，再提取每张图的四个拐点，按图片顺序存成 npz。
    输入：path：str，图片文件夹路径；calib_npz：str，鱼眼标定结果(K/D)路径
    输出：list，每个元素是该图四个拐点的 ((x, y), ...) 二维元组
    """
    def _file_index(f):                                     # 从文件名提取数字作排序键；无数字返回 0 兜底，避免 re.search 返回 None 报错
        m = re.search(r'\d+', os.path.basename(f))
        return int(m.group()) if m else 0
    files = sorted(glob.glob(os.path.join(path, "*.png")), key=_file_index)  # 按文件名数字自然排序，避免 img10 排在 img2 前
    calib = np.load(calib_npz)                              # 只读加载一次标定结果，避免每张图重复 IO
    K = calib["K"]                                          # (3,3) 内参矩阵
    D = calib["D"]                                          # (4,1) 畸变系数
    probe_size = None                                       # 用第一张可读图的尺寸准备去畸变表（所有图尺寸一致）
    for file_path in files:
        probe = cv2.imread(file_path)
        if probe is not None:
            probe_size = (probe.shape[1], probe.shape[0])   # (宽, 高)
            break
    if probe_size is None:
        return []                                           # 没有可读图，直接返回空结果
    map1, map2, y0, y1, x0, x1 = _prepare_undistort(K, D, probe_size, BALANCE)  # 循环外只算一次 map 与有效区包围盒，后续每张只 remap
    result = []
    count = 1
    for file_path in files:
        img = cv2.imread(file_path)                          # 读 BGR 彩色图，只取红色通道作灰度
        if img is None:
            print(f"跳过无法读取的图片：{file_path}")
            continue
        gray = img[:, :, 2]                                  # 红色通道(BGR 第 3 通道)
        gray = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)[y0:y1, x0:x1]  # 去畸变并裁黑边
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        binary = _remove_small(binary)
        skeleton = _thin(binary)
        cycle = _extract_main_cycle(skeleton)               # 提取主闭合环的有序像素序列
        points = _order_points(_corner_points_from_cycle(cycle, gray=gray))  # 沿环经 Steger 亚像素直线拟合挑拐点
        result.append(tuple(points))
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x,y) in points:
            cv2.circle(gray, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        keep_path= r"C:\python\3D_project_3\_debug"
        
        path = os.path.join(keep_path,f"img_{count}.png")
        count += 1
        cv2.imwrite(path, gray)
        print(f"{os.path.basename(file_path)}：拐点 {len(points)} 个")
    return result


if __name__ == "__main__":
    image_processing()
