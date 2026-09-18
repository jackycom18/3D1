"""临时脚本：定量诊断——每段光条的二值/骨架/去毛刺/填充面积，以及质点是否落在光条内。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\python\3D_project_3\核心代码")
import 图像处理 as ip

file_path = sorted(glob.glob(os.path.join(ip.IMAGE_DIR, "*.png")))[0]
gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ip.TOPHAT_KERNEL)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
binary = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, ip.ADAPTIVE_BLOCK, ip.ADAPTIVE_C)
binary = ip._remove_small(binary, ip.MIN_AREA)
skeleton = ip._thin(binary)
pruned, filled = ip._prune_and_simplify(skeleton)

nb, lb, sb, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
ns, ls, ss, _ = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), 8)
print(f"二值连通域 {nb - 1} 段，骨架连通域 {ns - 1} 段")

print("\n--- 各二值光条段 ---")
for i in range(1, nb):
    x, y, w, h, area = sb[i]
    m = lb[y:y+h, x:x+w] == i
    sk = (skeleton[y:y+h, x:x+w] > 0) & m
    pn = (pruned[y:y+h, x:x+w] > 0) & m
    fl = (filled[y:y+h, x:x+w] > 0) & m
    conv = cv2.filter2D(sk.astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8))
    ends = int(((sk) & (conv == 2)).sum())                 # 端点：仅 1 个邻居
    branch = int(((sk) & (conv >= 4)).sum())               # 分支点：≥3 个邻居
    path = ip._longest_path(sk, np.argwhere(sk & (conv == 2)))
    plen = 0 if path is None else len(path)
    print(f"段{i}: 面积{area:7d}px 包围盒{w}x{h} | 骨架{int(sk.sum()):6d}px "
          f"端点{ends:5d} 分支{branch:6d} 最长路径{plen:6d}px | "
          f"去毛刺{int(pn.sum()):6d}px 填充{int(fl.sum()):6d}px")
    print(f"      骨架/面积={sk.sum()/area:.3f}(→1 表示单像素网)  去毛刺保留={pn.sum()/max(sk.sum(),1):.1%}")

print("\n--- 各质点归属 ---")
for j, (cx, cy) in enumerate(ip._centroids(filled, ip.MIN_AREA)):
    xi, yi = int(round(cx)), int(round(cy))
    inside = bool(binary[yi, xi] > 0)
    seg = int(lb[yi, xi])
    # 到最近二值前景像素的距离
    inv = cv2.distanceTransform((binary == 0).astype(np.uint8), cv2.DIST_L2, 3)
    print(f"质点{j}: ({cx:7.1f},{cy:7.1f}) 落在二值光条内={inside} 所属段={seg} "
          f"距最近光条像素={inv[yi, xi]:.1f}px")
