"""临时脚本：把图像处理各阶段的中间结果导出成 png，便于肉眼检查质点是否落在光条上。"""
import glob
import os
import sys

import cv2

sys.path.insert(0, r"C:\python\3D_project_3\核心代码")
import 图像处理 as ip

OUT = r"C:\python\3D_project_3\_debug"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(ip.IMAGE_DIR, "*.png")))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ip.TOPHAT_KERNEL)

for file_path in files:
    name = os.path.splitext(os.path.basename(file_path))[0]
    gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    binary = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, ip.ADAPTIVE_BLOCK, ip.ADAPTIVE_C)
    binary = ip._remove_small(binary, ip.MIN_AREA)
    skeleton = ip._thin(binary)
    pruned, filled = ip._prune_and_simplify(skeleton)
    cents = ip._centroids(filled, ip.MIN_AREA)

    print(f"=== {name}  尺寸 {gray.shape} ===")
    print(f"  灰度范围 {gray.min()}~{gray.max()}  顶帽最大 {int(tophat.max())}")
    print(f"  二值前景 {int((binary > 0).sum())} px  骨架 {int((skeleton > 0).sum())} px"
          f"  去毛刺后 {int((pruned > 0).sum())} px  填充后 {int((filled > 0).sum())} px")
    for i, (cx, cy) in enumerate(cents):
        print(f"  质点{i}  x={cx:8.2f}  y={cy:8.2f}")

    for tag, img in (("1_灰度", gray), ("2_顶帽", tophat), ("3_二值", binary),
                     ("4_骨架", skeleton), ("5_去毛刺", pruned), ("6_填充", filled)):
        cv2.imwrite(os.path.join(OUT, f"{name}_{tag}.png"), img)

    # 质检图：灰度打底，品红=填充区(半透明) 绿=去毛刺后中心线 蓝=原始骨架 红十字=质点
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = canvas.copy()
    tint[filled > 0] = (255, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.5, tint, 0.5, 0)
    canvas[pruned > 0] = (0, 255, 0)
    canvas[skeleton > 0] = (255, 0, 0)
    for i, (cx, cy) in enumerate(cents):
        p = (int(round(cx)), int(round(cy)))
        cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_CROSS, 41, 3)
        cv2.putText(canvas, str(i), (p[0] + 25, p[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    cv2.imwrite(os.path.join(OUT, f"{name}_7_质检.png"), canvas)

print(f"\n图片已导出到 {OUT}")
