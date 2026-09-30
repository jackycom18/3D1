"""对比测试：对比原版灰度质心法 vs 新版 Steger 算法的耗时、精度与光滑度。
运行：py -3.8 对比测试.py
"""

import os
import time
import cv2
import numpy as np
import importlib.util
from skimage.morphology import skeletonize

# 加载原模块与新模块
p_orig = r"C:\python\3D_project_3\核心代码_标定2\图像处理_重建.py"
p_new = r"C:\python\3D_project_3\核心代码_标定2\图像处理_重建1.py"

spec1 = importlib.util.spec_from_file_location("orig", p_orig)
mod_orig = importlib.util.module_from_spec(spec1)
# 原版的细化用快速细化替代以便公平对比质心提取本身的耗时
mod_orig._thin = lambda b: (skeletonize(b > 0) * 255).astype(np.uint8)
spec1.loader.exec_module(mod_orig)

spec2 = importlib.util.spec_from_file_location("new_steger", p_new)
mod_new = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod_new)

CALIB_NPZ = r"C:\python\3D_project_3\calib_result1.npz"
IMG_PATH = r"C:\python\3D_project_3\takephoto\images\img10.png"
DEBUG_DIR = r"C:\python\3D_project_3\_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

calib = np.load(CALIB_NPZ)
K, D = calib['K'], calib['D']

print(f"正在对测试图进行对比：{os.path.basename(IMG_PATH)} ...")
img = cv2.imread(IMG_PATH)
gray = img[:, :, 2]
undist = mod_new._undistort_fisheye(gray, K, D, 0.0)
blur = cv2.GaussianBlur(undist, (5, 5), 0)
binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -15)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
binary = mod_new._remove_small(binary)
skeleton = mod_new._fast_thin(binary)
pruned = mod_new._prune_and_simplify(skeleton)

# 1. 测试原版灰度质心法
t0 = time.time()
pts_centroid = mod_orig._gray_centroid_subpixel(pruned, blur)
t_centroid = (time.time() - t0) * 1000

# 2. 测试新版 Steger 算法
t0 = time.time()
pts_steger = mod_new._steger_subpixel(pruned, undist, sigma=2.5)
t_steger = (time.time() - t0) * 1000

# 3. 计算光滑度指标（相邻三点二阶差分抖动）
def calc_roughness(pts):
    if len(pts) < 100:
        return 0.0
    p = pts[400:1000] # 取中间连续 600 个点
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.mean(np.hypot(d2[:, 0], d2[:, 1])))

r_centroid = calc_roughness(pts_centroid)
r_steger = calc_roughness(pts_steger)

print("\n" + "="*50)
print(f"【对比结果报告】")
print(f"1. 提取点数：")
print(f"   - 原版灰度质心法：{len(pts_centroid)} 个点")
print(f"   - 新版 Steger 算法：{len(pts_steger)} 个点")
print(f"\n2. 亚像素计算耗时：")
print(f"   - 原版灰度质心法：{t_centroid:.1f} ms")
print(f"   - 新版 Steger 算法：{t_steger:.1f} ms  (提速约 {t_centroid/max(1e-3, t_steger):.1f} 倍)")
print(f"\n3. 局部阶梯抖动与粗糙度 (越小越光滑)：")
print(f"   - 原版灰度质心法：{r_centroid:.4f}")
print(f"   - 新版 Steger 算法：{r_steger:.4f}  (光滑度提升约 {(1 - r_steger/r_centroid)*100:.1f}%)")
print("="*50)

# 4. 生成局部高倍放大对比图 (5x 放大)
ys, xs = np.nonzero(pruned)
yc, xc = int(np.mean(ys)), int(np.min(xs)) # 左侧立壁
zy0, zy1 = yc - 60, yc + 60
zx0, zx1 = xc - 30, xc + 30

patch = undist[zy0:zy1, zx0:zx1]
scale = 6
vis = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# 画原版质心（蓝色）
for y, x in pts_centroid:
    if zy0 <= y < zy1 and zx0 <= x < zx1:
        px = int(round((x - zx0) * scale))
        py = int(round((y - zy0) * scale))
        cv2.circle(vis, (px, py), 2, (255, 120, 0), -1)

# 画 Steger 点（绿色）
for y, x in pts_steger:
    if zy0 <= y < zy1 and zx0 <= x < zx1:
        px = int(round((x - zx0) * scale))
        py = int(round((y - zy0) * scale))
        cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)

# 加图例
cv2.putText(vis, "Blue: Centroid", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
cv2.putText(vis, "Green: Steger", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

out_vis = os.path.join(DEBUG_DIR, "steger_vs_centroid.png")
cv2.imwrite(out_vis, vis)
print(f"\n对比图已保存至：{out_vis}")
