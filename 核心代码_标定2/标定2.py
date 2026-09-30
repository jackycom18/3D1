"""标定2：批量处理结构光照片，提取每张图 4 个拐点(像素)，与位移台 xy.npz 的现实坐标一一对应，求像素->现实单应矩阵 H。

流程：
1. 复用 图像处理_四点提取.py 批量提取文件夹内每张图的 4 个拐点(像素坐标，左上→右上→右下→左下)
2. 读取位移台 xy.npz(字段 points，形状 (N,4,2)) 的现实坐标，第 i 张图的 4 拐点与第 i 组一一对应
3. 所有拐点像素坐标与对应现实坐标做 findHomography，得到像素->现实(mm) 单应矩阵 H
4. H 存入 npz
运行：python 标定2.py
"""

import importlib.util
import os

import cv2
import numpy as np

# ---------- 路径与参数（数值按需手改） ----------
_IMGPROC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "图像处理_四点提取.py")  # 复用的图像处理模块(四点提取)
IMAGE_DIR = r"C:\python\3D_project_3\images\images\images"                    # 多张待处理标定框照片文件夹
CALIB_NPZ = r"C:\python\3D_project_3\calib_result1.npz"          # 鱼眼标定结果(K/D)
XY_NPZ = r"C:\python\3D_project_3\xy.npz"                       # 位移台 xy 坐标(字段 points，(N,4,2))，按需改
OUTPUT_NPZ = r"C:\python\3D_project_3\核心代码_标定2\H.npz"     # 单应矩阵 H 保存路径


def _load_imgproc(imgproc_path=_IMGPROC_PATH):
    """功能：用 importlib 加载中文文件名的 图像处理_四点提取.py，复用其处理函数。
    输入：imgproc_path：str，图像处理模块文件路径
    输出：module，图像处理模块；找不到文件时返回 None
    """
    if not os.path.isfile(imgproc_path):
        print(f"找不到图像处理模块：{imgproc_path}")
        return None
    spec = importlib.util.spec_from_file_location("imgproc", imgproc_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_xy(xy_npz):
    """功能：读取位移台 xy.npz 的现实坐标。
    输入：xy_npz：str，npz 文件路径
    输出：(N,4,2) float64 数组，每行是一组四顶点现实坐标(左上→右上→右下→左下)；
          缺少 points 字段或形状不对时返回 None
    """
    data = np.load(xy_npz)
    if "points" not in data.files:
        print(f"xy.npz 缺少 points 字段，现有字段：{data.files}")
        return None
    real = np.asarray(data["points"], np.float64)             # (N,4,2)
    if real.ndim != 3 or real.shape[1:] != (4, 2):
        print(f"xy.npz 的 points 形状应为 (N,4,2)，实际 {real.shape}")
        return None
    return real


def calibrate2(image_dir=IMAGE_DIR, xy_npz=XY_NPZ, calib_npz=CALIB_NPZ,
               out_npz=OUTPUT_NPZ, imgproc_path=_IMGPROC_PATH):
    """功能：批量提取文件夹内每张图的 4 拐点(像素)，与 xy.npz 的 (N,4,2) 现实坐标一一对应，求单应矩阵 H 并存 npz。
    输入：image_dir：str，图片文件夹路径；xy_npz：str，位移台 xy 坐标 npz(字段 points，(N,4,2))；
          calib_npz：str，鱼眼标定结果(K/D)路径；out_npz：str，H 保存路径；
          imgproc_path：str，复用的图像处理模块(四点提取)路径
    输出：(3,3) float64 单应矩阵 H；模块/图片/坐标缺失或有效点不足时返回 None
    """
    imgproc = _load_imgproc(imgproc_path)
    if imgproc is None:
        return None

    # 批量提取每张图 4 拐点(像素)，返回 list，每项是 4 个 (x,y)，顺序 左上→右上→右下→左下
    quads_pix = imgproc.image_processing(image_dir, calib_npz=calib_npz)

    real = _load_xy(xy_npz)                                    # (N,4,2) 现实坐标(mm)
    if real is None:
        return None

    if len(quads_pix) != real.shape[0]:
        print(f"图片数({len(quads_pix)}) 与 xy.npz 组数({real.shape[0]}) 不一致，按较少数对应")

    pix_all = []                                               # 所有拐点像素坐标 (x,y)
    real_all = []                                              # 对应现实坐标 (x,y)
    n = min(len(quads_pix), real.shape[0])
    for i in range(n):
        quad = quads_pix[i]                                    # 本图 4 个拐点(像素)
        if len(quad) != 4:                                     # 拐点不足 4 个时跳过，避免与 xy.npz 错位
            print(f"第 {i} 张图只提取到 {len(quad)} 个拐点，跳过")
            continue
        pix_all.extend(quad)
        real_all.extend(real[i])

    if len(pix_all) < 4:                                       # 单应矩阵至少需要 4 个对应点
        print("有效对应点不足 4 个，无法求单应矩阵 H")
        return None
    pix_all = np.asarray(pix_all, np.float64)                  # (P,2)
    real_all = np.asarray(real_all, np.float64)                # (P,2)

    H, _ = cv2.findHomography(pix_all, real_all, method=0)     # 最小二乘，用全部点
    if H is None:
        print("findHomography 失败，未保存单应矩阵 H")
        return None
    np.savez_compressed(out_npz, H=H)
    print(f"已保存单应矩阵 H(像素->现实)到 {out_npz}，参与求解的点数 {len(pix_all)}")
    return H


def main():
    """功能：脚本入口，用默认路径常量跑一次标定2 流程。
    输入：无
    输出：无
    """
    calibrate2()


if __name__ == "__main__":
    main()
