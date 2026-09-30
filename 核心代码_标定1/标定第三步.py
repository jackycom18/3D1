"""标定第 3 步：标定位移台移动方向与光平面坐标系之间的夹角 ang_move。

流程：
1. 读 calib_result.npz 内参 K/D，读 centroids.npz 前 3 组四点交点像素坐标
2. cv2.fisheye.undistortPoints 把每组的 4 个点去畸变映射到 z=1 虚拟平面
3. 第 1、2 组四点按原始排序一一对应，求对应点距离平均值 dis1；第 2、3 组同理得 dis2
4. 读 初始四点坐标.npz 里的 ang1、ang2，结合位移台读数 X1、Y2 解方程
   (Y2 / X1) = (dis2·cos(ang2)) / (dis1·cos(ang1 + ang_move))
   反解出 ang_move
5. 保存 ang_move 到 npz
运行：py -3.8 标定第三步.py
"""

import numpy as np
import cv2

# ---------- 可调参数（数值按需手改） ----------
CALIB_NPZ = r"C:\python\3D_project_3\calib_result.npz"    # 鱼眼标定结果（第 1 步输出）
CENTROIDS_NPZ = r"C:\python\3D_project_3\centroids.npz"   # 多组四点交点像素坐标
INIT_NPZ = r"C:\python\3D_project_3\初始四点坐标.npz"      # 初始四点坐标（第 2 步输出，含 ang1/ang2）
OUT_NPZ = r"C:\python\3D_project_3\ang_move.npz"           # 求解出的夹角保存路径
X1 = 10.0       # 位移台第一次移动距离(mm)，纯 x 方向（y=0），按实际读数修改
Y2 = 10.0       # 位移台第二次移动距离(mm)，纯 y 方向（x=0），按实际读数修改


def _undistort_to_plane(points_px, K, D):
    """功能：把一组四点像素坐标去畸变投影到 z=1 虚拟平面。
    输入：points_px：(4,2) float，像素坐标(uv)；K：(3,3) float，内参矩阵；D：(4,1) float，畸变系数
    输出：(4,2) float，z=1 虚拟平面上的理想归一化坐标 (x,y)
    """
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)   # undistortPoints 要求 (N,1,2)
    norm = cv2.fisheye.undistortPoints(pts, K, D)               # 理想针孔归一化坐标，即 z=1 平面
    return norm.reshape(-1, 2)


def _mean_pair_dist(a, b):
    """功能：两组四点按原始排序一一对应，求对应点欧氏距离的平均值。
    输入：a、b：(4,2) float，两组虚拟平面坐标
    输出：float，四个对应点距离的平均值
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _solve_ang_move(dis1, dis2, ang1, ang2, x1, y2):
    """功能：解方程 (y2/x1) = (dis2·cos ang2)/(dis1·cos(ang1+ang_move)) 求 ang_move。
    输入：dis1、dis2：float，两次移动在虚拟平面上的平均位移；ang1、ang2：float，方向角(弧度)；
          x1、y2：float，位移台两次移动距离(单位一致即可，只用比值)
    输出：(ang_move, candidates)：ang_move 为绝对值最小的解(弧度)；candidates 为全部候选(弧度)；
          无实数解时返回 (None, None)
    """
    c = (dis2 * np.cos(ang2) * x1) / (dis1 * y2)    # cos(ang1 + ang_move) 的取值
    if abs(c) > 1.0:                                 # 超出余弦值域，方程无实数解
        return None, None
    theta = float(np.arccos(np.clip(c, -1.0, 1.0)))  # ang1 + ang_move 的绝对值
    cands = [theta - ang1, -theta - ang1]            # arccos 的两个对称候选
    best = min(cands, key=lambda v: abs(v))          # 取绝对值最小的作为主解
    return best, cands


def calibrate_step3(calib_npz=CALIB_NPZ, centroids_npz=CENTROIDS_NPZ,
                    init_npz=INIT_NPZ, x1=X1, y2=Y2, out_npz=OUT_NPZ):
    """功能：读内参、前三组交点和初始四点坐标，求解并保存位移台夹角 ang_move。
    输入：calib_npz：str，内参 npz；centroids_npz：str，交点 npz；init_npz：str，初始四点 npz；
          x1、y2：float，位移台两次移动距离；out_npz：str，结果保存路径
    输出：(ang_move, candidates) 弧度；数据不足或无解时返回 (None, None)
    """
    cal = np.load(calib_npz)
    K = cal["K"].astype(np.float64)                  # 内参矩阵
    D = cal["D"].astype(np.float64)                  # 畸变系数

    data = np.load(centroids_npz)
    if "points" in data:                             # 新版单键 points：(N,4,2)
        groups = np.asarray(data["points"], np.float64).reshape(-1, 4, 2)
    else:                                            # 旧版多个 imgX 键：按文件名排序拼接
        groups = np.array([np.asarray(data[k], np.float64).reshape(4, 2)
                           for k in sorted(data.files)], np.float64)
    if groups.shape[0] < 3:                          # 至少需要 3 组才能算 dis1、dis2
        print(f"交点只有 {groups.shape[0]} 组，不足 3 组，无法标定")
        return None, None

    planes = [_undistort_to_plane(groups[i], K, D) for i in range(3)]   # 前三组映射到虚拟平面
    dis1 = _mean_pair_dist(planes[0], planes[1])     # 第 1、2 组对应点平均位移
    dis2 = _mean_pair_dist(planes[1], planes[2])     # 第 2、3 组对应点平均位移

    init = np.load(init_npz)
    ang1 = float(init["ang1"])                       # 初始四点坐标里解出的方向角
    ang2 = float(init["ang2"])

    ang_move, cands = _solve_ang_move(dis1, dis2, ang1, ang2, x1, y2)
    if ang_move is None:
        print("方程无实数解，请检查位移读数 x1/y2 或交点顺序")
        return None, None
    np.savez_compressed(out_npz, ang_move=ang_move, candidates=np.asarray(cands))
    return ang_move, cands


if __name__ == "__main__":
    ang_move, cands = calibrate_step3()
    if ang_move is None:
        print("标定失败")
    else:
        print(f"ang_move = {np.degrees(ang_move):.4f}°")
        print(f"候选解 = {[f'{np.degrees(v):.4f}°' for v in cands]}")
