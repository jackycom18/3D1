---
name: 提取result角点_kcosine
overview: 在 图像处理.py 新增一个基于 k-cosine 的角点提取函数：先把 result 的骨架有序化成若干路径段，再对每个点取路径上前后各 k 个点拟合方向、求夹角，夹角超过阈值且彼此间距不小于 50px 的点作为角点，输出 (x,y) 点集。
todos:
  - id: ordered-paths
    content: 按 [skill:code-rules] 新增 _ordered_paths 骨架有序化函数，处理端点、交叉点与纯环
    status: completed
  - id: extract-corners
    content: 新增 _fit_direction 与 _extract_corners，实现 k-cosine 夹角判定与 50px 非极大值抑制
    status: completed
    dependencies:
      - ordered-paths
  - id: integrate-main
    content: 在 image_processing 中 result 之后接入 _extract_corners 并保存/输出角点点集
    status: completed
    dependencies:
      - extract-corners
---

## 产品概述

在现有图像处理脚本中新增一个角点提取函数，对 `result`（去毛刺骨架 + 端点直线连接后的二值图）中的曲线进行角点检测。

## 核心功能

- 输入 `result`（uint8、0/255 骨架图），输出角点点集 `points`。
- 采用标准 k-cosine 思路：先把骨架转成有序路径点列，对每个像素取路径上前 k 个、后 k 个点分别拟合方向，求两条方向夹角（取锐角）。
- 夹角超过阈值（默认 30°）的像素点判定为候选角点。
- 对候选角点做间距抑制：任意两个保留角点之间距离不小于 50 像素。
- 输出 `(N, 2)` 浮点数组，每行为 `(x, y)`（列坐标在前），与 `_centroids` 一致，便于直接用于 cv2。

## 技术栈

- 语言/库：Python 3 + NumPy + OpenCV（cv2），与项目现有依赖一致，不新增依赖。
- 复用现有实现：`_NEIGHBORS8` 8 邻域偏移、`cv2.connectedComponentsWithStats` 连通域、`cv2.filter2D` 邻域计数、`_centroids` 的 `(x, y)` 输出格式。

## 实现方案

### 新增函数（插入到 `_connect_endpoints` 之后、`_centroids` 之前）

1. `_ordered_paths(result)`：骨架有序化辅助函数。

- 输入：uint8、0/255 二值图；输出：`list[(seg, closed)]`，`seg` 为 `(P,2)` 的 `(x,y)` 有序点列，`closed` 表示该段是否为闭合环。
- 对每条 8 连通域，用 `filter2D` 求每个前景点的邻域前景计数，度 = 邻域前景数 − 1。
- 度=1 为端点、度≥3 为交叉点、度=2 为普通路径点；以「端点/交叉点」为节点，沿度=2 的点 DFS 提取节点间的无分支路径段。
- 整段全部度=2 的纯环：任取一点沿环走一圈，标记 `closed=True`。

2. `_fit_direction(seg)`：点集主方向拟合辅助函数。

- 输入：`(N,2)` 的 `(x,y)` 点集；输出：主方向角（弧度）。
- 用总最小二乘/PCA 闭式 `θ = 0.5·atan2(2·Sxy, Sxx−Syy)`，`Sxx/Syy/Sxy` 为点集相对均值的协方差，避免垂直线斜率无穷问题。

3. `_extract_corners(result, k=10, angle_deg=30, min_dist=50)`：主函数。

- 输入：uint8、0/255 二值图及三个可调参数；输出：`(N,2)` float32 的 `(x,y)` 角点集，无角点时返回 `(0,2)` 空数组。
- 对每条路径段上的点 `i` 取前 `k` 个点与后 `k` 个点分别拟合方向；闭合环用环形索引（取模），开放路径要求前后各有 `k` 个点。
- 夹角 `δ = min(|θ1−θ2|, π−|θ1−θ2|)`，转角度后与 `angle_deg` 比较，超过阈值的记为候选角点（保留其夹角响应值）。
- 非极大值抑制：候选角点按夹角响应从大到小排序，贪心选取，与已选角点欧氏距离 ≥ `min_dist` 才保留。

### 关键决策

- 有序化采用「节点 + 沿度=2 路径 DFS」，正确处理 `result` 中因端点连线产生的交叉点和闭合多边形，避免误按单路径假设。
- 方向拟合用 PCA 闭式解而非斜率，规避垂直线斜率无穷大。
- 闭合环用取模索引做 k-cosine，保证多边形顶点（直线拐角）能被检出。
- NMS 用夹角降序贪心，保留响应最强的角点，满足 50px 间距约束。

### 性能与边界

- 有序化 O(前景像素数)；k-cosine 对每段 O(P·k)，k=10 很小，整体足够快。
- `result` 全黑时返回 `(0,2)` 空数组；路径段长度不足 `2k+1` 时跳过该段。
- 三个新函数均按 code-rules 写「功能/输入/输出」注释，核心变量首次出现处注释含义。

## 目录结构

仅修改一个文件：

```
c:/python/3D_project_3/核心代码_标定1/图像处理.py
```

- `[MODIFY]` 在 `_connect_endpoints`（227 行）之后、`_centroids`（230 行）之前新增 `_ordered_paths`、`_fit_direction`、`_extract_corners` 三个函数。
- `[MODIFY]` 在 `image_processing` 主函数 `result = _connect_endpoints(cluster)`（281 行）之后接入 `_extract_corners(result)`，并将角点结果 `imwrite` 到 `_debug/corners.png`（可视化角点叠加图）或打印坐标，便于验证效果。

## Agent Extensions

### Skill

- **code-rules**
- 用途：写 `_ordered_paths`、`_fit_direction`、`_extract_corners` 三个函数时严格遵守用户个人代码规范（函数头「功能/输入/输出」注释、核心变量首现注释、只写核心代码、复用已有函数、改动最小化）。
- 预期结果：新增函数接口清晰、注释完整、无测试样例与多余调试输出，风格与现有 `_connect_endpoints`、`_centroids` 一致。