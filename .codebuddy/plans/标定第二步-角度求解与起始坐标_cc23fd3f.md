---
name: 标定第二步-角度求解与起始坐标
overview: 新建标定第二步脚本：读取鱼眼内参和多个角点组，去畸变投影到 z=1 虚拟平面，按给定两公式求解两个角度 ang1/ang2（多组平均），并用第一组点算出光平面上四点的现实坐标保存为 npz；同时改造平移台软件，启动时自动读取该 npz 作为初始坐标。
todos:
  - id: create-step2-script
    content: 按 [skill:code-rules] 新建 核心代码/标定第二步.py，实现去畸变、计算 ang/k2、解方程、多组平均、第一组重建四点坐标并保存 npz、返回角度
    status: completed
  - id: modify-stage-software
    content: 按 [skill:code-rules] 修改 平移台软件/平移台软件.py，新增从标定 npz 读取起始坐标（cm→mm）并回填/写入初始记录
    status: completed
  - id: syntax-check
    content: 用 py -3.8 对两个文件做语法编译校验并确认无语法错误
    status: completed
    dependencies:
      - create-step2-script
      - modify-stage-software
---

## 产品概述

新增「标定第 2 步」脚本：利用鱼眼内参把图像中的 4 个交点（光平面与镂空区域内侧四条边的交点）去畸变到 z=1 虚拟平面，通过图像中的夹角 ang、距离比 k2 与镂空区域长宽比 k1，反解出两个方向角 ang1、ang2；多组测量取平均后，用第一组点重建光平面上的 4 点现实坐标并保存为 npz。同时改造平移台软件，自动读取该 npz 作为起始坐标。

## 核心功能

- 读取 `calib_result.npz`（K、D、image_size）与 `centroids.npz`（多组、每组 4 个 uv 像素坐标，已按左上→右上→右下→左下排序）。
- 用 `cv2.fisheye.undistortPoints` 把每组 4 点去畸变到 z=1 虚拟平面。
- 对每组计算：左上角两条相邻边的夹角 `ang`、左上角到两相邻点距离比 `k2`。
- 解方程组 `k1·(sin ang1 / sin ang2) = k2` 与 `cos ang = cos ang1 · cos ang2`，得到 ang1、ang2；多组取平均。
- 用第一组点的 ang1、ang2、ang 重建 4 点现实坐标（左上为原点，左上→右上为 x 正半轴），单位为 cm，保存到 npz。
- 函数返回平均后的 ang1、ang2（不写文件）。
- 平移台软件新增从该 npz 读取起始坐标（cm 自动换算 mm），替代手动输入。

## 边界与约定

- `calib_result.npz` 由用户拍完照后生成，脚本按第 1 步保存的字段读取。
- `centroids.npz` 将重新生成为多组，脚本优先读取 `points` 键（形状 (N,4,2)），并兼容收集所有数组键。
- 镂空区域为正方形，`k1` 默认 1，`L`、`W` 默认相等；公式按用户给定实现。

## 技术栈

- 语言：Python 3.8（命令 `py -3.8`）
- 依赖：numpy 1.24.4、opencv 4.13.0（已安装）；不使用 scipy，方程组用解析解，减少依赖
- 图形界面：tkinter（沿用现有平移台软件）

## 实现方案

### 新建脚本 `核心代码/标定第二步.py`

采用与现有《鱼眼标定.py》《图像处理_四点提取.py》一致的「常量 + 小函数 + 主函数 + __main__」结构。

1. **去畸变**：`cv2.fisheye.undistortPoints(pts, K, D)`，输入 `(4,1,2)` float64 像素坐标，输出 `(4,1,2)` 理想针孔归一化坐标（即 z=1 平面 x,y）。注意 K、D 从 npz 读取后转 float64。

2. **几何量**：沿用左上→右上→右下→左下顺序，P0=左上、P1=右上、P2=右下、P3=左下；`v1=P1-P0`、`v3=P3-P0`；

- `ang = arccos((v1·v3)/(|v1||v3|))`
- `k2 = |v1| / |v3|`

3. **解方程（解析解）**：令 `r=k2/k1`、`C=cos(ang)`、`u=sin²(ang2)`。由 `sin(ang1)=r·sin(ang2)` 与 `cos(ang1)·cos(ang2)=C` 平方得二次方程 `r²u²-(1+r²)u+(1-C²)=0`，用求根公式求 `u`，枚举两个候选根，验证 `0<u≤1`、`sin(ang1)≤1`、`cos(ang1)·cos(ang2)≈C`，取锐角有效解得到 ang1、ang2。

4. **多组平均**：对 centroids 每组解出 ang1_i、ang2_i，对有效解求均值得到最终 ang1、ang2。

5. **现实坐标（仅第一组）**：

- `d1 = L / sin(ang1_0)`（左上→右上，x 正半轴）
- `d2 = W / sin(ang2_0)`（左上→左下；正方形时与原文 `L/sin` 等价）
- 四点：左上 `(0,0)`、右上 `(d1,0)`、左下 `(d2·cos ang_0, d2·sin ang_0)`、右下 `(d1+d2·cos ang_0, d2·sin ang_0)`，单位 cm。

6. **保存与返回**：`np.savez_compressed(out_npz, points=pts4, unit="cm")`；返回 `(ang1, ang2)`。

### 修改 `平移台软件/平移台软件.py`

- 新增方法 `_load_initial_from_calib()`：打开 npz，读取 `points`(4,2) 与 `unit`，按 `unit` 换算为内部 mm（复用 `_to_mm`），回填 `point_vars` 输入框，并写入 `records=[{"dx":0,"dy":0,"points":pts}]`，刷新表格。
- 在初始坐标区新增按钮「从标定结果读取初始坐标」绑定该方法；程序启动时若默认标定 npz 存在，可自动调用一次（存在才加载，不阻塞）。

## 性能与可靠性

- 每组仅 4 个点，计算量为 O(N)（N 为组数），无性能瓶颈。
- 方程可能有多个候选解，必须验证并选锐角有效解；无有效解时跳过该组并在多组平均时排除。
- 保持对旧版 `centroids.npz`（多个 `imgX` 键）的兼容读取，避免用户重新生成前脚本报错。
- 遵循 code-rules：函数头注释功能/输入/输出、核心变量注释、克制 print、复用已有函数、只写核心代码。

## 目录结构

```
c:/python/3D_project_3/
├── 核心代码/
│   └── 标定第二步.py          # [NEW] 标定第2步主脚本
└── 平移台软件/
    └── 平移台软件.py          # [MODIFY] 新增从标定npz读取起始坐标
```

## Agent Extensions

### Skill

- **code-rules**
- 用途：写/改代码时遵守用户个人规范（函数头注释、核心变量注释、克制 print、模块化、复用已有函数、改动最小化、收尾清理临时文件）。
- 预期结果：新脚本与平移台软件改动风格统一、注释齐全、无多余测试文件与临时文件残留。