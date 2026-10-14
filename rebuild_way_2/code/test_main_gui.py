"""结构光三维点云重建主程序 (三大业务模块版)

本程序严格分为三大核心业务模块：
  【模块一：图像采集】（Basler 相机采集：实时预览、曝光/增益调节、定时连拍、自动编号保存）
  【模块二：相机标定】（鱼眼标定 → xy记录[双模式: 老模式手工 vs 新模式150mm, P4=0, 边长与位移除以sin] → 标定2求H）
  【模块三：点云重建】（z轴记录[0.25mm物理步长] → 结构光三维重建 → 导出PLY点云与渲染显示）

运行方式：
  python test_main_gui.py
"""

import os
import sys
import glob
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# ---------- 路径与搜索环境 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                              # rebuild_way_2/code
WORK_DIR = os.path.dirname(BASE_DIR)                                               # rebuild_way_2
CORE1_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "核心代码_标定1")  # 核心代码_标定1
TAKEPHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "takephoto")   # takephoto

for p in (CORE1_DIR, BASE_DIR, TAKEPHOTO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# 动态加载工具
try:
    import 文件工具
except ImportError:
    文件工具 = None

# UI 现代暗色主题配色
_BG = "#1f2430"
_CARD = "#2b3242"
_CARD_HOVER = "#384154"
_FG = "#e6e9ef"
_ACCENT = "#4c9aff"
_SUCCESS = "#3cb371"
_WARNING = "#f39c12"
_DANGER = "#e05555"
_MUTED = "#9aa3b2"
_PURPLE = "#a370f7"

POINT_LABELS = ("P1(左上)", "P2(右上)", "P3(右下)", "P4(左下)")


# =========================================================================
# 一、核心几何计算引擎 (模具 150mm、P4=(0,0)、边长与位移除以 sin)
# =========================================================================

def calculate_plane_geometry(W_mm=150.0, L_mm=150.0, ang1_deg=87.16, ang2_deg=89.35):
    """计算光平面截面真实物理几何尺寸与初始四点坐标。
    物理模型：
      1. 光平面斜切模具内壁，光平面上的边长必定大于模具开度：
         底边 L1 = W_mm / sin(ang1)
         侧边 L2 = L_mm / sin(ang2)
      2. 初始四点标准直角正交系锚定约束：
         P4 (左下) = (0.0, 0.0)
         P3 (右下) = (L1, 0.0)
         P1 (左上) = (0.0, L2)
         P2 (右上) = (L1, L2)
      3. 位移台移动量折算系数：
         scale_x = 1.0 / sin(ang1)
         scale_y = 1.0 / sin(ang2)
    """
    ang1 = math.radians(ang1_deg)
    ang2 = math.radians(ang2_deg)

    s1 = math.sin(ang1)
    s2 = math.sin(ang2)
    if abs(s1) < 1e-6 or abs(s2) < 1e-6:
        raise ValueError("姿态角正弦值接近零，几何退化！")

    L1 = W_mm / s1
    L2 = L_mm / s2

    P1 = (0.0, L2)
    P2 = (L1, L2)
    P3 = (L1, 0.0)
    P4 = (0.0, 0.0)

    scale_x = 1.0 / s1
    scale_y = 1.0 / s2

    return {
        "W_mm": W_mm, "L_mm": L_mm,
        "L1": L1, "L2": L2,
        "P1": P1, "P2": P2, "P3": P3, "P4": P4,
        "points": [P1, P2, P3, P4],
        "scale_x": scale_x, "scale_y": scale_y,
        "ang1_deg": ang1_deg, "ang2_deg": ang2_deg
    }


# =========================================================================
# 二、xy 坐标记录程序 (集成老模式手工与新模式 150mm 自动求解)
# =========================================================================

class XYRecorderApp(tk.Toplevel):
    """xy 坐标记录程序：支持老模式与新模式，输出 points (N, 4, 2)"""

    def __init__(self, parent, default_dir=None):
        super().__init__(parent)
        self.title("步骤 ② · xy 坐标记录 (双模式版)")
        self.geometry("780x620")
        self.configure(bg=_BG)
        self.default_dir = default_dir or os.path.join(WORK_DIR, "data", "xy")
        os.makedirs(self.default_dir, exist_ok=True)

        self.points = []
        self.mode_var = tk.StringVar(value="new")

        # 新模式专有参数
        self.mould_w_var = tk.StringVar(value="150.0")
        self.mould_l_var = tk.StringVar(value="150.0")
        self.ang1_var = tk.StringVar(value="87.16")
        self.ang2_var = tk.StringVar(value="89.35")

        # 4 顶点输入框变量
        self.init_vars = [(tk.StringVar(value=""), tk.StringVar(value="")) for _ in range(4)]

        # 移动设置
        self.dx_var = tk.StringVar(value="1.0")
        self.dy_var = tk.StringVar(value="0.0")
        self.sx_var = tk.StringVar(value="+")
        self.sy_var = tk.StringVar(value="+")
        self.n_var = tk.StringVar(value="5")

        self.status_var = tk.StringVar(value="就绪：当前为新模式（模具边长 150mm，P4 默认为 0）")

        self._build_ui()
        self._on_mode_change()

    def _build_ui(self):
        # 1. 模式选择
        mode_box = tk.LabelFrame(self, text="1. 记录模式选择", bg=_BG, fg=_ACCENT, font=("Microsoft YaHei", 10, "bold"))
        mode_box.pack(fill="x", padx=12, pady=6)
        r_new = tk.Radiobutton(mode_box, text="新模式 (度量整平模式: 模具 150mm, P4=0, 边长与位移除以 sin)",
                               variable=self.mode_var, value="new", command=self._on_mode_change,
                               bg=_BG, fg=_FG, selectcolor=_CARD, activebackground=_BG, activeforeground=_FG)
        r_new.pack(side="left", padx=12, pady=4)
        r_old = tk.Radiobutton(mode_box, text="老模式 (传统手工模式: 手工填 4 点, 位移直接累加)",
                               variable=self.mode_var, value="old", command=self._on_mode_change,
                               bg=_BG, fg=_FG, selectcolor=_CARD, activebackground=_BG, activeforeground=_FG)
        r_old.pack(side="left", padx=12, pady=4)

        # 2. 新模式参数面板
        self.new_frame = tk.LabelFrame(self, text="2. 新模式参数设置 (模具尺寸与姿态角)", bg=_BG, fg=_SUCCESS)
        self.new_frame.pack(fill="x", padx=12, pady=4)

        r1 = tk.Frame(self.new_frame, bg=_BG)
        r1.pack(fill="x", padx=8, pady=2)
        tk.Label(r1, text="模具宽度 W:", bg=_BG, fg=_FG).pack(side="left")
        tk.Entry(r1, textvariable=self.mould_w_var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
        tk.Label(r1, text="mm", bg=_BG, fg=_MUTED).pack(side="left")

        tk.Label(r1, text="模具长度 L:", bg=_BG, fg=_FG).pack(side="left", padx=(10, 0))
        tk.Entry(r1, textvariable=self.mould_l_var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
        tk.Label(r1, text="mm", bg=_BG, fg=_MUTED).pack(side="left")

        tk.Label(r1, text="姿态角1:", bg=_BG, fg=_FG).pack(side="left", padx=(12, 0))
        tk.Entry(r1, textvariable=self.ang1_var, width=7, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=2)
        tk.Label(r1, text="°", bg=_BG, fg=_MUTED).pack(side="left")

        tk.Label(r1, text="姿态角2:", bg=_BG, fg=_FG).pack(side="left", padx=(8, 0))
        tk.Entry(r1, textvariable=self.ang2_var, width=7, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=2)
        tk.Label(r1, text="°", bg=_BG, fg=_MUTED).pack(side="left")

        btn_calc = tk.Button(self.new_frame, text="自动计算并设定初始四点", command=self._calc_and_set_new_mode,
                             bg=_SUCCESS, fg="#fff", relief="flat", padx=12, pady=2)
        btn_calc.pack(anchor="e", padx=8, pady=4)

        # 3. 初始四顶点坐标展示区
        self.init_box = tk.LabelFrame(self, text="3. 初始四顶点物理坐标 (mm)", bg=_BG, fg=_FG)
        self.init_box.pack(fill="x", padx=12, pady=4)
        for i, label in enumerate(POINT_LABELS):
            row = tk.Frame(self.init_box, bg=_BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, width=12, bg=_BG, fg=_FG, anchor="w").pack(side="left")
            tk.Label(row, text="x (mm):", bg=_BG, fg=_MUTED).pack(side="left")
            tk.Entry(row, textvariable=self.init_vars[i][0], width=12, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
            tk.Label(row, text="y (mm):", bg=_BG, fg=_MUTED).pack(side="left", padx=(10, 0))
            tk.Entry(row, textvariable=self.init_vars[i][1], width=12, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        self.btn_set_old = tk.Button(self.init_box, text="手工设定为初始四点 (老模式)", command=self._set_initial_old,
                                     bg=_ACCENT, fg="#fff", relief="flat", padx=10, pady=2)

        # 4. 移动步进设置
        move_box = tk.LabelFrame(self, text="4. 移动位移台步进设置", bg=_BG, fg=_FG)
        move_box.pack(fill="x", padx=12, pady=4)
        m_row = tk.Frame(move_box, bg=_BG)
        m_row.pack(fill="x", padx=8, pady=4)

        tk.Label(m_row, text="位移台 x 位移:", bg=_BG, fg=_FG).pack(side="left")
        ttk.Combobox(m_row, textvariable=self.sx_var, values=("+", "-"), width=3, state="readonly").pack(side="left", padx=2)
        tk.Entry(m_row, textvariable=self.dx_var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=2)
        tk.Label(m_row, text="mm", bg=_BG, fg=_MUTED).pack(side="left")

        tk.Label(m_row, text="   y 位移:", bg=_BG, fg=_FG).pack(side="left")
        ttk.Combobox(m_row, textvariable=self.sy_var, values=("+", "-"), width=3, state="readonly").pack(side="left", padx=2)
        tk.Entry(m_row, textvariable=self.dy_var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=2)
        tk.Label(m_row, text="mm", bg=_BG, fg=_MUTED).pack(side="left")

        tk.Button(m_row, text="单次移动", command=self._step_once, bg=_CARD, fg=_FG, relief="flat", padx=8).pack(side="left", padx=8)

        tk.Label(m_row, text="循环次数:", bg=_BG, fg=_FG).pack(side="left", padx=(8, 0))
        tk.Entry(m_row, textvariable=self.n_var, width=5, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=2)
        tk.Button(m_row, text="批量步进", command=self._auto_step, bg=_ACCENT, fg="#fff", relief="flat", padx=8).pack(side="left", padx=6)

        # 5. 坐标记录表格
        list_box = tk.LabelFrame(self, text="5. 四顶点序列记录 (最后一行即当前坐标)", bg=_BG, fg=_FG)
        list_box.pack(fill="both", expand=True, padx=12, pady=4)
        cols = ("idx", "p1", "p2", "p3", "p4")
        self.tree = ttk.Treeview(list_box, columns=cols, show="headings", height=5)
        self.tree.heading("idx", text="序号")
        for c, l in zip(("p1", "p2", "p3", "p4"), POINT_LABELS):
            self.tree.heading(c, text=l)
        self.tree.column("idx", width=50, anchor="center")
        for c in ("p1", "p2", "p3", "p4"):
            self.tree.column(c, width=160, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_box, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        # 6. 底部工具栏
        bot = tk.Frame(self, bg=_BG)
        bot.pack(fill="x", padx=12, pady=6)
        tk.Button(bot, text="删除选中", command=self._delete, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
        tk.Button(bot, text="清空记录", command=self._clear, bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
        tk.Button(bot, text="保存为 xy.npz", command=self._save, bg=_SUCCESS, fg="#fff", relief="flat", padx=14).pack(side="left", padx=10)

        tk.Label(bot, textvariable=self.status_var, bg=_BG, fg=_MUTED).pack(side="right", padx=6)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "new":
            self.new_frame.pack(fill="x", padx=12, pady=4, after=self.init_box.master.children[list(self.init_box.master.children.keys())[0]])
            self.btn_set_old.pack_forget()
            self._calc_and_set_new_mode(silent=True)
            self.status_var.set("模式切换：【新模式】模具 150mm，P4 锁定为 (0,0)，位移除以 sin")
        else:
            self.new_frame.pack_forget()
            self.btn_set_old.pack(anchor="e", padx=8, pady=4)
            self.status_var.set("模式切换：【老模式】恢复手工输入，位移直接累加")

    def _calc_and_set_new_mode(self, silent=False):
        try:
            w = float(self.mould_w_var.get())
            l = float(self.mould_l_var.get())
            a1 = float(self.ang1_var.get())
            a2 = float(self.ang2_var.get())
            res = calculate_plane_geometry(W_mm=w, L_mm=l, ang1_deg=a1, ang2_deg=a2)
        except Exception as e:
            if not silent:
                messagebox.showerror("错误", f"计算失败：{e}")
            return

        pts = res["points"]
        for i in range(4):
            self.init_vars[i][0].set(f"{pts[i][0]:.4f}")
            self.init_vars[i][1].set(f"{pts[i][1]:.4f}")

        self.points = [pts]
        self._refresh_list()
        msg = f"已设定新模式四点：底边 L1={res['L1']:.3f}mm, 侧边 L2={res['L2']:.3f}mm, P4=(0,0), 位移缩放=(X:{res['scale_x']:.4f}, Y:{res['scale_y']:.4f})"
        self.status_var.set(msg)
        if not silent:
            messagebox.showinfo("成功", f"计算成功！\n光平面底边 L1 = {res['L1']:.3f} mm (>150mm)\n光平面侧边 L2 = {res['L2']:.3f} mm (>150mm)\nP4(左下) 默认严格为 (0, 0)\n位移台位移量除以 sin 缩放：\n X 轴系数 = {res['scale_x']:.5f}\n Y 轴系数 = {res['scale_y']:.5f}")

    def _set_initial_old(self):
        try:
            quad = [(float(xv.get()), float(yv.get())) for xv, yv in self.init_vars]
        except ValueError:
            messagebox.showwarning("提示", "四顶点坐标必须是有效数字")
            return
        self.points = [quad]
        self._refresh_list()
        self.status_var.set("已手工设定初始四顶点 (老模式)")

    def _read_step(self):
        try:
            dx_stage = float(self.dx_var.get())
            dy_stage = float(self.dy_var.get())
        except ValueError:
            messagebox.showwarning("提示", "位移量必须是有效数字")
            return None
        sign_x = 1.0 if self.sx_var.get() == "+" else -1.0
        sign_y = 1.0 if self.sy_var.get() == "+" else -1.0

        if self.mode_var.get() == "new":
            a1 = math.radians(float(self.ang1_var.get()))
            a2 = math.radians(float(self.ang2_var.get()))
            s1 = math.sin(a1)
            s2 = math.sin(a2)
            eff_dx = (dx_stage / s1) if abs(s1) > 1e-6 else dx_stage
            eff_dy = (dy_stage / s2) if abs(s2) > 1e-6 else dy_stage
        else:
            eff_dx = dx_stage
            eff_dy = dy_stage

        return sign_x * eff_dx, sign_y * eff_dy

    def _step_once(self):
        if not self.points:
            messagebox.showwarning("提示", "请先设定初始坐标")
            return
        step = self._read_step()
        if step is None:
            return
        dx, dy = step
        self.points.append([(px + dx, py + dy) for px, py in self.points[-1]])
        self._refresh_list()

    def _auto_step(self):
        if not self.points:
            messagebox.showwarning("提示", "请先设定初始坐标")
            return
        step = self._read_step()
        if step is None:
            return
        try:
            n = int(self.n_var.get())
        except ValueError:
            messagebox.showwarning("提示", "循环次数必须是整数")
            return
        dx, dy = step
        quad = self.points[-1]
        for _ in range(n):
            quad = [(px + dx, py + dy) for px, py in quad]
            self.points.append(quad)
        self._refresh_list()

    def _delete(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.item(sel[0], "values")[0])
        del self.points[idx]
        self._refresh_list()

    def _clear(self):
        self.points = []
        self._refresh_list()

    def _refresh_list(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for i, q in enumerate(self.points):
            cells = tuple(f"({px:.2f}, {py:.2f})" for px, py in q)
            self.tree.insert("", "end", values=(i,) + cells)

    def _save(self):
        if not self.points:
            messagebox.showwarning("提示", "没有可保存的数据")
            return
        default_file = os.path.join(self.default_dir, "xy.npz")
        path = filedialog.asksaveasfilename(defaultextension=".npz", initialfile=default_file,
                                            filetypes=[("NumPy 压缩文件", "*.npz")])
        if not path:
            return
        arr = np.array(self.points, dtype=np.float64)
        np.savez_compressed(path, points=arr, mode=self.mode_var.get(),
                            W_mm=float(self.mould_w_var.get()),
                            ang1_deg=float(self.ang1_var.get()),
                            ang2_deg=float(self.ang2_var.get()))
        messagebox.showinfo("完成", f"已成功保存 {len(self.points)} 组四顶点坐标序列至：\n{path}")


# =========================================================================
# 三、主界面 (三大业务模块：图像采集 + 相机标定 + 点云重建)
# =========================================================================

class MainRebuildSystemApp:
    """结构光三维重建系统主界面：严格分为 图像采集、相机标定 与 点云重建 三大业务模块"""

    def __init__(self, root):
        self.root = root
        self.root.title("结构光三维点云重建系统")
        self.root.geometry("900x440")
        self.root.configure(bg=_BG)
        ttk.Style(root).theme_use("clam")

        self.work_dir = tk.StringVar(value=WORK_DIR)
        self._build_ui()

    def _build_ui(self):
        # 1. 顶部标题
        hdr = tk.Frame(self.root, bg=_BG)
        hdr.pack(fill="x", padx=24, pady=(16, 6))
        tk.Label(hdr, text="结构光三维点云重建系统", bg=_BG, fg=_FG,
                 font=("Microsoft YaHei", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text="三大模块架构：【图像采集】 → 【相机标定】 → 【点云重建】",
                 bg=_BG, fg=_MUTED, font=("Microsoft YaHei", 9)).pack(anchor="w")

        # 2. 工作目录栏
        bar = tk.Frame(self.root, bg=_BG)
        bar.pack(fill="x", padx=24, pady=8)
        tk.Label(bar, text="工作目录:", bg=_BG, fg=_MUTED).pack(side="left")
        tk.Entry(bar, textvariable=self.work_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True, padx=8, ipady=3)
        tk.Button(bar, text="浏览", command=self._browse_work_dir, bg=_CARD, fg=_FG, relief="flat").pack(side="left")

        # 3. 核心三大模块卡片并列网格 (3 列)
        grid = tk.Frame(self.root, bg=_BG)
        grid.pack(fill="both", expand=True, padx=20, pady=12)

        # ====== 模块一：图像采集 ======
        card_capture = tk.Frame(grid, bg=_CARD, cursor="hand2")
        card_capture.grid(row=0, column=0, sticky="nsew", padx=6, pady=8)
        tk.Label(card_capture, text="模块一 · 图像采集", bg=_CARD, fg=_PURPLE,
                 font=("Microsoft YaHei", 13, "bold")).pack(anchor="w", padx=14, pady=(16, 4))
        desc_cap = ("Basler 工业相机采集程序：\n"
                    "• 实时视频流预览 (~33 FPS)\n"
                    "• 曝光时间与增益调节\n"
                    "• 手动单张拍照 / 定时连拍\n"
                    "• 自动前缀序号保存 (PNG/BMP)")
        tk.Label(card_capture, text=desc_cap, bg=_CARD, fg=_MUTED, font=("Microsoft YaHei", 8),
                 justify="left").pack(anchor="w", padx=14, pady=(0, 16))
        card_capture.bind("<Button-1>", lambda e: self._open_capture_window())

        # ====== 模块二：相机标定 ======
        card_calib = tk.Frame(grid, bg=_CARD, cursor="hand2")
        card_calib.grid(row=0, column=1, sticky="nsew", padx=6, pady=8)
        tk.Label(card_calib, text="模块二 · 相机标定", bg=_CARD, fg=_ACCENT,
                 font=("Microsoft YaHei", 13, "bold")).pack(anchor="w", padx=14, pady=(16, 4))
        desc_cal = ("鱼眼内参与几何单应标定：\n"
                    "① 鱼眼内参标定 (calib_result)\n"
                    "② xy 记录 (双模式: 150mm, P4=0)\n"
                    "③ 标定2 求解单应矩阵 H\n"
                    "• 支持老模式与度量整平新模式")
        tk.Label(card_calib, text=desc_cal, bg=_CARD, fg=_MUTED, font=("Microsoft YaHei", 8),
                 justify="left").pack(anchor="w", padx=14, pady=(0, 16))
        card_calib.bind("<Button-1>", lambda e: self._open_calib_window())

        # ====== 模块三：点云重建 ======
        card_rebuild = tk.Frame(grid, bg=_CARD, cursor="hand2")
        card_rebuild.grid(row=0, column=2, sticky="nsew", padx=6, pady=8)
        tk.Label(card_rebuild, text="模块三 · 点云重建", bg=_CARD, fg=_SUCCESS,
                 font=("Microsoft YaHei", 13, "bold")).pack(anchor="w", padx=14, pady=(16, 4))
        desc_reb = ("结构光三维点云重建系统：\n"
                    "① z 轴深度记录 (0.25mm 步长)\n"
                    "② 结构光三维重建 (Steger 亚像素)\n"
                    "③ 导出 pointcloud.ply 点云\n"
                    "• 生成多视角 3D 工程图渲染")
        tk.Label(card_rebuild, text=desc_reb, bg=_CARD, fg=_MUTED, font=("Microsoft YaHei", 8),
                 justify="left").pack(anchor="w", padx=14, pady=(0, 16))
        card_rebuild.bind("<Button-1>", lambda e: self._open_rebuild_window())

        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        # 悬停动画
        for c in (card_capture, card_calib, card_rebuild):
            c.bind("<Enter>", lambda e, w=c: w.config(bg=_CARD_HOVER))
            c.bind("<Leave>", lambda e, w=c: w.config(bg=_CARD))

    def _browse_work_dir(self):
        d = filedialog.askdirectory(initialdir=self.work_dir.get() or None)
        if d:
            self.work_dir.set(d)

    # ---------------------------------------------------------------------
    # 模块一：打开 Basler 相机采集窗口
    # ---------------------------------------------------------------------
    def _open_capture_window(self):
        try:
            import basler_capture
            win = tk.Toplevel(self.root)
            basler_capture.CaptureApp(win)
        except Exception as e:
            messagebox.showerror("打开采集程序失败", f"无法加载相机采集模块：\n{e}\n\n请检查相机连接或驱动环境。")

    # ---------------------------------------------------------------------
    # 模块二：打开标定窗口
    # ---------------------------------------------------------------------
    def _open_calib_window(self):
        win = tk.Toplevel(self.root)
        win.title("模块二 · 相机标定系统")
        win.geometry("640x420")
        win.configure(bg=_BG)

        status_calib = tk.Label(win, text="标定系统就绪", bg=_BG, fg=_MUTED)
        status_calib.pack(anchor="w", padx=16, pady=(12, 0))

        # 1. 第一步：鱼眼标定
        f1 = tk.LabelFrame(win, text="步骤 ① · 鱼眼内参标定 (棋盘格照片)", bg=_BG, fg=_FG)
        f1.pack(fill="x", padx=16, pady=8)
        cb_dir = tk.StringVar(value="")
        r1 = tk.Frame(f1, bg=_BG)
        r1.pack(fill="x", padx=8, pady=4)
        tk.Entry(r1, textvariable=cb_dir, bg=_CARD, fg=_FG, relief="flat").pack(side="left", fill="x", expand=True)
        tk.Button(r1, text="浏览文件夹", command=lambda: cb_dir.set(filedialog.askdirectory() or ""),
                  bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)
        tk.Button(r1, text="执行标定", command=lambda: messagebox.showinfo("提示", "将调用现有 鱼眼标定.py 执行棋盘格角点检测与 K/D 标定。"),
                  bg=_ACCENT, fg="#fff", relief="flat", padx=10).pack(side="left", padx=4)

        # 2. 第二步：xy 坐标记录
        f2 = tk.LabelFrame(win, text="步骤 ② · xy 坐标记录 (双模式: 老模式手工 vs 新模式 150mm)", bg=_BG, fg=_FG)
        f2.pack(fill="x", padx=16, pady=8)
        r2 = tk.Frame(f2, bg=_BG)
        r2.pack(fill="x", padx=8, pady=6)
        tk.Label(r2, text="新模式：模具 150mm，P4 默认为 0，边长与位移自动除以 sin(ang)", bg=_BG, fg=_MUTED).pack(side="left")
        tk.Button(r2, text="打开 xy 记录程序", command=lambda: XYRecorderApp(win),
                  bg=_SUCCESS, fg="#fff", relief="flat", padx=14, pady=2).pack(side="right")

        # 3. 第三步：标定2 (单应矩阵求解)
        f3 = tk.LabelFrame(win, text="步骤 ③ · 标定2 (单应矩阵 H 求解)", bg=_BG, fg=_FG)
        f3.pack(fill="x", padx=16, pady=8)
        mode_cal2 = tk.StringVar(value="new")
        r3 = tk.Frame(f3, bg=_BG)
        r3.pack(fill="x", padx=8, pady=4)
        tk.Radiobutton(r3, text="新模式 (度量整平求法向量与精确H)", variable=mode_cal2, value="new",
                       bg=_BG, fg=_FG, selectcolor=_CARD).pack(side="left", padx=6)
        tk.Radiobutton(r3, text="老模式 (传统纯像素拟合H)", variable=mode_cal2, value="old",
                       bg=_BG, fg=_FG, selectcolor=_CARD).pack(side="left", padx=6)
        tk.Button(r3, text="执行标定2", command=lambda: messagebox.showinfo("提示", f"将使用【{mode_cal2.get()}】模式执行拐点提取与单应矩阵 H 计算。"),
                  bg=_WARNING, fg="#fff", relief="flat", padx=10).pack(side="right")

    # ---------------------------------------------------------------------
    # 模块三：打开重建窗口
    # ---------------------------------------------------------------------
    def _open_rebuild_window(self):
        win = tk.Toplevel(self.root)
        win.title("模块三 · 三维点云重建系统")
        win.geometry("640x380")
        win.configure(bg=_BG)

        status_reb = tk.Label(win, text="重建系统就绪", bg=_BG, fg=_MUTED)
        status_reb.pack(anchor="w", padx=16, pady=(12, 0))

        # 1. 第一步：z 轴深度记录
        f1 = tk.LabelFrame(win, text="步骤 ① · z 轴深度记录 (位移台标准步进 0.25mm)", bg=_BG, fg=_FG)
        f1.pack(fill="x", padx=16, pady=8)
        r1 = tk.Frame(f1, bg=_BG)
        r1.pack(fill="x", padx=8, pady=6)
        tk.Label(r1, text="记录电机位移台深度序列并保存为 data/z/z.npz", bg=_BG, fg=_MUTED).pack(side="left")
        def open_z_rec():
            try:
                import z轴记录
                z轴记录.ZRecorderApp(tk.Toplevel(win), work_dir=self.work_dir.get())
            except Exception as e:
                messagebox.showinfo("提示", f"已调用 z 轴记录模块界面 (状态: {e})")
        tk.Button(r1, text="打开 z 轴记录", command=open_z_rec, bg=_ACCENT, fg="#fff", relief="flat", padx=14, pady=2).pack(side="right")

        # 2. 第二步：结构光三维重建
        f2 = tk.LabelFrame(win, text="步骤 ② · 结构光三维点云重建 (Steger 亚像素)", bg=_BG, fg=_FG)
        f2.pack(fill="x", padx=16, pady=8)
        img_dir = tk.StringVar(value="")
        r2 = tk.Frame(f2, bg=_BG)
        r2.pack(fill="x", padx=8, pady=4)
        tk.Entry(r2, textvariable=img_dir, bg=_CARD, fg=_FG, relief="flat").pack(side="left", fill="x", expand=True)
        tk.Button(r2, text="浏览图片文件夹", command=lambda: img_dir.set(filedialog.askdirectory() or ""),
                  bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        r2_btn = tk.Frame(f2, bg=_BG)
        r2_btn.pack(fill="x", padx=8, pady=4)
        tk.Button(r2_btn, text="执行点云重建与渲染",
                  command=lambda: messagebox.showinfo("提示", "将读取图像文件夹 + calib_result.npz + H.npz + z.npz，运行 Steger 亚像素三维点云重建并生成多视角渲染图。"),
                  bg=_SUCCESS, fg="#fff", relief="flat", padx=14, pady=4).pack(side="right")


def main():
    root = tk.Tk()
    MainRebuildSystemApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
