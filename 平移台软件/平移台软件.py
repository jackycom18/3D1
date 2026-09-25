"""平移台坐标记录软件：跟踪四边形四顶点(P1~P4)随 xy 平移台移动后的坐标变化。

功能：设置初始四点坐标 → 每次输入平移台 x/y 移动方向与距离 → 自动算出新四点坐标 →
      表格展示历史记录(含初始) → 保存/加载 npz → 手动删除某条记录。
单位：cm / mm 可切换。
运行：python 平移台软件.py
"""

import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np


# ---------- 常量 ----------
UNITS = ("mm", "cm")                        # 可选单位：毫米、厘米
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0}        # 各单位换算到内部基准单位 mm 的倍率
POINT_NAMES = ("P1", "P2", "P3", "P4")      # 四个顶点名称，按固定顺序
DEFAULT_NPZ = "records.npz"                 # 默认保存文件名
CALIB_NPZ = r"C:\python\3D_project_3\初始四点坐标.npz"   # 标定第 2 步输出的起始四点坐标


def _to_mm(value, unit):
    """功能：把按 unit 单位表示的长度换算成内部基准单位 mm。
    输入：value：float，待换算数值；unit：str，取值 'mm' 或 'cm'
    输出：float，换算后的 mm 值
    """
    return value * UNIT_TO_MM[unit]


def _from_mm(value, unit):
    """功能：把内部 mm 值换算成按 unit 单位显示的数值。
    输入：value：float，mm 值；unit：str，取值 'mm' 或 'cm'
    输出：float，换算后的显示值
    """
    return value / UNIT_TO_MM[unit]


def _fmt(value):
    """功能：把浮点数格式化成最多 4 位小数的字符串，去掉多余的 0。
    输入：value：float，待格式化数值
    输出：str，格式化结果
    """
    return f"{value:.4f}".rstrip("0").rstrip(".")


class StageApp:
    """平移台坐标记录主窗口：负责界面构建、单位换算、记录计算、表格刷新与 npz 存取。"""

    def __init__(self, root):
        """功能：初始化窗口与内部状态，构建界面。
        输入：root：tk.Tk，主窗口对象
        输出：无
        """
        self.root = root
        root.title("平移台坐标记录软件")
        root.resizable(True, True)

        # 内部状态（坐标统一存 mm）
        self.records = []                    # 记录列表，每条 dict：dx/dy(本次移动量 mm) + points(绝对四点坐标 mm)
        self.unit = tk.StringVar(value="mm") # 当前显示/输入单位
        self.point_vars = []                 # 初始四点坐标输入框变量，[(xvar,yvar), ...]
        self.move_x_sign = tk.StringVar(value="+")  # 本次 x 移动方向 +/-
        self.move_y_sign = tk.StringVar(value="+")  # 本次 y 移动方向 +/-
        self.move_x_val = tk.StringVar(value="0")   # 本次 x 移动距离（当前单位）
        self.move_y_val = tk.StringVar(value="0")   # 本次 y 移动距离（当前单位）

        self._build_ui()
        self.unit.trace_add("write", lambda *a: self._refresh_table())  # 切换单位时自动刷新表格显示
        self._refresh_table()
        self._load_initial_from_calib(silent=True)   # 存在标定 npz 时自动作为起始坐标，不存在则静默跳过

    # ---------- 界面构建 ----------
    def _build_ui(self):
        """功能：构建所有界面控件并绑定回调。
        输入：无
        输出：无
        """
        # 顶部：初始四点坐标设置区
        top = ttk.LabelFrame(self.root, text="1. 初始四点坐标")
        top.pack(fill="x", padx=8, pady=6)
        grid = ttk.Frame(top)
        grid.pack(fill="x", padx=8, pady=4)
        ttk.Label(grid, text="点").grid(row=0, column=0)
        ttk.Label(grid, text="x").grid(row=0, column=1)
        ttk.Label(grid, text="y").grid(row=0, column=2)
        for i, name in enumerate(POINT_NAMES):
            ttk.Label(grid, text=name).grid(row=i + 1, column=0, padx=4, pady=2)
            xv = tk.StringVar(value="0")
            yv = tk.StringVar(value="0")
            ttk.Entry(grid, textvariable=xv, width=10).grid(row=i + 1, column=1, padx=4)
            ttk.Entry(grid, textvariable=yv, width=10).grid(row=i + 1, column=2, padx=4)
            self.point_vars.append((xv, yv))

        # 单位、方向配置
        cfg = ttk.Frame(top)
        cfg.pack(fill="x", padx=8, pady=4)
        ttk.Label(cfg, text="单位").pack(side="left")
        ttk.Combobox(cfg, textvariable=self.unit, values=UNITS, width=5,
                     state="readonly").pack(side="left", padx=(2, 12))
        ttk.Button(cfg, text="设置初始坐标", command=self._set_initial).pack(side="left", padx=12)
        ttk.Button(cfg, text="从标定结果读取初始坐标", command=self._load_initial_from_calib).pack(side="left", padx=12)

        # 中部：移动输入区
        mid = ttk.LabelFrame(self.root, text="2. 输入移动距离")
        mid.pack(fill="x", padx=8, pady=6)
        row = ttk.Frame(mid)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="x 方向").pack(side="left")
        ttk.Combobox(row, textvariable=self.move_x_sign, values=("+", "-"), width=3,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(row, text="距离").pack(side="left")
        ttk.Entry(row, textvariable=self.move_x_val, width=10).pack(side="left", padx=2)
        ttk.Label(row, text="   y 方向").pack(side="left")
        ttk.Combobox(row, textvariable=self.move_y_sign, values=("+", "-"), width=3,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(row, text="距离").pack(side="left")
        ttk.Entry(row, textvariable=self.move_y_val, width=10).pack(side="left", padx=2)
        ttk.Button(row, text="添加移动记录", command=self._add_move).pack(side="left", padx=12)

        # 主体：记录表格
        table_frame = ttk.LabelFrame(self.root, text="3. 记录")
        table_frame.pack(fill="both", expand=True, padx=8, pady=6)
        cols = ("idx", "dx", "dy", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "p4x", "p4y")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
        headers = {"idx": "序号", "dx": "dx", "dy": "dy"}
        for i, name in enumerate(POINT_NAMES):
            headers[f"p{i + 1}x"] = f"{name}.x"
            headers[f"p{i + 1}y"] = f"{name}.y"
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        # 底部：操作按钮
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=6)
        ttk.Button(bottom, text="保存到 npz", command=self._save).pack(side="left", padx=4)
        ttk.Button(bottom, text="加载 npz", command=self._load).pack(side="left", padx=4)
        ttk.Button(bottom, text="删除选中记录", command=self._delete).pack(side="left", padx=4)

    # ---------- 核心逻辑 ----------
    def _set_initial(self):
        """功能：读取四点输入框并按当前单位写入第一条初始记录。
        输入：无
        输出：无
        """
        try:
            pts = []
            for xv, yv in self.point_vars:
                x = _to_mm(float(xv.get()), self.unit.get())
                y = _to_mm(float(yv.get()), self.unit.get())
                pts.append((x, y))
        except ValueError:
            messagebox.showerror("错误", "初始坐标必须是数字")
            return
        # 重置为仅含初始记录
        self.records = [{"dx": 0.0, "dy": 0.0, "points": pts}]
        self._refresh_table()

    def _load_initial_from_calib(self, path=CALIB_NPZ, silent=False):
        """功能：从标定结果 npz 读取四点现实坐标作为初始坐标，按文件单位换算后回填输入框并写入初始记录。
        输入：path：str，标定结果 npz 路径；silent：bool，为 True 时失败不弹窗（用于启动自动加载）
        输出：bool，读取并写入成功返回 True，否则 False
        """
        if not os.path.exists(path):
            if not silent:
                messagebox.showwarning("提示", f"未找到标定结果文件：\n{path}")
            return False
        try:
            data = np.load(path)
            pts = np.asarray(data["points"], np.float64)      # 四点现实坐标 (4,2)
            unit = str(data["unit"]) if "unit" in data else "cm"  # 文件里坐标的单位
            if pts.shape != (4, 2):
                raise ValueError("points 形状应为 (4,2)")
        except Exception:
            if not silent:
                messagebox.showerror("错误", "标定结果文件格式不正确，缺少 points/unit 字段")
            return False
        # 回填输入框（按当前显示单位）
        for (xv, yv), (x, y) in zip(self.point_vars, pts):
            xv.set(_fmt(_from_mm(_to_mm(float(x), unit), self.unit.get())))
            yv.set(_fmt(_from_mm(_to_mm(float(y), unit), self.unit.get())))
        # 写入初始记录（内部统一 mm）
        mm_pts = [(_to_mm(float(x), unit), _to_mm(float(y), unit)) for x, y in pts]
        self.records = [{"dx": 0.0, "dy": 0.0, "points": mm_pts}]
        self._refresh_table()
        return True

    def _add_move(self):
        """功能：按输入的方向与距离，基于最后一条记录算出新四点坐标并追加记录。
        输入：无
        输出：无
        """
        if not self.records:
            messagebox.showwarning("提示", "请先设置初始坐标")
            return
        try:
            dist_x = _to_mm(float(self.move_x_val.get()), self.unit.get())
            dist_y = _to_mm(float(self.move_y_val.get()), self.unit.get())
        except ValueError:
            messagebox.showerror("错误", "移动距离必须是数字")
            return
        sign_x = 1 if self.move_x_sign.get() == "+" else -1   # 本次 x 移动方向符号
        sign_y = 1 if self.move_y_sign.get() == "+" else -1   # 本次 y 移动方向符号
        # 实际坐标增量 = 本次方向 × 距离
        dx = sign_x * dist_x
        dy = sign_y * dist_y
        last = self.records[-1]["points"]                       # 上一条绝对坐标
        pts = [(x + dx, y + dy) for x, y in last]               # 整体平移
        self.records.append({"dx": dx, "dy": dy, "points": pts})
        self._refresh_table()

    def _delete(self):
        """功能：删除表格中选中的某条记录，其余记录不受影响。
        输入：无
        输出：无
        """
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的记录")
            return
        idx = int(self.tree.item(sel[0], "values")[0])          # 选中行对应的记录下标
        if idx == 0:
            # 删除初始记录：清空全部，回到未设置状态
            self.records = []
        else:
            del self.records[idx]
        self._refresh_table()

    def _refresh_table(self):
        """功能：按当前单位把内部记录刷新到表格显示。
        输入：无
        输出：无
        """
        unit = self.unit.get()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, rec in enumerate(self.records):
            vals = [i, _fmt(_from_mm(rec["dx"], unit)), _fmt(_from_mm(rec["dy"], unit))]
            for x, y in rec["points"]:
                vals += [_fmt(_from_mm(x, unit)), _fmt(_from_mm(y, unit))]
            self.tree.insert("", "end", values=vals)

    # ---------- 保存 / 加载 ----------
    def _save(self):
        """功能：把全部记录及单位保存到 npz 文件。
        输入：无
        输出：无
        """
        if not self.records:
            messagebox.showwarning("提示", "没有可保存的记录")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            initialfile=DEFAULT_NPZ,
                                            filetypes=[("NumPy 压缩文件", "*.npz")])
        if not path:
            return
        pts = np.array([r["points"] for r in self.records], dtype=np.float64)  # (N,4,2) mm
        dx = np.array([r["dx"] for r in self.records], dtype=np.float64)
        dy = np.array([r["dy"] for r in self.records], dtype=np.float64)
        np.savez_compressed(path, points=pts, dx=dx, dy=dy, unit=self.unit.get())
        messagebox.showinfo("完成", f"已保存到 {path}")

    def _load(self):
        """功能：从 npz 文件读取记录与配置并恢复界面状态。
        输入：无
        输出：无
        """
        path = filedialog.askopenfilename(filetypes=[("NumPy 压缩文件", "*.npz")])
        if not path:
            return
        try:
            data = np.load(path)
            for key in ("points", "dx", "dy"):
                if key not in data:
                    raise KeyError(key)
        except Exception:
            messagebox.showerror("错误", "文件格式不正确，缺少必要字段")
            return
        pts = data["points"]
        dx = data["dx"]
        dy = data["dy"]
        self.records = [{"dx": float(dx[i]), "dy": float(dy[i]),
                         "points": [tuple(map(float, p)) for p in pts[i]]}
                        for i in range(len(dx))]
        if "unit" in data:
            self.unit.set(str(data["unit"]))
        self._refresh_table()


def main():
    """功能：程序入口，创建主窗口并进入事件循环。
    输入：无
    输出：无
    """
    root = tk.Tk()
    StageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
