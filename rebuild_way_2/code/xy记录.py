"""xy 记录程序：设定 4 个初始顶点坐标后，按方向与位移量逐次/批量移动（四顶点整体平移）并记录，
保存为 points (N,4,2) 的 npz。

运行：python xy记录.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np


DEFAULT_NPZ = "xy.npz"    # 默认保存文件名
POINT_LABELS = ("P1(左上)", "P2(右上)", "P3(右下)", "P4(左下)")  # 四顶点顺序，与图像拐点排序一致


class XYRecorderApp:
    """xy 记录主窗口：设定 4 个初始顶点后，按方向与位移量记录四顶点坐标序列，保存为 (N,4,2) npz。"""

    def __init__(self, root):
        """功能：初始化窗口与内部状态，构建界面。
        输入：root：tk.Tk，主窗口对象
        输出：无
        """
        self.root = root
        self.points = []                          # 已累积四顶点坐标序列，每项 [(x,y),(x,y),(x,y),(x,y)]，单位 mm
        self.init_vars = []                       # 4 个顶点的 (x_var, y_var) 输入框变量，顺序同 POINT_LABELS
        for _ in range(4):
            self.init_vars.append((tk.StringVar(value=""), tk.StringVar(value="")))
        self.dx_var = tk.StringVar(value="")      # 每次 x 位移量(mm)输入框
        self.dy_var = tk.StringVar(value="")      # 每次 y 位移量(mm)输入框
        self.sx_var = tk.StringVar(value="+")     # x 方向符号，+/-
        self.sy_var = tk.StringVar(value="+")     # y 方向符号，+/-
        self.n_var = tk.StringVar(value="")       # 自动步进循环次数输入框
        root.title("xy 记录（四顶点）")
        root.resizable(True, True)
        self._build_ui()
        self._refresh_list()

    # ---------- 界面 ----------
    def _build_ui(self):
        """功能：构建所有界面控件并绑定回调。
        输入：无
        输出：无
        """
        init_box = ttk.LabelFrame(self.root, text="1. 初始四顶点坐标（顺序=左上→右上→右下→左下，开始记录前设定）")
        init_box.pack(fill="x", padx=8, pady=6)
        for i, label in enumerate(POINT_LABELS):
            row = ttk.Frame(init_box)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=label, width=12).pack(side="left")
            ttk.Label(row, text="x (mm)").pack(side="left")
            ttk.Entry(row, textvariable=self.init_vars[i][0], width=12).pack(side="left", padx=4)
            ttk.Label(row, text="y (mm)").pack(side="left")
            ttk.Entry(row, textvariable=self.init_vars[i][1], width=12).pack(side="left", padx=4)
        ttk.Button(init_box, text="设定为初始四顶点坐标", command=self._set_initial).pack(padx=8, pady=6)

        move_box = ttk.LabelFrame(self.root, text="2. 移动设置（方向 + 位移量，四顶点整体平移）")
        move_box.pack(fill="x", padx=8, pady=6)
        move_row = ttk.Frame(move_box)
        move_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(move_row, text="x 方向").pack(side="left")
        ttk.Combobox(move_row, textvariable=self.sx_var, values=("+", "-"), width=3,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(move_row, text="位移(mm)").pack(side="left")
        ttk.Entry(move_row, textvariable=self.dx_var, width=10).pack(side="left", padx=2)
        ttk.Label(move_row, text="   y 方向").pack(side="left")
        ttk.Combobox(move_row, textvariable=self.sy_var, values=("+", "-"), width=3,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(move_row, text="位移(mm)").pack(side="left")
        ttk.Entry(move_row, textvariable=self.dy_var, width=10).pack(side="left", padx=2)
        ttk.Button(move_row, text="移动并记录（单次）", command=self._step_once).pack(side="left", padx=12)

        auto_row = ttk.Frame(move_box)
        auto_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(auto_row, text="循环次数").pack(side="left")
        ttk.Entry(auto_row, textvariable=self.n_var, width=8).pack(side="left", padx=6)
        ttk.Button(auto_row, text="自动步进（多次）", command=self._auto_step).pack(side="left", padx=6)

        list_box = ttk.LabelFrame(self.root, text="3. 四顶点坐标记录（最后一行即当前坐标）")
        list_box.pack(fill="both", expand=True, padx=8, pady=6)
        cols = ("idx", "p1", "p2", "p3", "p4")
        self.tree = ttk.Treeview(list_box, columns=cols, show="headings", height=14)
        self.tree.heading("idx", text="序号")
        for col, label in zip(("p1", "p2", "p3", "p4"), POINT_LABELS):
            self.tree.heading(col, text=label)
        self.tree.column("idx", width=60, anchor="center")
        for col in ("p1", "p2", "p3", "p4"):
            self.tree.column(col, width=150, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_box, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=6)
        ttk.Button(bottom, text="删除选中", command=self._delete).pack(side="left", padx=4)
        ttk.Button(bottom, text="清空序列", command=self._clear).pack(side="left", padx=4)
        ttk.Button(bottom, text="保存到 npz", command=self._save).pack(side="left", padx=4)

    # ---------- 逻辑 ----------
    def _set_initial(self):
        """功能：读取 4 个顶点输入框，写入序列第一条记录（初始四顶点坐标）。
        输入：无
        输出：无
        """
        try:
            quad = [(float(xv.get()), float(yv.get())) for xv, yv in self.init_vars]
        except ValueError:
            messagebox.showwarning("提示", "四顶点坐标必须是数字")
            return
        if self.points and not messagebox.askyesno("确认", "已有记录，设定初始坐标将清空并重新开始，是否继续？"):
            return
        self.points = [quad]
        self._refresh_list()

    def _read_step(self):
        """功能：读取 x/y 方向与位移量，计算本次带符号的坐标增量。
        输入：无
        输出：tuple(dx, dy) 带符号增量；输入非法时返回 None
        """
        try:
            dist_x = float(self.dx_var.get())
            dist_y = float(self.dy_var.get())
        except ValueError:
            messagebox.showwarning("提示", "位移量必须是数字")
            return None
        sign_x = 1 if self.sx_var.get() == "+" else -1
        sign_y = 1 if self.sy_var.get() == "+" else -1
        return sign_x * dist_x, sign_y * dist_y

    def _step_once(self):
        """功能：按当前方向与位移量单次移动，四顶点整体平移后追加记录。
        输入：无
        输出：无
        """
        if not self.points:
            messagebox.showwarning("提示", "请先设定初始四顶点坐标")
            return
        step = self._read_step()
        if step is None:
            return
        dx, dy = step
        self.points.append([(px + dx, py + dy) for px, py in self.points[-1]])
        self._refresh_list()

    def _auto_step(self):
        """功能：按当前方向与位移量自动循环 N 次，每次四顶点在上一次基础上整体平移并记录。
        输入：无
        输出：无
        """
        if not self.points:
            messagebox.showwarning("提示", "请先设定初始四顶点坐标")
            return
        step = self._read_step()
        if step is None:
            return
        try:
            n = int(self.n_var.get())
        except ValueError:
            messagebox.showwarning("提示", "循环次数必须是整数")
            return
        if n < 0:
            messagebox.showwarning("提示", "循环次数不能为负数")
            return
        dx, dy = step
        quad = self.points[-1]
        for _ in range(n):
            quad = [(px + dx, py + dy) for px, py in quad]
            self.points.append(quad)
        self._refresh_list()

    def _delete(self):
        """功能：删除列表中选中的一条记录。
        输入：无
        输出：无
        """
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的记录")
            return
        idx = int(self.tree.item(sel[0], "values")[0])
        del self.points[idx]
        self._refresh_list()

    def _clear(self):
        """功能：清空整个坐标序列。
        输入：无
        输出：无
        """
        if self.points and not messagebox.askyesno("确认", "确定清空整个坐标序列吗？"):
            return
        self.points = []
        self._refresh_list()

    def _refresh_list(self):
        """功能：把内部四顶点坐标序列刷新到表格显示。
        输入：无
        输出：无
        """
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, quad in enumerate(self.points):
            cells = tuple(f"{px:.3f}, {py:.3f}" for px, py in quad)
            self.tree.insert("", "end", values=(i,) + cells)

    def _save(self):
        """功能：把四顶点坐标序列保存为 points (N,4,2) 的 npz 文件。
        输入：无
        输出：无
        """
        if not self.points:
            messagebox.showwarning("提示", "没有可保存的坐标")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            initialfile=DEFAULT_NPZ,
                                            filetypes=[("NumPy 压缩文件", "*.npz")])
        if not path:
            return
        arr = np.array(self.points, dtype=np.float64)   # (N,4,2)
        np.savez_compressed(path, points=arr)
        messagebox.showinfo("完成", f"已保存 {len(self.points)} 组四顶点坐标 (N,4,2) 到\n{path}")


def main():
    """功能：程序入口，创建窗口并进入事件循环。
    输入：无
    输出：无
    """
    root = tk.Tk()
    XYRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
