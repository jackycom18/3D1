"""z 轴记录程序：设定初始 z 坐标后，按偏移量逐次/批量位移并累积真实深度序列，
保存为 z.npz（存入 data/z/）。

运行：python z轴记录.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# 保证同目录的 文件工具 可被导入（无论以何种方式运行或加载本文件）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import 文件工具


class ZRecorderApp:
    """z 轴记录主窗口：设定初始坐标后，按偏移量记录真实深度序列，保存到 data/z/zN.npz。"""

    def __init__(self, root, work_dir=None):
        """功能：初始化窗口与内部状态，构建界面。
        输入：root：tkinter 窗口对象（Tk 或 Toplevel）；work_dir：str 或 None，工作目录
        输出：无
        """
        self.root = root
        self.work_dir = work_dir              # 工作目录，None 时需用户先选择
        self.z_values = []                    # 已累积的真实深度序列（单位 mm）
        self.dir_var = tk.StringVar(value=work_dir or "")   # 工作目录输入框
        self.z0_var = tk.StringVar(value="")  # 初始 z 轴坐标输入框
        self.d_var = tk.StringVar(value="")   # 位移量（偏移量）输入框
        self.n_var = tk.StringVar(value="")   # 自动步进循环次数输入框

        root.title("z 轴记录")
        root.resizable(True, True)
        self._build_ui()
        self._refresh_list()

    # ---------- 界面 ----------
    def _build_ui(self):
        """功能：构建所有界面控件并绑定回调。
        输入：无
        输出：无
        """
        # 1. 工作目录
        dir_box = ttk.LabelFrame(self.root, text="1. 工作目录（data/z 将建在此目录下）")
        dir_box.pack(fill="x", padx=8, pady=6)
        dir_row = ttk.Frame(dir_box)
        dir_row.pack(fill="x", padx=8, pady=4)
        ttk.Entry(dir_row, textvariable=self.dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(dir_row, text="浏览", command=self._choose_work_dir).pack(side="left", padx=6)

        # 2. 初始 z 轴坐标（开始记录前设定）
        init_box = ttk.LabelFrame(self.root, text="2. 初始 z 轴坐标（真实深度，开始记录前设定）")
        init_box.pack(fill="x", padx=8, pady=6)
        init_row = ttk.Frame(init_box)
        init_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(init_row, text="初始坐标 (mm)").pack(side="left")
        ttk.Entry(init_row, textvariable=self.z0_var, width=12).pack(side="left", padx=6)
        ttk.Button(init_row, text="设定为初始坐标", command=self._set_initial).pack(side="left", padx=6)
        ttk.Label(init_row, text="（设定后作为序列第一条基准记录）", foreground="gray").pack(side="left", padx=6)

        # 3. 记录位移（偏移量 = 每次 z 轴移动距离）
        step_box = ttk.LabelFrame(self.root, text="3. 记录位移（偏移量 = 每次 z 轴移动距离）")
        step_box.pack(fill="x", padx=8, pady=6)
        step_row = ttk.Frame(step_box)
        step_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(step_row, text="位移量 (mm)").pack(side="left")
        ttk.Entry(step_row, textvariable=self.d_var, width=12).pack(side="left", padx=6)
        ttk.Button(step_row, text="位移并记录（单次）", command=self._step_once).pack(side="left", padx=6)
        ttk.Label(step_row, text="循环次数").pack(side="left", padx=(16, 0))
        ttk.Entry(step_row, textvariable=self.n_var, width=8).pack(side="left", padx=6)
        ttk.Button(step_row, text="自动步进（多次）", command=self._auto_step).pack(side="left", padx=6)

        # 4. 已累积的 z 序列
        list_box = ttk.LabelFrame(self.root, text="4. 已累积的 z 序列（最后一行即当前 z 坐标）")
        list_box.pack(fill="both", expand=True, padx=8, pady=6)
        self.tree = ttk.Treeview(list_box, columns=("idx", "z"), show="headings", height=14)
        self.tree.heading("idx", text="序号")
        self.tree.heading("z", text="z (mm)")
        self.tree.column("idx", width=80, anchor="center")
        self.tree.column("z", width=160, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_box, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        # 底部操作
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=6)
        ttk.Button(bottom, text="删除选中", command=self._delete_z).pack(side="left", padx=4)
        ttk.Button(bottom, text="清空序列", command=self._clear).pack(side="left", padx=4)
        ttk.Button(bottom, text="保存到 data/z", command=self._save).pack(side="left", padx=4)

    # ---------- 逻辑 ----------
    def _choose_work_dir(self):
        """功能：弹出目录选择框，更新工作目录。
        输入：无
        输出：无
        """
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get() or None)
        if chosen:
            self.dir_var.set(chosen)
            self.work_dir = chosen

    def _set_initial(self):
        """功能：设定初始 z 轴坐标，作为序列第一条基准记录。
        输入：无
        输出：无
        """
        try:
            z0 = float(self.z0_var.get())
        except ValueError:
            messagebox.showwarning("提示", "初始坐标必须是数字")
            return
        if self.z_values and not messagebox.askyesno("确认", "已有记录，设定初始坐标将清空并重新开始，是否继续？"):
            return
        self.z_values = [z0]
        self._refresh_list()

    def _step_once(self):
        """功能：手动位移一次：当前深度加上偏移量，追加记录并显示。
        输入：无
        输出：无
        """
        if not self.z_values:
            messagebox.showwarning("提示", "请先设定初始 z 轴坐标")
            return
        try:
            d = float(self.d_var.get())
        except ValueError:
            messagebox.showwarning("提示", "位移量必须是数字")
            return
        self.z_values.append(self.z_values[-1] + d)
        self._refresh_list()

    def _auto_step(self):
        """功能：按固定位移量自动循环 N 次，每次在上一次深度基础上累加并记录。
        输入：无
        输出：无
        """
        if not self.z_values:
            messagebox.showwarning("提示", "请先设定初始 z 轴坐标")
            return
        try:
            d = float(self.d_var.get())
        except ValueError:
            messagebox.showwarning("提示", "位移量必须是数字")
            return
        try:
            n = int(self.n_var.get())
        except ValueError:
            messagebox.showwarning("提示", "循环次数必须是整数")
            return
        if n < 0:
            messagebox.showwarning("提示", "循环次数不能为负数")
            return
        current = self.z_values[-1]
        for _ in range(n):
            current += d
            self.z_values.append(current)
        self._refresh_list()

    def _delete_z(self):
        """功能：删除列表中选中的 z 值。
        输入：无
        输出：无
        """
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的记录")
            return
        idx = int(self.tree.item(sel[0], "values")[0])
        del self.z_values[idx]
        self._refresh_list()

    def _clear(self):
        """功能：清空整个 z 序列。
        输入：无
        输出：无
        """
        if self.z_values and not messagebox.askyesno("确认", "确定清空整个 z 序列吗？"):
            return
        self.z_values = []
        self._refresh_list()

    def _refresh_list(self):
        """功能：把内部 z 序列刷新到表格显示。
        输入：无
        输出：无
        """
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, z in enumerate(self.z_values):
            self.tree.insert("", "end", values=(i, f"{z:.4f}"))

    def _save(self):
        """功能：把 z 序列保存到 data/z/zN.npz（自动编号）。
        输入：无
        输出：无
        """
        if not self.z_values:
            messagebox.showwarning("提示", "没有可保存的 z 值")
            return
        if not self.dir_var.get().strip():
            messagebox.showwarning("提示", "请先选择工作目录")
            return
        self.work_dir = self.dir_var.get().strip()
        z_dir = os.path.join(self.work_dir, "data", "z")
        path = 文件工具.next_path(z_dir, "z", ".npz")
        np.savez_compressed(path, z=np.array(self.z_values, dtype=np.float64))
        messagebox.showinfo("完成", f"已保存 {len(self.z_values)} 个 z 值到\n{path}")


def main():
    """功能：程序入口，创建窗口并进入事件循环。
    输入：无
    输出：无
    """
    root = tk.Tk()
    ZRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
