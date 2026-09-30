"""环形结构光 3D 点云系统主程序：统一启动器，7 个功能模块各自弹窗。

运行：py -3.8 main.py
"""

import importlib.util
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- 常量 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 项目根目录
CORE_DIR = os.path.join(BASE_DIR, "核心代码_标定1")      # 核心算法目录

MODULES = (                                               # 7 个模块：(标题, 说明)
    ("采集图像", "查找/连接相机并拍照保存"),
    ("位移台记录", "记录 xy 移动量与四点坐标"),
    ("z 轴记录", "手动输入 z 深度累积保存"),
    ("图形处理", "提取四点交点像素坐标"),
    ("标定", "鱼眼标定 + 第 2/3/4 步"),
    ("重建", "用 H + 像素 + z 生成 PLY"),
    ("显示点云", "open3d 打开 PLY"),
)

# 配色
_BG, _CARD, _HOVER = "#1f2430", "#2b3242", "#384154"
_FG, _ACCENT, _MUTED = "#e6e9ef", "#4c9aff", "#9aa3b2"


def _load(name, path):
    """功能：按绝对路径动态加载模块，规避中文文件名 import 限制。
    输入：name：str 模块唯一名；path：str .py 绝对路径
    输出：module 对象
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tools():
    """功能：加载并返回 文件工具 模块。
    输入：无
    输出：module
    """
    return _load("文件工具", os.path.join(CORE_DIR, "文件工具.py"))


class Launcher:
    """主启动器：顶部工作目录选择 + 7 个模块卡片。"""

    def __init__(self, root):
        self.root = root
        self.tools = _tools()
        self.work_dir = tk.StringVar(value=BASE_DIR)

        root.title("环形结构光 3D 点云系统")
        root.geometry("720x520")
        root.configure(bg=_BG)
        ttk.Style(root).theme_use("clam")
        self._build_ui()

    def _build_ui(self):
        """功能：构建标题栏、工作目录区与卡片网格。
        输入：无
        输出：无
        """
        hdr = tk.Frame(self.root, bg=_BG)
        hdr.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(hdr, text="环形结构光 3D 点云系统", bg=_BG, fg=_FG,
                 font=("Microsoft YaHei", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text="采集 → 记录 → 处理 → 标定 → 重建 → 显示", bg=_BG, fg=_MUTED,
                 font=("Microsoft YaHei", 9)).pack(anchor="w")

        bar = tk.Frame(self.root, bg=_BG)
        bar.pack(fill="x", padx=24, pady=10)
        tk.Label(bar, text="工作目录", bg=_BG, fg=_MUTED).pack(side="left")
        tk.Entry(bar, textvariable=self.work_dir, bg=_CARD, fg=_FG, insertbackground=_FG,
                 relief="flat").pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        tk.Button(bar, text="浏览", command=self._browse, bg=_CARD, fg=_FG, relief="flat",
                  activebackground=_HOVER).pack(side="left", padx=4)

        grid = tk.Frame(self.root, bg=_BG)
        grid.pack(fill="both", expand=True, padx=24, pady=8)
        for i, (title, desc) in enumerate(MODULES):
            card = tk.Frame(grid, bg=_CARD, cursor="hand2")
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            tk.Label(card, text=title, bg=_CARD, fg=_ACCENT,
                     font=("Microsoft YaHei", 13, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
            tk.Label(card, text=desc, bg=_CARD, fg=_MUTED,
                     font=("Microsoft YaHei", 9)).pack(anchor="w", padx=16, pady=(0, 14))
            card.bind("<Button-1>", lambda e, t=title: self._open(t))
            card.bind("<Enter>", lambda e, c=card: c.config(bg=_HOVER))
            card.bind("<Leave>", lambda e, c=card: c.config(bg=_CARD))
            grid.rowconfigure(i // 2, weight=1)
            grid.columnconfigure(i % 2, weight=1)

    def _browse(self):
        chosen = filedialog.askdirectory(initialdir=self.work_dir.get() or None)
        if chosen:
            self.work_dir.set(chosen)

    def _open(self, title):
        """功能：按标题分发到对应模块。
        输入：title：str 模块标题
        输出：无
        """
        {
            "采集图像": self._open_capture,
            "位移台记录": self._open_stage,
            "z 轴记录": self._open_z,
            "图形处理": self._open_image_proc,
            "标定": self._open_calib,
            "重建": self._open_rebuild,
            "显示点云": self._open_view,
        }[title]()

    # ---------- 复用现有 GUI ----------
    def _open_capture(self):
        """功能：打开采集窗口（复用 CaptureApp，pypylon/PIL 缺失时提示）。"""
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, "takephoto"))
            from basler_capture import CaptureApp
        except Exception as exc:
            messagebox.showwarning("提示", f"采集模块依赖 pypylon/PIL，当前环境不可用：\n{exc}")
            return
        CaptureApp(tk.Toplevel(self.root))

    def _open_stage(self):
        """功能：打开位移台记录窗口（复用 StageApp）。"""
        stage = _load("平移台软件", os.path.join(BASE_DIR, "平移台软件", "平移台软件.py"))
        stage.StageApp(tk.Toplevel(self.root))

    def _open_z(self):
        """功能：打开 z 轴记录窗口（复用 ZRecorderApp）。"""
        zrec = _load("z轴记录", os.path.join(CORE_DIR, "z轴记录.py"))
        zrec.ZRecorderApp(tk.Toplevel(self.root), work_dir=self.work_dir.get())

    # ---------- 算法模块封装 ----------
    def _open_image_proc(self):
        """功能：打开图形处理窗口（调用 image_processing 提取四点交点）。"""
        ip = _load("图像处理", os.path.join(CORE_DIR, "图像处理_四点提取.py"))
        win = tk.Toplevel(self.root)
        win.title("图形处理")
        win.configure(bg=_BG)
        self.tools.ensure_data_dirs(self.work_dir.get())

        tk.Label(win, text="照片文件夹", bg=_BG, fg=_MUTED).pack(anchor="w", padx=12, pady=(12, 0))
        row = tk.Frame(win, bg=_BG)
        row.pack(fill="x", padx=12)
        img_dir = tk.StringVar(value="")
        tk.Entry(row, textvariable=img_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(row, text="浏览", command=lambda: img_dir.set(
            filedialog.askdirectory(initialdir=self.work_dir.get() or None)) or "",
            bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def run():
            if not img_dir.get().strip():
                messagebox.showwarning("提示", "请先选择照片文件夹")
                return
            out = self.tools.next_path(
                os.path.join(self.work_dir.get(), "data", "centroids"), "centroids", ".npz")
            messagebox.showinfo("提示", "将逐图弹出窗口，按任意键确认每张图，最后保存四点坐标")
            try:
                ip.image_processing(path=img_dir.get().strip(), out_npz=out)
                messagebox.showinfo("完成", f"已保存到\n{out}")
            except Exception as exc:
                messagebox.showerror("错误", str(exc))

        tk.Button(win, text="执行提取", command=run, bg=_ACCENT, fg="#fff", relief="flat",
                  padx=16, pady=6).pack(pady=16)

    def _open_calib(self):
        """功能：打开标定窗口（鱼眼标定 + 第 2/3/4 步）。"""
        fish = _load("鱼眼标定", os.path.join(CORE_DIR, "鱼眼标定.py"))
        s2 = _load("标定第二步", os.path.join(CORE_DIR, "标定第二步.py"))
        s3 = _load("标定第三步", os.path.join(CORE_DIR, "标定第三步.py"))
        s4 = _load("标定第四步", os.path.join(CORE_DIR, "标定第四步.py"))

        win = tk.Toplevel(self.root)
        win.title("标定")
        win.configure(bg=_BG)
        wd = self.work_dir.get()
        self.tools.ensure_data_dirs(wd)
        data = os.path.join(wd, "data")
        status = tk.Label(win, text="就绪", bg=_BG, fg=_MUTED)
        status.pack(anchor="w", padx=12, pady=(12, 0))

        def latest(sub, base, ext):
            return self.tools.latest_file(os.path.join(data, sub), base, ext)

        def wrap(fn):
            """功能：执行算法；失败时把异常显示到状态栏（成功提示由各步骤自行设置）。"""
            def run():
                try:
                    fn()
                except Exception as exc:
                    status.config(text=f"失败：{exc}", fg="#e05555")
            return run

        # 步骤1：鱼眼标定
        f1 = tk.LabelFrame(win, text="第一步 · 鱼眼标定（棋盘格）", bg=_BG, fg=_FG)
        f1.pack(fill="x", padx=12, pady=8)
        cb_dir = tk.StringVar(value="")
        r1 = tk.Frame(f1, bg=_BG)
        r1.pack(fill="x", padx=8, pady=6)
        tk.Entry(r1, textvariable=cb_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r1, text="浏览", command=lambda: cb_dir.set(
            filedialog.askdirectory() or ""), bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def step1():
            out = self.tools.next_path(os.path.join(data, "calib_result"), "calib_result", ".npz")
            fish.calibrate(path=cb_dir.get().strip(), out_npz=out)
            status.config(text=f"完成：鱼眼标定 → {out}", fg="#3cb371")
        tk.Button(f1, text="执行", command=wrap(step1), bg=_ACCENT,
                  fg="#fff", relief="flat").pack(anchor="e", padx=8, pady=6)

        # 步骤2：初始四点坐标（k1/L/W）
        f2 = tk.LabelFrame(win, text="第二步 · 初始四点坐标", bg=_BG, fg=_FG)
        f2.pack(fill="x", padx=12, pady=8)
        r2 = tk.Frame(f2, bg=_BG)
        r2.pack(fill="x", padx=8, pady=6)
        k1 = tk.StringVar(value="1.0"); lm = tk.StringVar(value="160.0"); wm = tk.StringVar(value="160.0")
        for label, var in (("k1(长宽比)", k1), ("L(mm)", lm), ("W(mm)", wm)):
            tk.Label(r2, text=label, bg=_BG, fg=_MUTED).pack(side="left", padx=4)
            tk.Entry(r2, textvariable=var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left")

        def step2():
            cb = latest("calib_result", "calib_result", ".npz")
            ce = latest("centroids", "centroids", ".npz")
            out = self.tools.next_path(os.path.join(data, "初始四点坐标"), "初始四点坐标", ".npz")
            s2.calibrate_step2(calib_npz=cb, centroids_npz=ce, k1=float(k1.get()),
                               L_mm=float(lm.get()), W_mm=float(wm.get()), out_npz=out)
            status.config(text=f"完成：初始四点坐标 → {out}", fg="#3cb371")
        tk.Button(f2, text="执行", command=wrap(step2), bg=_ACCENT,
                  fg="#fff", relief="flat").pack(anchor="e", padx=8, pady=6)

        # 步骤3：ang_move（x1/y2）
        f3 = tk.LabelFrame(win, text="第三步 · 位移台夹角 ang_move", bg=_BG, fg=_FG)
        f3.pack(fill="x", padx=12, pady=8)
        r3 = tk.Frame(f3, bg=_BG)
        r3.pack(fill="x", padx=8, pady=6)
        x1 = tk.StringVar(value="10.0"); y2 = tk.StringVar(value="10.0")
        for label, var in (("x1(mm)", x1), ("y2(mm)", y2)):
            tk.Label(r3, text=label, bg=_BG, fg=_MUTED).pack(side="left", padx=4)
            tk.Entry(r3, textvariable=var, width=8, bg=_CARD, fg=_FG, relief="flat").pack(side="left")

        def step3():
            cb = latest("calib_result", "calib_result", ".npz")
            ce = latest("centroids", "centroids", ".npz")
            init = latest("初始四点坐标", "初始四点坐标", ".npz")
            out = self.tools.next_path(os.path.join(data, "ang_move"), "ang_move", ".npz")
            s3.calibrate_step3(calib_npz=cb, centroids_npz=ce, init_npz=init,
                               x1=float(x1.get()), y2=float(y2.get()), out_npz=out)
            status.config(text=f"完成：ang_move → {out}", fg="#3cb371")
        tk.Button(f3, text="执行", command=wrap(step3), bg=_ACCENT,
                  fg="#fff", relief="flat").pack(anchor="e", padx=8, pady=6)

        # 步骤4：H
        f4 = tk.LabelFrame(win, text="第四步 · 单应矩阵 H", bg=_BG, fg=_FG)
        f4.pack(fill="x", padx=12, pady=8)
        r4 = tk.Frame(f4, bg=_BG)
        r4.pack(fill="x", padx=8, pady=6)
        rec = tk.StringVar(value=latest("records", "records", ".npz") or "")
        tk.Label(r4, text="records", bg=_BG, fg=_MUTED).pack(side="left", padx=4)
        tk.Entry(r4, textvariable=rec, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r4, text="浏览", command=lambda: rec.set(
            filedialog.askopenfilename(filetypes=[("npz", "*.npz")]) or ""),
            bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def step4():
            cb = latest("calib_result", "calib_result", ".npz")
            ce = latest("centroids", "centroids", ".npz")
            init = latest("初始四点坐标", "初始四点坐标", ".npz")
            am = latest("ang_move", "ang_move", ".npz")
            out = self.tools.next_path(os.path.join(data, "H"), "H", ".npz")
            s4.calibrate_step4(calib_npz=cb, centroids_npz=ce, init_npz=init,
                               ang_move_npz=am, records_npz=rec.get().strip(), out_npz=out)
            status.config(text=f"完成：H → {out}", fg="#3cb371")
        tk.Button(f4, text="执行", command=wrap(step4), bg=_ACCENT,
                  fg="#fff", relief="flat").pack(anchor="e", padx=8, pady=6)

    def _open_rebuild(self):
        """功能：打开重建窗口（调用 reconstruct 生成 PLY）。"""
        recon = _load("点云重建", os.path.join(CORE_DIR, "点云重建.py"))
        win = tk.Toplevel(self.root)
        win.title("重建")
        win.configure(bg=_BG)
        wd = self.work_dir.get()
        self.tools.ensure_data_dirs(wd)
        data = os.path.join(wd, "data")
        status = tk.Label(win, text="就绪", bg=_BG, fg=_MUTED)
        status.pack(anchor="w", padx=12, pady=(12, 0))

        # points 文件（激光线像素坐标）可手动指定，默认取 data/points 最新
        tk.Label(win, text="points（激光线像素坐标）", bg=_BG, fg=_MUTED).pack(anchor="w", padx=12, pady=(8, 0))
        rp = tk.Frame(win, bg=_BG)
        rp.pack(fill="x", padx=12)
        pts = tk.StringVar(value=self.tools.latest_file(
            os.path.join(data, "points"), "points", ".npz") or "")
        tk.Entry(rp, textvariable=pts, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(rp, text="浏览", command=lambda: pts.set(
            filedialog.askopenfilename(filetypes=[("npz", "*.npz")]) or ""),
            bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def run():
            cb = self.tools.latest_file(os.path.join(data, "calib_result"), "calib_result", ".npz")
            h = self.tools.latest_file(os.path.join(data, "H"), "H", ".npz")
            z = self.tools.latest_file(os.path.join(data, "z"), "z", ".npz")
            if not all([cb, h, pts.get().strip(), z]):
                status.config(text="缺少输入：请确认 calib_result / H / points / z 均已生成", fg="#e05555")
                return
            out = self.tools.next_path(os.path.join(data, "pointcloud"), "pointcloud", ".ply")
            try:
                recon.reconstruct(calib_npz=cb, h_npz=h, points_npz=pts.get().strip(),
                                  z_npz=z, out_ply=out)
                status.config(text=f"完成：点云 → {out}", fg="#3cb371")
            except Exception as exc:
                status.config(text=f"失败：{exc}", fg="#e05555")

        tk.Button(win, text="执行重建", command=run, bg=_ACCENT, fg="#fff", relief="flat",
                  padx=16, pady=6).pack(pady=16)

    def _open_view(self):
        """功能：打开显示点云窗口（open3d 打开 PLY）。"""
        win = tk.Toplevel(self.root)
        win.title("显示点云")
        win.configure(bg=_BG)
        data = os.path.join(self.work_dir.get(), "data")
        ply = tk.StringVar(value=self.tools.latest_file(
            os.path.join(data, "pointcloud"), "pointcloud", ".ply") or "")
        tk.Label(win, text="PLY 文件", bg=_BG, fg=_MUTED).pack(anchor="w", padx=12, pady=(12, 0))
        r = tk.Frame(win, bg=_BG)
        r.pack(fill="x", padx=12)
        tk.Entry(r, textvariable=ply, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r, text="浏览", command=lambda: ply.set(
            filedialog.askopenfilename(filetypes=[("ply", "*.ply")]) or ""),
            bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def show():
            if not os.path.isfile(ply.get().strip()):
                messagebox.showerror("错误", "请先选择存在的 PLY 文件")
                return
            try:
                import open3d as o3d
            except ImportError:
                messagebox.showerror("错误", "未安装 open3d")
                return
            pcd = o3d.io.read_point_cloud(ply.get().strip())
            if not pcd.has_points():
                messagebox.showerror("错误", "点云为空或读取失败")
                return
            o3d.visualization.draw_geometries([pcd], window_name="点云显示")

        tk.Button(win, text="显示", command=show, bg=_ACCENT, fg="#fff", relief="flat",
                  padx=16, pady=6).pack(pady=16)


def main():
    """功能：程序入口，创建主窗口并进入事件循环。
    输入：无
    输出：无
    """
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
