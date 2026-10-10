"""标定2 主程序：标定（鱼眼标定→标定2）与重建（z 轴记录→重建点云）两个步骤化模块。

标定：①棋盘格文件夹→鱼眼标定→calib_result.npz  ②标定框文件夹唯一图→人工点 8 锚点→H.npz
重建：①z 轴记录保存 z.npz  ②重建图片文件夹 + calib/H/z → ply → open3d 显示
运行：py -3.8 main.py
"""

import glob
import importlib.util
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- 常量 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                              # 本文件所在目录（rebuild_way_2/code）
WORK_DIR = os.path.dirname(BASE_DIR)                                               # 工作目录（rebuild_way_2），data/ 在其下
CORE1_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "核心代码_标定1")  # 标定1 目录，含 文件工具.py

# 把标定1 目录加入搜索路径，供 z轴记录.py 内部 `import 文件工具` 解析
if CORE1_DIR not in sys.path:
    sys.path.insert(0, CORE1_DIR)

# 本目录下复用的处理模块路径
_IMGPROC_PATH = os.path.join(BASE_DIR, "图像处理_四点提取.py")            # 标定2 复用
_IMGPROC_REBUILD_PATH = os.path.join(BASE_DIR, "图像处理_重建.py")        # 重建复用

# 配色
_BG, _CARD, _HOVER = "#1f2430", "#2b3242", "#384154"
_FG, _ACCENT, _MUTED = "#e6e9ef", "#4c9aff", "#9aa3b2"


def _load(name, path):
    """功能：按绝对路径动态加载中文文件名模块。
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
    return _load("文件工具", os.path.join(CORE1_DIR, "文件工具.py"))


class Launcher:
    """主启动器：工作目录选择 + 标定/重建两个模块卡片。"""

    def __init__(self, root):
        self.root = root
        self.tools = _tools()
        self.work_dir = tk.StringVar(value=WORK_DIR)       # 工作目录，产物存到其下 data/

        root.title("结构光标定与重建")
        root.geometry("560x360")
        root.configure(bg=_BG)
        ttk.Style(root).theme_use("clam")
        self._build_ui()

    def _build_ui(self):
        """功能：构建标题、工作目录栏与两个模块卡片。
        输入：无
        输出：无
        """
        hdr = tk.Frame(self.root, bg=_BG)
        hdr.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(hdr, text="结构光标定与重建", bg=_BG, fg=_FG,
                 font=("Microsoft YaHei", 18, "bold")).pack(anchor="w")
        tk.Label(hdr, text="标定：鱼眼标定 → 标定2（H）    重建：z 轴记录 → 点云", bg=_BG, fg=_MUTED,
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
        for i, (title, desc, fn) in enumerate((
                ("标定", "鱼眼标定 + 标定2，产出 calib_result 与 H", self._open_calib),
                ("重建", "z 轴记录 + 重建点云，open3d 显示", self._open_rebuild),
        )):
            card = tk.Frame(grid, bg=_CARD, cursor="hand2")
            card.grid(row=0, column=i, sticky="nsew", padx=6, pady=6)
            tk.Label(card, text=title, bg=_CARD, fg=_ACCENT,
                     font=("Microsoft YaHei", 14, "bold")).pack(anchor="w", padx=16, pady=(16, 2))
            tk.Label(card, text=desc, bg=_CARD, fg=_MUTED,
                     font=("Microsoft YaHei", 9)).pack(anchor="w", padx=16, pady=(0, 16))
            card.bind("<Button-1>", lambda e, f=fn: f())
            card.bind("<Enter>", lambda e, c=card: c.config(bg=_HOVER))
            card.bind("<Leave>", lambda e, c=card: c.config(bg=_CARD))
            grid.rowconfigure(0, weight=1)
            grid.columnconfigure(i, weight=1)

    def _browse(self):
        """功能：选择工作目录。
        输入：无
        输出：无
        """
        chosen = filedialog.askdirectory(initialdir=self.work_dir.get() or None)
        if chosen:
            self.work_dir.set(chosen)

    def _open_calib(self):
        """功能：打开标定模块两步窗口（鱼眼标定→标定2），第二步锁定。
        输入：无
        输出：无
        """
        fish = _load("鱼眼标定", os.path.join(BASE_DIR, "鱼眼标定.py"))
        cal2 = _load("标定", os.path.join(BASE_DIR, "标定.py"))

        win = tk.Toplevel(self.root)
        win.title("标定")
        win.geometry("640x360")
        win.configure(bg=_BG)
        wd = self.work_dir.get()
        self.tools.ensure_data_dirs(wd)
        data = os.path.join(wd, "data")
        status = tk.Label(win, text="就绪", bg=_BG, fg=_MUTED)
        status.pack(anchor="w", padx=12, pady=(12, 0))

        def latest(sub, base, ext):
            return self.tools.latest_file(os.path.join(data, sub), base, ext)

        def wrap(fn):
            """功能：执行步骤，异常显示到状态栏。"""
            def run():
                try:
                    fn()
                except Exception as exc:
                    status.config(text=f"失败：{exc}", fg="#e05555")
            return run

        # 第一步：鱼眼标定
        f1 = tk.LabelFrame(win, text="第一步 · 鱼眼标定（棋盘格文件夹）", bg=_BG, fg=_FG)
        f1.pack(fill="x", padx=12, pady=8)
        cb_dir = tk.StringVar(value="")
        r1 = tk.Frame(f1, bg=_BG)
        r1.pack(fill="x", padx=8, pady=6)
        tk.Entry(r1, textvariable=cb_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r1, text="浏览", command=lambda: cb_dir.set(
            filedialog.askdirectory() or ""), bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def step1():
            if not cb_dir.get().strip():
                status.config(text="请先选择棋盘格文件夹", fg="#e05555")
                return
            out = self.tools.next_path(os.path.join(data, "neican"), "calib_result", ".npz")
            fish.calibrate(path=cb_dir.get().strip(), out_npz=out)
            status.config(text=f"完成：鱼眼标定 → {out}", fg="#3cb371")
            btn2.config(state="normal")                          # 解锁第二步

        tk.Button(f1, text="执行", command=wrap(step1), bg=_ACCENT, fg="#fff", relief="flat",
                  padx=16, pady=4).pack(anchor="e", padx=8, pady=6)

        # 第二步：标定2
        f2 = tk.LabelFrame(win, text="第二步 · 标定2（标定框文件夹，仅放一张图）", bg=_BG, fg=_FG)
        f2.pack(fill="x", padx=12, pady=8)
        box_dir = tk.StringVar(value="")
        r2 = tk.Frame(f2, bg=_BG)
        r2.pack(fill="x", padx=8, pady=6)
        tk.Entry(r2, textvariable=box_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r2, text="浏览", command=lambda: box_dir.set(
            filedialog.askdirectory() or ""), bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def step2():
            d = box_dir.get().strip()
            if not d:
                status.config(text="请先选择标定框文件夹", fg="#e05555")
                return
            imgs = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
            if not imgs:
                status.config(text="标定框文件夹内没有图片", fg="#e05555")
                return
            cb = latest("neican", "calib_result", ".npz")
            if not cb:
                status.config(text="缺少 calib_result，请先执行第一步", fg="#e05555")
                return
            out = self.tools.next_path(os.path.join(data, "H"), "H", ".npz")
            H = cal2.calibrate2(image_dir=d, calib_npz=cb, out_npz=out, imgproc_path=_IMGPROC_PATH)
            if H is not None:
                status.config(text=f"完成：单应矩阵 H → {out}", fg="#3cb371")
            else:
                status.config(text="标定2 未完成（锚点不足或未采到角点）", fg="#e05555")

        btn2 = tk.Button(f2, text="执行", command=wrap(step2), bg=_ACCENT, fg="#fff", relief="flat",
                         padx=16, pady=4, state="disabled")
        btn2.pack(anchor="e", padx=8, pady=6)

    def _open_rebuild(self):
        """功能：打开重建模块两步窗口（z 轴记录→重建点云并显示）。
        输入：无
        输出：无
        """
        recon = _load("重建", os.path.join(BASE_DIR, "重建.py"))
        zrec = _load("z轴记录", os.path.join(BASE_DIR, "z轴记录.py"))

        win = tk.Toplevel(self.root)
        win.title("重建")
        win.geometry("640x360")
        win.configure(bg=_BG)
        wd = self.work_dir.get()
        self.tools.ensure_data_dirs(wd)
        data = os.path.join(wd, "data")
        status = tk.Label(win, text="就绪", bg=_BG, fg=_MUTED)
        status.pack(anchor="w", padx=12, pady=(12, 0))

        def latest(sub, base, ext):
            return self.tools.latest_file(os.path.join(data, sub), base, ext)

        # 第一步：z 轴记录
        f1 = tk.LabelFrame(win, text="第一步 · z 轴记录", bg=_BG, fg=_FG)
        f1.pack(fill="x", padx=12, pady=8)
        tk.Label(f1, text="记录 z 坐标序列并保存到 data/z/", bg=_BG, fg=_MUTED).pack(
            anchor="w", padx=8, pady=6)
        tk.Button(f1, text="打开 z 轴记录", command=lambda: zrec.ZRecorderApp(tk.Toplevel(win), work_dir=wd),
                  bg=_ACCENT, fg="#fff", relief="flat", padx=16, pady=4).pack(anchor="e", padx=8, pady=6)

        # 第二步：重建点云
        f2 = tk.LabelFrame(win, text="第二步 · 重建点云", bg=_BG, fg=_FG)
        f2.pack(fill="x", padx=12, pady=8)
        img_dir = tk.StringVar(value="")
        r2 = tk.Frame(f2, bg=_BG)
        r2.pack(fill="x", padx=8, pady=6)
        tk.Label(r2, text="重建图片文件夹", bg=_BG, fg=_MUTED).pack(side="left", padx=4)
        tk.Entry(r2, textvariable=img_dir, bg=_CARD, fg=_FG, relief="flat").pack(
            side="left", fill="x", expand=True)
        tk.Button(r2, text="浏览", command=lambda: img_dir.set(
            filedialog.askdirectory() or ""), bg=_CARD, fg=_FG, relief="flat").pack(side="left", padx=4)

        def run():
            if not img_dir.get().strip():
                status.config(text="请先选择重建图片文件夹", fg="#e05555")
                return
            cb = latest("neican", "calib_result", ".npz")
            h = latest("H", "H", ".npz")
            z = latest("z", "z", ".npz")
            missing = []
            if not cb:
                missing.append("calib_result（先执行标定第一步）")
            if not h:
                missing.append("H（先执行标定第二步）")
            if not z:
                missing.append("z（先执行 z 轴记录）")
            if missing:
                status.config(text="缺少前置产物：" + "、".join(missing), fg="#e05555")
                return
            out = self.tools.next_path(os.path.join(data, "pointcloud"), "pointcloud", ".ply")
            try:
                ply = recon.reconstruct(image_dir=img_dir.get().strip(),
                                        h_npz=h, z_npz=z, out_ply=out, proc_path=_IMGPROC_REBUILD_PATH)
            except Exception as exc:
                status.config(text=f"失败：{exc}", fg="#e05555")
                return
            if not ply:
                status.config(text="重建未生成点云", fg="#e05555")
                return
            status.config(text=f"完成：点云 → {out}", fg="#3cb371")
            self._show_ply(ply)

        tk.Button(f2, text="执行重建并显示", command=run, bg=_ACCENT, fg="#fff", relief="flat",
                  padx=16, pady=4).pack(anchor="e", padx=8, pady=6)

    def _show_ply(self, ply):
        """功能：用 open3d 弹出窗口显示点云。
        输入：ply：str，ply 文件路径
        输出：无
        """
        try:
            import open3d as o3d
        except ImportError:
            messagebox.showerror("错误", "未安装 open3d")
            return
        pcd = o3d.io.read_point_cloud(ply)
        if not pcd.has_points():
            messagebox.showerror("错误", "点云为空或读取失败")
            return
        o3d.visualization.draw_geometries([pcd], window_name="点云显示")


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
