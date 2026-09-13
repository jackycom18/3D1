"""Basler 相机采集程序（tkinter 界面）。

功能：实时预览、单张拍摄、按“前缀+序号”自动命名保存、定时连拍、曝光与增益调节。
运行：python basler_capture.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from basler_camera import DEFAULT_FORMAT, SUPPORTED_FORMATS, BaslerCamera

# 预览区分辨率（像素），图像按比例缩放到该区域内显示
PREVIEW_W, PREVIEW_H = 800, 600

# 预览刷新周期（毫秒），约 33 帧/秒
REFRESH_MS = 30

# 默认保存文件夹：脚本所在目录下的 images
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# 默认文件名前缀
DEFAULT_PREFIX = "img"

# 曝光 / 增益的自动模式枚举值（GenICam 标准）
AUTO_ON, AUTO_OFF = "Continuous", "Off"


class CaptureApp:
    """采集程序主界面。

    属性：
        cam：BaslerCamera 实例
        root：tkinter 主窗口
    """

    def __init__(self, root):
        """功能：构建界面并连接相机。
        输入：root：tk.Tk，tkinter 主窗口
        输出：无
        """
        self.root = root
        self.cam = BaslerCamera()
        self._photo = None                # 当前显示的 PhotoImage，必须保留引用否则被回收
        self._timer_job = None            # 定时拍照的 after 任务 id，None 表示未启动
        self._timer_left = 0              # 定时拍照剩余张数
        self._saved_count = 0             # 本次运行已保存张数
        self._timer_interval = 1.0        # 定时拍照间隔（秒），启动定时后由输入框覆盖

        root.title("Basler 相机采集")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self._open_camera()

    # ---------- 界面搭建 ----------

    def _build_ui(self):
        """功能：创建预览区与右侧控制面板的全部控件。
        输入：无
        输出：无
        """
        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

        # 左侧预览区
        self.preview = tk.Label(self.root, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#202020", fg="#888888", text="等待相机图像…")
        self.preview.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        panel = ttk.Frame(self.root, padding=8)
        panel.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="n")

        # 相机状态
        self.cam_label = ttk.Label(panel, text="相机：未连接", foreground="#b00")
        self.cam_label.pack(anchor="w", pady=(0, 8))

        # 保存设置
        box = ttk.LabelFrame(panel, text="保存设置", padding=6)
        box.pack(fill="x", pady=4)

        ttk.Label(box, text="文件夹").grid(row=0, column=0, sticky="w")
        self.dir_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
        ttk.Entry(box, textvariable=self.dir_var, width=26).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(box, text="浏览", width=6, command=self._browse_dir).grid(row=0, column=2)

        ttk.Label(box, text="前缀").grid(row=1, column=0, sticky="w")
        self.prefix_var = tk.StringVar(value=DEFAULT_PREFIX)
        ttk.Entry(box, textvariable=self.prefix_var, width=26).grid(row=1, column=1, padx=4, pady=2)
        ttk.Button(box, text="应用", width=6, command=self._apply_naming).grid(row=1, column=2)

        ttk.Label(box, text="格式").grid(row=2, column=0, sticky="w")
        self.fmt_var = tk.StringVar(value=DEFAULT_FORMAT)
        fmt_combo = ttk.Combobox(box, textvariable=self.fmt_var, width=24, state="readonly",
                                 values=list(SUPPORTED_FORMATS.keys()))
        fmt_combo.grid(row=2, column=1, columnspan=2, padx=4, pady=2, sticky="w")
        fmt_combo.bind("<<ComboboxSelected>>", lambda event: self._apply_naming())

        # 拍摄
        box = ttk.LabelFrame(panel, text="拍摄", padding=6)
        box.pack(fill="x", pady=4)
        ttk.Button(box, text="拍摄一张", command=self.snap_once).pack(fill="x", pady=2)
        self.saved_label = ttk.Label(box, text="已保存 0 张")
        self.saved_label.pack(anchor="w")

        # 定时拍照
        box = ttk.LabelFrame(panel, text="定时拍照", padding=6)
        box.pack(fill="x", pady=4)

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="间隔(秒)").pack(side="left")
        self.interval_var = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.interval_var, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="张数").pack(side="left")
        self.count_var = tk.StringVar(value="10")
        ttk.Entry(row, textvariable=self.count_var, width=6).pack(side="left", padx=4)

        self.timer_btn = ttk.Button(box, text="开始定时拍照", command=self.toggle_timer)
        self.timer_btn.pack(fill="x", pady=(4, 0))

        # 曝光
        self.expo_auto_var = tk.BooleanVar(value=False)
        self.expo_entry, self.expo_var, self.expo_info = self._build_param_box(
            panel, "曝光", "ExposureTime", "ExposureAuto", self.expo_auto_var)

        # 增益
        self.gain_auto_var = tk.BooleanVar(value=False)
        self.gain_entry, self.gain_var, self.gain_info = self._build_param_box(
            panel, "增益", "Gain", "GainAuto", self.gain_auto_var)

    def _build_param_box(self, parent, title, value_node, auto_node, auto_var):
        """功能：生成一个“数值输入 + 自动复选框 + 应用”的参数调节区。
        输入：parent：父容器；title：str，分组标题；value_node：str，数值节点名；
              auto_node：str，自动模式节点名；auto_var：BooleanVar，自动模式状态
        输出：tuple(Entry, StringVar, Label)，数值输入框、其绑定变量与范围提示标签
        """
        box = ttk.LabelFrame(parent, text=title, padding=6)
        box.pack(fill="x", pady=4)

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="数值").pack(side="left")
        var = tk.StringVar(value="0")
        entry = ttk.Entry(row, textvariable=var, width=12)
        entry.pack(side="left", padx=4)
        ttk.Button(row, text="应用", width=6,
                   command=lambda: self._apply_param(value_node, var, auto_node)).pack(side="left")

        ttk.Checkbutton(box, text="自动", variable=auto_var,
                        command=lambda: self._toggle_auto(auto_node, value_node, var, auto_var, entry)
                        ).pack(anchor="w", pady=(4, 0))
        info = ttk.Label(box, text="", foreground="#666")
        info.pack(anchor="w")
        return entry, var, info

    # ---------- 相机 ----------

    def _open_camera(self):
        """功能：打开相机，成功后读取曝光与增益的当前值与范围。
        输入：无
        输出：无
        """
        info = self.cam.open()
        if not info:
            self.cam_label.config(text="相机：未连接（请检查电源与网线）", foreground="#b00")
            return
        self.cam_label.config(text=f"相机：{info}", foreground="#080")
        self.cam.start()
        self.cam.setup_save(self.dir_var.get(), self.prefix_var.get(), self.fmt_var.get())
        self._sync_param("ExposureTime", "ExposureAuto", self.expo_entry, self.expo_var,
                         self.expo_auto_var, self.expo_info)
        self._sync_param("Gain", "GainAuto", self.gain_entry, self.gain_var,
                         self.gain_auto_var, self.gain_info)
        self._refresh_preview()

    def _sync_param(self, value_node, auto_node, entry, value_var, auto_var, info_label):
        """功能：把相机当前的参数值与自动模式回读到界面控件。
        输入：value_node/auto_node：str，节点名；entry：Entry，数值输入框；
              value_var：StringVar，数值；auto_var：BooleanVar，自动状态；
              info_label：Label，范围提示
        输出：无
        """
        auto = self.cam.get_node_value(auto_node)
        auto_var.set(auto == AUTO_ON)
        value = self.cam.get_node_value(value_node)
        if value is not None:
            value_var.set(f"{value:.0f}" if isinstance(value, (int, float)) else str(value))
        entry.config(state="disabled" if auto_var.get() else "normal")
        rng = self.cam.get_node_range(value_node)
        info_label.config(text="" if rng is None else f"范围 {rng[0]:.0f} ~ {rng[1]:.0f}")

    # ---------- 预览 ----------

    def _refresh_preview(self):
        """功能：定时把最新一帧缩放到预览区显示。
        输入：无
        输出：无
        """
        frame = self.cam.get_latest_frame()
        if frame is not None:
            image = Image.fromarray(frame[:, :, ::-1])          # BGR -> RGB
            image.thumbnail((PREVIEW_W, PREVIEW_H), Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(image)             # 保留引用防止被回收
            self.preview.config(image=self._photo, text="")
        self.root.after(REFRESH_MS, self._refresh_preview)

    # ---------- 保存与拍摄 ----------

    def _browse_dir(self):
        """功能：弹出文件夹选择对话框并更新保存目录。
        输入：无
        输出：无
        """
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get())
        if chosen:
            self.dir_var.set(chosen)
            self._apply_naming()

    def _apply_naming(self):
        """功能：校验保存目录、应用前缀与格式，并重置照片序号。
        输入：无
        输出：无
        """
        save_dir = self.dir_var.get().strip()
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            messagebox.showerror("错误", f"无法创建文件夹：{save_dir}")
            return
        start = self.cam.setup_save(save_dir, self.prefix_var.get().strip(), self.fmt_var.get())
        self.saved_label.config(text=f"已保存 {self._saved_count} 张，下一张序号 {start}")

    def snap_once(self):
        """功能：请求保存一张当前画面。
        输入：无
        输出：无
        """
        if self.cam.camera is None:
            messagebox.showwarning("提示", "相机未连接")
            return
        if not os.path.isdir(self.dir_var.get().strip()):
            self._apply_naming()
        self.cam.request_save(1)
        self._saved_count += 1
        self.saved_label.config(text=f"已保存 {self._saved_count} 张")

    # ---------- 定时拍照 ----------

    def toggle_timer(self):
        """功能：启动或停止定时拍照。
        输入：无
        输出：无
        """
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
            self.timer_btn.config(text="开始定时拍照")
            return
        try:
            interval = float(self.interval_var.get())
            count = int(self.count_var.get())
            if interval <= 0 or count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "间隔需为正数，张数需为正整数")
            return
        if self.cam.camera is None:
            messagebox.showwarning("提示", "相机未连接")
            return
        self._timer_left = count
        self._timer_interval = interval                  # 定时拍照间隔，单位秒
        self.timer_btn.config(text="停止定时拍照")
        self._timer_tick()

    def _timer_tick(self):
        """功能：定时拍照的单个节拍，拍一张后按间隔安排下一拍。
        输入：无
        输出：无
        """
        if self._timer_left <= 0:
            self.toggle_timer()
            return
        self.snap_once()
        self._timer_left -= 1
        self.timer_btn.config(text=f"停止定时拍照（剩 {self._timer_left} 张）")
        self._timer_job = self.root.after(int(self._timer_interval * 1000), self._timer_tick)

    # ---------- 参数调节 ----------

    def _apply_param(self, value_node, value_var, auto_node):
        """功能：把输入框中的数值写入相机参数。
        输入：value_node：str，数值节点名；value_var：StringVar，输入值；auto_node：str，自动节点名
        输出：无
        """
        try:
            value = float(value_var.get())
        except ValueError:
            messagebox.showwarning("提示", "请输入数字")
            return
        if self.cam.get_node_value(auto_node) == AUTO_ON:
            messagebox.showinfo("提示", "请先取消“自动”再手动设置数值")
            return
        if not self.cam.set_node_value(value_node, value):
            messagebox.showwarning("提示", f"相机不支持设置 {value_node}")

    def _toggle_auto(self, auto_node, value_node, value_var, auto_var, entry):
        """功能：切换参数的自动 / 手动模式，切回手动时回读当前值。
        输入：auto_node/value_node：str，节点名；value_var：StringVar，数值；
              auto_var：BooleanVar，自动状态；entry：Entry，数值输入框
        输出：无
        """
        target = AUTO_ON if auto_var.get() else AUTO_OFF
        if not self.cam.set_node_value(auto_node, target):
            messagebox.showwarning("提示", f"相机不支持设置 {auto_node}")
            auto_var.set(False)
            return
        if not auto_var.get():
            value = self.cam.get_node_value(value_node)
            if value is not None:
                value_var.set(f"{value:.0f}")
        entry.config(state="disabled" if auto_var.get() else "normal")

    # ---------- 退出 ----------

    def on_close(self):
        """功能：关闭窗口前停止定时任务、采集线程与相机。
        输入：无
        输出：无
        """
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        self.cam.close()
        self.root.destroy()


def main():
    """功能：程序入口，创建窗口并进入事件循环。
    输入：无
    输出：无
    """
    root = tk.Tk()
    CaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
