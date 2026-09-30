"""Basler 相机封装：打开设备 / 读写曝光增益 / 后台连续采集 / 按序保存图片。

界面层只需要调用本模块的公开方法，不直接接触 pypylon。
"""

import os
import threading

from PIL import Image

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True                           # pypylon 是否可用；缺失时相机功能降级为不可用
    # 显示用像素格式：把相机原始格式统一转成 BGR8，便于 OpenCV / PIL 处理
    DISPLAY_PIXEL_FORMAT = pylon.PixelType_BGR8packed
except ImportError:
    pylon = None
    PYLON_AVAILABLE = False
    DISPLAY_PIXEL_FORMAT = None

# 可选的保存格式：界面显示名 -> 文件扩展名
SUPPORTED_FORMATS = {
    "PNG（无损）": ".png",
    "BMP（无损）": ".bmp",
    "TIFF（无损）": ".tiff",
    "JPG（有损，体积小）": ".jpg",
}

# 默认保存格式
DEFAULT_FORMAT = "PNG（无损）"

# 关闭自动曝光/增益后使用的固定起点值（标定要求亮度稳定，用户可通过界面再调整）
DEFAULT_EXPOSURE_US = 10000.0   # 曝光时间，单位微秒
DEFAULT_GAIN_DB = 0.0           # 增益，单位 dB，标定优先低噪点


class BaslerCamera:
    """单台 Basler 相机的采集封装。

    属性：
        camera：pylon.InstantCamera 实例，未打开时为 None
        latest_frame：numpy 数组 (H, W, 3)，最近一帧 BGR 图像，可能为 None
    """

    def __init__(self):
        """功能：初始化对象，创建格式转换器，不连接硬件。
        输入：无
        输出：无
        """
        self.camera = None                     # 相机句柄，None 表示未打开
        self.latest_frame = None               # 最近一帧 BGR 图，供界面读取
        self._lock = threading.Lock()          # 保护 latest_frame / _pending_save 的互斥锁
        self._grabbing = False                 # 采集线程运行标志
        self._thread = None                    # 采集线程对象
        self._converter = None
        if PYLON_AVAILABLE:
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = DISPLAY_PIXEL_FORMAT
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._pending_save = 0                 # 待保存帧数，界面点一次拍摄加 1，采集线程取一帧减 1
        self._save_dir = ""                    # 当前保存文件夹
        self._prefix = ""                      # 当前文件名前缀
        self._ext = SUPPORTED_FORMATS[DEFAULT_FORMAT]   # 当前保存格式的扩展名
        self._index = 1                        # 下一个要用的照片序号

    # ---------- 设备连接 ----------

    @staticmethod
    def find_devices():
        """功能：枚举当前连接的 Basler 相机。
        输入：无
        输出：list[str]，每台相机的型号(序列号)，无设备或 pypylon 缺失时返回空列表
        """
        if not PYLON_AVAILABLE:
            return []
        try:
            infos = pylon.TlFactory.GetInstance().EnumerateDevices()
            return [f"{d.GetModelName()}({d.GetSerialNumber()})" for d in infos]
        except Exception:
            return []

    def open(self):
        """功能：打开第一台可用相机并设为连续自由采集模式。
        输入：无
        输出：str，成功时返回“型号(序列号)”，失败或 pypylon 缺失时返回空字符串
        """
        if not PYLON_AVAILABLE:
            self.camera = None
            return ""
        # 已打开则直接返回设备信息：重复点“连接”时设备已被本进程占用，
        # 再走 CreateFirstDevice 会失败并把 camera 覆盖成 None，出现“能出图却显示断开”
        if self.camera is not None:
            try:
                if self.camera.IsOpen():
                    return self.device_info()
            except Exception:
                pass
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            # 标定场景：关闭自动曝光/自动增益，使用固定手动值，保证亮度稳定可复现
            for auto_node in ("ExposureAuto", "GainAuto"):
                try:
                    getattr(self.camera, auto_node).SetValue("Off")
                except Exception:
                    pass
            # 关闭自动后给固定曝光/增益起点，避免相机默认曝光过小导致画面全黑
            try:
                self.camera.ExposureTime.SetValue(DEFAULT_EXPOSURE_US)
            except Exception:
                pass
            try:
                self.camera.Gain.SetValue(DEFAULT_GAIN_DB)
            except Exception:
                pass
            # 采集线程来不及取图时只保留最新帧，避免延迟累积
            self.camera.MaxNumBuffer = 5
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            self.camera = None
            return ""
        return self.device_info()

    def device_info(self):
        """功能：返回当前相机“型号(序列号)”，读取失败时返回空字符串。
        输入：无
        输出：str
        """
        if self.camera is None:
            return ""
        try:
            info = self.camera.GetDeviceInfo()
            return f"{info.GetModelName()}({info.GetSerialNumber()})"
        except Exception:
            return ""

    def close(self):
        """功能：停止采集线程并关闭相机。
        输入：无
        输出：无
        """
        self._grabbing = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.Close()
            except Exception:
                pass
            self.camera = None

    def start(self):
        """功能：启动后台采集线程（需先 open 成功）。
        输入：无
        输出：无
        """
        if self.camera is None or self._grabbing:
            return
        self._grabbing = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    # ---------- 采集线程 ----------

    def _grab_loop(self):
        """功能：采集线程主循环，取帧后按需保存并更新 latest_frame。
        超时未出图或单帧异常都不会终止循环，保证预览持续可用。
        输入：无
        输出：无
        """
        camera = self.camera            # 固定本循环使用的相机句柄，避免与 close() 抢引用
        if camera is None:
            return
        fails = 0                       # 连续取帧异常次数，用于判断相机是否掉线
        while self._grabbing:
            try:
                result = camera.RetrieveResult(200, pylon.TimeoutHandling_Return)
            except Exception:
                fails += 1
                if fails > 50:          # 连续约 10 秒取不到帧，判定相机异常并退出
                    break
                continue
            if result is None or not result.IsValid():   # 超时无数据，属正常情况
                continue
            fails = 0
            try:
                if not result.GrabSucceeded():
                    continue
                frame = self._converter.Convert(result).GetArray()
                with self._lock:
                    self.latest_frame = frame
                    need_save = self._pending_save > 0
                    if need_save:
                        self._pending_save -= 1
                if need_save:
                    self._save_frame(frame)
            except Exception:
                continue                # 单帧转换或保存异常不应拖垮整个采集
            finally:
                result.Release()

    def _save_frame(self, frame):
        """功能：把一帧 BGR 图像按“前缀+序号”命名写入保存文件夹。
        输入：frame：numpy 数组 (H, W, 3)，BGR 图像
        输出：str，保存成功的完整路径；文件夹无效或写盘失败时返回空字符串
        """
        if not self._save_dir or not os.path.isdir(self._save_dir):
            return ""
        with self._lock:
            index = self._index
            self._index += 1
        path = os.path.join(self._save_dir, f"{self._prefix}{index}{self._ext}")
        try:
            Image.fromarray(frame[:, :, ::-1]).save(path)   # 转成 RGB 后用 PIL 保存，兼容中文路径
        except Exception:
            return ""
        return path

    def get_latest_frame(self):
        """功能：取最近一帧的副本，供界面显示。
        输入：无
        输出：numpy 数组 (H, W, 3) 或 None（尚无图像）
        """
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame

    # ---------- 拍摄与命名 ----------

    def setup_save(self, save_dir, prefix, fmt):
        """功能：设置保存文件夹、文件名前缀与图片格式，并把序号接到已有文件之后。
        输入：save_dir：str，保存文件夹；prefix：str，文件名前缀；
              fmt：str，格式显示名，需是 SUPPORTED_FORMATS 的键
        输出：int，本次拍摄将使用的起始序号
        """
        self._save_dir = save_dir
        self._prefix = prefix
        self._ext = SUPPORTED_FORMATS.get(fmt, SUPPORTED_FORMATS[DEFAULT_FORMAT])
        self._index = self._next_index(save_dir, prefix, self._ext)
        return self._index

    @staticmethod
    def _next_index(save_dir, prefix, ext):
        """功能：扫描文件夹中已有的“前缀+数字+扩展名”文件，返回最大序号加一。
        输入：save_dir：str，保存文件夹；prefix：str，文件名前缀；ext：str，扩展名如 ".png"
        输出：int，起始序号；文件夹不存在或无匹配文件时返回 1
        """
        if not os.path.isdir(save_dir):
            return 1
        biggest = 0
        for name in os.listdir(save_dir):
            if not name.startswith(prefix) or not name.lower().endswith(ext):
                continue
            stem = name[: -len(ext)][len(prefix):]
            if stem.isdigit():
                biggest = max(biggest, int(stem))
        return biggest + 1

    def request_save(self, count=1):
        """功能：请求采集线程在接下来的若干帧上执行保存。
        输入：count：int，需要保存的帧数，默认 1
        输出：无
        """
        with self._lock:
            self._pending_save += count

    # ---------- 相机参数 ----------

    def get_node_value(self, name):
        """功能：读取相机某个参数节点的当前值。
        输入：name：str，GenICam 节点名，如 "ExposureTime"
        输出：节点值，节点不存在或不可读时返回 None
        """
        try:
            return getattr(self.camera, name).GetValue()
        except Exception:
            return None

    def get_node_range(self, name):
        """功能：读取相机某个参数节点的取值范围。
        输入：name：str，GenICam 节点名
        输出：tuple(min, max)，节点不存在或不是数值节点时返回 None
        """
        try:
            node = getattr(self.camera, name)
            return (node.GetMin(), node.GetMax())
        except Exception:
            return None

    def set_node_value(self, name, value):
        """功能：写入相机参数节点。
        输入：name：str，GenICam 节点名；value：目标值，数值或字符串枚举
        输出：bool，写入成功 True，失败 False
        """
        try:
            getattr(self.camera, name).SetValue(value)
            return True
        except Exception:
            return False
