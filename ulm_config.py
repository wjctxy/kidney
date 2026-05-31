"""ULM 项目的统一配置。

本文件只保存参数，不参与任何一步算法计算。所有 step 模块都从这里读取
默认路径和默认阈值，避免在不同脚本里写出不一致的接口。
"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
HUMAN_DIR = BASE_DIR / "human_dcm"
MOUSE_DIR = BASE_DIR / "mouse_dcm"

DICOM_NAME = "11.0.dcm"

# step0 预处理默认使用 DICOM 中的超声区域标签；如果标签缺失，使用该 ROI。
DEFAULT_ROI = (602, 0, 1204, 590)  # x0, y0, x1, y1

# 当前 DICOM metadata 中给出的时间和物理尺寸。
DEFAULT_FRAME_TIME_MS = 33.3333333333333
DEFAULT_FPS = 1000.0 / DEFAULT_FRAME_TIME_MS
DEFAULT_PIXEL_SIZE_X = 0.0063787375415282396
DEFAULT_PIXEL_SIZE_Y = 0.0063787375415282396

# 人类链路：时域带通滤波参数。单位为 Hz，当前使用 0.5-8.5 Hz。
HUMAN_BANDPASS_LOW_HZ = 0.5
HUMAN_BANDPASS_HIGH_HZ = 8.5

# 人类链路：二维高斯平滑和局部峰值检测。
GAUSSIAN_SIGMA = 1.0
PEAK_MIN_DISTANCE = 3
PEAK_THRESHOLD_STD = 2.0

# 小鼠链路：SVD 保留中间奇异值。低阈值 SVD 可把 low_rank_cut 调小。
SVD_LOW_RANK_CUT = 2
SVD_HIGH_RANK_CUT = 80

# 小鼠链路：径向对称法定位 patch 半径。
RADIAL_PATCH_RADIUS = 4

# 追踪与结果重建。
MAX_FRAME_DISPLACEMENT_PX = 12.0
MAX_MISSING_FRAMES = 2
MIN_TRACK_LENGTH = 5
SPEED_THRESHOLD = 1.0
SUPER_RES_FACTOR = 4
DIRECTION_TOLERANCE_DEG = 20.0
