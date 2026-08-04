"""状态估计：四足没有 GPS，靠 IMU 与支撑腿把自己定位出来。

三层：

* :mod:`state_estimator.simulation` —— 生成带完整真值的运动数据。真机上
  没有真值，所以验证估计器必须先造一个"什么都知道"的世界。
* :mod:`state_estimator.leg_odometry` —— 纯腿部里程计，手推、纯 NumPy。
  它既是最简单的可用估计器，也是衡量"融合到底带来了什么"的对照组。
* :mod:`state_estimator.eskf` —— 误差状态卡尔曼滤波，融合 IMU 与腿部
  运动学，把足端位置纳入状态。这是工业标准做法。
"""

from .eskf import ErrorStateKF, ESKFConfig, ESKFState, so3_exp
from .leg_odometry import LegOdometry, base_velocity_from_legs, foot_position_in_base
from .simulation import GroundTruth, TrajectoryConfig, generate_trot, simulate_imu

__all__ = [
    "ESKFConfig",
    "ESKFState",
    "ErrorStateKF",
    "GroundTruth",
    "LegOdometry",
    "TrajectoryConfig",
    "base_velocity_from_legs",
    "foot_position_in_base",
    "generate_trot",
    "simulate_imu",
    "so3_exp",
]
