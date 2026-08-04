"""摆动腿规划：脚在空中怎么走。

* :mod:`swing_planner.trajectory` —— 四种足端轨迹，并把落地冲击、越障
  能力、关节负担三项指标量化对比。
* :mod:`swing_planner.swing_leg` —— 摆动腿控制器，把轨迹落实成关节指令，
  第一次把里程碑 1 的逆运动学放进控制回路。

分工：里程碑 4 决定"什么时候"，本模块决定"脚怎么过去"，
里程碑 6 决定"脚落在哪"。
"""

from .swing_leg import SwingLegConfig, SwingLegController
from .trajectory import (
    SWING_TRAJECTORIES,
    BezierSwing,
    CycloidSwing,
    QuinticSwing,
    SineHeightSwing,
    SwingTrajectory,
)

__all__ = [
    "SWING_TRAJECTORIES",
    "BezierSwing",
    "CycloidSwing",
    "QuinticSwing",
    "SineHeightSwing",
    "SwingLegConfig",
    "SwingLegController",
    "SwingTrajectory",
]
