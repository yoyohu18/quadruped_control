"""落脚点规划：脚该落在哪里。

三块：

* :mod:`footstep_planner.lipm` —— 线性倒立摆与捕获点，动态平衡的数学基础。
  它解释了里程碑 4 那个尴尬结论：trot 静态稳定裕度恒为负，机器人却能走。
* :mod:`footstep_planner.raibert` —— Raibert 启发式与基于捕获点的放置策略，
  并揭示前者其实是后者的一个欠调近似。
* :mod:`footstep_planner.planner` —— 整合步态相位、转弯、工作空间限制。

分工：M4 决定"什么时候"，M5 决定"脚怎么过去"，本模块决定"落在哪"。
"""

from .lipm import LIPMParams, capture_point, lipm_step, simulate_lipm, time_to_boundary
from .planner import FootstepPlanner, FootstepPlannerConfig
from .raibert import (
    capture_point_footstep,
    deadbeat_feedback_gain,
    exact_footstep,
    exact_stride_coefficient,
    optimal_feedback_gain,
    raibert_footstep,
    steady_state_velocity_ratio,
)

__all__ = [
    "FootstepPlanner",
    "FootstepPlannerConfig",
    "LIPMParams",
    "capture_point",
    "capture_point_footstep",
    "deadbeat_feedback_gain",
    "exact_footstep",
    "exact_stride_coefficient",
    "lipm_step",
    "optimal_feedback_gain",
    "raibert_footstep",
    "simulate_lipm",
    "steady_state_velocity_ratio",
    "time_to_boundary",
]
