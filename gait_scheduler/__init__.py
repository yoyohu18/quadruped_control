"""步态调度：整个控制栈共同的时间基准。

* :mod:`gait_scheduler.gait` —— 步态定义与标准步态库。周期、占空比、
  相位偏移三个参数即可描述一切周期性四足步态。
* :mod:`gait_scheduler.scheduler` —— 相位调度器。回答"现在谁踩地"、
  "摆动到百分之几"、"还有多久落地"、"未来 N 步的接触序列"。
* :mod:`gait_scheduler.stability` —— 支撑多边形与静态稳定裕度，
  解释步态选择背后的物理权衡。
"""

from .gait import GAITS, GaitDefinition, get_gait
from .scheduler import GaitScheduler
from .stability import (
    convex_hull_2d,
    point_in_polygon,
    static_stability_margin,
    support_polygon,
)

__all__ = [
    "GAITS",
    "GaitDefinition",
    "GaitScheduler",
    "convex_hull_2d",
    "get_gait",
    "point_in_polygon",
    "static_stability_margin",
    "support_polygon",
]
