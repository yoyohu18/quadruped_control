"""全身控制：把 MPC 的接触力落实成 12 个关节力矩。

* :mod:`whole_body_controller.tasks` —— 加速度任务的统一表示，以及
  运算空间 PD 与 SO(3) 姿态误差。
* :mod:`whole_body_controller.wbc` —— 基于完整 18 自由度动力学的加权 QP。

它是里程碑 2 那个方程被完整用上的地方，也是"S 的前 6 列全为零"最终
变成 QP 等式约束的地方。
"""

from .tasks import Task, orientation_error, pd_acceleration
from .wbc import WBCConfig, WBCResult, WholeBodyController

__all__ = [
    "Task",
    "WBCConfig",
    "WBCResult",
    "WholeBodyController",
    "orientation_error",
    "pd_acceleration",
]
