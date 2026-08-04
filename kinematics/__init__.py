"""运动学：正运动学、逆运动学与雅可比。

刻意分成两层：

* :mod:`kinematics.leg_kinematics` —— 闭式解，单腿，纯 NumPy。
  单次调用微秒级。真实控制器跑的就是这一层。
* :mod:`kinematics.robot_model` —— 通用，由 URDF 驱动，基于 Pinocchio。
  适用于任意机器人、任意坐标系，是校验闭式解的基准真值。

:mod:`kinematics.go2` 针对宇树 Go2 把两者粘合起来。
"""

from .go2 import GO2_URDF, HIP_OFFSETS, LEG_GEOMETRY, STANDING_JOINT_ANGLES, load_go2
from .leg_kinematics import LegGeometry, forward_kinematics, inverse_kinematics, leg_jacobian
from .robot_model import (
    ISAAC_JOINT_ORDER,
    LEGS,
    PINOCCHIO_JOINT_ORDER,
    QuadrupedModel,
    isaac_to_pinocchio,
    pinocchio_to_isaac,
)

__all__ = [
    "GO2_URDF",
    "HIP_OFFSETS",
    "ISAAC_JOINT_ORDER",
    "LEGS",
    "LEG_GEOMETRY",
    "LegGeometry",
    "PINOCCHIO_JOINT_ORDER",
    "QuadrupedModel",
    "STANDING_JOINT_ANGLES",
    "forward_kinematics",
    "inverse_kinematics",
    "isaac_to_pinocchio",
    "leg_jacobian",
    "load_go2",
    "pinocchio_to_isaac",
]
