"""Go2 专属常数与工厂函数。

所有与具体机器人绑定的东西都放在这里，这样 ``leg_kinematics`` 保持为纯几何
库，``robot_model`` 保持与 URDF 无关。要换成 A1、B2 或自研机器人，只需再写
一个和本文件同样结构的文件。

这里所有数值都直接读自 ``robot_description/go2/go2.urdf``（宇树 Go2），
并由 ``tests/test_kinematics.py`` 反向校验 —— 绝不允许它们悄悄漂移。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .leg_kinematics import LegGeometry
from .robot_model import LEGS, QuadrupedModel

__all__ = [
    "GO2_URDF",
    "ABAD_OFFSET_Y",
    "THIGH_LENGTH",
    "CALF_LENGTH",
    "HIP_OFFSETS",
    "LEG_GEOMETRY",
    "STANDING_JOINT_ANGLES",
    "load_go2",
]

GO2_URDF = Path(__file__).resolve().parents[1] / "robot_description" / "go2" / "go2.urdf"

ABAD_OFFSET_Y = 0.0955  # 侧摆轴 -> 髋俯仰轴，沿 y
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

#: 各侧摆轴在躯干系下的位置。
HIP_OFFSETS: dict[str, np.ndarray] = {
    "FL": np.array([0.1934, 0.0465, 0.0]),
    "FR": np.array([0.1934, -0.0465, 0.0]),
    "RL": np.array([-0.1934, 0.0465, 0.0]),
    "RR": np.array([-0.1934, -0.0465, 0.0]),
}

#: 每条腿的带符号几何参数；``l0`` 的符号把右侧腿镜像过来。
LEG_GEOMETRY: dict[str, LegGeometry] = {
    leg: LegGeometry(
        l0=ABAD_OFFSET_Y * (1.0 if leg.endswith("L") else -1.0),
        l1=THIGH_LENGTH,
        l2=CALF_LENGTH,
    )
    for leg in LEGS
}

#: 标称站立姿态，按 Pinocchio 关节顺序，躯干高度约 0.31 m。
STANDING_JOINT_ANGLES = np.array(
    [
        0.0, 0.8, -1.5,  # FL
        0.0, 0.8, -1.5,  # FR
        0.0, 0.8, -1.5,  # RL
        0.0, 0.8, -1.5,  # RR
    ]
)


def load_go2(floating_base: bool = False) -> QuadrupedModel:
    """构建 Go2 的 :class:`~kinematics.robot_model.QuadrupedModel`。

    Args:
        floating_base: 是否在躯干处加一个自由飞行关节。

    Returns:
        已注册好四个足端坐标系、可直接使用的模型。
    """
    return QuadrupedModel(GO2_URDF, floating_base=floating_base)
