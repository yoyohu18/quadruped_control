"""步态定义：用三个参数描述一切四足步态。

一个周期性步态只需要三样东西就能完全确定：

* **周期** :math:`T` —— 走完一个完整循环需要多久；
* **占空比** :math:`D` —— 每条腿有多大比例的时间踩在地上；
* **相位偏移** :math:`\\phi_i` —— 每条腿在循环里的哪个时刻落地。

四足的各种步态之间，差别**全部**在相位偏移上。trot 和 pace 的周期、
占空比可以完全一样，只是配对方式不同：trot 是对角腿同相，pace 是同侧腿
同相。就这一个数字的差别，决定了机器人是稳还是晃。

约定：偏移 :math:`\\phi_i` 是该腿**落地**的相位。于是

.. math::  s_i(t) = \\left(\\frac{t}{T} - \\phi_i\\right) \\bmod 1

    \\text{接触} \\iff s_i(t) < D

这个约定要一次定死 —— 有的文献把偏移定义成离地时刻，符号全反。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinematics import LEGS

__all__ = ["GaitDefinition", "GAITS", "get_gait"]


@dataclass(frozen=True)
class GaitDefinition:
    """一个周期性步态。

    Attributes:
        name: 步态名。
        period: 步态周期，秒。
        duty_factor: 占空比，即支撑相时长占周期的比例，取值 (0, 1]。
            ``1.0`` 表示始终触地（站立）。
        phase_offsets: 每条腿的落地相位，取值 [0, 1)。
        description: 一句话说明这个步态的特点。
    """

    name: str
    period: float
    duty_factor: float
    phase_offsets: dict[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"步态周期必须为正，收到 {self.period}")
        if not 0.0 < self.duty_factor <= 1.0:
            raise ValueError(f"占空比必须在 (0, 1] 内，收到 {self.duty_factor}")
        missing = set(LEGS) - set(self.phase_offsets)
        if missing:
            raise ValueError(f"缺少这些腿的相位偏移：{sorted(missing)}")
        for leg, off in self.phase_offsets.items():
            if not 0.0 <= off < 1.0:
                raise ValueError(f"腿 {leg} 的相位偏移必须在 [0, 1) 内，收到 {off}")

    @property
    def stance_duration(self) -> float:
        """单条腿的支撑相时长，秒。"""
        return self.period * self.duty_factor

    @property
    def swing_duration(self) -> float:
        """单条腿的摆动相时长，秒。"""
        return self.period * (1.0 - self.duty_factor)

    @property
    def offsets_array(self) -> np.ndarray:
        """按 :data:`kinematics.LEGS` 顺序排列的相位偏移，形状 (4,)。"""
        return np.array([self.phase_offsets[leg] for leg in LEGS])

    def is_statically_stable_capable(self) -> bool:
        """占空比是否足以保证任意时刻至少三条腿触地。

        三条腿才能构成支撑三角形，才谈得上静态稳定。四条腿均匀分布时，
        需要 :math:`D \\ge 0.75`。这是 crawl 与 trot 的分水岭。
        """
        return self.duty_factor >= 0.75


#: 标准步态库。相位偏移按 (FL, FR, RL, RR) 给出。
GAITS: dict[str, GaitDefinition] = {
    "stand": GaitDefinition(
        name="stand",
        period=1.0,
        duty_factor=1.0,
        phase_offsets={"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
        description="四脚always触地，用于站立与力分配测试。",
    ),
    "crawl": GaitDefinition(
        name="crawl",
        period=1.0,
        duty_factor=0.75,
        phase_offsets={"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75},
        description="一次只抬一条腿，恒有三条腿支撑，静态稳定但慢。",
    ),
    "trot": GaitDefinition(
        name="trot",
        period=0.4,
        duty_factor=0.5,
        phase_offsets={"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5},
        description="对角腿成对交替，四足最常用的中速步态。",
    ),
    "pace": GaitDefinition(
        name="pace",
        period=0.4,
        duty_factor=0.5,
        phase_offsets={"FL": 0.0, "RL": 0.0, "FR": 0.5, "RR": 0.5},
        description="同侧腿成对交替，横滚方向不稳定，骆驼与部分犬类采用。",
    ),
    "bound": GaitDefinition(
        name="bound",
        period=0.32,
        duty_factor=0.4,
        phase_offsets={"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5},
        description="前后腿成对交替，俯仰方向剧烈起伏，兔子与猎豹的高速步态。",
    ),
    "pronk": GaitDefinition(
        name="pronk",
        period=0.3,
        duty_factor=0.4,
        phase_offsets={"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
        description="四腿完全同步，存在腾空相，最考验起跳与落地控制。",
    ),
    "gallop": GaitDefinition(
        name="gallop",
        period=0.3,
        duty_factor=0.3,
        phase_offsets={"FL": 0.0, "FR": 0.1, "RL": 0.5, "RR": 0.6},
        description="四条腿依次落地，速度最高，接触序列不对称。",
    ),
}


def get_gait(name: str) -> GaitDefinition:
    """按名字取出标准步态。

    Args:
        name: 步态名，见 :data:`GAITS`。

    Returns:
        对应的 :class:`GaitDefinition`。

    Raises:
        KeyError: 步态名不存在。
    """
    if name not in GAITS:
        raise KeyError(f"未知步态 '{name}'，可选：{sorted(GAITS)}")
    return GAITS[name]
