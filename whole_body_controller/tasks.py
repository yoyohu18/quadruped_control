"""全身控制的任务定义：把"我想让机器人怎么动"写成加速度目标。

WBC 的输入不是"力矩"，而是一组**任务**。每个任务说的是"某个量的加速度
应该是多少"，形式统一：

.. math::  J_{\\text{task}}\\,a + \\dot J_{\\text{task}}\\,v = a_{\\text{des}}

于是所有任务都变成对广义加速度 :math:`a` 的**线性**约束或代价项。
这是 WBC 能写成 QP 的原因，和里程碑 7 凸 MPC 的思路完全一致 —— **只要
把非线性都塞进"已知量"，剩下的就是线性代数。**

任务的加速度目标怎么来？用一个 PD 律把位置/姿态误差转成加速度：

.. math::  a_{\\text{des}} = \\ddot x_{\\text{ref}}
           + K_d(\\dot x_{\\text{ref}} - \\dot x) + K_p(x_{\\text{ref}} - x)

这就是"运算空间控制"（operational space control）的标准做法：**PD 不直接
产生力矩，而是产生期望加速度，再由动力学翻译成力矩。** 好处是增益的物理
含义清晰（$K_p$ 的单位是 1/s²），且与机器人的惯量解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Task", "pd_acceleration", "orientation_error"]


@dataclass
class Task:
    """一个加速度任务。

    Attributes:
        name: 任务名，仅用于诊断。
        jacobian: 任务雅可比 :math:`J_{\\text{task}}`，形状 (m, nv)。
        drift: 漂移项 :math:`\\dot J_{\\text{task}} v`，形状 (m,)。
        target: 期望任务加速度 :math:`a_{\\text{des}}`，形状 (m,)。
        weight: 该任务在代价里的权重，标量或形状 (m,) 的对角元。
    """

    name: str
    jacobian: np.ndarray
    drift: np.ndarray
    target: np.ndarray
    weight: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        J = np.atleast_2d(np.asarray(self.jacobian, dtype=float))
        m = J.shape[0]
        self.jacobian = J
        self.drift = np.asarray(self.drift, dtype=float).reshape(m)
        self.target = np.asarray(self.target, dtype=float).reshape(m)
        w = np.asarray(self.weight, dtype=float)
        if w.ndim == 0:
            w = np.full(m, float(w))
        if w.shape != (m,):
            raise ValueError(f"任务 '{self.name}' 的权重维度应为 {m}，收到 {w.shape}")
        if np.any(w < 0.0):
            raise ValueError(f"任务 '{self.name}' 的权重不能为负")
        self.weight = w

    @property
    def dim(self) -> int:
        """任务维度。"""
        return self.jacobian.shape[0]

    def residual(self, acceleration: np.ndarray) -> np.ndarray:
        """给定广义加速度，返回该任务的残差 :math:`Ja + \\dot Jv - a_{des}`。"""
        a = np.asarray(acceleration, dtype=float)
        return self.jacobian @ a + self.drift - self.target

    def cost(self, acceleration: np.ndarray) -> float:
        """该任务的加权平方残差。"""
        r = self.residual(acceleration)
        return float(r @ (self.weight * r))


def pd_acceleration(
    position_error: np.ndarray,
    velocity_error: np.ndarray,
    kp: float | np.ndarray,
    kd: float | np.ndarray,
    feedforward: np.ndarray | None = None,
) -> np.ndarray:
    """把位置/速度误差转成期望加速度。

    .. math::  a_{\\text{des}} = a_{\\text{ff}} + K_p e_p + K_d e_v

    Args:
        position_error: :math:`x_{\\text{ref}} - x`。
        velocity_error: :math:`\\dot x_{\\text{ref}} - \\dot x`。
        kp: 位置增益，单位 1/s^2。
        kd: 速度增益，单位 1/s。临界阻尼时 :math:`K_d = 2\\sqrt{K_p}`。
        feedforward: 前馈加速度，缺省为零。

    Returns:
        期望加速度，形状与误差相同。
    """
    ep = np.asarray(position_error, dtype=float)
    ev = np.asarray(velocity_error, dtype=float)
    ff = np.zeros_like(ep) if feedforward is None else np.asarray(feedforward, dtype=float)
    return ff + np.asarray(kp) * ep + np.asarray(kd) * ev


def orientation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """姿态误差的旋转矢量表示，形状 (3,)。

    .. math::  e = \\log\\left(R_{\\text{des}} R^\\top\\right)^\\vee

    **不能用欧拉角相减。** 姿态活在 SO(3) 上，欧拉角相减在大角度下没有
    意义，还会遇到万向锁 —— 这和里程碑 3 里"误差状态必须活在切空间"
    是同一个道理，本项目第三次遇到它。

    Args:
        R_desired: 期望姿态矩阵。
        R_current: 当前姿态矩阵。

    Returns:
        旋转矢量形式的误差，方向是转轴，模长是转角（弧度）。
    """
    R_err = np.asarray(R_desired, dtype=float) @ np.asarray(R_current, dtype=float).T
    cos_angle = np.clip(0.5 * (np.trace(R_err) - 1.0), -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        # 小角度：log 退化为反对称部分
        return 0.5 * np.array(
            [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]
        )
    axis = np.array(
        [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]
    ) / (2.0 * np.sin(angle))
    return angle * axis
