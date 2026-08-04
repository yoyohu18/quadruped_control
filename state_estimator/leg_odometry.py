"""支撑腿里程计：把"脚踩在地上不动"翻译成躯干速度。

这是四足独有的一种传感器。四足没有 GPS，室内也没有外部定位，但它有一件
无人机没有的东西：**只要脚踩在地上不打滑，那只脚在世界系里的速度就是零**。
这是一个免费的速度观测。

推导只用到里程碑 1 的雅可比。设躯干位姿为 :math:`(p, R)`，某只支撑脚在
躯干系下的位置为 :math:`p^B_f`，则它在世界系下的位置是

.. math::  p^W_f = p + R\\, p^B_f

对时间求导，并令支撑脚速度为零：

.. math::  0 = \\dot p + R\\,(\\omega \\times p^B_f) + R\\, J(q)\\,\\dot q

于是

.. math::  \\boxed{\\dot p = -R\\left(J(q)\\,\\dot q + \\omega \\times p^B_f\\right)}

三项的物理含义很清楚：关节在动（:math:`J\\dot q`）、躯干在转
（:math:`\\omega \\times p^B_f`），二者合起来必须由躯干平动抵消掉，
才能让脚在世界里保持静止。

**这个方法的致命弱点：只要脚打滑，上式的前提就不成立，速度估计立刻错，
而位置是速度的积分，于是位置误差永久累积且无法自我纠正。** 四足没有任何
绝对位置参考，所以这个误差只能靠外部传感器（视觉、激光）消除 —— 见
``docs/03_state_estimation.md`` 的可观测性一节。
"""

from __future__ import annotations

import numpy as np

from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS, forward_kinematics, leg_jacobian

__all__ = ["foot_position_in_base", "base_velocity_from_legs", "LegOdometry"]


def foot_position_in_base(joint_pos: np.ndarray, leg: str) -> np.ndarray:
    """足端在**躯干系**下的位置（含髋部安装偏置）。

    Args:
        joint_pos: 12 维关节角，Pinocchio 顺序。
        leg: 腿标识。

    Returns:
        形状 (3,) 的位置矢量。
    """
    i = LEGS.index(leg)
    q = np.asarray(joint_pos, dtype=float)[3 * i : 3 * i + 3]
    return HIP_OFFSETS[leg] + forward_kinematics(q, LEG_GEOMETRY[leg])


def base_velocity_from_legs(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    R: np.ndarray,
    omega_body: np.ndarray,
    contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由支撑腿反推躯干速度（世界系）。

    每条支撑腿都独立给出一个估计，最后取平均。**保留每条腿的单独估计
    很重要** —— 它们之间的离散程度就是最好用的打滑检测量：不打滑时四条
    腿应当高度一致，某条腿一滑，它给出的估计立刻偏离其余几条。

    Args:
        joint_pos: 12 维关节角。
        joint_vel: 12 维关节角速度。
        R: 机体到世界的旋转矩阵，形状 (3, 3)。
        omega_body: 机体系角速度，形状 (3,)。
        contact: 四条腿的接触标志，形状 (4,)。

    Returns:
        ``(v_world, per_leg)`` —— 平均后的躯干速度，以及每条支撑腿各自
        给出的估计（形状 (4, 3)，非支撑腿处为 NaN）。

    Raises:
        ValueError: 没有任何一条腿处于支撑相。
    """
    joint_pos = np.asarray(joint_pos, dtype=float)
    joint_vel = np.asarray(joint_vel, dtype=float)
    R = np.asarray(R, dtype=float)
    omega_body = np.asarray(omega_body, dtype=float).reshape(3)
    contact = np.asarray(contact, dtype=bool)

    per_leg = np.full((4, 3), np.nan)
    for i, leg in enumerate(LEGS):
        if not contact[i]:
            continue
        q = joint_pos[3 * i : 3 * i + 3]
        dq = joint_vel[3 * i : 3 * i + 3]
        p_foot_base = HIP_OFFSETS[leg] + forward_kinematics(q, LEG_GEOMETRY[leg])
        v_foot_base = leg_jacobian(q, LEG_GEOMETRY[leg]) @ dq
        per_leg[i] = -R @ (v_foot_base + np.cross(omega_body, p_foot_base))

    if not contact.any():
        raise ValueError("没有支撑腿，无法由腿部里程计估计速度（四足腾空相）。")
    return np.nanmean(per_leg, axis=0), per_leg


class LegOdometry:
    """纯腿部里程计：只用编码器和姿态，不用加速度计。

    刻意做得很简单，它的作用是当 ESKF 的对照组 —— 让"融合到底带来了
    什么"这个问题有一个可量化的答案，而不是空谈。

    Args:
        initial_position: 躯干初始位置。位置本身不可观测，必须外部给定。
        dt: 采样周期。
    """

    def __init__(self, initial_position: np.ndarray, dt: float) -> None:
        self.position = np.asarray(initial_position, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.dt = float(dt)

    def update(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        R: np.ndarray,
        omega_body: np.ndarray,
        contact: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """推进一步。

        Returns:
            ``(position, velocity)`` —— 当前的躯干位置与速度估计。
        """
        if np.asarray(contact, dtype=bool).any():
            self.velocity, _ = base_velocity_from_legs(joint_pos, joint_vel, R, omega_body, contact)
        # 腾空相没有观测，只能保持上一次的速度做外推。
        self.position = self.position + self.dt * self.velocity
        return self.position.copy(), self.velocity.copy()

    @staticmethod
    def slip_indicator(per_leg: np.ndarray) -> float:
        """打滑指标：各支撑腿速度估计之间的最大离散度，单位 m/s。

        不打滑时应当接近传感器噪声水平；某条腿打滑时会显著跳高。
        真实控制器会用它来动态调整该腿观测的协方差，而不是硬性剔除。
        """
        valid = per_leg[~np.isnan(per_leg).any(axis=1)]
        if len(valid) < 2:
            return 0.0
        return float(np.max(np.linalg.norm(valid - valid.mean(axis=0), axis=1)))
