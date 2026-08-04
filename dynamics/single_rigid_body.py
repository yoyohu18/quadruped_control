"""单刚体动力学（SRBD）—— 凸 MPC 真正使用的那个模型。

上一层的 ``rigid_body_dynamics`` 描述的是全部 18 个自由度。凸 MPC 不用它：
18 维非线性动力学放进预测时域会得到一个非凸的大问题，1 kHz 下解不动。

于是做一个大胆的近似：**把整台机器人当成一个刚体，腿视为无质量，
唯一的作用是把地面反力施加到这个刚体上**。于是只剩牛顿-欧拉两式：

.. math::

    m \\ddot{p} = \\sum_i f_i + m g

    \\frac{d}{dt}(I_w \\omega) = \\sum_i (r_i - p) \\times f_i

关键在于：**在足端位置 r_i 已知的前提下，这两式对接触力 f_i 是线性的。**
所以最优化问题变成一个凸 QP —— 这就是 MIT Cheetah 那篇凸 MPC 的全部诀窍。

本模块是手推的纯 NumPy 实现，是 ``rigid_body_dynamics`` 的交叉验证对象，
同时量化"腿无质量"这个假设到底带来多大误差
（对 Go2 来说误差不小 —— 腿占了 54.8% 的质量，见 ``docs/02_dynamics.md``）。

无人机类比：这一步等价于把四旋翼当成一个刚体、螺旋桨只提供力和力矩。
区别在于无人机的力作用点固定在机体上，而四足的作用点 r_i 每走一步就换
一次，而且只有支撑腿才有力 —— 力臂随步态变化，正是四足 MPC 的难点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SRBDParams",
    "skew",
    "rpy_to_matrix",
    "matrix_to_rpy",
    "rpy_rates_from_angular_velocity",
    "srbd_acceleration",
    "srbd_acceleration_mpc",
    "contact_wrench",
]


def skew(w: np.ndarray) -> np.ndarray:
    """反对称矩阵，满足 ``skew(a) @ b == np.cross(a, b)``。"""
    x, y, z = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """由 roll-pitch-yaw 构造旋转矩阵，约定 ``R = Rz(yaw) Ry(pitch) Rx(roll)``。

    与 Pinocchio 的 ``pin.rpy.rpyToMatrix`` 完全一致（测试中有验证）。
    """
    r, p, y = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> roll-pitch-yaw，是 :func:`rpy_to_matrix` 的逆。

    在 ``pitch = ±90°`` 处存在万向锁；四足躯干不会走到那里，但摆动腿
    规划器如果用欧拉角就要小心。
    """
    R = np.asarray(R, dtype=float)
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def rpy_rates_from_angular_velocity(rpy: np.ndarray, omega_world: np.ndarray) -> np.ndarray:
    """把**世界系**角速度换算成欧拉角变化率 ``[roll_dot, pitch_dot, yaw_dot]``。

    角速度不是任何量的导数 —— 它和欧拉角速率之间隔着一个依赖姿态的矩阵。
    把 ``omega`` 直接当成 ``rpy_dot`` 用，是姿态相关代码里最常见的错误之一，
    小角度下看不出来，一大角度就发散。
    """
    _, p, y = np.asarray(rpy, dtype=float).reshape(3)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    if abs(cp) < 1e-9:
        raise ValueError("接近万向锁 (pitch = ±90°)，欧拉角速率无定义。")
    # 三个欧拉角速率各自绕哪根轴（世界系）：
    #   roll_dot  绕 Rz(yaw) Ry(pitch) e_x
    #   pitch_dot 绕 Rz(yaw) e_y
    #   yaw_dot   绕 e_z
    # 把这三根轴按列拼起来就是 E，满足 omega_world = E @ rpy_dot。
    E = np.array(
        [
            [cy * cp, -sy, 0.0],
            [sy * cp, cy, 0.0],
            [-sp, 0.0, 1.0],
        ]
    )
    return np.linalg.solve(E, np.asarray(omega_world, dtype=float).reshape(3))


@dataclass(frozen=True)
class SRBDParams:
    """单刚体模型的参数。

    Attributes:
        mass: 整机总质量，kg。
        inertia_body: **机体系**下、关于整机质心的复合转动惯量，形状 (3, 3)。
            必须用复合惯量，不能用躯干连杆自身的惯量。
        gravity: 重力加速度大小，m/s^2。
    """

    mass: float
    inertia_body: np.ndarray
    gravity: float = 9.81

    def __post_init__(self) -> None:
        I = np.asarray(self.inertia_body, dtype=float)
        if I.shape != (3, 3):
            raise ValueError(f"inertia_body 必须是 3x3，收到 {I.shape}")
        if not np.allclose(I, I.T, atol=1e-9):
            raise ValueError("inertia_body 必须对称")
        if np.min(np.linalg.eigvalsh(I)) <= 0.0:
            raise ValueError("inertia_body 必须正定")

    @property
    def gravity_vector(self) -> np.ndarray:
        """世界系下的重力加速度矢量。"""
        return np.array([0.0, 0.0, -self.gravity])

    def inertia_world(self, R: np.ndarray) -> np.ndarray:
        """把机体系惯量旋转到世界系：``I_w = R I_b R^T``。"""
        R = np.asarray(R, dtype=float)
        return R @ np.asarray(self.inertia_body, dtype=float) @ R.T


def contact_wrench(
    forces: np.ndarray, foot_positions: np.ndarray, com: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把若干接触力汇总成作用于质心的合力与合力矩。

    Args:
        forces: 各支撑脚的接触力，形状 (n, 3)，世界系。
        foot_positions: 对应的足端位置，形状 (n, 3)，世界系。
        com: 质心位置，形状 (3,)。

    Returns:
        ``(合力, 关于质心的合力矩)``，均为形状 (3,)。
    """
    forces = np.asarray(forces, dtype=float).reshape(-1, 3)
    feet = np.asarray(foot_positions, dtype=float).reshape(-1, 3)
    if forces.shape != feet.shape:
        raise ValueError(f"力与足端位置数量不一致：{forces.shape} vs {feet.shape}")
    total_force = forces.sum(axis=0)
    total_torque = np.cross(feet - np.asarray(com, dtype=float).reshape(3), forces).sum(axis=0)
    return total_force, total_torque


def srbd_acceleration(
    com: np.ndarray,
    R: np.ndarray,
    omega_world: np.ndarray,
    forces: np.ndarray,
    foot_positions: np.ndarray,
    params: SRBDParams,
) -> tuple[np.ndarray, np.ndarray]:
    """完整（非线性）单刚体动力学。

    .. math::

        \\dot{v} = \\frac{1}{m}\\sum_i f_i + g

        \\dot{\\omega} = I_w^{-1}\\Big(\\sum_i (r_i - p)\\times f_i
                         - \\omega \\times (I_w \\omega)\\Big)

    Args:
        com: 质心位置，世界系。
        R: 机体到世界的旋转矩阵。
        omega_world: 世界系角速度。
        forces: 接触力，形状 (n, 3)。
        foot_positions: 足端位置，形状 (n, 3)。
        params: 单刚体参数。

    Returns:
        ``(线加速度, 角加速度)``，均为世界系下形状 (3,) 的矢量。
    """
    f_total, tau_total = contact_wrench(forces, foot_positions, com)
    lin_acc = f_total / params.mass + params.gravity_vector

    I_w = params.inertia_world(R)
    omega = np.asarray(omega_world, dtype=float).reshape(3)
    gyroscopic = np.cross(omega, I_w @ omega)
    ang_acc = np.linalg.solve(I_w, tau_total - gyroscopic)
    return lin_acc, ang_acc


def srbd_acceleration_mpc(
    com: np.ndarray,
    yaw: float,
    omega_world: np.ndarray,
    forces: np.ndarray,
    foot_positions: np.ndarray,
    params: SRBDParams,
) -> tuple[np.ndarray, np.ndarray]:
    """凸 MPC 实际使用的那个近似版本（MIT Cheetah 3 的做法）。

    在 :func:`srbd_acceleration` 之上再做两条近似，目的都是让模型对
    决策变量 ``f_i`` **保持线性**：

    1. **丢掉陀螺项** ``omega x (I_w omega)``。它对 ``omega`` 是二次的，
       在预测时域内会破坏线性。四足躯干角速度通常较小，所以代价可接受。
    2. **姿态只保留偏航** ``I_w ~= Rz(yaw) I_b Rz(yaw)^T``。假设横滚俯仰
       接近零；机器人在平地小幅摆动时成立，爬陡坡时开始失效。

    这两条近似的实际误差在 ``tests/test_dynamics.py`` 与
    ``scripts/viz_dynamics.py`` 中被定量测量出来 —— 不要凭感觉相信它们。

    Args:
        com: 质心位置。
        yaw: 偏航角，弧度。
        omega_world: 世界系角速度。
        forces: 接触力，形状 (n, 3)。
        foot_positions: 足端位置，形状 (n, 3)。
        params: 单刚体参数。

    Returns:
        ``(线加速度, 角加速度)``。
    """
    R_yaw = rpy_to_matrix(np.array([0.0, 0.0, float(yaw)]))
    f_total, tau_total = contact_wrench(forces, foot_positions, com)
    lin_acc = f_total / params.mass + params.gravity_vector
    I_w = params.inertia_world(R_yaw)
    ang_acc = np.linalg.solve(I_w, tau_total)  # 陀螺项被丢弃
    return lin_acc, ang_acc
