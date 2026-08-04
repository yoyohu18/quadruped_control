"""误差状态卡尔曼滤波（ESKF）：融合 IMU 与腿部运动学。

这是四足状态估计的工业标准做法（Bloesch 等 2013）。核心思路：

* **IMU 负责高频推进。** 加速度计与陀螺仪以 1 kHz 直接积分出位姿。
  好处是不依赖任何模型假设，坏处是零偏和噪声会让它在几秒内飘掉。
* **腿部运动学负责低频纠正。** 支撑脚在世界系里不动，这提供了一个
  几何观测，把 IMU 的漂移拉回来。
* **足端位置进入状态向量。** 这是 Bloesch 那篇论文的关键设计：不把
  "脚不动"当成硬约束，而是把每只脚的世界位置作为状态、给它一个很小的
  过程噪声。轻微打滑于是被自然吸收成状态的缓慢漂移，而不是让滤波器发散。

为什么是**误差状态**而不是直接状态？因为姿态活在 SO(3) 上，不是向量空间。
直接对四元数做卡尔曼更新会破坏单位模长，还会遇到奇异。误差状态把估计
拆成"大的名义值（在流形上）+ 小的误差（在切空间里）"，卡尔曼只处理后者，
于是所有线性代数都合法。**这与里程碑 2 里"浮动基座求导必须用
pin.integrate"是同一个道理。**

状态布局（名义状态）::

    p       (3)   躯干位置，世界系
    v       (3)   躯干速度，世界系
    R       (3,3) 机体到世界的旋转
    b_a     (3)   加速度计零偏，机体系
    b_g     (3)   陀螺仪零偏，机体系
    p_f     (4,3) 四只脚的世界位置

误差状态共 27 维，顺序为::

    [dp(3), dv(3), dtheta(3), db_a(3), db_g(3), dp_f(12)]

姿态误差采用**世界系（左）扰动**约定：:math:`R = \\exp(\\delta\\theta^\\wedge)\\hat R`。
换成机体系（右）扰动，下面每一个雅可比都要改 —— 这类约定必须写死并测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import skew
from kinematics import LEGS

from .leg_odometry import foot_position_in_base

__all__ = ["so3_exp", "ESKFConfig", "ESKFState", "ErrorStateKF"]

# 误差状态各分块的下标
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_THETA = slice(6, 9)
IDX_BA = slice(9, 12)
IDX_BG = slice(12, 15)
IDX_FEET = slice(15, 27)
N_ERROR = 27


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) 指数映射（罗德里格斯公式）：旋转矢量 -> 旋转矩阵。

    小角度处用泰勒展开，避免 ``sin(x)/x`` 在 ``x -> 0`` 时的数值问题。
    与 ``pin.exp3`` 一致（由测试验证）。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    K = skew(phi)
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return (
        np.eye(3)
        + (np.sin(angle) / angle) * K
        + ((1.0 - np.cos(angle)) / angle**2) * K @ K
    )


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """把因浮点累积而略微失正交的旋转矩阵投影回 SO(3)。"""
    U, _, Vt = np.linalg.svd(R)
    R_new = U @ Vt
    if np.linalg.det(R_new) < 0:
        U[:, -1] *= -1
        R_new = U @ Vt
    return R_new


@dataclass
class ESKFConfig:
    """滤波器噪声参数。

    这些数字来自 IMU 数据手册与实测，是**调参的主要旋钮**。经验法则：
    观测噪声反映你对腿部运动学的信任度，过程噪声反映你对 IMU 的信任度。

    Attributes:
        accel_noise: 加速度计噪声密度，m/s^2。
        gyro_noise: 陀螺仪噪声密度，rad/s。
        accel_bias_walk: 加速度计零偏随机游走，m/s^2/√s。
        gyro_bias_walk: 陀螺仪零偏随机游走，rad/s/√s。
        foot_noise_stance: 支撑脚位置的过程噪声，m/√s。**这是吸收轻微
            打滑的旋钮**：调大则滤波器容忍打滑但精度下降，调小则精度高
            但一打滑就发散。
        foot_noise_swing: 摆动脚位置的过程噪声。取一个很大的值，等价于
            "这只脚在哪我完全不知道"。
        kinematics_noise: 腿部运动学观测噪声，m。含编码器误差、连杆
            变形、足端半径等未建模因素。
        gravity: 重力加速度大小。
    """

    accel_noise: float = 0.02
    gyro_noise: float = 0.002
    accel_bias_walk: float = 1e-3
    gyro_bias_walk: float = 1e-4
    foot_noise_stance: float = 1e-3
    foot_noise_swing: float = 1e2
    kinematics_noise: float = 5e-3
    gravity: float = 9.81


@dataclass
class ESKFState:
    """名义状态。"""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foot_position: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))

    def copy(self) -> "ESKFState":
        return ESKFState(
            self.position.copy(),
            self.velocity.copy(),
            self.rotation.copy(),
            self.accel_bias.copy(),
            self.gyro_bias.copy(),
            self.foot_position.copy(),
        )


class ErrorStateKF:
    """融合 IMU 与腿部运动学的误差状态卡尔曼滤波器。

    Args:
        dt: 采样周期，秒。
        config: 噪声参数。
        initial_state: 名义状态初值。位置与偏航不可观测，必须由外部给定
            一个起点 —— 滤波器只能保证它们的**变化量**正确。
        initial_covariance: 误差状态协方差初值，形状 (27, 27)。
    """

    def __init__(
        self,
        dt: float,
        config: ESKFConfig | None = None,
        initial_state: ESKFState | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.dt = float(dt)
        self.cfg = config or ESKFConfig()
        self.state = (initial_state or ESKFState()).copy()
        if initial_covariance is None:
            P = np.eye(N_ERROR) * 1e-4
            P[IDX_FEET, IDX_FEET] = np.eye(12) * 1e-2
            self.P = P
        else:
            self.P = np.asarray(initial_covariance, dtype=float).copy()
        self._prev_contact = np.zeros(4, dtype=bool)

    # -- 预测 -----------------------------------------------------------------

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, contact: np.ndarray) -> None:
        """用一帧 IMU 数据推进名义状态与协方差。

        Args:
            accel_meas: 加速度计读数（比力），机体系，形状 (3,)。
            gyro_meas: 陀螺仪读数，机体系，形状 (3,)。
            contact: 四条腿的接触标志，决定足端状态的过程噪声。
        """
        dt = self.dt
        s = self.state
        cfg = self.cfg

        a_body = np.asarray(accel_meas, dtype=float).reshape(3) - s.accel_bias
        w_body = np.asarray(gyro_meas, dtype=float).reshape(3) - s.gyro_bias
        g_world = np.array([0.0, 0.0, -cfg.gravity])

        # 加速度计测的是比力，转到世界系后要把重力加回去。
        a_world = s.rotation @ a_body + g_world

        # --- 名义状态推进 ---
        s.position = s.position + s.velocity * dt + 0.5 * a_world * dt**2
        s.velocity = s.velocity + a_world * dt
        s.rotation = _orthonormalize(s.rotation @ so3_exp(w_body * dt))
        # 零偏与足端位置按常值推进（其不确定性由过程噪声体现）

        # --- 误差状态传播 ---
        F = np.eye(N_ERROR)
        F[IDX_P, IDX_V] = np.eye(3) * dt
        F[IDX_V, IDX_THETA] = -skew(s.rotation @ a_body) * dt
        F[IDX_V, IDX_BA] = -s.rotation * dt
        F[IDX_THETA, IDX_BG] = -s.rotation * dt

        Q = np.zeros((N_ERROR, N_ERROR))
        Q[IDX_V, IDX_V] = np.eye(3) * (cfg.accel_noise**2 * dt)
        Q[IDX_THETA, IDX_THETA] = np.eye(3) * (cfg.gyro_noise**2 * dt)
        Q[IDX_BA, IDX_BA] = np.eye(3) * (cfg.accel_bias_walk**2 * dt)
        Q[IDX_BG, IDX_BG] = np.eye(3) * (cfg.gyro_bias_walk**2 * dt)
        contact = np.asarray(contact, dtype=bool)
        for i in range(4):
            noise = cfg.foot_noise_stance if contact[i] else cfg.foot_noise_swing
            j = 15 + 3 * i
            Q[j : j + 3, j : j + 3] = np.eye(3) * (noise**2 * dt)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # 强制对称，抑制浮点漂移

    # -- 更新 -----------------------------------------------------------------

    def update(self, joint_pos: np.ndarray, contact: np.ndarray) -> np.ndarray:
        """用腿部运动学观测做一次更新。

        观测模型：对每只支撑脚，编码器给出它在躯干系下的位置，而状态里
        存着它的世界位置，二者必须自洽：

        .. math::  z = R^T (p_f - p)

        Args:
            joint_pos: 12 维关节角。
            contact: 四条腿的接触标志。

        Returns:
            本次更新的观测残差（新息），形状 (3 * 支撑腿数,)。没有支撑腿
            时返回空数组。
        """
        contact = np.asarray(contact, dtype=bool)
        s = self.state

        # 刚落地的脚：用当前位姿重新初始化它的世界位置，并放大其协方差。
        newly_landed = contact & ~self._prev_contact
        for i, leg in enumerate(LEGS):
            if newly_landed[i]:
                s.foot_position[i] = s.position + s.rotation @ foot_position_in_base(joint_pos, leg)
                j = 15 + 3 * i
                self.P[j : j + 3, :] = 0.0
                self.P[:, j : j + 3] = 0.0
                self.P[j : j + 3, j : j + 3] = np.eye(3) * self.cfg.kinematics_noise**2 * 10
        self._prev_contact = contact.copy()

        stance = np.flatnonzero(contact)
        if len(stance) == 0:
            return np.array([])

        m = 3 * len(stance)
        H = np.zeros((m, N_ERROR))
        residual = np.zeros(m)
        Rt = s.rotation.T

        for k, i in enumerate(stance):
            z_meas = foot_position_in_base(joint_pos, LEGS[i])
            delta = s.foot_position[i] - s.position
            z_pred = Rt @ delta
            residual[3 * k : 3 * k + 3] = z_meas - z_pred

            # 对 R = exp(dtheta^) R_hat 做一阶展开得到下面三块。
            H[3 * k : 3 * k + 3, IDX_P] = -Rt
            H[3 * k : 3 * k + 3, IDX_THETA] = Rt @ skew(delta)
            j = 15 + 3 * i
            H[3 * k : 3 * k + 3, j : j + 3] = Rt

        R_cov = np.eye(m) * self.cfg.kinematics_noise**2
        S = H @ self.P @ H.T + R_cov
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T  # 等价于 P H' S^-1，但更稳
        dx = K @ residual

        self._inject(dx)

        # Joseph 形式，保证协方差始终对称正定
        I_KH = np.eye(N_ERROR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return residual

    def _inject(self, dx: np.ndarray) -> None:
        """把误差状态注入名义状态，然后把误差清零。

        这是 ESKF 的精髓：误差永远保持在零附近的小量，线性化因此始终有效。
        """
        s = self.state
        s.position = s.position + dx[IDX_P]
        s.velocity = s.velocity + dx[IDX_V]
        # 世界系（左）扰动：R <- exp(dtheta^) R
        s.rotation = _orthonormalize(so3_exp(dx[IDX_THETA]) @ s.rotation)
        s.accel_bias = s.accel_bias + dx[IDX_BA]
        s.gyro_bias = s.gyro_bias + dx[IDX_BG]
        s.foot_position = s.foot_position + dx[IDX_FEET].reshape(4, 3)

    # -- 便捷接口 -------------------------------------------------------------

    def step(
        self,
        accel_meas: np.ndarray,
        gyro_meas: np.ndarray,
        joint_pos: np.ndarray,
        contact: np.ndarray,
    ) -> ESKFState:
        """预测 + 更新，一个完整控制周期。"""
        self.predict(accel_meas, gyro_meas, contact)
        self.update(joint_pos, contact)
        return self.state

    @property
    def position_std(self) -> np.ndarray:
        """位置估计的标准差，形状 (3,)。"""
        return np.sqrt(np.diag(self.P)[IDX_P])

    @property
    def velocity_std(self) -> np.ndarray:
        """速度估计的标准差，形状 (3,)。"""
        return np.sqrt(np.diag(self.P)[IDX_V])

    @property
    def attitude_std(self) -> np.ndarray:
        """姿态估计的标准差（横滚、俯仰、偏航），单位弧度。"""
        return np.sqrt(np.diag(self.P)[IDX_THETA])
