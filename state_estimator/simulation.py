"""生成带真值的运动数据，用来验证状态估计器。

真机上你**没有**躯干位姿的真值 —— 这正是要做状态估计的原因。所以验证
估计器必须先造一段"什么都知道"的数据：解析地给定躯干轨迹与步态，反推
出足端位置、关节角、IMU 读数，然后看估计器能不能从这些量把躯干轨迹还原
出来。

这里的做法是**运动学一致**的正向构造：

1. 解析给定躯干位姿 p(t)、R(t)，因而速度、加速度也解析已知；
2. 给定 trot 步态时序，决定每条腿处于支撑相还是摆动相；
3. 支撑脚钉在地面不动；摆动脚走一条抬起-落下的弧线；
4. 由足端位置**逆运动学**解出关节角（复用里程碑 1）；
5. 由躯干加速度与角速度合成 IMU 读数（复用里程碑 2 的姿态工具）。

这样得到的数据严格满足 FK(关节角) == 足端位置，估计器面对的是一个
自洽的世界，任何误差都只能来自估计器本身或人为注入的噪声与打滑。

.. note::
   这里的 trot 时序是为了造数据而写的最小实现。真正的步态调度器是
   里程碑 4 的内容，那时会有相位、占空比、步态切换等完整功能。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics import rpy_to_matrix
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS, inverse_kinematics, leg_jacobian

__all__ = [
    "TrajectoryConfig",
    "GroundTruth",
    "generate_trot",
    "simulate_imu",
    "inject_slip",
    "add_encoder_noise",
]

#: trot 步态中同相位的两组对角腿。
TROT_PAIRS = (("FL", "RR"), ("FR", "RL"))


@dataclass
class TrajectoryConfig:
    """轨迹与步态参数。

    Attributes:
        duration: 总时长，秒。
        dt: 采样周期，秒。1 kHz 对应 0.001。
        base_height: 躯干标称离地高度，米。
        forward_speed: 前进速度，m/s。
        lateral_amplitude: 横向摆动幅值，米。
        height_amplitude: 竖直起伏幅值，米。
        yaw_rate: 偏航角速度，rad/s。
        roll_amplitude: 横滚摆动幅值，弧度。
        pitch_amplitude: 俯仰摆动幅值，弧度。
        body_frequency: 躯干摆动频率，Hz。
        gait_period: 一个完整步态周期，秒。
        swing_height: 摆动腿抬起高度，米。
        stance_width: 足端相对髋部的横向外扩量，米。
    """

    duration: float = 3.0
    dt: float = 1e-3
    base_height: float = 0.30
    forward_speed: float = 0.4
    lateral_amplitude: float = 0.02
    height_amplitude: float = 0.015
    yaw_rate: float = 0.15
    roll_amplitude: float = 0.04
    pitch_amplitude: float = 0.03
    body_frequency: float = 1.3
    gait_period: float = 0.4
    swing_height: float = 0.06
    stance_width: float = 0.0

    @property
    def n_steps(self) -> int:
        """采样点个数。"""
        return int(round(self.duration / self.dt))


@dataclass
class GroundTruth:
    """一段带完整真值的运动数据。

    所有数组的第 0 维都是时间。角速度与加速度的坐标系已在字段名中标明，
    混用坐标系是状态估计里最常见的错误来源。
    """

    t: np.ndarray  # (N,)
    base_position: np.ndarray  # (N, 3) 世界系
    base_rpy: np.ndarray  # (N, 3)
    base_rotation: np.ndarray  # (N, 3, 3) 机体 -> 世界
    base_velocity_world: np.ndarray  # (N, 3)
    base_acceleration_world: np.ndarray  # (N, 3)
    omega_body: np.ndarray  # (N, 3) 机体系角速度（陀螺仪测的就是它）
    joint_position: np.ndarray  # (N, 12) Pinocchio 顺序
    joint_velocity: np.ndarray  # (N, 12)
    foot_position_world: np.ndarray  # (N, 4, 3)
    contact: np.ndarray  # (N, 4) bool
    config: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    def __len__(self) -> int:
        return len(self.t)


def _base_pose(t: np.ndarray, cfg: TrajectoryConfig):
    """解析给定的躯干位姿及其一、二阶导数。

    全部用正弦，所以导数是精确的 —— 不引入数值微分误差，估计器的误差
    才不会和"造数据的误差"混在一起。
    """
    w = 2 * np.pi * cfg.body_frequency

    pos = np.column_stack(
        [
            cfg.forward_speed * t,
            cfg.lateral_amplitude * np.sin(w * t),
            cfg.base_height + cfg.height_amplitude * np.sin(2 * w * t),
        ]
    )
    vel = np.column_stack(
        [
            np.full_like(t, cfg.forward_speed),
            cfg.lateral_amplitude * w * np.cos(w * t),
            cfg.height_amplitude * 2 * w * np.cos(2 * w * t),
        ]
    )
    acc = np.column_stack(
        [
            np.zeros_like(t),
            -cfg.lateral_amplitude * w**2 * np.sin(w * t),
            -cfg.height_amplitude * (2 * w) ** 2 * np.sin(2 * w * t),
        ]
    )

    rpy = np.column_stack(
        [
            cfg.roll_amplitude * np.sin(w * t),
            cfg.pitch_amplitude * np.sin(1.7 * w * t),
            cfg.yaw_rate * t,
        ]
    )
    rpy_dot = np.column_stack(
        [
            cfg.roll_amplitude * w * np.cos(w * t),
            cfg.pitch_amplitude * 1.7 * w * np.cos(1.7 * w * t),
            np.full_like(t, cfg.yaw_rate),
        ]
    )
    return pos, vel, acc, rpy, rpy_dot


def _omega_body_from_rpy(rpy: np.ndarray, rpy_dot: np.ndarray) -> np.ndarray:
    """由欧拉角及其变化率求**机体系**角速度。

    对 R = Rz(y) Ry(p) Rx(r)，机体系角速度是三段旋转轴在机体系下的叠加。
    这与里程碑 2 里的世界系版本互为转置关系，别搞混。
    """
    r, p = rpy[:, 0], rpy[:, 1]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    rd, pd, yd = rpy_dot[:, 0], rpy_dot[:, 1], rpy_dot[:, 2]
    return np.column_stack(
        [
            rd - yd * sp,
            pd * cr + yd * cp * sr,
            -pd * sr + yd * cp * cr,
        ]
    )


def _contact_schedule(t: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    """trot 时序：两组对角腿交替支撑，占空比 0.5。"""
    phase = (t / cfg.gait_period) % 1.0
    contact = np.zeros((len(t), 4), dtype=bool)
    for leg in LEGS:
        i = LEGS.index(leg)
        offset = 0.0 if leg in TROT_PAIRS[0] else 0.5
        leg_phase = (phase + offset) % 1.0
        contact[:, i] = leg_phase < 0.5  # 前半周期支撑
    return contact


def generate_trot(cfg: TrajectoryConfig | None = None) -> GroundTruth:
    """生成一段 trot 步行的完整真值数据。

    Args:
        cfg: 轨迹与步态参数，默认使用 :class:`TrajectoryConfig` 的默认值。

    Returns:
        :class:`GroundTruth`，其中 ``FK(joint_position)`` 严格等于
        ``foot_position_world``（由测试验证）。
    """
    cfg = cfg or TrajectoryConfig()
    n = cfg.n_steps
    t = np.arange(n) * cfg.dt

    pos, vel, acc, rpy, rpy_dot = _base_pose(t, cfg)
    R = np.array([rpy_to_matrix(r) for r in rpy])
    omega_body = _omega_body_from_rpy(rpy, rpy_dot)
    contact = _contact_schedule(t, cfg)

    foot_world = np.zeros((n, 4, 3))
    joint_pos = np.zeros((n, 12))

    # 每条腿维护一个"当前支撑点"，落地时刷新，支撑期间保持不动。
    anchor = np.zeros((4, 3))
    liftoff = np.zeros((4, 3))
    for i, leg in enumerate(LEGS):
        anchor[i] = pos[0] + R[0] @ HIP_OFFSETS[leg]
        anchor[i, 2] = 0.0
        liftoff[i] = anchor[i]

    for k in range(n):
        for i, leg in enumerate(LEGS):
            hip_world = pos[k] + R[k] @ HIP_OFFSETS[leg]
            if contact[k, i]:
                if k > 0 and not contact[k - 1, i]:
                    anchor[i] = foot_world[k - 1, i]  # 刚落地，钉住
                    anchor[i, 2] = 0.0
                foot_world[k, i] = anchor[i]
            else:
                if k > 0 and contact[k - 1, i]:
                    liftoff[i] = foot_world[k - 1, i]  # 刚离地
                # 摆动相位 0 -> 1
                phase = (t[k] / cfg.gait_period) % 1.0
                offset = 0.0 if leg in TROT_PAIRS[0] else 0.5
                s = ((phase + offset) % 1.0 - 0.5) / 0.5
                # 落脚点：髋部投影 + 半个支撑期的前移量（Raibert 启发式的雏形）
                target = hip_world.copy()
                target[2] = 0.0
                target[:2] += vel[k, :2] * cfg.gait_period * 0.25
                alpha = 3 * s**2 - 2 * s**3
                foot_world[k, i] = (1 - alpha) * liftoff[i] + alpha * target
                foot_world[k, i, 2] = cfg.swing_height * np.sin(np.pi * s)

            # 足端位置 -> 髋系 -> 逆运动学
            p_hip = R[k].T @ (foot_world[k, i] - hip_world)
            joint_pos[k, 3 * i : 3 * i + 3] = inverse_kinematics(p_hip, LEG_GEOMETRY[leg], clamp=True)

    # 关节速度由中心差分得到；端点用单侧差分。
    joint_vel = np.gradient(joint_pos, cfg.dt, axis=0)

    return GroundTruth(
        t=t,
        base_position=pos,
        base_rpy=rpy,
        base_rotation=R,
        base_velocity_world=vel,
        base_acceleration_world=acc,
        omega_body=omega_body,
        joint_position=joint_pos,
        joint_velocity=joint_vel,
        foot_position_world=foot_world,
        contact=contact,
        config=cfg,
    )


def simulate_imu(
    gt: GroundTruth,
    accel_noise: float = 0.02,
    gyro_noise: float = 0.002,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
    gravity: float = 9.81,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """由真值合成 IMU 读数。

    加速度计测的是**比力（specific force）**，不是加速度：

    .. math::  a_{meas} = R^T (a_{world} - g_{world}) + b_a + n_a

    静止放在桌上时它读出的是 :math:`+9.81\\,m/s^2` 向上，而不是零 ——
    这个符号搞反是 IMU 相关代码里最经典的错误。

    Args:
        gt: 真值数据。
        accel_noise: 加速度计噪声标准差，m/s^2。
        gyro_noise: 陀螺仪噪声标准差，rad/s。
        accel_bias: 加速度计常值零偏，形状 (3,)，默认无零偏。
        gyro_bias: 陀螺仪常值零偏，形状 (3,)，默认无零偏。
        gravity: 重力加速度大小。
        seed: 随机种子。

    Returns:
        ``(accel_meas, gyro_meas)``，形状均为 (N, 3)，表达在**机体系**下。
    """
    rng = np.random.default_rng(seed)
    n = len(gt)
    b_a = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float)
    b_g = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float)
    g_world = np.array([0.0, 0.0, -gravity])

    accel = np.einsum("nji,nj->ni", gt.base_rotation, gt.base_acceleration_world - g_world)
    accel = accel + b_a + rng.normal(0.0, accel_noise, (n, 3))
    gyro = gt.omega_body + b_g + rng.normal(0.0, gyro_noise, (n, 3))
    return accel, gyro


def foot_velocity_in_base(joint_pos: np.ndarray, joint_vel: np.ndarray, leg: str) -> np.ndarray:
    """由关节速度求足端在躯干系下的速度：``J(q) qdot``。"""
    i = LEGS.index(leg)
    q = joint_pos[3 * i : 3 * i + 3]
    dq = joint_vel[3 * i : 3 * i + 3]
    return leg_jacobian(q, LEG_GEOMETRY[leg]) @ dq


def inject_slip(
    gt: GroundTruth,
    leg: str,
    t_start: float,
    t_end: float,
    velocity: np.ndarray,
) -> GroundTruth:
    """让某条支撑腿在一段时间内打滑，返回修改后的新数据。

    打滑的物理含义是：**脚仍然被判定为"接触"，但它在世界系里确实在动**。
    机器人的编码器忠实地反映了这一点（关节角随之改变），但估计器如果
    坚持"支撑脚不动"这个假设，就会把脚的移动错误地归因为躯干的移动。

    这正是腿部里程计无法自我纠正的误差来源：没有任何外部参考能告诉
    估计器"是脚动了，不是身体动了"。

    Args:
        gt: 原始真值数据。
        leg: 打滑的腿。
        t_start: 打滑起始时刻，秒。
        t_end: 打滑结束时刻，秒。
        velocity: 打滑速度，形状 (3,)，世界系，m/s。

    Returns:
        新的 :class:`GroundTruth`；躯干真值完全不变，只有该腿的足端位置
        与关节角被修改。
    """
    i = LEGS.index(leg)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    foot = gt.foot_position_world.copy()
    joints = gt.joint_position.copy()

    window = (gt.t >= t_start) & (gt.t <= t_end) & gt.contact[:, i]
    if not window.any():
        raise ValueError(f"腿 {leg} 在 [{t_start}, {t_end}] 内没有支撑相，无法注入打滑。")

    # 打滑量随时间累积，并在窗口结束后保持（脚确实挪到了新位置）。
    offset = np.zeros((len(gt), 3))
    accumulated = np.zeros(3)
    for k in range(1, len(gt)):
        if window[k]:
            accumulated = accumulated + velocity * gt.config.dt
        offset[k] = accumulated

    # 只在该腿处于接触时施加偏移；离地后重新规划，偏移自然消失。
    apply = gt.contact[:, i] & (gt.t >= t_start)
    foot[apply, i] += offset[apply]

    # 关节角必须与被挪动后的足端位置保持一致，否则数据自相矛盾。
    for k in np.flatnonzero(apply):
        hip_world = gt.base_position[k] + gt.base_rotation[k] @ HIP_OFFSETS[leg]
        p_hip = gt.base_rotation[k].T @ (foot[k, i] - hip_world)
        joints[k, 3 * i : 3 * i + 3] = inverse_kinematics(p_hip, LEG_GEOMETRY[leg], clamp=True)

    joint_vel = np.gradient(joints, gt.config.dt, axis=0)
    return GroundTruth(
        t=gt.t,
        base_position=gt.base_position,
        base_rpy=gt.base_rpy,
        base_rotation=gt.base_rotation,
        base_velocity_world=gt.base_velocity_world,
        base_acceleration_world=gt.base_acceleration_world,
        omega_body=gt.omega_body,
        joint_position=joints,
        joint_velocity=joint_vel,
        foot_position_world=foot,
        contact=gt.contact,
        config=gt.config,
    )


def add_encoder_noise(
    gt: GroundTruth, position_noise: float = 1e-3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """给关节角加上编码器噪声，并重新求出关节速度。

    真实编码器的分辨率有限，还有齿轮回差与连杆挠曲。这些误差直接进入
    腿部运动学观测，是腿部里程计精度的实际上限。

    Args:
        gt: 真值数据。
        position_noise: 关节角噪声标准差，弧度。
        seed: 随机种子。

    Returns:
        ``(joint_pos_noisy, joint_vel_noisy)``。
    """
    rng = np.random.default_rng(seed)
    jp = gt.joint_position + rng.normal(0.0, position_noise, gt.joint_position.shape)
    jv = np.gradient(jp, gt.config.dt, axis=0)
    return jp, jv
