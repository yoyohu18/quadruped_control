"""自定义奖励项：把里程碑 4/5/6 的物理先验接进 Isaac Lab 的 RewardManager。

每个函数都是三行胶水 —— **从 ``env`` 里取张量、调用 :mod:`rl.reward_kernels`
里的纯函数、返回**。数学一行都不写在这里，因为这里没法被单元测试。

Isaac Lab 的约定：奖励函数返回 ``(num_envs,)``，最终奖励是
:math:`\\sum_i w_i\\, r_i(s,a)\\,\\Delta t`。**权重里的负号要自己带**，
函数本身返回非负的"惩罚量"。这个约定不遵守的话，调权重时会疯掉。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from rl.reward_kernels import (
    air_time_reward,
    capture_point_error,
    foot_clearance_reward,
    foot_slip_penalty,
    gait_contact_reward,
    reference_contact,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "capture_point_stability",
    "feet_air_time",
    "gait_phase_tracking",
    "foot_clearance",
    "foot_slip",
    "joint_power",
]


def _contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """从接触传感器读出布尔接触状态 ``(N, 4)``。

    用 ``net_forces_w_history`` 的历史最大值而不是当前帧：接触力在
    500 Hz 物理步上会抖，单帧判据会让"触地"这个信号闪烁，
    进而让步态奖励充满噪声。取 3 帧最大值相当于一个极简的消抖滤波器 ——
    与 M3 里给足端接触检测加滞环是同一个动机。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).max(dim=1)[0] > threshold


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.2,
    command_deadzone: float = 0.1,
) -> torch.Tensor:
    """腾空时间奖励。数学在 :func:`rl.reward_kernels.air_time_reward`。

    Isaac Lab 的 ``isaaclab_tasks`` 里有同名实现，这里自己写一份是为了让
    "只在落地那一帧结算"这条规则能被单元测试钉住 —— 它是最容易写错的
    一条（写错的最优解是"把脚永远举在空中"）。

    ``compute_first_contact(dt)`` 返回"本控制周期内刚从空中落地"的标志，
    ``last_air_time`` 是这次落地之前累计的腾空时长。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return air_time_reward(last_air_time, first_contact, threshold, command_norm, command_deadzone)


def capture_point_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """里程碑 6 的捕获点惩罚：发散分量偏离支撑足形心的距离平方。

    这一项是本仓库与 Isaac Lab 官方 Go2 配置**最实质的区别**。官方奖励里
    没有任何显式的平衡度量，稳定性完全靠"躯干触地就终止"这个稀疏信号
    反向推出来。捕获点提供的是**稠密**且**有物理含义**的信号：它在机器人
    真正摔倒前几百毫秒就开始变大。

    权重不能给大 —— 它与速度跟踪本质上冲突（想跑就必须让捕获点跑到支撑区
    外面去）。经验值在 -0.1 量级，作用是"在不影响跟踪的前提下别乱晃"。
    """
    asset = env.scene[asset_cfg.name]
    contact = _contact_state(env, sensor_cfg)

    base_pos = asset.data.root_pos_w[:, :2]
    base_vel = asset.data.root_lin_vel_w[:, :2]
    foot_pos = asset.data.body_pos_w[:, sensor_cfg.body_ids, :2]
    # 相对地形的高度：足端平均高度当作局部地面
    foot_height = asset.data.body_pos_w[:, sensor_cfg.body_ids, 2]
    base_height = asset.data.root_pos_w[:, 2] - foot_height.min(dim=1)[0]

    return capture_point_error(base_pos, base_vel, base_height, foot_pos, contact)


def gait_phase_tracking(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    period: float = 0.4,
    duty_factor: float = 0.5,
    command_deadzone: float = 0.1,
) -> torch.Tensor:
    """里程碑 4 的步态奖励：实际接触与 trot 参考时序的一致比例。

    参考相位由**每个环境自己的 episode 时间**驱动
    （``episode_length_buf * step_dt``），不是全局步数 —— 各环境重置时刻
    不同，用全局时间会让参考步态与机器人的实际节奏完全脱节。

    指令接近零时不计此项：站着不动时强行要求踏步是自相矛盾的。
    """
    contact = _contact_state(env, sensor_cfg)
    episode_time = env.episode_length_buf.float() * env.step_dt
    reference = reference_contact(episode_time, period=period, duty_factor=duty_factor)

    reward = gait_contact_reward(contact, reference)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return reward * (command_norm > command_deadzone).float()


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.08,
) -> torch.Tensor:
    """里程碑 5 的抬脚高度惩罚，按足端水平速度加权。"""
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    # 以四足最低点为局部地面，避免地形高度直接进入奖励
    ground = foot_z.min(dim=1, keepdim=True)[0]
    return foot_clearance_reward(foot_z - ground, foot_vel, target_height)


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """支撑脚打滑惩罚。M7 的摩擦锥硬约束在这里退化成软惩罚。"""
    asset = env.scene[asset_cfg.name]
    contact = _contact_state(env, sensor_cfg)
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return foot_slip_penalty(foot_vel, contact)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机械功率 :math:`\\sum_i |\\tau_i \\dot q_i|`。

    比常用的 ``joint_torques_l2`` 更接近真实能耗：静态站立时力矩很大但
    速度为零，功率接近 0 —— 二次力矩惩罚会错误地惩罚"稳稳站着"。

    做过运输能耗（COT）评估的话会认出来：这一项正比于 COT 的分子。
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )
