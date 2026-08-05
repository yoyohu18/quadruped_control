"""奖励核函数：纯 torch，不依赖 Isaac Sim。

## 为什么单独拆一层

Isaac Lab 的奖励项签名长这样::

    def my_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor

它把"物理量提取"和"数学"揉在一起，结果是**没法单元测试** —— 想测一行公式
得先启动整个仿真器。

所以这里把数学单独拿出来：输入输出全是张量，没有 ``env``，没有
``SceneEntityCfg``，import 只有 torch。:mod:`isaac.mdp.rewards` 里的
manager 项退化成三行胶水：取数据、调这里的函数、返回。

这与 M1 把闭式运动学和 Pinocchio 分层、M7 把 QP 组装和求解分层是同一个
判断：**能被测试的东西必须能脱离运行时环境构造出来**。

## 里程碑 1–8 在这里以"奖励"的身份回来

| 来源 | 概念 | 这里的用法 |
|---|---|---|
| M4 步态调度器 | 相位、占空比 | 参考接触序列 → 步态奖励 |
| M5 摆动规划器 | 摆动高度 | 足端离地高度奖励 |
| M6 落脚点规划器 | 捕获点 :math:`\\xi = x + \\dot x/\\omega` | 平衡奖励 |
| M6 LIPM | :math:`\\omega=\\sqrt{g/h}` | 捕获点的尺度 |

**这正是这条路线与"纯黑盒 RL"的区别。** 不给步态奖励，PPO 也能学会走 ——
学出来的往往是抖动的、不对称的、听觉上很难听的步态。前八个里程碑给了我们
一组有物理含义的先验，把它们写成奖励，就是在给搜索加正则。
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "exp_tracking_reward",
    "capture_point",
    "capture_point_error",
    "reference_contact",
    "gait_contact_reward",
    "foot_clearance_reward",
    "foot_slip_penalty",
    "air_time_reward",
    "TROT_OFFSETS",
    "LEG_ORDER",
]

#: 与 :data:`kinematics.LEGS` 一致的腿序，也与 Isaac Lab 的 ``.*_foot``
#: 排序一致（冒烟测试里已核对过真实仿真器的 body 顺序）。
LEG_ORDER = ("FL", "FR", "RL", "RR")

#: trot 的相位偏移，取自 :data:`gait_scheduler.GAITS`。对角腿成对：
#: FL 与 RR 同相，FR 与 RL 同相。
TROT_OFFSETS = (0.0, 0.5, 0.5, 0.0)


def exp_tracking_reward(squared_error: torch.Tensor, sigma: float) -> torch.Tensor:
    """指数型跟踪奖励 :math:`\\exp(-e^2/\\sigma)`。

    **为什么不用 :math:`-e^2`？** 二次惩罚在误差大时梯度也大，策略会
    优先去修那个最离谱的自由度，其余奖励项全被压制 —— 训练早期机器人还
    站不稳的时候，速度误差动辄几 m/s，二次项会把所有其他项淹没。

    指数型的关键性质是**有界且饱和**：误差大时奖励趋近 0，梯度也趋近 0，
    于是"先学会站住"（存活奖励与姿态项）自然获得优先级，"再学会跟上指令"
    在站稳之后才成为主要梯度来源。这是一种隐式的课程学习。

    :math:`\\sigma` 决定"多准算准"。Go2 上常取 0.25，对应误差 0.5 m/s 时
    奖励掉到 :math:`e^{-1}\\approx 0.37`。

    Args:
        squared_error: 误差平方和，任意形状。
        sigma: 宽度参数，正数。
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma 必须为正，收到 {sigma}")
    return torch.exp(-squared_error / sigma)


def capture_point(
    position_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    height: torch.Tensor | float,
    gravity: float = 9.81,
) -> torch.Tensor:
    """捕获点 :math:`\\xi = x + \\dot x/\\omega`，:math:`\\omega=\\sqrt{g/h}`。

    这是 :func:`footstep_planner.capture_point` 的向量化 torch 版本，
    一次算 N 个环境。测试断言两者在相同输入下逐元素一致。

    Args:
        position_xy: ``(N, 2)`` 质心（这里用基座近似）水平位置。
        velocity_xy: ``(N, 2)`` 水平速度。
        height: 标量或 ``(N,)``，质心高度。**必须为正** ——
            机器人蹲下时 h 变小、ω 变大、捕获点更靠近脚下，这个耦合是
            物理的，不要把 h 写死成常数。
        gravity: 重力加速度。

    Returns:
        ``(N, 2)`` 捕获点的水平位置。
    """
    h = torch.as_tensor(height, dtype=position_xy.dtype, device=position_xy.device)
    h = h.clamp(min=1e-3)
    omega = torch.sqrt(gravity / h)
    if omega.ndim == 0:
        omega = omega.expand(position_xy.shape[0])
    return position_xy + velocity_xy / omega.unsqueeze(-1)


def capture_point_error(
    base_pos_xy: torch.Tensor,
    base_vel_xy: torch.Tensor,
    base_height: torch.Tensor,
    foot_pos_xy: torch.Tensor,
    contact: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    """捕获点到**支撑足形心**的水平距离平方。

    M6 的结论是：把脚落在捕获点上，机器人渐近停下。反过来看，
    :math:`\\lVert\\xi - \\bar p_{\\text{stance}}\\rVert` 就是一个直接的
    "当前有多接近发散"的度量 —— 它比"基座高度掉了多少"要早得多地发出警报，
    因为高度是位置量，捕获点里含速度。

    把它取负当惩罚项，等于告诉策略：**不要让发散分量跑出支撑区域**。

    Args:
        base_pos_xy: ``(N, 2)``。
        base_vel_xy: ``(N, 2)`` 世界系水平速度。
        base_height: ``(N,)`` 基座离地高度。
        foot_pos_xy: ``(N, 4, 2)`` 四个足端的水平位置。
        contact: ``(N, 4)`` 布尔或 0/1，当前触地的腿。
        gravity: 重力加速度。

    Returns:
        ``(N,)`` 距离平方。**没有支撑腿时（腾空相）返回 0** ——
        腾空时谈支撑区域没有意义，硬算会给出无意义的巨大惩罚。
    """
    xi = capture_point(base_pos_xy, base_vel_xy, base_height, gravity)

    c = contact.float()
    n_stance = c.sum(dim=-1, keepdim=True)
    centroid = (foot_pos_xy * c.unsqueeze(-1)).sum(dim=1) / n_stance.clamp(min=1.0)

    error = torch.sum((xi - centroid) ** 2, dim=-1)
    return torch.where(n_stance.squeeze(-1) > 0, error, torch.zeros_like(error))


def reference_contact(
    time: torch.Tensor,
    period: float = 0.4,
    duty_factor: float = 0.5,
    offsets: tuple[float, ...] = TROT_OFFSETS,
) -> torch.Tensor:
    """M4 步态调度器的 torch 版：给定时刻的参考接触状态。

    与 :meth:`gait_scheduler.GaitScheduler.contact` **必须逐元素一致** ——
    测试就是这么断言的。同一个定义在 numpy 侧驱动 MPC，在 torch 侧驱动
    RL 奖励，两边不许漂移。

    Args:
        time: ``(N,)`` 每个环境自己的 episode 时间，秒。
            注意各环境重置时刻不同，**不能用全局步数**。
        period: 步态周期。
        duty_factor: 占空比。
        offsets: 四条腿的相位偏移，按 :data:`LEG_ORDER`。

    Returns:
        ``(N, 4)`` 布尔张量。
    """
    off = torch.as_tensor(offsets, dtype=time.dtype, device=time.device)
    phase = ((time.unsqueeze(-1) / period) - off) % 1.0
    return phase < duty_factor


def gait_contact_reward(
    contact: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """实际接触与参考接触的一致程度，取值 [0, 1]。

    **这是把 M4 塞进 RL 的最简形式。** 更讲究的做法（Unitree、MIT 的一些
    工作）用连续的相位相关量，甚至把相位本身放进观测让策略去"读钟"。
    这里保持最朴素的版本，因为它的效果已经很明显：没有它，PPO 学出来的
    步态在低速时经常退化成"四条腿同时小碎步"，看着很不自然。

    Args:
        contact: ``(N, 4)`` 实际接触。
        reference: ``(N, 4)`` 参考接触。

    Returns:
        ``(N,)``，四条腿匹配比例。
    """
    return (contact.bool() == reference.bool()).float().mean(dim=-1)


def foot_clearance_reward(
    foot_height: torch.Tensor,
    foot_vel_xy_norm: torch.Tensor,
    target_height: float = 0.08,
) -> torch.Tensor:
    """摆动腿抬脚高度惩罚，按足端水平速度加权。

    .. math::  r = -\\sum_i (h_i - h^*)^2 \\, \\lVert v_{xy,i}\\rVert

    **速度加权是关键**：脚站着不动时高度是多少无所谓（它就该贴地），
    只有在"正在往前迈"的时候才要求抬够。不加权的版本会逼着支撑脚
    也去够那个目标高度，直接把机器人顶起来。

    目标高度取 M5 摆动轨迹的最高点量级（Go2 上 0.08 m）。

    Args:
        foot_height: ``(N, 4)`` 足端离地高度。
        foot_vel_xy_norm: ``(N, 4)`` 足端水平速度大小。
        target_height: 期望抬脚高度。

    Returns:
        ``(N,)`` 惩罚（非负，使用时配负权重）。
    """
    return torch.sum((foot_height - target_height) ** 2 * foot_vel_xy_norm, dim=-1)


def foot_slip_penalty(foot_vel_xy: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """支撑脚打滑惩罚 :math:`\\sum_i \\lVert v_{xy,i}\\rVert^2 \\,[\\text{contact}_i]`。

    M7 的摩擦锥约束在 MPC 里是硬约束；到了 RL 这边没有约束这回事，
    只能用惩罚项来表达"支撑脚不该动"。这是无模型方法的普遍代价：
    **约束退化成惩罚，可行性退化成概率**。

    Args:
        foot_vel_xy: ``(N, 4, 2)`` 足端水平速度。
        contact: ``(N, 4)`` 接触标志。

    Returns:
        ``(N,)`` 惩罚。
    """
    return torch.sum(torch.sum(foot_vel_xy**2, dim=-1) * contact.float(), dim=-1)


def air_time_reward(
    air_time: torch.Tensor,
    first_contact: torch.Tensor,
    threshold: float = 0.5,
    command_norm: torch.Tensor | None = None,
    command_deadzone: float = 0.1,
) -> torch.Tensor:
    """腾空时间奖励：鼓励迈大步而不是高频小碎步。

    只在**刚落地那一帧**结算 :math:`(t_{air} - t^*)`，落地之前不给分 ——
    否则策略会发现"把脚永远举在空中"是最优解。这个坑几乎每个人都踩过一次。

    ``command_norm`` 传入时，指令接近零的环境不计此项：站着不动的时候
    奖励抬脚是矛盾的目标。

    Args:
        air_time: ``(N, 4)`` 各脚当前的腾空累计时间，秒。
        first_contact: ``(N, 4)`` 本帧是否刚从空中落地。
        threshold: 期望腾空时长，秒。取步态摆动相时长的量级
            （trot 周期 0.4 s、占空比 0.5 → 0.2 s）。
        command_norm: ``(N,)`` 速度指令模长。
        command_deadzone: 小于它认为是"站立指令"。

    Returns:
        ``(N,)`` 奖励，可正可负。
    """
    reward = torch.sum((air_time - threshold) * first_contact.float(), dim=-1)
    if command_norm is not None:
        reward = reward * (command_norm > command_deadzone).float()
    return reward


def _omega(height: float, gravity: float = 9.81) -> float:
    """LIPM 固有频率，供文档与测试引用。"""
    return math.sqrt(gravity / height)
