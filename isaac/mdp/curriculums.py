"""里程碑 10 的课程项：残差幅值与推力强度。

Isaac Lab 的课程项每次 episode 重置时被调用，签名是
``term(env, env_ids, ...) -> float | Tensor``，返回值会被记进日志。
它可以**任意改写环境的配置** —— 课程学习在这个框架里就是"训练中途改配置"。

## 两条课程各自在解决什么

### 残差幅值 :math:`\\alpha`：从"信任模型"到"信任数据"

训练开始时策略是随机的，:math:`\\alpha` 大意味着随机噪声被放大后直接
破坏名义步态，机器人从第一帧就摔 —— 残差 RL 的优势（从一个会走路的
控制器起步）当场清零。所以 :math:`\\alpha` 从小开始，随着策略变好再放开。

这与信任域的思想是同一件事，只是被控对象换了：PPO 的自适应学习率约束的是
"策略每次更新走多远"，:math:`\\alpha` 课程约束的是"策略允许偏离先验多远"。
**两者都是在管理同一种风险：在还不知道对不对的时候，别走太远。**

### 推力：抗扰能力只能被推出来

不推，策略永远见不到大的速度扰动，学不出恢复策略；一上来就推很大，
早期全部摔倒、没有有效梯度。所以推力也要跟着能力涨。

判据用**存活率**而不是迭代数：能扛住就加大，扛不住就退回。这与
``terrain_levels_vel`` 按"走过的距离"升降级是同一套逻辑 ——
**课程的自变量必须是能力的度量，不是时间。** 按迭代数线性拉满是最常见的
错误做法：策略学得快时被拖慢，学得慢时被推垮。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["residual_scale_curriculum", "push_velocity_curriculum", "push_survival_rate"]


def residual_scale_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_name: str = "joint_pos",
    start_scale: float = 0.02,
    end_scale: float = 0.10,
    start_step: int = 0,
    end_step: int = 2_000_000,
) -> float:
    """把残差幅值 :math:`\\alpha` 从 ``start_scale`` 线性拉到 ``end_scale``。

    这一条**按环境交互步数**推进，而不是按能力 —— 因为"残差该放多大"没有
    直接的能力度量可用（回报同时受地形课程影响，分不干净）。属于诚实的
    工程折中：能按能力走的（地形、推力）就按能力，不能的就按步数，
    但要知道自己在近似什么。

    Args:
        env: 环境。
        env_ids: 本次重置的环境（本项不使用，α 是全局量）。
        action_name: 残差动作项的名字。
        start_scale / end_scale: 起止幅值。
        start_step / end_step: 起止的总交互步数。

    Returns:
        当前的 :math:`\\alpha`，会出现在训练日志里。
    """
    term = env.action_manager.get_term(action_name)
    total_steps = env.common_step_counter * env.num_envs

    if total_steps <= start_step:
        alpha = start_scale
    elif total_steps >= end_step:
        alpha = end_scale
    else:
        ratio = (total_steps - start_step) / max(end_step - start_step, 1)
        alpha = start_scale + ratio * (end_scale - start_scale)

    term.residual_scale = float(alpha)
    return float(alpha)


def push_velocity_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str = "push_robot",
    start_velocity: float = 0.3,
    max_velocity: float = 2.0,
    step_velocity: float = 0.1,
    survival_threshold: float = 0.9,
) -> float:
    """按存活率升降推力强度。

    每次重置时统计这批环境里"因超时结束"（= 没摔）的比例：

    * 比例 > ``survival_threshold`` → 推力 +``step_velocity``；
    * 比例 < ``survival_threshold`` / 2 → 推力 −``step_velocity``。

    **升快降慢是刻意的**（升 1 档要 90% 存活，降 1 档要跌破 45%）：
    推力课程一旦冲太快，策略会陷入"全都摔"的区域，那里梯度接近纯噪声，
    自己爬不回来。宁可慢一点。

    Args:
        env: 环境。
        env_ids: 本次重置的环境。
        event_name: 推力事件项的名字。
        start_velocity: 初始推力（速度突变的幅值），m/s。
        max_velocity: 上限。
        step_velocity: 每次升降的步长。
        survival_threshold: 升级所需的存活率。

    Returns:
        当前推力幅值。
    """
    # ``EventManager.active_terms`` 是 ``{模式: [名字, ...]}``，不是名字列表。
    #
    # 这里曾经写成 ``if event_name not in env.event_manager.active_terms``，
    # 于是在比对**模式名**（startup/reset/interval），永远不匹配，
    # **整条推力课程被静默关掉**：训练日志里 ``push_velocity`` 全程为 0，
    # 而训练照跑、回报照涨，没有任何报错。
    #
    # 教训：防御性代码写错比不写更糟 —— 它把一个会报错的 bug 变成了
    # 一个不报错的 bug。写这类降级分支时，一定要有一个测试去撞它。
    if not any(event_name in names for names in env.event_manager.active_terms.values()):
        # 推力事件确实被关掉了（例如评估配置）—— 退化成"没有课程"。
        return 0.0

    term_cfg = env.event_manager.get_term_cfg(event_name)
    current = term_cfg.params["velocity_range"]["x"][1]
    if current == 0.0:
        current = start_velocity

    if len(env_ids) > 0:
        # time_out 为真表示跑满了 episode，即没摔
        survived = env.termination_manager.time_outs[env_ids].float().mean().item()
        if survived > survival_threshold:
            current = min(current + step_velocity, max_velocity)
        elif survived < survival_threshold * 0.5:
            current = max(current - step_velocity, start_velocity)

    term_cfg.params["velocity_range"] = {"x": (-current, current), "y": (-current, current)}
    return float(current)


def push_survival_rate(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """纯监控项：本批重置里没摔的比例。不改任何配置。

    单独列出来是因为**课程项的返回值是唯一会进日志的通道**。想看某个量的
    演化又不想改配置时，写一个只读的课程项是 Isaac Lab 里最省事的做法。
    """
    if len(env_ids) == 0:
        return torch.zeros((), device=env.device)
    return env.termination_manager.time_outs[env_ids].float().mean()
