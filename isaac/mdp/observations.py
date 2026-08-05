"""残差 RL 需要的额外观测项。

残差策略比纯 RL 策略多需要两样东西，而且**两样都是必需的，不是可选的**：

1. **名义动作 :math:`q_{\\text{nom}}`。** 策略要修正一个东西，就得知道那个
   东西是什么。不给的话，策略只能从关节位置反推名义控制器此刻想干什么 ——
   可以学，但白白浪费容量和样本。
2. **步态相位。** 名义控制器是**时间驱动**的：同样的机器人状态，在摆动相
   前段和后段需要完全不同的修正。不给相位，MDP 对策略而言就不是马尔可夫的
   —— 这与里程碑 9 里"上一步动作必须进观测"是同一类问题，只是这次
   隐藏状态是相位而不是执行器滞后。

相位用 :math:`(\\sin 2\\pi\\phi, \\cos 2\\pi\\phi)` 而不是 :math:`\\phi` 本身：
相位是**环形量**，0.99 和 0.01 只差 0.02 个周期，但数值上差 0.98。直接把
锯齿波喂给 MLP，网络必须浪费容量去学那个跳变。这与四元数不能用欧拉角替代、
角度误差要用 :math:`\\operatorname{atan2}(\\sin, \\cos)` 是同一个道理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["nominal_action", "gait_phase_sin_cos"]


def nominal_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """名义控制器当前给出的 12 维关节角，减去默认站姿。

    减默认站姿是为了让这一项零均值、量级与 ``joint_pos_rel`` 一致 ——
    观测归一化虽然会在线做同样的事，但从一开始就把量级摆正能少烧几十次迭代。
    """
    term = env.action_manager.get_term(action_name)
    default = env.scene["robot"].data.default_joint_pos
    return term.nominal_actions - default


def gait_phase_sin_cos(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """四条腿相位的 ``(sin, cos)`` 编码，8 维。

    直接复用动作项里那个名义控制器的相位定义，**不另算一份** ——
    两处各算一次迟早会漂移，这是本仓库反复强调的"约定只写一次"。
    """
    term = env.action_manager.get_term(action_name)
    phase = term.nominal.phase(env.episode_length_buf.float() * env.step_dt)  # (N, 4)
    angle = 2.0 * torch.pi * phase
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
