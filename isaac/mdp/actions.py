"""残差动作项：把策略的输出叠加到名义步态控制器上。

## 一行公式

.. math::  q^{des} = \\underbrace{q_{\\text{nom}}(s)}_{\\text{M1/4/5/6 解析步态}}
           + \\underbrace{\\alpha\\,a}_{\\text{策略学的残差}}

对比里程碑 9 的纯 RL：

.. math::  q^{des} = q^{\\text{default}} + 0.25\\,a

差别只有一项：**基线从"固定站姿"换成了"一个已经会走路的控制器"**。
但这一项改变了三件事：

1. **探索的起点。** 纯 RL 的第一帧是站着不动 + 高斯噪声；残差 RL 的第一帧
   已经在按 trot 迈步。策略不需要从零发明步态。
2. **最坏情况有界。** :math:`\\lVert q^{des} - q_{\\text{nom}}\\rVert_\\infty
   \\le \\alpha\\lVert a\\rVert_\\infty`。给定动作限幅，**策略再离谱也不会
   离名义步态太远** —— 这是纯 RL 给不了的、可以写进安全论证的界。
3. **可解释性。** 残差本身就是一个可以画出来、可以统计的量："模型在哪里
   不够用"。这是里程碑 1–8 的建模路线与 RL 路线的真正接口。

## 与无人机的类比（这次几乎是同构的）

你在无人机上做过的 **NMPC + 学习残差动力学**：名义模型给出前馈，
神经网络补上未建模的气动效应。这里一模一样，只是残差补在**动作**上而不是
**动力学**上：

| 无人机 | 四足残差 RL |
|---|---|
| 名义刚体模型 | M1/4/5/6 的解析步态 |
| 学到的气动残差 | 策略输出的关节角残差 |
| 补偿的是：地效、桨叶挥舞 | 补偿的是：接触、打滑、地形、腿的惯量 |
| 模型错了会怎样 | 残差要先"抵消"名义，反而更慢 |

最后一行是**残差 RL 唯一的真陷阱**：名义控制器若给出的是有害的动作，
策略必须先花样本把它抵消掉，收敛比纯 RL 还慢。所以 :math:`\\alpha` 的取值
和名义控制器的质量必须一起考虑，见 `docs/10_residual_rl.md` 第 6 节。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from rl.nominal_controller import NominalGaitConfig, NominalGaitController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["ResidualJointPositionAction", "ResidualJointPositionActionCfg"]


class ResidualJointPositionAction(JointPositionAction):
    """名义步态 + 策略残差的关节位置动作。

    继承 :class:`JointPositionAction` 只为了复用它的关节解析与
    ``set_joint_position_target``；``process_actions`` 被整个换掉。

    名义控制器需要三个输入，全部从 ``env`` 现取：

    * **episode 时间** —— ``episode_length_buf * step_dt``。用每个环境自己的
      计时器，不是全局步数：各环境重置时刻不同，用全局时间会让名义步态与
      机器人的实际节奏脱节（里程碑 9 的 ``gait_phase`` 奖励踩过同样的坑）。
    * **速度指令** —— 来自 CommandManager。
    * **实测躯干水平速度** —— 仿真里直接读。**真机上这一项要靠里程碑 3 的
      ESKF**，也就是说残差 RL 的部署门槛比里程碑 9 的纯 RL 更高：
      纯 RL 的策略观测里可以完全不含线速度，但名义控制器的 Raibert 项离不开它。
      这是一个必须说清楚的代价。
    """

    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        gait_cfg = cfg.gait or NominalGaitConfig()
        # 名义控制器按本动作项解析出的关节顺序输出，避免任何顺序假设
        gait_cfg.joint_order = tuple(self._joint_names)
        self.nominal = NominalGaitController(gait_cfg, device=self.device)
        self._foot_ids, _ = self._asset.find_bodies(cfg.foot_body_names)

        #: 残差幅值 :math:`\\alpha`。课程项可以在训练中改写它。
        self.residual_scale = float(cfg.residual_scale)
        self._nominal_actions = torch.zeros_like(self._raw_actions)

    # ------------------------------------------------------------------ 属性

    @property
    def nominal_actions(self) -> torch.Tensor:
        """最近一次的名义关节角 ``(N, 12)``，供观测项与奖励项读取。"""
        return self._nominal_actions

    # ------------------------------------------------------------------ 操作

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self.cfg.clip is not None:
            self._raw_actions[:] = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        env = self._env
        asset = env.scene[self.cfg.asset_name]
        time = env.episode_length_buf.float() * env.step_dt
        command = env.command_manager.get_command(self.cfg.command_name)
        base_vel = asset.data.root_lin_vel_b[:, :2]
        # 足端的竖直基准用**实测**离地高度：以四足最低点当局部地面，
        # 这样地形起伏和位置控制下的躯干下沉都被自动吸收。
        # 用固定的 stand_height 会让摆动腿提前触地并把机身往后推，
        # 详见 :meth:`rl.nominal_controller.NominalGaitController.foot_targets`。
        foot_z = asset.data.body_pos_w[:, self._foot_ids, 2]
        base_height = asset.data.root_pos_w[:, 2] - foot_z.min(dim=1)[0]

        self._nominal_actions = self.nominal.compute(time, command, base_vel, base_height)
        self._processed_actions = self._nominal_actions + self.residual_scale * self._raw_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is not None:
            self._nominal_actions[env_ids] = 0.0


@configclass
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    """残差动作项的配置。

    Attributes:
        residual_scale: :math:`\\alpha`，残差的幅值上限系数。0 表示纯名义
            控制器（可用来单独评估名义控制器的水平）。

            **缺省取 0.25，与里程碑 9 的 ``action_scale`` 完全相同。**
            这是刻意的：两条路线的动作权限一模一样，唯一的差别是叠加在
            什么基线上，于是"残差 RL 到底值不值"这个问题才被问干净了。

            取更小的值（0.05~0.1）会把策略**铐在先验上** —— 先验好时这是
            优点（搜索空间小、行为有界），先验差时就是灾难：策略连抵消
            先验的权限都没有。α 的选择本质上是"你有多信任那个名义控制器"。
        command_name: 速度指令项的名字。
        gait: 名义步态参数；``None`` 用缺省的 Go2 trot。
        foot_body_names: 足端 body 的正则，用来估计躯干离地高度。
    """

    class_type: type[ActionTerm] = ResidualJointPositionAction

    residual_scale: float = 0.25
    command_name: str = "base_velocity"
    gait: NominalGaitConfig | None = None
    foot_body_names: str = ".*_foot"

    #: 基类的 ``scale`` / ``use_default_offset`` 在这里没有意义 —— 偏置由
    #: 名义控制器逐步给出，不是一个常数站姿。显式钉死以免被误配。
    scale: float = 1.0
    use_default_offset: bool = False
    asset_name: str = MISSING
