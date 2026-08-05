"""残差强化学习的 Go2 环境：里程碑 9 的环境 + 名义步态基线。

**只改三处**，其余原样继承 —— 这本身就是一个结论：残差 RL 不是另一套方法，
它是同一个 MDP 换了一个动作的参数化。

1. **动作** —— :class:`~isaac.mdp.actions.ResidualJointPositionAction`
   把策略输出叠加到解析步态上，而不是叠加到固定站姿上。
2. **观测** —— 多两项：名义动作（12 维）与步态相位的 sin/cos（8 维）。
   两项都是必需的，理由见 :mod:`isaac.mdp.observations`。
3. **课程** —— 残差幅值 :math:`\\alpha` 与推力强度各一条。

奖励**一项都没改**。这是刻意的：里程碑 9 和 10 的唯一自变量必须是动作的
参数化，否则两者的对比就不干净。唯一"新"的一项 ``action_l2`` 其实是内置项，
只是在残差设定下有了新含义 —— 见下。

## ``action_l2`` 在残差设定下是"贴着先验走"的正则

纯 RL 里 ``action_l2`` 惩罚的是"关节离默认站姿多远"，含义模糊，
所以里程碑 9 里权重给 0（没启用）。

残差 RL 里它变成

.. math::  -w\\lVert a\\rVert^2 \\propto -\\lVert q^{des} - q_{\\text{nom}}\\rVert^2

**即"偏离名义步态的代价"**。这是一个有明确含义的正则：策略只在偏离能换来
更高回报的地方才偏离。权重给得越大，学出来的东西越接近名义控制器；给 0
则退化成"名义控制器只是一个初始化"。这个旋钮是残差 RL 里最值得调的一个。

## 一个必须说清楚的部署代价

名义控制器的 Raibert 项需要**实测躯干水平速度**。里程碑 9 的纯 RL 策略
可以完全不依赖它（线速度只进 critic），残差 RL 不行 —— 名义控制器在
真机上也要跑，也要那个速度。

**所以残差 RL 把里程碑 3 的 ESKF 从"可选"变成了"必需"。** 换来的是更好的
样本效率和有界的行为。这个交换划不划算取决于你的状态估计有多可信 ——
这正是前八个里程碑的价值在 RL 路线上的体现。
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaac.mdp as mdp
from isaac.go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg
from rl.nominal_controller import NominalGaitConfig

__all__ = [
    "Go2ResidualFlatEnvCfg",
    "Go2ResidualFlatEnvCfg_PLAY",
    "Go2ResidualFlatPushEnvCfg",
    "Go2ResidualRoughEnvCfg",
    "Go2ResidualRoughEnvCfg_PLAY",
]

#: 名义步态：trot，与里程碑 4 的定义、里程碑 9 的 ``gait_phase`` 奖励同参数。
#: 三处用同一组数字不是巧合 —— 名义控制器、步态奖励、参考接触序列本来就该
#: 说同一件事。
GO2_TROT = NominalGaitConfig(
    period=0.4,
    duty_factor=0.5,
    phase_offsets=(0.0, 0.5, 0.5, 0.0),
    stand_height=0.30,
    swing_height=0.08,
    raibert_gain=0.03,
    max_stride=0.20,
)


def _apply_residual(cfg, push_curriculum: bool = False) -> None:
    """把一个里程碑 9 的环境配置改造成残差版本。就地修改。

    Args:
        cfg: 里程碑 9 的环境配置。
        push_curriculum: 是否启用推力课程。**缺省关闭**，因为开启后环境的
            扰动强度会与里程碑 9 不同，"残差 vs 纯 RL"的对比就不干净了 ——
            实测开启后推力会一路升到上限 2.0 m/s 且每 3~6 秒一次，回报从
            35.0 掉到 19.5，而这个差别完全来自环境变难，与算法无关。
            抗推研究请用单独注册的 ``Go2-Residual-*-Push-v0``。
    """
    # -- 1. 动作 --------------------------------------------------------
    cfg.actions.joint_pos = mdp.ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        residual_scale=0.25,
        command_name="base_velocity",
        gait=GO2_TROT,
    )

    # -- 2. 观测 --------------------------------------------------------
    # policy: 45 + 12 + 8 = 65；critic: 48 + 12 + 8 = 68
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.nominal_action = ObsTerm(func=mdp.nominal_action, params={"action_name": "joint_pos"})
        group.gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"action_name": "joint_pos"})

    # -- 3. 正则：偏离名义步态的代价 ------------------------------------
    cfg.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)

    # -- 4. 课程 --------------------------------------------------------
    cfg.curriculum.residual_scale = CurrTerm(
        func=mdp.residual_scale_curriculum,
        params={
            "action_name": "joint_pos",
            "start_scale": 0.05,
            "end_scale": 0.25,
            "start_step": 0,
            "end_step": 5_000_000,
        },
    )
    if push_curriculum:
        cfg.curriculum.push_velocity = CurrTerm(
            func=mdp.push_velocity_curriculum,
            params={
                "event_name": "push_robot",
                "start_velocity": 0.3,
                "max_velocity": 2.0,
                "step_velocity": 0.1,
                "survival_threshold": 0.9,
            },
        )
        # 推力课程接管这一项，起始范围必须与 ``start_velocity`` 一致，
        # 且推得更频繁 —— 抗扰能力是被推出来的，一条 episode 只推一次太少。
        cfg.events.push_robot.interval_range_s = (3.0, 6.0)
        cfg.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class Go2ResidualFlatEnvCfg(Go2FlatEnvCfg):
    """平地残差 RL。与里程碑 9 的平地环境逐项可比。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_residual(self)


@configclass
class Go2ResidualRoughEnvCfg(Go2RoughEnvCfg):
    """崎岖地形残差 RL。名义控制器完全不知道地形长什么样 ——
    **地形自适应的全部内容都在残差里**，这让残差的可解释性变得很有用：
    把残差按地形等级统计一下，就能看出策略在什么地形上偏离先验最多。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_residual(self)


@configclass
class Go2ResidualFlatPushEnvCfg(Go2ResidualFlatEnvCfg):
    """平地残差 RL + 推力课程。**专供抗推研究，不用来和里程碑 9 对比。**

    推力按存活率自适应升降，实测会一路升到上限 2.0 m/s、每 3~6 秒一次。
    环境比里程碑 9 难得多，回报不可直接比。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_residual(self, push_curriculum=True)


@configclass
class Go2ResidualFlatEnvCfg_PLAY(Go2ResidualFlatEnvCfg):
    """平地评估。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        # 评估时把课程全部关掉：α 固定在终值，推力由评估脚本自己控制
        self.curriculum.residual_scale = None
        self.curriculum.push_velocity = None


@configclass
class Go2ResidualRoughEnvCfg_PLAY(Go2ResidualRoughEnvCfg):
    """崎岖地形评估。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.curriculum.residual_scale = None
        self.curriculum.push_velocity = None
        self.curriculum.terrain_levels = None
