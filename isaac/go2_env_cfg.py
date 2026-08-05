"""Go2 速度跟踪环境：从零写的 manager-based 配置。

## 为什么不直接继承 Isaac Lab 的 ``LocomotionVelocityRoughEnvCfg``

继承当然更省事，官方配置也确实能训出能走的策略。但那样一来，**这个里程碑
学到的东西只有"改几个权重"**。从零写一遍，才会被迫回答每一个设计问题：
观测里该放什么、动作是什么、奖励里每一项在补偿什么、什么时候该终止。

写完之后再和官方配置逐项对比，差异本身就是最好的教材 —— 下面每一处
"与 Isaac Lab 官方不同"的注释，都是一个可以在面试里讲十分钟的点。

## 三个最重要的设计决策

### 1. 动作是关节目标位置，不是力矩

.. math::  \\tau = K_p\\,(q^{des} - q) - K_d\\,\\dot q,\\qquad
           q^{des} = q^{default} + s\\cdot a

策略以 50 Hz 输出 :math:`a`，PD 以 500 Hz 跑。**为什么不直接输出力矩？**

* PD 提供了一个"内环"，把执行器的高频动态挡在策略之外。策略只需要决定
  "腿该摆到哪"，不需要决定"每一毫秒该出多大力"。
* 力矩策略对 sim-to-real gap 极其敏感：仿真里 1 Nm 的建模误差在位置控制下
  被 PD 自动补偿，在力矩控制下直接变成轨迹误差。
* 50 Hz 的力矩指令在真机上会激起结构共振。

> 这与无人机的分层完全一致：外环给姿态角/角速度指令，内环 PID 跑到
> 几百 Hz 出电机转速。**没人让神经网络直接输出 PWM。**

``scale=0.25`` 意味着策略最多把关节从默认位置挪 ±0.25 rad × (动作幅值)。
它同时是一个隐式的安全限幅。

### 2. 基座线速度不进 policy 观测

Isaac Lab 官方把 ``base_lin_vel`` 放在 policy 组里。**真机上没有这个量** ——
IMU 只给加速度和角速度，线速度必须靠里程碑 3 的 ESKF 融合估出来，
而且估计值有偏差、有延迟、在打滑时会发散。

所以这里把它挪进 critic 组（训练时可见，部署时不需要）。代价是策略更难学
（少了最直接的反馈量），收益是**策略学到的东西真机上跑得起来**。这正是
非对称 Actor-Critic 存在的意义。

> 想在真机上用官方那套配置的话，M3 的状态估计就是必需品 —— 前八个里程碑
> 与 RL 的第一个真实交汇点。

### 3. 奖励 = 任务项 + 正则项 + 物理先验项

* **任务项**（正权重）：线速度、角速度跟踪，指数核。
* **正则项**（负权重）：垂直速度、横滚俯仰角速度、力矩、关节加速度、
  动作变化率、关节限位。它们不定义任务，只是把解从"能拿分但很丑"的
  区域里推开。
* **物理先验项**：捕获点、步态相位、抬脚高度、打滑 —— 里程碑 4/5/6 的
  成果。它们是本项目相对官方配置的增量。

**权重调不好，什么算法都白搭。** 一个实用的判据：训练结束时打印每一项的
累计贡献，任何一项如果绝对值超过任务项的 30%，基本就是权重给大了。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaac.mdp as mdp

__all__ = [
    "Go2FlatEnvCfg",
    "Go2FlatEnvCfg_PLAY",
    "Go2RoughEnvCfg",
    "Go2RoughEnvCfg_PLAY",
]

#: 足端 body 名的正则。Go2 的 body 顺序在冒烟测试里核对过：
#: ``FL_foot, FR_foot, RL_foot, RR_foot`` —— 与 :data:`kinematics.LEGS` 同序。
FOOT_BODIES = ".*_foot"


##
# 场景
##


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """地形、机器人、传感器、灯光。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #: 高度扫描：以基座为中心的 1.6 × 1.0 m 网格，分辨率 0.1 m → 187 个点。
    #: 只在崎岖地形版本里启用。真机上要靠深度相机 + 高程图重建，
    #: 噪声和空洞比仿真里恶劣得多 —— 这是崎岖地形 sim-to-real 的主要难点。
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    #: ``history_length=3`` 给接触消抖，``track_air_time`` 提供腾空时间。
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """速度指令：MDP 的"任务"就藏在这里。

    ``heading_command=True`` 时，角速度指令由航向误差经比例控制生成，
    而不是独立采样。这样机器人学到的是"朝着某个方向走"，而不是
    "原地按指定角速度转"，前者才是导航层真正会下发的指令。

    ``rel_standing_envs=0.02`` 让 2% 的环境收到零指令 —— **必须有**，
    否则策略永远学不会站住不动，一停下来就抖。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.7, 0.7),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """动作：12 个关节的目标位置偏移。

    ``use_default_offset=True`` 让动作叠加在默认站姿上，于是**零动作
    对应站立**。配合网络输出层的小增益初始化，训练第一帧机器人是站着的，
    而不是四条腿乱抽。
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """观测：非对称 Actor-Critic。

    * ``policy`` 组 —— 真机上拿得到的量，且全部加了噪声。
    * ``critic`` 组 —— 训练专用，含特权信息，**不加噪声**
      （critic 不需要鲁棒性，它只需要准）。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """45 维：IMU 3+3、指令 3、关节 12+12、上一步动作 12。

        噪声幅值不是随便填的，对应真实传感器的量级：陀螺 0.2 rad/s、
        关节编码器 0.01 rad、关节速度 1.5 rad/s（差分出来的，噪声很大）。
        **域随机化里最有效的一项往往就是观测噪声** —— 它比动力学随机化
        更直接地告诉策略"别太相信任何单个传感器"。
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """48 维：policy 的 45 维 + 基座线速度 3 维（特权）。

        基座线速度是**最有价值的特权信息**：速度跟踪任务的误差直接由它
        决定，critic 有了它，价值估计的误差能降一个量级，优势的方差随之
        下降。真机部署时这一组网络整个丢掉。
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """域随机化：sim-to-real 的全部希望所在。

    分三种触发模式：

    * ``startup`` —— 场景建好时执行一次。用于**不随时间变化**的参数：
      摩擦系数、连杆质量、质心位置。
    * ``reset`` —— 每次 episode 重置时执行。用于初始状态分布。
    * ``interval`` —— 训练中按时间间隔触发。用于外部扰动。

    **随机化范围怎么定？** 不是拍脑袋，而是"真机上这个量的不确定度有多大"。
    Go2 整机 15 kg，负载变化 ±3 kg 是现实的；地面摩擦系数在瓷砖和地毯之间
    可以从 0.4 变到 1.2。范围给太宽会让策略过度保守（走得又慢又蹲），
    太窄则一上真机就摔。
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )

    #: 随机推一把。抗推恢复能力主要来自这一项 —— 里程碑 10 会把它做成课程。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 13.0),
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
    )


@configclass
class RewardsCfg:
    """奖励。分组与权重量级见模块文档。"""

    # -------------------------------------------------- 任务项
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -------------------------------------------------- 正则项
    #: 垂直速度：躯干上下颠是最难看也最费电的失败模式。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    #: **动作变化率是抖动的直接对手。** 权重不够大，学出来的策略在真机上
    #: 会发出刺耳的高频嗡鸣，电机温度飙升。这是 sim-to-real 最常见的翻车点。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )

    # -------------------------------------------------- 里程碑 4/5/6 的先验
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.2,  # trot 摆动相时长 = 0.4 s × (1 - 0.5)
        },
    )
    gait_phase = RewTerm(
        func=mdp.gait_phase_tracking,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "command_name": "base_velocity",
            "period": 0.4,
            "duty_factor": 0.5,
        },
    )
    capture_point = RewTerm(
        func=mdp.capture_point_stability,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES)},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "target_height": 0.08},
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )


@configclass
class TerminationsCfg:
    """终止条件。

    ``time_out=True`` 这个标记**极其重要** —— 它让 Isaac Lab 把这一项归入
    "截断"而非"终止"，进而在 ``extras["time_outs"]`` 里报出来，PPO 才能
    正确 bootstrap。写漏了的话训练照跑，但策略会学到"活到 20 秒是坏事"。
    详见 :func:`rl.storage.compute_gae`。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """课程学习：让任务难度跟着能力涨。

    ``terrain_levels_vel`` 的规则很朴素：一条 episode 里走过的距离超过
    指令距离的一半就升一级，不到一半就降一级。**这是把"学不会就换个简单的"
    自动化了** —— 没有它，4096 个环境全扔在最难的地形上，早期没有任何一条
    轨迹能拿到有效奖励，梯度接近纯噪声。

    平地版本不需要地形课程，但仍然可以做**指令课程**（速度范围逐步放宽），
    见 :class:`Go2FlatEnvCfg`。
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# 环境配置
##


@configclass
class Go2RoughEnvCfg(ManagerBasedRLEnvCfg):
    """崎岖地形上的 Go2 速度跟踪。"""

    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 物理 200 Hz，策略 50 Hz（decimation=4）。
        # 策略频率不是越高越好：50 Hz 已经远高于四足的机械带宽（~10 Hz），
        # 再高只会让 episode 变长、样本变相关、训练变慢。
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器更新周期必须与使用它的频率对齐：
        # 高度扫描按策略频率（50 Hz）就够，接触力必须按物理频率，
        # 否则腾空时间的统计会掉帧。
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 崎岖地形版本给 policy 加上高度扫描（187 维）
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        # 地形尺度按 Go2 的体型缩小 —— 官方默认是给 ANYmal（大一圈）调的
        gen = self.scene.terrain.terrain_generator
        if gen is not None:
            gen.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            gen.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            gen.sub_terrains["random_rough"].noise_step = 0.01
            gen.curriculum = getattr(self.curriculum, "terrain_levels", None) is not None


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    """平地版本。**先在这里跑通，再上崎岖地形。**

    平地训练 300 次迭代就能走得像样 —— RTX 5080 上实测 **2 分 45 秒**，
    是调试整条管线的正确入口。崎岖地形要 1500 次以上。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # 平地上没有地形课程，改用指令课程：先学慢的，再逐步放开速度范围。
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)


@configclass
class Go2RoughEnvCfg_PLAY(Go2RoughEnvCfg):
    """评估用：环境少、无噪声、无扰动、无课程。

    **评估必须关掉随机化。** 否则你看到的"策略表现"里混着运气成分，
    两次跑分不可比，调参就成了玄学。
    """

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


@configclass
class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    """平地评估。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
