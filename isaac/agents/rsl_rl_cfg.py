"""rsl-rl 的训练配置。

这里的超参数与 :class:`rl.ppo.PPOConfig` **保持一致** —— 同一套超参数下，
两个 PPO 实现在同一个环境上应该给出统计上无法区分的学习曲线。这是本里程碑
的交叉验证方式，和前八个里程碑"闭式解 vs Pinocchio"是同一个套路。

## 写的是 rsl-rl ≥ 5.0 的新式配置

rsl-rl 4.0 起把原来的 ``policy=RslRlPpoActorCriticCfg(...)`` 拆成了并列的
``actor`` / ``critic`` 两个模型配置，旧写法已弃用。Isaac Lab 官方任务包里
的 Go2 配置**还是旧写法**，靠 ``handle_deprecated_rsl_rl_cfg()`` 在运行时
转换。本仓库直接写新式的，少一层转换，也少一处将来会坏掉的地方。

## ``obs_groups`` 就是非对称 Actor-Critic 的接线图

.. code-block:: python

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

一行配置决定了"策略只能看真机上拿得到的量，价值函数可以看特权信息"。
环境侧对应 :class:`isaac.go2_env_cfg.ObservationsCfg` 的两个组。

网络宽度上，平地版用 ``[128, 128, 128]``，崎岖地形因为多了 187 维高度扫描
用 ``[512, 256, 128]``。**策略网络的大小几乎从来不是四足 RL 的瓶颈** ——
瓶颈在奖励设计和域随机化。见过太多人在这里调网络结构调一周，
不如把 ``action_rate_l2`` 的权重改一下。
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

__all__ = ["Go2FlatPPORunnerCfg", "Go2RoughPPORunnerCfg"]


@configclass
class Go2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """崎岖地形。

    ``num_steps_per_env=24`` × ``num_envs=4096`` = 每次迭代 98304 步。
    1500 次迭代 ≈ 1.5 亿步 ≈ 仿真里的 34 天，实机上不可能采集到 ——
    这就是仿真训练的全部意义。
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_rough"

    #: 非对称 Actor-Critic 的接线：actor 看 ``policy`` 组，critic 看含特权
    #: 信息的 ``critic`` 组。
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    #: 动作限幅。高斯策略的长尾偶尔会给出幅值 10+ 的动作，
    #: 乘上 ``action_scale`` 后足以让仿真器出 NaN。
    clip_actions = 100.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,  # critic 输出确定性的标量值
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2FlatPPORunnerCfg(Go2RoughPPORunnerCfg):
    """平地。300 次迭代足够走得像样，是调试整条管线的正确入口。"""

    def __post_init__(self) -> None:
        self.max_iterations = 300
        self.experiment_name = "go2_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]
