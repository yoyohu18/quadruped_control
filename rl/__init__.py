"""强化学习：从数据里直接学出运动控制器。

前八个里程碑走的是"建模 → 优化"的路线：写下动力学，写下代价，解 QP。
这里换一条完全不同的路 —— **不建模，直接从交互数据里学**。

模块划分刻意与 Isaac Sim 解耦：

* :mod:`rl.networks` —— Actor-Critic，对角高斯策略。
* :mod:`rl.normalization` —— 观测的在线均值/方差。
* :mod:`rl.storage` —— rollout 缓冲与 GAE（含超时 bootstrap 的正确处理）。
* :mod:`rl.ppo` —— PPO 更新：clipped surrogate + 自适应 KL 学习率。
* :mod:`rl.runner` —— on-policy 训练循环。
* :mod:`rl.toy_env` —— 纯 torch 的向量化玩具环境，让整套 PPO 可以在
  **不启动仿真器**的情况下被回归测试。
* :mod:`rl.reward_kernels` —— 奖励的数学核，M4/M5/M6 的成果在这里以
  奖励项的身份复用。
* :mod:`rl.nominal_controller` —— 里程碑 10 的名义步态控制器：M1 的闭式
  IK、M4 的相位、M5 的摆动轨迹、M6 的落脚点，全部重写成批量 torch，
  一次算 4096 个环境。残差 RL 的 :math:`a_{\\text{nom}}` 就是它。
* :mod:`rl.vec_env` —— Isaac Lab 环境到本项目训练循环的适配层。

Isaac Lab 相关的一切（场景、MDP 配置、注册）都在 :mod:`isaac` 里，
本包不 import 它，因此 ``pytest tests/test_rl.py`` 毫秒级完成。
"""

from .networks import ActorCritic, build_mlp
from .nominal_controller import (
    NominalGaitConfig,
    NominalGaitController,
    batched_inverse_kinematics,
)
from .normalization import EmpiricalNormalization, RunningMeanStd
from .ppo import PPO, PPOConfig
from .reward_kernels import (
    LEG_ORDER,
    TROT_OFFSETS,
    air_time_reward,
    capture_point,
    capture_point_error,
    exp_tracking_reward,
    foot_clearance_reward,
    foot_slip_penalty,
    gait_contact_reward,
    reference_contact,
)
from .runner import OnPolicyRunner, RunnerConfig
from .storage import Batch, RolloutStorage, compute_gae
from .toy_env import ToyEnvConfig, VelocityTrackingToyEnv

__all__ = [
    "ActorCritic",
    "Batch",
    "EmpiricalNormalization",
    "NominalGaitConfig",
    "NominalGaitController",
    "LEG_ORDER",
    "OnPolicyRunner",
    "PPO",
    "PPOConfig",
    "RolloutStorage",
    "RunnerConfig",
    "RunningMeanStd",
    "TROT_OFFSETS",
    "ToyEnvConfig",
    "VelocityTrackingToyEnv",
    "air_time_reward",
    "batched_inverse_kinematics",
    "build_mlp",
    "capture_point",
    "capture_point_error",
    "compute_gae",
    "exp_tracking_reward",
    "foot_clearance_reward",
    "foot_slip_penalty",
    "gait_contact_reward",
    "reference_contact",
]
