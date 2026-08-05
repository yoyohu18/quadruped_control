"""Isaac Lab 环境：Go2 速度跟踪任务的注册入口。

import 本包会向 gymnasium 注册八个任务 id：

纯强化学习（里程碑 9），动作叠加在固定站姿上：

* ``Go2-Velocity-Flat-v0`` / ``Go2-Velocity-Flat-Play-v0``
* ``Go2-Velocity-Rough-v0`` / ``Go2-Velocity-Rough-Play-v0``

残差强化学习（里程碑 10），动作叠加在解析步态控制器上：

* ``Go2-Residual-Flat-v0`` / ``Go2-Residual-Flat-Play-v0``
* ``Go2-Residual-Rough-v0`` / ``Go2-Residual-Rough-Play-v0``

两组环境**除了动作的参数化之外完全相同**（同样的观测项、奖励项、终止条件、
域随机化），这样两条路线的对比才干净。

**注意：本包只能在 Isaac Sim 启动之后 import。** ``isaaclab.*`` 依赖
``pxr``，而 ``pxr`` 由 ``AppLauncher`` 注入 ``sys.path``。所有脚本都必须
先起 App 再 import 这里 —— 这是 Isaac Lab 用户第一天必踩的坑，
错误信息 ``ModuleNotFoundError: No module named 'pxr'`` 说的就是这件事。

:mod:`rl` 包**不 import 本包**，因此 PPO 的全部测试无需仿真器即可运行。
"""

import gymnasium as gym

from .agents import (
    Go2FlatPPORunnerCfg,
    Go2ResidualFlatPPORunnerCfg,
    Go2ResidualRoughPPORunnerCfg,
    Go2RoughPPORunnerCfg,
)
from .go2_env_cfg import Go2FlatEnvCfg, Go2FlatEnvCfg_PLAY, Go2RoughEnvCfg, Go2RoughEnvCfg_PLAY
from .go2_residual_env_cfg import (
    Go2ResidualFlatEnvCfg,
    Go2ResidualFlatEnvCfg_PLAY,
    Go2ResidualFlatPushEnvCfg,
    Go2ResidualRoughEnvCfg,
    Go2ResidualRoughEnvCfg_PLAY,
)

__all__ = [
    "Go2FlatEnvCfg",
    "Go2FlatEnvCfg_PLAY",
    "Go2FlatPPORunnerCfg",
    "Go2ResidualFlatEnvCfg",
    "Go2ResidualFlatEnvCfg_PLAY",
    "Go2ResidualFlatPPORunnerCfg",
    "Go2ResidualFlatPushEnvCfg",
    "Go2ResidualRoughEnvCfg",
    "Go2ResidualRoughEnvCfg_PLAY",
    "Go2ResidualRoughPPORunnerCfg",
    "Go2RoughEnvCfg",
    "Go2RoughEnvCfg_PLAY",
    "Go2RoughPPORunnerCfg",
]

_ENTRY = "isaaclab.envs:ManagerBasedRLEnv"

gym.register(
    id="Go2-Velocity-Flat-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2FlatEnvCfg, "rsl_rl_cfg_entry_point": Go2FlatPPORunnerCfg},
)

gym.register(
    id="Go2-Velocity-Flat-Play-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2FlatEnvCfg_PLAY, "rsl_rl_cfg_entry_point": Go2FlatPPORunnerCfg},
)

gym.register(
    id="Go2-Velocity-Rough-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2RoughEnvCfg, "rsl_rl_cfg_entry_point": Go2RoughPPORunnerCfg},
)

gym.register(
    id="Go2-Velocity-Rough-Play-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2RoughEnvCfg_PLAY, "rsl_rl_cfg_entry_point": Go2RoughPPORunnerCfg},
)

gym.register(
    id="Go2-Residual-Flat-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2ResidualFlatEnvCfg, "rsl_rl_cfg_entry_point": Go2ResidualFlatPPORunnerCfg},
)

gym.register(
    id="Go2-Residual-Flat-Play-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2ResidualFlatEnvCfg_PLAY, "rsl_rl_cfg_entry_point": Go2ResidualFlatPPORunnerCfg},
)

gym.register(
    id="Go2-Residual-Rough-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2ResidualRoughEnvCfg, "rsl_rl_cfg_entry_point": Go2ResidualRoughPPORunnerCfg},
)

gym.register(
    id="Go2-Residual-Rough-Play-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2ResidualRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": Go2ResidualRoughPPORunnerCfg,
    },
)

gym.register(
    id="Go2-Residual-Flat-Push-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2ResidualFlatPushEnvCfg, "rsl_rl_cfg_entry_point": Go2ResidualFlatPPORunnerCfg},
)
