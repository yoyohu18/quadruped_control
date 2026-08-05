"""训练算法配置。目前只有 rsl-rl 一家，将来可以并列 skrl / sb3。"""

from .rsl_rl_cfg import Go2FlatPPORunnerCfg, Go2RoughPPORunnerCfg

__all__ = ["Go2FlatPPORunnerCfg", "Go2RoughPPORunnerCfg"]
