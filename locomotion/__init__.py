"""里程碑 11：把 M2–M8 串成闭环控制器，在 MuJoCo 全身物理仿真里验证。"""

from .controller import ControlOutput, LocomotionConfig, LocomotionController

__all__ = ["ControlOutput", "LocomotionConfig", "LocomotionController"]
