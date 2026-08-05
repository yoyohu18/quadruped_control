"""本项目的 MDP 组件，叠加在 Isaac Lab 内置项之上。

三层来源，按覆盖顺序：

1. ``isaaclab.envs.mdp`` —— 与机器人无关的通用项（速度跟踪、关节正则、
   接触终止、域随机化事件）。绝大多数项来自这里。
2. ``isaaclab_tasks...velocity.mdp`` —— **只借一个**：地形课程
   ``terrain_levels_vel``。它要读地形管理器的内部状态，重写一遍没有
   任何教学价值，属于"该复用就复用"。
3. :mod:`isaac.mdp.rewards` —— 本项目自己的奖励项，里程碑 4/5/6 的物理
   先验从这里进入 RewardManager。同名项（如 ``feet_air_time``）会覆盖
   上面两层。

这样配置里统一写 ``mdp.xxx``，不用关心某一项到底出自哪一层。
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import (  # noqa: F401
    terrain_levels_vel,
)

from .rewards import *  # noqa: F401, F403
