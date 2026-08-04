"""凸模型预测控制：在预测时域上求解接触力。

* :mod:`mpc.constraints` —— 摩擦金字塔与法向力限制，把物理事实写成线性
  不等式。这是 QP 能成立的另一半原因。
* :mod:`mpc.srbd_mpc` —— 基于单刚体模型的凸 MPC，条件化后用 OSQP 求解。

三个输入分别来自前面的里程碑：
  单刚体模型（M2）+ 接触序列（M4）+ 足端位置（M6）
"""

from .constraints import (
    FrictionConstraints,
    build_force_constraints,
    check_friction_cone,
    friction_pyramid_matrix,
)
from .srbd_mpc import ConvexMPC, MPCConfig, MPCResult

__all__ = [
    "ConvexMPC",
    "FrictionConstraints",
    "MPCConfig",
    "MPCResult",
    "build_force_constraints",
    "check_friction_cone",
    "friction_pyramid_matrix",
]
