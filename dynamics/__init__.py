"""动力学：质量矩阵、逆动力学、接触，以及凸 MPC 用的单刚体模型。

与运动学一样，刻意分成两层：

* :mod:`dynamics.rigid_body_dynamics` —— 完整的 18 自由度浮动基座动力学，
  由 Pinocchio 的 CRBA / RNEA / ABA 支撑。这是真值。
* :mod:`dynamics.single_rigid_body` —— 手推的单刚体近似，纯 NumPy。
  这是凸 MPC 真正求解的那个模型。

:mod:`dynamics.go2_srbd` 负责从整机模型里**正确地**提取单刚体参数，
并把两者的差距量化出来。
"""

from .go2_srbd import (
    NOMINAL_BASE_HEIGHT,
    load_go2_dynamics,
    nominal_configuration,
    srbd_params_from_model,
)
from .rigid_body_dynamics import FLOATING_BASE_DOF, RigidBodyDynamics
from .single_rigid_body import (
    SRBDParams,
    contact_wrench,
    matrix_to_rpy,
    rpy_rates_from_angular_velocity,
    rpy_to_matrix,
    skew,
    srbd_acceleration,
    srbd_acceleration_mpc,
)

__all__ = [
    "FLOATING_BASE_DOF",
    "NOMINAL_BASE_HEIGHT",
    "RigidBodyDynamics",
    "SRBDParams",
    "contact_wrench",
    "load_go2_dynamics",
    "matrix_to_rpy",
    "nominal_configuration",
    "rpy_rates_from_angular_velocity",
    "rpy_to_matrix",
    "skew",
    "srbd_acceleration",
    "srbd_acceleration_mpc",
    "srbd_params_from_model",
]
