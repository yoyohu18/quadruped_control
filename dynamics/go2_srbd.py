"""从整机模型中正确提取 Go2 的单刚体（SRBD）参数。

看起来只是"取个质量和惯量"，但这里恰恰是四足 MPC 最容易埋雷的地方：

* 质量必须是**整机**质量，不是躯干连杆的质量。Go2 躯干只有 7.279 kg，
  整机 16.087 kg —— 腿占了 54.8%。
* 惯量必须是**关于整机质心的复合刚体惯量**，而且要在**具体位形下**算。
  躯干连杆自身的惯量比它小 5 到 7 倍，用错会让 MPC 的姿态响应完全失真。
* 复合惯量随腿的摆动而变化。凸 MPC 把它当常数，所以要在**标称站立位形**
  下取值 —— 这个近似的实际误差在测试与可视化脚本里被量化了。
"""

from __future__ import annotations

import numpy as np

from kinematics import STANDING_JOINT_ANGLES, load_go2

from .rigid_body_dynamics import RigidBodyDynamics
from .single_rigid_body import SRBDParams

__all__ = ["NOMINAL_BASE_HEIGHT", "load_go2_dynamics", "nominal_configuration", "srbd_params_from_model"]

#: 标称站立时躯干原点的离地高度，与 :data:`kinematics.STANDING_JOINT_ANGLES` 对应。
NOMINAL_BASE_HEIGHT = 0.3113


def load_go2_dynamics(floating_base: bool = True) -> RigidBodyDynamics:
    """构建 Go2 的整机动力学对象，默认使用浮动基座。"""
    return RigidBodyDynamics(load_go2(floating_base=floating_base))


def nominal_configuration(rbd: RigidBodyDynamics, height: float = NOMINAL_BASE_HEIGHT) -> np.ndarray:
    """标称站立位形：四脚着地，躯干水平。"""
    if not rbd.model.floating_base:
        return STANDING_JOINT_ANGLES.copy()
    return rbd.model.make_configuration(STANDING_JOINT_ANGLES, base_position=[0.0, 0.0, height])


def srbd_params_from_model(
    rbd: RigidBodyDynamics, q: np.ndarray | None = None
) -> SRBDParams:
    """在给定位形下，从整机模型中提取单刚体参数。

    Args:
        rbd: 整机动力学对象。
        q: 提取参数所用的位形，默认取标称站立位形。

    Returns:
        可直接喂给凸 MPC 的 :class:`~dynamics.single_rigid_body.SRBDParams`。

    Note:
        返回的惯量表达在**世界系轴向**下。在标称位形里躯干与世界对齐，
        因此它同时也是机体系下的惯量 —— 这正是我们要在水平位形下提取
        参数的原因。换成别的位形就必须自己做一次坐标旋转。
    """
    q = nominal_configuration(rbd) if q is None else np.asarray(q, dtype=float)
    return SRBDParams(
        mass=rbd.total_mass,
        inertia_body=rbd.centroidal_inertia(q),
        gravity=rbd.gravity,
    )
