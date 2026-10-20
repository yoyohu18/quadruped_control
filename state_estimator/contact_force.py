"""足端接触力估计：广义动量观测器 + 滞回接触检测。

Go2 的脚上没有力传感器。要知道"这只脚现在踩实了没有、踩得多重"，只能
从关节力矩反推。朴素做法是直接解动力学方程

.. math::  J_c^\\top f = M a + h - S^\\top \\tau

但它要**加速度** :math:`a` —— 只能对编码器速度做数值微分，噪声放大到没法用。

## 广义动量观测器（De Luca & Mattone, 2003）

绕开加速度的关键是换一个变量：广义动量 :math:`p = M(q)\\,v`。利用
:math:`\\dot M = C + C^\\top`，

.. math::  \\dot p = S^\\top \\tau + C^\\top v - g + \\tau_{ext}

右边**没有加速度**。于是定义残差

.. math::  r(t) = K\\Big[\\,p(t) - p(0) - \\int_0^t \\big(S^\\top\\tau + C^\\top v - g + r\\big)\\,ds\\Big]

代进去就得到 :math:`\\dot r = K(\\tau_{ext} - r)` —— **残差是外力矩经过一个
带宽为 K 的一阶低通**。K 越大越快，但对未建模项（关节摩擦、转子惯量）也
越敏感。

> 无人机类比：这就是你做过的**外力/风扰观测器**。多旋翼用
> :math:`m\\dot v = R f_T + mg + f_{ext}` 估风，这里用整机的动量估接触力。
> 区别在于四足的 "外力" 作用在四个不同点上，要靠雅可比 :math:`J_c^\\top`
> 把 18 维的外力矩拆回到每只脚。

## 从外力矩到足端力

外力只来自四只脚：:math:`\\tau_{ext} = J_c^\\top f`，:math:`J_c` 是 12×18。
18 个方程、12 个未知数，**超定** —— 用最小二乘。浮动基座那 6 行在这里
是信息而不是负担：它们让"哪只脚在出力"变得可分辨，即使那只脚的关节
力矩很小。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import RigidBodyDynamics
from kinematics import LEGS

__all__ = ["MomentumObserver", "ContactDetector", "ContactDetectorConfig"]


class MomentumObserver:
    """整机广义动量观测器。

    Args:
        dynamics: 浮动基座整机动力学。
        gain: 观测器带宽 K，rad/s。缺省 200（时间常数 5 ms）—— 补偿转子
            惯量之后，带宽拉高不再放大误差，见 docs/11 第 6 节的扫描。
        joint_armature: 电机转子折算到关节的惯量（kg·m²），加到 M 的关节
            对角块上。URDF 不含这一项，但它是电机手册上的已知参数 ——
            Go2 约 0.01。漏掉它，摆动腿大加速度时会被误读成接触力。

    调用时序：``update`` 传入的是**施加 τ 之后**的状态 —— 即第 k 步的
    力矩配第 k+1 步的 (q, v)。反过来会引入一步延迟。
    """

    def __init__(
        self,
        dynamics: RigidBodyDynamics,
        gain: float = 200.0,
        joint_armature: float | np.ndarray = 0.0,
    ) -> None:
        if not dynamics.model.floating_base:
            raise ValueError("动量观测器需要浮动基座模型 —— 基座 6 行是区分各脚受力的关键")
        self.dyn = dynamics
        self.gain = float(gain)
        self.nv = dynamics.nv
        self._armature = np.zeros(self.nv)
        self._armature[dynamics.actuated_dofs] = joint_armature
        self._integral = np.zeros(self.nv)
        self._p0: np.ndarray | None = None
        self.residual = np.zeros(self.nv)

    def _mass(self, q: np.ndarray) -> np.ndarray:
        M = self.dyn.mass_matrix(q)
        M[np.diag_indices(self.nv)] += self._armature
        return M

    def reset(self, q: np.ndarray, v: np.ndarray) -> None:
        self._p0 = self._mass(q) @ v
        self._integral[:] = 0.0
        self.residual[:] = 0.0

    def update(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float) -> np.ndarray:
        """推进一步，返回当前外力矩估计 :math:`\\hat\\tau_{ext}`（18 维）。

        Args:
            tau: 这一步**实际施加**的 12 个关节力矩（饱和之后的）。
        """
        if self._p0 is None:
            self.reset(q, v)
        M = self._mass(q)
        C = self.dyn.coriolis_matrix(q, v)
        g = self.dyn.gravity_vector(q)
        S_tau = np.zeros(self.nv)
        S_tau[self.dyn.actuated_dofs] = tau
        self._integral += (S_tau + C.T @ v - g + self.residual) * dt
        self.residual = self.gain * (M @ v - self._p0 - self._integral)
        return self.residual.copy()

    def foot_forces(self, q: np.ndarray, legs: tuple[str, ...] = LEGS) -> np.ndarray:
        """把外力矩最小二乘拆成各脚受到的地面反力，形状 (len(legs), 3)，世界系。"""
        J = self.dyn.contact_jacobian(q, legs)  # (3n, 18)
        f, *_ = np.linalg.lstsq(J.T, self.residual, rcond=None)
        return f.reshape(len(legs), 3)


@dataclass
class ContactDetectorConfig:
    """滞回阈值：法向力超过 ``on`` 判触地，低于 ``off`` 判离地。

    单阈值在临界处会来回跳（chattering），这对 WBC 是灾难 —— 接触约束
    一会儿加一会儿撤。滞回是最便宜的解法。
    """

    on: float = 25.0
    off: float = 10.0


class ContactDetector:
    """基于估计法向力的滞回接触检测。"""

    def __init__(self, config: ContactDetectorConfig | None = None) -> None:
        self.cfg = config or ContactDetectorConfig()
        self.state = np.ones(4, dtype=bool)

    def update(self, normal_forces: np.ndarray) -> np.ndarray:
        fz = np.asarray(normal_forces, dtype=float).reshape(4)
        self.state = np.where(self.state, fz > self.cfg.off, fz > self.cfg.on)
        return self.state.copy()
