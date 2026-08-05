"""全身控制器：把 MPC 的接触力落实成 12 个关节力矩。

这是经典技术栈的最后一块，也是里程碑 2 那个方程终于被完整用上的地方：

.. math::  M(q)\\,a + C(q,v)\\,v + g(q) = S^\\top \\tau + \\sum_i J_i(q)^\\top f_i

## 决策变量

.. math::  z = [\\,a \\;(18)\\;;\\; f \\;(3 n_c)\\,]

力矩 :math:`\\tau` **不是**决策变量 —— 它由动力学方程的后 12 行**唯一确定**：

.. math::  \\tau = M_{j}a + h_{j} - J_{c,j}^\\top f

把 :math:`\\tau` 消掉可以让 QP 小 12 维。代价是力矩限幅变成对 :math:`(a, f)`
的线性不等式，而不是简单的箱式约束 —— 换来的规模缩减更划算。

## 硬约束：那 6 行终于用上了

从里程碑 2 开始反复说的"**S 的前 6 列全为零**"，在这里变成 QP 的
**等式约束**：

.. math::  M_{b}\\,a + h_{b} = J_{c,b}^\\top f

躯干的 6 个自由度上没有电机，所以这 6 行右端**没有** :math:`\\tau`。它不是
可以妥协的目标，而是物理定律 —— 违反它意味着算出来的力矩在真实机器人上
根本产生不了那个运动。

再加上支撑脚不动：

.. math::  J_c\\,a + \\dot J_c\\,v = 0

## 软任务：为什么 MPC 的力不能当硬约束

里程碑 2 已经量化过：关节速度 8 rad/s 时，单刚体模型的躯干角加速度误差
中位数达 **43%**。MPC 是基于那个模型算出来的力，**它本身就是近似的**。

如果把 MPC 的力当**硬**约束，WBC 就必须精确复现一组基于错误模型算出的力，
结果是摆动腿的跟踪任务被牺牲掉，甚至 QP 直接不可行。

所以正确做法是把它当**软**任务 —— 权重高，但可以让步。**MPC 负责"大方向
对"，WBC 负责"物理上真的能做到"。** 两个模型不一致时，让完整模型赢。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dynamics import RigidBodyDynamics
from kinematics import LEGS
from mpc import FrictionConstraints, friction_pyramid_matrix

from .tasks import Task

__all__ = ["WBCConfig", "WBCResult", "WholeBodyController"]

FLOATING_BASE_DOF = 6


@dataclass
class WBCConfig:
    """全身控制器参数。

    Attributes:
        force_tracking_weight: 跟踪 MPC 接触力的权重。取大值（但**不是**
            硬约束）—— 见模块文档关于软/硬的讨论。
        acceleration_regularisation: 广义加速度的正则权重，保证 QP 严格凸。
        force_regularisation: 接触力的正则权重。
        torque_limit: 单关节力矩上限，N·m。Go2 的膝关节约 45 N·m。
        friction: 摩擦与法向力限制，与里程碑 7 共用同一套定义。
        solver: QP 求解器名，缺省自动挑选。
    """

    force_tracking_weight: float = 1.0
    acceleration_regularisation: float = 1e-4
    force_regularisation: float = 1e-6
    torque_limit: float = 45.0
    friction: FrictionConstraints = field(default_factory=FrictionConstraints)
    solver: str | None = None

    def __post_init__(self) -> None:
        if self.force_tracking_weight < 0.0:
            raise ValueError("力跟踪权重不能为负")
        if self.acceleration_regularisation <= 0.0:
            raise ValueError("加速度正则必须为正，否则 QP 可能不严格凸")
        if self.torque_limit <= 0.0:
            raise ValueError("力矩上限必须为正")
        if self.solver is None:
            from mpc.srbd_mpc import _pick_solver

            self.solver = _pick_solver()


@dataclass
class WBCResult:
    """一次求解的结果。

    Attributes:
        torque: 12 个关节力矩，N·m。
        acceleration: 广义加速度，形状 (nv,)。
        contact_force: 各支撑腿的接触力，形状 (n_contact, 3)。
        solve_time: 求解耗时，秒。**1 kHz 下必须远小于 1 ms。**
        success: 求解是否成功。
        task_costs: 各任务的加权残差平方，用于诊断谁在让步。
        n_variables: 决策变量个数。
    """

    torque: np.ndarray
    acceleration: np.ndarray
    contact_force: np.ndarray
    solve_time: float
    success: bool
    task_costs: dict[str, float]
    n_variables: int


class WholeBodyController:
    """基于完整刚体动力学的加权 QP 全身控制器。

    Args:
        dynamics: 里程碑 2 的 :class:`~dynamics.RigidBodyDynamics`，
            必须是浮动基座模型。
        config: 控制器参数。
    """

    def __init__(self, dynamics: RigidBodyDynamics, config: WBCConfig | None = None) -> None:
        if not dynamics.model.floating_base:
            raise ValueError("全身控制需要浮动基座模型 —— 欠驱动结构是它存在的前提")
        self.dyn = dynamics
        self.cfg = config or WBCConfig()
        self.nv = dynamics.nv

    # -- 求解 -----------------------------------------------------------------

    def solve(
        self,
        q: np.ndarray,
        v: np.ndarray,
        stance_legs: tuple[str, ...] | list[str],
        desired_forces: np.ndarray | None = None,
        tasks: list[Task] | None = None,
    ) -> WBCResult:
        """求解一次全身控制 QP。

        Args:
            q: 位置向量（浮动基座，19 维）。
            v: 速度向量（18 维）。
            stance_legs: 当前支撑腿，来自步态调度器（M4）。
            desired_forces: MPC 给出的接触力，形状 (len(stance_legs), 3)。
                缺省则不加力跟踪任务。
            tasks: 加速度任务列表（摆动腿跟踪、躯干姿态等）。

        Returns:
            :class:`WBCResult`。
        """
        import qpsolvers

        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        tasks = list(tasks or [])
        stance_legs = tuple(stance_legs)
        nc = len(stance_legs)
        nf = 3 * nc
        n = self.nv + nf

        M = self.dyn.mass_matrix(q)
        h = self.dyn.nonlinear_effects(q, v)
        Jc = self.dyn.contact_jacobian(q, stance_legs)
        dJv = self.dyn.contact_jacobian_dot_v(q, v, stance_legs)

        # --- 等式约束 -------------------------------------------------------
        # 1) 浮动基座 6 行：M_b a + h_b = J_b^T f   （右端没有 tau！）
        A_base = np.zeros((FLOATING_BASE_DOF, n))
        A_base[:, : self.nv] = M[:FLOATING_BASE_DOF]
        if nc:
            A_base[:, self.nv :] = -Jc[:, :FLOATING_BASE_DOF].T
        b_base = -h[:FLOATING_BASE_DOF]

        # 2) 支撑脚不动：J_c a + dJ v = 0
        blocks_A, blocks_b = [A_base], [b_base]
        if nc:
            A_contact = np.zeros((nf, n))
            A_contact[:, : self.nv] = Jc
            blocks_A.append(A_contact)
            blocks_b.append(-dJv)
        A_eq = np.vstack(blocks_A)
        b_eq = np.concatenate(blocks_b)

        # --- 不等式约束 -----------------------------------------------------
        G_rows, h_rows = [], []

        # 摩擦金字塔与法向力上下界
        if nc:
            pyramid = friction_pyramid_matrix(self.cfg.friction.pyramid_mu)
            for i in range(nc):
                block = np.zeros((6, n))
                block[:4, self.nv + 3 * i : self.nv + 3 * i + 3] = pyramid
                block[4, self.nv + 3 * i + 2] = -1.0   # -f_z <= -f_min
                block[5, self.nv + 3 * i + 2] = 1.0    #  f_z <=  f_max
                G_rows.append(block)
                h_rows.append(
                    np.array([0.0, 0.0, 0.0, 0.0, -self.cfg.friction.f_min, self.cfg.friction.f_max])
                )

        # 力矩限幅：tau = M_j a + h_j - J_j^T f，写成对 (a, f) 的线性不等式
        act = self.dyn.actuated_dofs
        T = np.zeros((12, n))
        T[:, : self.nv] = M[act]
        if nc:
            T[:, self.nv :] = -Jc[:, act].T
        tau_offset = h[act]
        G_rows.append(T)
        h_rows.append(self.cfg.torque_limit - tau_offset)
        G_rows.append(-T)
        h_rows.append(self.cfg.torque_limit + tau_offset)

        G = np.vstack(G_rows)
        h_ineq = np.concatenate(h_rows)

        # --- 代价 -----------------------------------------------------------
        P = np.zeros((n, n))
        qv = np.zeros(n)
        P[: self.nv, : self.nv] += np.eye(self.nv) * self.cfg.acceleration_regularisation
        if nc:
            P[self.nv :, self.nv :] += np.eye(nf) * self.cfg.force_regularisation

        for task in tasks:
            J = np.zeros((task.dim, n))
            J[:, : self.nv] = task.jacobian
            r0 = task.drift - task.target
            W = np.diag(task.weight)
            P += J.T @ W @ J
            qv += J.T @ (task.weight * r0)

        if nc and desired_forces is not None:
            f_des = np.asarray(desired_forces, dtype=float).reshape(nf)
            w = self.cfg.force_tracking_weight
            P[self.nv :, self.nv :] += np.eye(nf) * w
            qv[self.nv :] += -w * f_des

        P = 0.5 * (P + P.T)

        t0 = time.perf_counter()
        z = qpsolvers.solve_qp(P, qv, G=G, h=h_ineq, A=A_eq, b=b_eq, solver=self.cfg.solver)
        solve_time = time.perf_counter() - t0

        success = z is not None
        if not success:
            z = np.zeros(n)
        a = z[: self.nv]
        f = z[self.nv :].reshape(nc, 3) if nc else np.zeros((0, 3))
        tau = M[act] @ a + h[act] - (Jc[:, act].T @ z[self.nv :] if nc else 0.0)

        return WBCResult(
            torque=tau,
            acceleration=a,
            contact_force=f,
            solve_time=solve_time,
            success=success,
            task_costs={t.name: t.cost(a) for t in tasks},
            n_variables=n,
        )

    # -- 对照：忽略动力学的朴素做法 -------------------------------------------

    def naive_torque(
        self,
        q: np.ndarray,
        stance_legs: tuple[str, ...] | list[str],
        desired_forces: np.ndarray,
    ) -> np.ndarray:
        """朴素做法：直接用 :math:`\\tau = -J^\\top f` 把接触力换算成力矩。

        这是很多入门实现的做法，**它忽略了三样东西**：

        1. 腿本身的惯量 —— 摆动腿要加速就需要力矩；
        2. 科氏力与离心力；
        3. 腿自身的重力。

        提供它是为了做定量对照：``scripts/viz_wbc.py`` 会量出这三项在
        Go2 上到底值多少 N·m。

        Args:
            q: 位置向量。
            stance_legs: 支撑腿。
            desired_forces: 接触力，形状 (n_contact, 3)。

        Returns:
            12 个关节力矩。
        """
        Jc = self.dyn.contact_jacobian(q, tuple(stance_legs))
        f = np.asarray(desired_forces, dtype=float).reshape(-1)
        return -Jc[:, self.dyn.actuated_dofs].T @ f

    # -- 任务构造的便捷函数 ---------------------------------------------------

    def swing_foot_task(
        self,
        q: np.ndarray,
        v: np.ndarray,
        leg: str,
        desired_acceleration: np.ndarray,
        weight: float = 100.0,
    ) -> Task:
        """构造一条摆动腿的足端加速度任务。

        Args:
            q: 位置向量。
            v: 速度向量。
            leg: 腿标识。
            desired_acceleration: 期望足端加速度（世界系），形状 (3,)。
                通常由里程碑 5 的轨迹加上 PD 修正得到。
            weight: 任务权重。

        Returns:
            :class:`~whole_body_controller.tasks.Task`。
        """
        J = self.dyn.model.full_foot_jacobian(q, leg)
        drift = self.dyn.contact_jacobian_dot_v(q, v, (leg,))
        return Task(f"swing_{leg}", J, drift, desired_acceleration, weight)

    def body_task(
        self,
        desired_linear_acceleration: np.ndarray,
        desired_angular_acceleration: np.ndarray,
        linear_weight: float = 10.0,
        angular_weight: float = 20.0,
    ) -> Task:
        """构造躯干的 6 维加速度任务。

        躯干的加速度就是广义加速度的前 6 个分量，所以任务雅可比是
        :math:`[I_6\\ \\ 0]`，漂移项为零 —— 最简单的一种任务。
        """
        J = np.zeros((6, self.nv))
        J[:, :FLOATING_BASE_DOF] = np.eye(6)
        target = np.concatenate(
            [
                np.asarray(desired_linear_acceleration, dtype=float).reshape(3),
                np.asarray(desired_angular_acceleration, dtype=float).reshape(3),
            ]
        )
        weight = np.concatenate([np.full(3, linear_weight), np.full(3, angular_weight)])
        return Task("body", J, np.zeros(6), target, weight)
