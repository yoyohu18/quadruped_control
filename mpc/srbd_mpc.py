"""凸模型预测控制：在预测时域上求解接触力。

这是整个项目的核心，也是与你 NMPC 背景最直接对应的地方。区别只有一个，
但它决定了一切：

    **四足的 MPC 是凸的，NMPC 不是。**

凸性来自里程碑 2 那句话：**足端位置已知时，单刚体动力学对接触力是线性的。**
再加上里程碑 7 把摩擦锥内接成金字塔（线性不等式），整个问题就退化成一个
**二次规划（QP）** —— 有全局最优、有多项式时间算法、求解时间可预测。

## 状态与模型

采用 MIT Cheetah 3 的 13 维状态：

.. math::  x = [\\Theta;\\; p;\\; \\omega;\\; \\dot p;\\; g] \\in \\mathbb{R}^{13}

最后那个 :math:`g` 是常数 :math:`-9.81`，把它塞进状态里是为了让重力这个
**仿射项**变成**线性项** —— 这样才能用标准的 :math:`\\dot x = Ax + Bu`
形式，条件化时不必单独处理常数项。这是个很实用的小技巧。

连续时间模型：

.. math::

    \\dot\\Theta &= R_z(\\psi)^\\top \\omega \\\\
    \\dot p &= \\dot p \\\\
    \\dot\\omega &= I_w^{-1}\\sum_i [r_i - p]_\\times f_i \\\\
    \\ddot p &= \\frac{1}{m}\\sum_i f_i + g

前两式是运动学，后两式就是里程碑 2 的
:func:`~dynamics.single_rigid_body.srbd_acceleration_mpc`。

## 条件化（condensing）

把状态全部消掉，只留下控制量作为决策变量：

.. math::  X = A_{qp}\\,x_0 + B_{qp}\\,U

于是代价函数变成 :math:`U` 的二次型，约束也只作用在 :math:`U` 上。

**为什么要条件化？** 稀疏形式（同时把 X 和 U 当决策变量）变量更多但矩阵
更稀疏，稠密形式变量少但矩阵稠密。对四足这种短时域（N=10~16）、小状态
（13 维）的问题，条件化后的稠密 QP 反而更快 —— 这与你在 acados 里遇到的
权衡是同一个。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dynamics import SRBDParams, skew

from .constraints import FrictionConstraints, build_force_constraints

__all__ = ["MPCConfig", "MPCResult", "ConvexMPC"]

N_STATE = 13
N_FORCE = 12

#: 稠密 QP 求解器的优先顺序。条件化之后的问题是稠密的，稀疏求解器无优势。
_SOLVER_PREFERENCE = ("daqp", "quadprog", "proxqp", "osqp")


def _pick_solver() -> str:
    """挑一个可用的稠密 QP 求解器。"""
    import qpsolvers

    for name in _SOLVER_PREFERENCE:
        if name in qpsolvers.available_solvers:
            return name
    raise RuntimeError(
        f"没有可用的 QP 求解器，已装：{qpsolvers.available_solvers}。"
        "请 pip install qpsolvers[daqp] 或 quadprog。"
    )


@dataclass
class MPCConfig:
    """凸 MPC 参数。

    Attributes:
        horizon: 预测步数 :math:`N`。太短则看不见未来的接触切换，太长则
            单刚体模型的误差累积。MIT Cheetah 用 10，本项目默认 10。
        dt: 预测步长，秒。注意它**远大于**控制周期 —— MPC 跑 50 Hz 左右，
            WBC 跑 1 kHz。
        state_weights: 13 维状态的权重对角元。
        force_weight: 接触力的正则权重。它同时起两个作用：让 QP 严格凸，
            以及在静不定的力分配中挑出"最省力"的那一组。
        friction: 接触力的物理限制。
        solver: ``qpsolvers`` 支持的求解器名。缺省自动挑选可用的稠密
            求解器 —— 条件化之后的 QP 是**稠密**的，所以稀疏求解器
            （OSQP 等）在这里没有优势。
    """

    horizon: int = 10
    dt: float = 0.03
    state_weights: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                # 姿态 roll pitch yaw
                80.0, 80.0, 20.0,
                # 位置 x y z
                5.0, 5.0, 250.0,
                # 角速度
                1.0, 1.0, 8.0,
                # 线速度
                12.0, 12.0, 6.0,
                # 重力常数项，不参与优化
                0.0,
            ]
        )
    )
    force_weight: float = 1e-5
    friction: FrictionConstraints = field(default_factory=FrictionConstraints)
    solver: str | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"预测步数必须为正，收到 {self.horizon}")
        if self.dt <= 0.0:
            raise ValueError(f"预测步长必须为正，收到 {self.dt}")
        w = np.asarray(self.state_weights, dtype=float).reshape(N_STATE)
        if np.any(w < 0.0):
            raise ValueError("状态权重不能为负")
        self.state_weights = w
        if self.force_weight <= 0.0:
            raise ValueError("力权重必须为正，否则 QP 可能不严格凸")
        if self.solver is None:
            self.solver = _pick_solver()


@dataclass
class MPCResult:
    """一次求解的结果。

    Attributes:
        forces: 整个预测时域的接触力，形状 (horizon, 4, 3)，世界系。
        solve_time: 求解耗时，秒。**这是能否上 50 Hz 的关键指标。**
        cost: 最优目标函数值。
        success: 求解器是否成功。
        n_variables: 决策变量个数。
        n_constraints: 不等式约束个数。
    """

    forces: np.ndarray
    solve_time: float
    cost: float
    success: bool
    n_variables: int
    n_constraints: int

    @property
    def current_forces(self) -> np.ndarray:
        """只取第一步的力 —— MPC 的滚动时域原则：算 N 步，只用第一步。"""
        return self.forces[0]


class ConvexMPC:
    """基于单刚体模型的凸 MPC。

    Args:
        params: 单刚体参数，由里程碑 2 的
            :func:`~dynamics.go2_srbd.srbd_params_from_model` 提取。
        config: MPC 参数。
    """

    def __init__(self, params: SRBDParams, config: MPCConfig | None = None) -> None:
        self.params = params
        self.cfg = config or MPCConfig()
        self._inertia_inv_body = np.linalg.inv(np.asarray(params.inertia_body, dtype=float))

    # -- 连续时间模型 ---------------------------------------------------------

    def continuous_matrices(
        self, yaw: float, foot_positions: np.ndarray, com: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """连续时间的 :math:`A_c`、:math:`B_c`。

        Args:
            yaw: 偏航角，弧度。模型只保留偏航（里程碑 2 的近似之二）。
            foot_positions: 四只脚的世界位置，形状 (4, 3)。
            com: 质心世界位置，形状 (3,)。

        Returns:
            ``(A_c, B_c)``，形状 (13, 13) 与 (13, 12)。
        """
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        I_world = Rz @ np.asarray(self.params.inertia_body, dtype=float) @ Rz.T
        I_inv = np.linalg.inv(I_world)

        A = np.zeros((N_STATE, N_STATE))
        A[0:3, 6:9] = Rz.T   # Theta_dot = Rz^T omega（小横滚俯仰近似）
        A[3:6, 9:12] = np.eye(3)  # p_dot = v
        A[11, 12] = 1.0      # v_z 受重力常数项驱动

        B = np.zeros((N_STATE, N_FORCE))
        feet = np.asarray(foot_positions, dtype=float).reshape(4, 3)
        com = np.asarray(com, dtype=float).reshape(3)
        for i in range(4):
            r = feet[i] - com
            B[6:9, 3 * i : 3 * i + 3] = I_inv @ skew(r)
            B[9:12, 3 * i : 3 * i + 3] = np.eye(3) / self.params.mass
        return A, B

    def discrete_matrices(
        self, yaw: float, foot_positions: np.ndarray, com: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """一阶欧拉离散化 :math:`A_d = I + A_c\\Delta t`，:math:`B_d = B_c\\Delta t`。

        MIT Cheetah 用的就是一阶离散。步长 0.03 s 下它足够准，而且矩阵指数
        每步都要重算的话开销不划算 —— 记住 :math:`B_c` 依赖足端位置，
        **每个预测步都不一样**。
        """
        A, B = self.continuous_matrices(yaw, foot_positions, com)
        dt = self.cfg.dt
        return np.eye(N_STATE) + A * dt, B * dt

    # -- 条件化 ---------------------------------------------------------------

    def build_prediction_matrices(
        self,
        yaw: float,
        foot_trajectory: np.ndarray,
        com_trajectory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """构造 :math:`X = A_{qp}x_0 + B_{qp}U`。

        Args:
            yaw: 当前偏航角。整个预测时域内按常值处理 —— 转弯较慢时可接受，
                高速转向时应当按 ``yaw + yaw_rate * k * dt`` 逐步更新。
            foot_trajectory: 各预测步的足端位置，形状 (horizon, 4, 3)。
                **这正是里程碑 6 的输出。**
            com_trajectory: 各预测步的参考质心位置，形状 (horizon, 3)。

        Returns:
            ``(A_qp, B_qp)``，形状 (13N, 13) 与 (13N, 12N)。
        """
        N = self.cfg.horizon
        A_qp = np.zeros((N_STATE * N, N_STATE))
        B_qp = np.zeros((N_STATE * N, N_FORCE * N))

        A_list, B_list = [], []
        for k in range(N):
            Ad, Bd = self.discrete_matrices(yaw, foot_trajectory[k], com_trajectory[k])
            A_list.append(Ad)
            B_list.append(Bd)

        # A_qp 的第 k 块是 A_{k-1} ... A_0
        powers = [np.eye(N_STATE)]
        for k in range(N):
            powers.append(A_list[k] @ powers[-1])
        for k in range(N):
            A_qp[N_STATE * k : N_STATE * (k + 1)] = powers[k + 1]

        # B_qp 的 (k, j) 块 = A_k ... A_{j+1} B_j，j <= k
        for k in range(N):
            for j in range(k + 1):
                prod = np.eye(N_STATE)
                for m in range(j + 1, k + 1):
                    prod = A_list[m] @ prod
                B_qp[
                    N_STATE * k : N_STATE * (k + 1), N_FORCE * j : N_FORCE * (j + 1)
                ] = prod @ B_list[j]
        return A_qp, B_qp

    # -- 求解 -----------------------------------------------------------------

    def solve(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        contact_schedule: np.ndarray,
        foot_trajectory: np.ndarray,
        com_trajectory: np.ndarray | None = None,
        yaw: float | None = None,
    ) -> MPCResult:
        """求解一次 MPC。

        Args:
            state: 当前 13 维状态 ``[Theta, p, omega, v, g]``。
            reference: 参考轨迹，形状 (horizon, 13)。
            contact_schedule: 接触序列，形状 (horizon, 4)。
                **这正是里程碑 4 的 ``contact_schedule`` 输出。**
            foot_trajectory: 各预测步的足端位置，形状 (horizon, 4, 3)。
            com_trajectory: 各预测步的质心位置，缺省用参考轨迹里的位置。
            yaw: 线性化用的偏航角，缺省取当前状态里的。

        Returns:
            :class:`MPCResult`。
        """
        import qpsolvers

        N = self.cfg.horizon
        x0 = np.asarray(state, dtype=float).reshape(N_STATE)
        ref = np.asarray(reference, dtype=float).reshape(N, N_STATE)
        contact = np.asarray(contact_schedule, dtype=bool).reshape(N, 4)
        feet = np.asarray(foot_trajectory, dtype=float).reshape(N, 4, 3)
        com_traj = ref[:, 3:6] if com_trajectory is None else np.asarray(com_trajectory).reshape(N, 3)
        yaw_lin = float(x0[2]) if yaw is None else float(yaw)

        A_qp, B_qp = self.build_prediction_matrices(yaw_lin, feet, com_traj)

        # 代价：||X - X_ref||^2_Q + ||U||^2_R
        Q = np.tile(self.cfg.state_weights, N)
        H = B_qp.T @ (Q[:, None] * B_qp)
        H[np.diag_indices_from(H)] += self.cfg.force_weight
        H = 0.5 * (H + H.T)  # 强制对称，抑制浮点漂移
        residual = A_qp @ x0 - ref.reshape(-1)
        g = B_qp.T @ (Q * residual)

        # 约束：逐步堆叠
        C_blocks, lb_blocks, ub_blocks = [], [], []
        for k in range(N):
            C, lb, ub = build_force_constraints(contact[k], self.cfg.friction)
            C_blocks.append(C)
            lb_blocks.append(lb)
            ub_blocks.append(ub)
        n_rows = sum(c.shape[0] for c in C_blocks)
        C_all = np.zeros((n_rows, N_FORCE * N))
        row = 0
        for k, C in enumerate(C_blocks):
            C_all[row : row + C.shape[0], N_FORCE * k : N_FORCE * (k + 1)] = C
            row += C.shape[0]
        lb_all = np.concatenate(lb_blocks)
        ub_all = np.concatenate(ub_blocks)

        # qpsolvers 的 box 形式：lb <= C u <= ub 拆成两个单边不等式
        G = np.vstack([C_all, -C_all])
        h = np.concatenate([ub_all, -lb_all])
        finite = np.isfinite(h)
        G, h = G[finite], h[finite]

        t0 = time.perf_counter()
        u = qpsolvers.solve_qp(H, g, G=G, h=h, solver=self.cfg.solver)
        solve_time = time.perf_counter() - t0

        success = u is not None
        if not success:
            u = np.zeros(N_FORCE * N)
        cost = float(0.5 * u @ H @ u + g @ u)

        return MPCResult(
            forces=u.reshape(N, 4, 3),
            solve_time=solve_time,
            cost=cost,
            success=success,
            n_variables=N_FORCE * N,
            n_constraints=G.shape[0],
        )

    # -- 便捷接口 -------------------------------------------------------------

    @staticmethod
    def make_state(
        rpy: np.ndarray,
        com: np.ndarray,
        omega: np.ndarray,
        velocity: np.ndarray,
        gravity: float = -9.81,
    ) -> np.ndarray:
        """拼装 13 维状态向量。"""
        return np.concatenate(
            [
                np.asarray(rpy, dtype=float).reshape(3),
                np.asarray(com, dtype=float).reshape(3),
                np.asarray(omega, dtype=float).reshape(3),
                np.asarray(velocity, dtype=float).reshape(3),
                [gravity],
            ]
        )

    def make_reference(
        self,
        state: np.ndarray,
        velocity_command: np.ndarray,
        yaw_rate_command: float = 0.0,
        height: float | None = None,
    ) -> np.ndarray:
        """按指令速度外推出一条参考轨迹，形状 (horizon, 13)。

        参考很简单：姿态保持水平、按指令偏航率转、质心按指令速度匀速前进、
        高度保持不变。**MPC 的作用不是跟踪一条精心设计的轨迹，而是在这个
        朴素参考的基础上算出可行的接触力。**
        """
        N, dt = self.cfg.horizon, self.cfg.dt
        x0 = np.asarray(state, dtype=float).reshape(N_STATE)
        v_cmd = np.asarray(velocity_command, dtype=float).reshape(-1)
        v_cmd = np.array([v_cmd[0], v_cmd[1], 0.0]) if len(v_cmd) == 2 else v_cmd
        z = x0[5] if height is None else height

        ref = np.zeros((N, N_STATE))
        for k in range(N):
            t = (k + 1) * dt
            ref[k, 0:2] = 0.0                       # 横滚俯仰保持水平
            ref[k, 2] = x0[2] + yaw_rate_command * t
            ref[k, 3:5] = x0[3:5] + v_cmd[:2] * t
            ref[k, 5] = z
            ref[k, 8] = yaw_rate_command
            ref[k, 9:12] = v_cmd
            ref[k, 12] = x0[12]
        return ref
