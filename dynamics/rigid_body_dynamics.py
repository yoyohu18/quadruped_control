"""浮动基座四足的刚体动力学，由 Pinocchio 提供底层算法。

整个后续技术栈（MPC、WBC）都建立在下面这一个方程上：

    M(q) a + C(q, v) v + g(q) = S^T tau + sum_i J_i(q)^T f_i

各项含义：

    M(q)      质量矩阵（nv x nv），由 CRBA 算出
    C(q, v) v 科氏力与离心力
    g(q)      重力项
    S         选择矩阵（12 x nv），把 12 个电机力矩映射到广义力
    J_i       第 i 只脚的接触雅可比
    f_i       第 i 只脚受到的地面反力

**浮动基座意味着 S 的前 6 列全为零。** 躯干的 6 个自由度上没有任何电机，
唯一能驱动它们的是脚上的接触力。这一条就是四足与无人机最本质的分野，
也是 WBC 与凸 MPC 存在的全部理由 —— 详见 ``docs/02_dynamics.md``。

本模块只负责把这些量算对、算全，并把约定钉死；具体的控制器放在
``mpc/`` 与 ``whole_body_controller/``。
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from kinematics.robot_model import LEGS, QuadrupedModel

__all__ = ["RigidBodyDynamics"]

#: 浮动基座下，广义坐标里不可驱动的自由度个数（3 平动 + 3 转动）。
FLOATING_BASE_DOF = 6


class RigidBodyDynamics:
    """四足整机的刚体动力学。

    Args:
        model: 一个 :class:`~kinematics.robot_model.QuadrupedModel`。
            强烈建议使用浮动基座模型 —— 固定基座下欠驱动结构不存在，
            后面所有关于接触力的讨论都无从谈起。

    Attributes:
        model: 传入的 :class:`QuadrupedModel`。
        nv: 速度向量维数（浮动基座下为 18）。
        total_mass: 整机总质量，单位 kg。
    """

    def __init__(self, model: QuadrupedModel) -> None:
        self.model = model
        self._m = model.model
        self._d = model.data
        self.nv = self._m.nv
        self.nq = self._m.nq
        self.total_mass = float(pin.computeTotalMass(self._m))
        self.gravity = float(-self._m.gravity.linear[2])

        # 驱动自由度在广义速度向量中的下标。
        if model.floating_base:
            self.actuated_dofs = np.arange(FLOATING_BASE_DOF, self.nv)
            self.underactuated_dofs = np.arange(FLOATING_BASE_DOF)
        else:
            self.actuated_dofs = np.arange(self.nv)
            self.underactuated_dofs = np.array([], dtype=int)

    # -- 动力学各项 -----------------------------------------------------------

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """质量矩阵 ``M(q)``，由 CRBA 算出。

        Pinocchio 的 ``crba`` 只填充上三角，这里补成完整的对称阵 ——
        直接拿 ``crba`` 的返回值去做矩阵乘法是一个非常常见的错误。

        Args:
            q: 位置向量。

        Returns:
            对称正定矩阵，形状 (nv, nv)。
        """
        M = pin.crba(self._m, self._d, np.asarray(q, dtype=float))
        return np.triu(M) + np.triu(M, 1).T

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """广义重力项 ``g(q)``，等于势能对 q 的梯度。"""
        return pin.computeGeneralizedGravity(self._m, self._d, np.asarray(q, dtype=float)).copy()

    def coriolis_matrix(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """科氏矩阵 ``C(q, v)``，满足 ``Mdot - 2C`` 反对称。

        注意 ``C`` 并不唯一 —— 只有乘积 ``C v`` 是唯一确定的。Pinocchio 选的
        是那个使反对称性质成立的因式分解，这也是 Lyapunov 稳定性证明里
        用到的那一个。
        """
        return pin.computeCoriolisMatrix(
            self._m, self._d, np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        ).copy()

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """非线性项 ``C(q, v) v + g(q)``，一次 RNEA 即可算出（比分别算快）。"""
        return pin.nonLinearEffects(
            self._m, self._d, np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        ).copy()

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """逆动力学（RNEA）：给定运动，求所需广义力。

        复杂度 O(n)，不需要显式构造 ``M``。它是 WBC 中把期望加速度换算成
        力矩的那一步。

        Args:
            q: 位置向量。
            v: 速度向量。
            a: 加速度向量。

        Returns:
            广义力 ``tau``，形状 (nv,)。浮动基座下前 6 个分量**不是**电机
            力矩，而是躯干上必须由接触力提供的那部分广义力。
        """
        return pin.rnea(
            self._m,
            self._d,
            np.asarray(q, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(a, dtype=float),
        ).copy()

    def forward_dynamics(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """正动力学（ABA）：给定广义力，求加速度。同样是 O(n)。"""
        return pin.aba(
            self._m,
            self._d,
            np.asarray(q, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(tau, dtype=float),
        ).copy()

    # -- 欠驱动结构 -----------------------------------------------------------

    def selection_matrix(self) -> np.ndarray:
        """选择矩阵 ``S``，把 12 个电机力矩映射为广义力：``tau_gen = S.T @ tau_act``。

        Returns:
            形状 (12, nv) 的矩阵。浮动基座下前 6 列全为零 —— 这就是欠驱动。
        """
        S = np.zeros((len(self.actuated_dofs), self.nv))
        S[np.arange(len(self.actuated_dofs)), self.actuated_dofs] = 1.0
        return S

    # -- 接触 -----------------------------------------------------------------

    def contact_jacobian(self, q: np.ndarray, legs: tuple[str, ...] | list[str]) -> np.ndarray:
        """把若干只脚的平动雅可比按行堆叠起来。

        Args:
            q: 位置向量。
            legs: 处于支撑相的腿，例如 ``("FL", "RR")``。

        Returns:
            形状 (3 * len(legs), nv) 的矩阵。
        """
        if len(legs) == 0:
            return np.zeros((0, self.nv))
        return np.vstack([self.model.full_foot_jacobian(q, leg) for leg in legs])

    def contact_jacobian_dot_v(
        self, q: np.ndarray, v: np.ndarray, legs: tuple[str, ...] | list[str]
    ) -> np.ndarray:
        """接触约束里的漂移项 ``Jdot @ v``。

        支撑脚不动意味着 ``J a + Jdot v = 0``，所以这一项是约束右端项，
        绝不能忘 —— 漏掉它，机器人一转弯支撑脚就开始打滑。
        """
        if len(legs) == 0:
            return np.zeros(0)
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        pin.computeJointJacobiansTimeVariation(self._m, self._d, q, v)
        pin.updateFramePlacements(self._m, self._d)
        out = []
        for leg in legs:
            dJ = pin.getFrameJacobianTimeVariation(
                self._m, self._d, self.model.foot_frame_ids[leg], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            out.append(dJ[:3, :] @ v)
        return np.concatenate(out)

    def constrained_forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        legs: tuple[str, ...] | list[str],
        damping: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """带接触约束的正动力学，求解如下 KKT 系统：

        .. math::

            \\begin{bmatrix} M & -J^T \\\\ J & 0 \\end{bmatrix}
            \\begin{bmatrix} a \\\\ f \\end{bmatrix}
            = \\begin{bmatrix} S^T\\tau - Cv - g \\\\ -\\dot{J}v \\end{bmatrix}

        这里把接触当作**双边刚性约束**：脚既不会滑动，也不会离地。真实
        地面只能推不能拉（``f_z >= 0``），还有摩擦锥约束 —— 那些不等式
        要等到 MPC 与 WBC 里用 QP 处理。这个函数给的是仿真意义上的"如果
        脚焊在地上会怎样"。

        Args:
            q: 位置向量。
            v: 速度向量。
            tau: 广义力（长度 nv，浮动基座下前 6 项应为零）。
            legs: 支撑腿。
            damping: KKT 求解的正则项。

        Returns:
            ``(a, f)`` —— 广义加速度，以及按 ``legs`` 顺序堆叠的接触力。
        """
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        J = self.contact_jacobian(q, legs)
        # Pinocchio 内部解的是 J a + gamma = 0，所以 gamma 直接传 Jdot·v，
        # **不要取负**。传错符号时约束残差是 O(0.1)，机器人看起来能站住，
        # 但支撑脚会缓慢滑移 —— 这类 bug 极难从现象反推。
        gamma = self.contact_jacobian_dot_v(q, v, legs)
        a = pin.forwardDynamics(self._m, self._d, q, v, np.asarray(tau, dtype=float), J, gamma, damping)
        return a.copy(), self._d.lambda_c.copy()

    # -- 质心量（凸 MPC 的接口） ----------------------------------------------

    def center_of_mass(self, q: np.ndarray) -> np.ndarray:
        """整机质心在世界系下的位置。"""
        return pin.centerOfMass(self._m, self._d, np.asarray(q, dtype=float)).copy()

    def com_velocity(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """质心速度。"""
        pin.centerOfMass(self._m, self._d, np.asarray(q, dtype=float), np.asarray(v, dtype=float))
        return self._d.vcom[0].copy()

    def centroidal_momentum_matrix(self, q: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """质心动量矩阵 ``A_g(q)``，满足 ``h_g = A_g v``。

        前三行是线动量（等于 ``m * v_com``），后三行是关于质心的角动量。
        """
        v = np.zeros(self.nv) if v is None else np.asarray(v, dtype=float)
        return pin.ccrba(self._m, self._d, np.asarray(q, dtype=float), v).copy()

    def centroidal_momentum(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """质心动量 ``h_g = [线动量; 关于质心的角动量]``，形状 (6,)。"""
        pin.ccrba(self._m, self._d, np.asarray(q, dtype=float), np.asarray(v, dtype=float))
        return self._d.hg.vector.copy()

    def centroidal_inertia(self, q: np.ndarray) -> np.ndarray:
        """关于质心的**复合刚体转动惯量**，在世界系轴向下表达，形状 (3, 3)。

        这就是凸 MPC 该用的那个惯量。**不要用躯干连杆自身的惯量** ——
        对 Go2 而言两者相差 5 到 7 倍，因为腿张开在外，占了一半以上的质量。
        """
        pin.ccrba(self._m, self._d, np.asarray(q, dtype=float), np.zeros(self.nv))
        return np.array(self._d.Ig.inertia, dtype=float)

    # -- 能量 -----------------------------------------------------------------

    def kinetic_energy(self, q: np.ndarray, v: np.ndarray) -> float:
        """动能 ``0.5 * v^T M(q) v``。"""
        return float(
            pin.computeKineticEnergy(self._m, self._d, np.asarray(q, dtype=float), np.asarray(v, dtype=float))
        )

    def potential_energy(self, q: np.ndarray) -> float:
        """重力势能。"""
        return float(pin.computePotentialEnergy(self._m, self._d, np.asarray(q, dtype=float)))

    def total_energy(self, q: np.ndarray, v: np.ndarray) -> float:
        """机械能总和。无接触、无驱动时它必须守恒 —— 这是最好用的积分器体检项。"""
        return self.kinetic_energy(q, v) + self.potential_energy(q)

    # -- 静力学 ---------------------------------------------------------------

    def gravity_compensation_torque(
        self, q: np.ndarray, legs: tuple[str, ...] | list[str] = LEGS
    ) -> tuple[np.ndarray, np.ndarray]:
        """静止站立时，使机器人保持平衡所需的接触力与关节力矩。

        求解 ``J^T f = g``（对基座那 6 行），四足支撑时接触力有 12 个未知量
        而只有 6 个方程 —— **静不定**，这里取最小二范数解。真实控制器会用
        QP，把摩擦锥和 ``f_z >= 0`` 也加进去（见 M7 与 M8）。

        Args:
            q: 位置向量。
            legs: 支撑腿。

        Returns:
            ``(f, tau_act)`` —— 堆叠的接触力，以及对应的 12 个关节力矩。
        """
        q = np.asarray(q, dtype=float)
        g = self.gravity_vector(q)
        J = self.contact_jacobian(q, legs)
        # 基座那 6 行没有电机，只能靠接触力平衡 -> 由它们定出 f。
        f = np.linalg.lstsq(J[:, : FLOATING_BASE_DOF].T, g[:FLOATING_BASE_DOF], rcond=None)[0]
        # 关节那些行：电机补上重力与接触力之差。
        tau_act = g[self.actuated_dofs] - J[:, self.actuated_dofs].T @ f
        return f, tau_act
