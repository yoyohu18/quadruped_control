"""摆动腿控制器：把足端轨迹落实成关节指令。

这是**第一次把里程碑 1 的逆运动学放进真正的控制回路**。整条链路是：

.. code-block:: text

    步态调度器(M4)  ──摆动进度 s──┐
    落脚点规划器(M6) ──落地目标──┤
    状态估计器(M3)   ──躯干位姿──┤
                                  v
                          足端轨迹(本模块)
                                  │ 世界系足端位置
                                  v
                          转换到髋系 (需要躯干位姿)
                                  │
                                  v
                          逆运动学 IK (M1) ──> 关节角
                          雅可比逆 J^-1 (M1) ──> 关节角速度

有三个纯"工程"但缺一不可的环节，教科书里通常不提：

1. **离地点必须在离地那一刻锁存。** 轨迹起点不是"当前足端位置"，而是
   这条腿离开地面时的位置。每个周期实时读当前位置会让轨迹自我追逐。
2. **落地目标可以中途更新。** 落脚点规划器每个周期都在重算目标，轨迹
   必须能跟着变，否则机器人无法响应速度指令的变化。
3. **迟落地要继续下探。** 地面比预期低时，摆动进度已经到 1 却还没碰到
   地。此时必须以受控速度继续向下探，而不是停在空中 —— 否则机器人会
   "踩空"，重心失去支撑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gait_scheduler import GaitScheduler
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS, inverse_kinematics, leg_jacobian

from .trajectory import SWING_TRAJECTORIES, SwingTrajectory

__all__ = ["SwingLegConfig", "SwingLegController"]


@dataclass
class SwingLegConfig:
    """摆动腿规划参数。

    Attributes:
        swing_height: 抬腿高度，米。越障能力与能耗的直接权衡。
        trajectory: 轨迹类型，见 :data:`~swing_planner.trajectory.SWING_TRAJECTORIES`。
        probe_velocity: 迟落地时的下探速度，m/s。太慢会踩空，太快等于
            自己制造冲击。
        max_probe_depth: 最大下探深度，米。超过则认为地面塌陷或标定错误。
        clamp_ik: 逆运动学是否对不可达目标做钳制而不是抛异常。控制回路里
            必须为 ``True`` —— 见里程碑 1 关于 ``clamp`` 的说明。
    """

    swing_height: float = 0.08
    trajectory: str = "bezier"
    probe_velocity: float = 0.15
    max_probe_depth: float = 0.06
    clamp_ik: bool = True


@dataclass
class _LegState:
    """单条腿的摆动状态。"""

    liftoff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    touchdown: np.ndarray = field(default_factory=lambda: np.zeros(3))
    trajectory: SwingTrajectory | None = None
    was_stance: bool = True
    probe_depth: float = 0.0


class SwingLegController:
    """按步态相位生成足端轨迹，并解算为关节指令。

    Args:
        scheduler: 里程碑 4 的步态调度器，提供摆动进度。
        config: 摆动腿参数。
    """

    def __init__(self, scheduler: GaitScheduler, config: SwingLegConfig | None = None) -> None:
        self.scheduler = scheduler
        self.cfg = config or SwingLegConfig()
        if self.cfg.trajectory not in SWING_TRAJECTORIES:
            raise ValueError(
                f"未知轨迹类型 '{self.cfg.trajectory}'，可选：{sorted(SWING_TRAJECTORIES)}"
            )
        self._legs = {leg: _LegState() for leg in LEGS}

    # -- 状态推进 -------------------------------------------------------------

    def update(
        self,
        t: float,
        foot_positions_world: np.ndarray,
        touchdown_targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        """每个控制周期调用一次，维护离地点锁存与落地目标。

        Args:
            t: 当前时刻。
            foot_positions_world: 四只脚当前的世界位置，形状 (4, 3)。
            touchdown_targets: 各腿的落地目标（世界系），由落脚点规划器
                提供。缺省时沿用上一次的目标。
        """
        contact = self.scheduler.contact(t)
        feet = np.asarray(foot_positions_world, dtype=float).reshape(4, 3)

        for i, leg in enumerate(LEGS):
            state = self._legs[leg]

            if contact[i]:
                # 支撑腿的"目标"就是它当前所在的位置 —— 保持不动。
                # 必须每周期锁存，否则开机瞬间尚未摆动过的腿会拿到零向量，
                # 逆运动学随即解出一个荒唐的姿态。
                state.touchdown = feet[i].copy()
                state.was_stance = True
                state.probe_depth = 0.0
                state.trajectory = None
                continue

            # 刚离地：锁存起点。轨迹起点是离地位置，不是当前位置。
            if state.was_stance:
                state.liftoff = feet[i].copy()
                state.was_stance = False
                if touchdown_targets is None or leg not in touchdown_targets:
                    # 没有规划器时退化为原地落回，仅用于单元测试
                    state.touchdown = feet[i].copy()

            if touchdown_targets is not None and leg in touchdown_targets:
                state.touchdown = np.asarray(touchdown_targets[leg], dtype=float).reshape(3).copy()

            state.trajectory = SWING_TRAJECTORIES[self.cfg.trajectory](
                state.liftoff, state.touchdown, self.cfg.swing_height
            )

    # -- 足端指令 -------------------------------------------------------------

    def foot_target(self, t: float, leg: str, in_contact: bool | None = None) -> np.ndarray:
        """该腿在当前时刻的期望足端位置（世界系），形状 (3,)。

        支撑腿返回其锁存的落地点（表示"待在原地"）。摆动腿返回轨迹上的点。
        """
        state = self._legs[leg]
        i = LEGS.index(leg)
        contact = self.scheduler.contact(t)[i] if in_contact is None else in_contact
        if contact or state.trajectory is None:
            return state.touchdown.copy()
        s = self.scheduler.swing_phase(t)[i]
        if np.isnan(s):
            return state.touchdown.copy()
        return state.trajectory.position(float(s))

    def foot_velocity(self, t: float, leg: str) -> np.ndarray:
        """期望足端速度（世界系），形状 (3,)。支撑腿返回零。"""
        state = self._legs[leg]
        i = LEGS.index(leg)
        if self.scheduler.contact(t)[i] or state.trajectory is None:
            return np.zeros(3)
        s = self.scheduler.swing_phase(t)[i]
        if np.isnan(s):
            return np.zeros(3)
        return state.trajectory.velocity(float(s), self.scheduler.gait.swing_duration)

    def probe(self, leg: str, dt: float) -> np.ndarray:
        """迟落地时的下探增量，形状 (3,)。

        摆动进度已到 1 却仍未检测到接触时调用。以恒定速度向下探，直到
        触地或达到 :attr:`SwingLegConfig.max_probe_depth`。

        Args:
            leg: 腿标识。
            dt: 控制周期。

        Returns:
            本周期应叠加到足端目标上的位移，形状 (3,)。已达最大深度时为零。
        """
        state = self._legs[leg]
        step = self.cfg.probe_velocity * dt
        if state.probe_depth + step > self.cfg.max_probe_depth:
            step = max(0.0, self.cfg.max_probe_depth - state.probe_depth)
        state.probe_depth += step
        return np.array([0.0, 0.0, -step])

    def probe_depth(self, leg: str) -> float:
        """该腿当前已下探的深度，米。"""
        return self._legs[leg].probe_depth

    # -- 关节指令：这里用上里程碑 1 的 IK 与雅可比 ----------------------------

    def joint_command(
        self,
        t: float,
        leg: str,
        base_position: np.ndarray,
        base_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把世界系足端目标解算成该腿的关节角与关节角速度。

        Args:
            t: 当前时刻。
            leg: 腿标识。
            base_position: 躯干位置（世界系），来自状态估计器。
            base_rotation: 躯干姿态矩阵（机体到世界）。

        Returns:
            ``(q, dq)`` —— 三个关节角与角速度。

        Note:
            关节角速度由 :math:`\\dot q = J^{-1} v` 求出。在腿接近伸直
            （雅可比接近奇异）时 :math:`J^{-1}` 会放大，这正是里程碑 1
            里那张条件数图所刻画的风险。这里用最小二乘求解而非直接求逆，
            让奇异附近的行为退化得更平缓。
        """
        base_position = np.asarray(base_position, dtype=float).reshape(3)
        base_rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
        geom = LEG_GEOMETRY[leg]

        p_world = self.foot_target(t, leg)
        v_world = self.foot_velocity(t, leg)

        # 世界系 -> 髋系（扣掉躯干位姿与髋部安装偏置）
        hip_world = base_position + base_rotation @ HIP_OFFSETS[leg]
        p_hip = base_rotation.T @ (p_world - hip_world)
        v_hip = base_rotation.T @ v_world

        q = inverse_kinematics(p_hip, geom, clamp=self.cfg.clamp_ik)
        J = leg_jacobian(q, geom)
        dq = np.linalg.lstsq(J, v_hip, rcond=None)[0]
        return q, dq

    # -- 可行性检查 -----------------------------------------------------------

    def check_swing_feasibility(
        self,
        leg: str,
        liftoff: np.ndarray,
        touchdown: np.ndarray,
        base_position: np.ndarray,
        base_rotation: np.ndarray,
        joint_limits: tuple[np.ndarray, np.ndarray],
        n: int = 401,
    ) -> dict:
        """扫描一整段摆动，检查关节限位、雅可比条件数与关节速度。

        规划器给出的轨迹在**笛卡尔空间**看起来总是漂亮的，但它未必在关节
        空间可行。这个检查用来离线标定抬腿高度与步长的上限。

        **刻意做成无状态的。** 它直接按几何构造轨迹并沿 :math:`s\\in[0,1]`
        扫描，不读取控制器的内部状态，也不依赖当前时刻 —— 否则"回放一段
        已经过去的摆动"会拿到被清空的轨迹，得到一个看似通过、实则什么也
        没检查的平凡结果。

        Args:
            leg: 腿标识。
            liftoff: 离地点（世界系）。
            touchdown: 落地点（世界系）。
            base_position: 摆动期间的躯干位置（世界系）。这里按定值处理 ——
                离线标定关心的是腿本身的能力边界。
            base_rotation: 躯干姿态矩阵。
            joint_limits: ``(lower, upper)``，各形状 (3,)。
            n: 采样点数。

        Returns:
            含 ``feasible``、``worst_margin``、``max_condition_number``、
            ``max_joint_speed``、``touchdown_speed`` 等字段的字典。
        """
        lower, upper = (np.asarray(x, dtype=float).reshape(3) for x in joint_limits)
        base_position = np.asarray(base_position, dtype=float).reshape(3)
        base_rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3)
        geom = LEG_GEOMETRY[leg]
        hip_world = base_position + base_rotation @ HIP_OFFSETS[leg]

        traj = SWING_TRAJECTORIES[self.cfg.trajectory](liftoff, touchdown, self.cfg.swing_height)
        swing_duration = self.scheduler.gait.swing_duration
        s = np.linspace(0.0, 1.0, n)
        positions = traj.position(s)
        velocities = traj.velocity(s, swing_duration)

        conds, margins, dq_norms = [], [], []
        feasible = True
        for p_world, v_world in zip(positions, velocities):
            p_hip = base_rotation.T @ (p_world - hip_world)
            q = inverse_kinematics(p_hip, geom, clamp=self.cfg.clamp_ik)
            margin = float(min(np.min(q - lower), np.min(upper - q)))
            margins.append(margin)
            if margin < 0.0:
                feasible = False
            J = leg_jacobian(q, geom)
            conds.append(float(np.linalg.cond(J)))
            dq_norms.append(float(np.linalg.norm(np.linalg.lstsq(J, base_rotation.T @ v_world, rcond=None)[0])))

        return {
            "feasible": feasible,
            "worst_margin": float(np.min(margins)),
            "max_condition_number": float(np.max(conds)),
            "max_joint_speed": float(np.max(dq_norms)),
            "touchdown_speed": float(np.linalg.norm(traj.touchdown_velocity(swing_duration))),
            "peak_foot_acceleration": traj.peak_acceleration(swing_duration),
        }
