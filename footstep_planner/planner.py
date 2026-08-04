"""落脚点规划器：把策略、步态、工作空间限制整合成可用的接口。

它是**里程碑 4、5、6 三者的汇合点**：

* 从步态调度器（M4）拿"这条腿还有多久落地"；
* 用 Raibert 或捕获点策略算出"落在哪"；
* 把结果交给摆动腿规划器（M5）去走那条弧线。

除了策略本身，还有三件真实机器人必须处理的事：

1. **转弯**。躯干在摆动期间会转过 :math:`\\dot\\psi \\cdot T` 的角度，
   髋关节的位置随之改变。落脚点必须按**落地时刻**的髋位置算，而不是
   当前时刻的。
2. **工作空间钳制**。策略算出的点可能超出腿的可达范围。里程碑 5 已经
   量出这个边界（步长上限约 50 cm，受关节限位而非几何限制）。超了必须
   钳制，而不是让逆运动学去救。
3. **地形高度**。落脚点的 z 不一定是 0。这里留出接口，由感知模块填。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gait_scheduler import GaitScheduler
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS

from .lipm import LIPMParams, capture_point
from .raibert import (
    capture_point_footstep,
    deadbeat_feedback_gain,
    exact_footstep,
    optimal_feedback_gain,
    raibert_footstep,
)

__all__ = ["FootstepPlannerConfig", "FootstepPlanner"]


@dataclass
class FootstepPlannerConfig:
    """落脚点规划参数。

    Attributes:
        strategy: ``"raibert"``、``"capture_point"`` 或 ``"exact"``。
            ``"exact"`` 使用倒立摆极限环的精确系数，稳态速度误差为零；
            另两个是它的渐近近似，各有约 11% 与 93% 的系数偏差。
        feedback_gain: Raibert 速度反馈增益，秒。设为 ``None`` 时自动
            取 :func:`~footstep_planner.raibert.optimal_feedback_gain`。
        nominal_height: 标称质心高度，米。决定倒立摆的 :math:`\\omega`。
        max_step_radius: 落脚点相对髋部投影的最大水平偏移，米。
            默认 0.22 m —— 里程碑 5 实测步长上限约 0.50 m，即半程 0.25 m，
            这里留 12% 的安全余量。
        lateral_offset_scale: 侧向落脚点相对标称的缩放，用于调整步宽。
    """

    strategy: str = "raibert"
    feedback_gain: float | None = None
    nominal_height: float = 0.30
    max_step_radius: float = 0.22
    lateral_offset_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in ("raibert", "capture_point", "exact"):
            raise ValueError(
                f"未知策略 '{self.strategy}'，可选 raibert / capture_point / exact"
            )
        if self.max_step_radius <= 0.0:
            raise ValueError("最大步长半径必须为正")


class FootstepPlanner:
    """决定每条摆动腿该落在哪里。

    Args:
        scheduler: 步态调度器（M4）。
        config: 规划参数。
    """

    def __init__(self, scheduler: GaitScheduler, config: FootstepPlannerConfig | None = None) -> None:
        self.scheduler = scheduler
        self.cfg = config or FootstepPlannerConfig()
        self.lipm = LIPMParams(height=self.cfg.nominal_height)
        if self.cfg.feedback_gain is None:
            # 两种写法的稳定条件不同，默认增益必须分别选取。
            stance = scheduler.gait.stance_duration
            if self.cfg.strategy == "exact":
                self.cfg.feedback_gain = deadbeat_feedback_gain(self.lipm, stance)
            else:
                self.cfg.feedback_gain = optimal_feedback_gain(self.lipm, stance)

    # -- 髋部投影 -------------------------------------------------------------

    def nominal_hip_projection(
        self,
        leg: str,
        base_position: np.ndarray,
        yaw: float,
        yaw_rate: float = 0.0,
        time_ahead: float = 0.0,
    ) -> np.ndarray:
        """落地时刻髋关节在地面的投影位置。

        **必须用落地时刻的偏航角，而不是当前时刻的。** 转弯时躯干在摆动
        期间会转过 :math:`\\dot\\psi \\cdot \\Delta t`，忽略这一点会让机器人
        转弯时步宽越走越歪。

        Args:
            leg: 腿标识。
            base_position: 当前躯干位置（世界系），形状 (3,)。
            yaw: 当前偏航角，弧度。
            yaw_rate: 偏航角速度，rad/s。
            time_ahead: 距离落地还有多久，秒。

        Returns:
            髋部地面投影的水平位置，形状 (2,)。
        """
        future_yaw = yaw + yaw_rate * time_ahead
        c, s = np.cos(future_yaw), np.sin(future_yaw)
        R = np.array([[c, -s], [s, c]])

        # 标称足端在躯干系下的水平位置：髋偏置 + 侧摆偏置
        offset = np.array(
            [
                HIP_OFFSETS[leg][0],
                (HIP_OFFSETS[leg][1] + LEG_GEOMETRY[leg].l0) * self.cfg.lateral_offset_scale,
            ]
        )
        base_xy = np.asarray(base_position, dtype=float).reshape(-1)[:2]
        return base_xy + R @ offset

    # -- 落脚点 ---------------------------------------------------------------

    def plan_leg(
        self,
        leg: str,
        base_position: np.ndarray,
        velocity: np.ndarray,
        velocity_command: np.ndarray,
        yaw: float = 0.0,
        yaw_rate: float = 0.0,
        time_to_touchdown: float = 0.0,
        terrain_height: float = 0.0,
    ) -> np.ndarray:
        """单条腿的落脚点（世界系，形状 (3,)）。

        Args:
            leg: 腿标识。
            base_position: 躯干位置（世界系）。质心在这里用躯干位置近似 ——
                里程碑 2 算出二者只差 2 cm，对落脚点而言可以忽略。
            velocity: 当前躯干水平速度，来自状态估计器（M3）。
            velocity_command: 期望水平速度。
            yaw: 当前偏航角。
            yaw_rate: 偏航角速度。
            time_to_touchdown: 距离落地的时间，来自步态调度器（M4）。
            terrain_height: 落脚处的地面高度，由感知给出。

        Returns:
            落脚点，形状 (3,)。
        """
        hip = self.nominal_hip_projection(leg, base_position, yaw, yaw_rate, time_to_touchdown)
        stance = self.scheduler.gait.stance_duration

        if self.cfg.strategy == "raibert":
            target = raibert_footstep(hip, velocity, velocity_command, stance, self.cfg.feedback_gain)
        elif self.cfg.strategy == "exact":
            target = exact_footstep(
                hip, velocity, velocity_command, self.lipm, stance, self.cfg.feedback_gain
            )
        else:
            com_xy = np.asarray(base_position, dtype=float).reshape(-1)[:2]
            target = capture_point_footstep(
                hip, com_xy, velocity, velocity_command, self.lipm, stance
            )

        target = self._clamp_to_workspace(target, hip)
        return np.array([target[0], target[1], terrain_height])

    def plan(
        self,
        t: float,
        base_position: np.ndarray,
        velocity: np.ndarray,
        velocity_command: np.ndarray,
        yaw: float = 0.0,
        yaw_rate: float = 0.0,
        terrain_height: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """为**所有摆动腿**规划落脚点，直接喂给摆动腿规划器（M5）。

        Args:
            t: 当前时刻。
            base_position: 躯干位置（世界系）。
            velocity: 当前水平速度。
            velocity_command: 期望水平速度。
            yaw: 当前偏航角。
            yaw_rate: 偏航角速度。
            terrain_height: 各腿落脚处的地面高度，缺省为 0。

        Returns:
            ``{腿: 落脚点(3,)}``，只包含当前处于摆动相的腿。
        """
        contact = self.scheduler.contact(t)
        ttd = self.scheduler.time_to_touchdown(t)
        heights = terrain_height or {}
        out = {}
        for i, leg in enumerate(LEGS):
            if contact[i]:
                continue
            out[leg] = self.plan_leg(
                leg,
                base_position,
                velocity,
                velocity_command,
                yaw=yaw,
                yaw_rate=yaw_rate,
                time_to_touchdown=float(ttd[i]),
                terrain_height=heights.get(leg, 0.0),
            )
        return out

    # -- 工作空间 -------------------------------------------------------------

    def _clamp_to_workspace(self, target: np.ndarray, hip: np.ndarray) -> np.ndarray:
        """把落脚点钳制在髋部周围的可达圆内。

        钳制发生在**规划层**而不是逆运动学层。里程碑 1 的 ``clamp`` 是
        最后一道安全网；真正该做的是规划器一开始就不要提出够不到的目标 ——
        那样至少方向还是对的，只是幅度被限制。
        """
        delta = target - hip
        radius = float(np.linalg.norm(delta))
        if radius <= self.cfg.max_step_radius:
            return target
        return hip + delta * (self.cfg.max_step_radius / radius)

    def is_clamped(self, target: np.ndarray, hip: np.ndarray) -> bool:
        """该落脚点是否已被工作空间限制截断。

        持续被钳制说明机器人在试图走得比腿允许的更快 —— 上层应当降低
        速度指令或切换到更高占空比的步态，而不是硬撑。
        """
        delta = np.asarray(target, dtype=float)[:2] - np.asarray(hip, dtype=float)[:2]
        return bool(np.linalg.norm(delta) >= self.cfg.max_step_radius - 1e-9)

    # -- 诊断 -----------------------------------------------------------------

    def capture_point_world(self, base_position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """当前捕获点（世界系水平位置），形状 (2,)。

        它是最有用的一个在线诊断量：捕获点跑出可达范围，就意味着这一步
        无论落在哪里都救不回来了。
        """
        return capture_point(base_position, velocity, self.lipm)
