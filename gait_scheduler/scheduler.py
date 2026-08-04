"""步态调度器：整个控制栈共同的时间基准。

它回答四个问题，分别服务四个下游模块：

============================  ==========================================
问题                          谁在问
============================  ==========================================
现在哪几条腿踩着地？          WBC（约束集合）、状态估计（观测集合）
这条腿的摆动进行到百分之几？  摆动腿规划器（M5）
这条腿还有多久落地？          落脚点规划器（M6）
未来 N 步的接触序列是什么？   凸 MPC（M7）
============================  ==========================================

最后一条是四足 MPC 与无人机 MPC 最本质的差别。无人机的控制分配矩阵不随
时间变化，MPC 每一步的结构完全相同；四足的接触集合每走一步就变一次，
**MPC 必须提前知道整个预测时域内的接触序列**，才能构造出那个 QP。所以
步态调度器不是"辅助模块"，它是 MPC 的输入之一。

这里只做**时间上的**调度。脚落在**哪里**是落脚点规划器（M6）的事，
脚**怎么过去**是摆动腿规划器（M5）的事。三者严格分工。
"""

from __future__ import annotations

import numpy as np

from kinematics import LEGS

from .gait import GaitDefinition, get_gait

__all__ = ["GaitScheduler"]


class GaitScheduler:
    """周期性步态的相位调度器，支持在周期边界安全切换步态。

    Args:
        gait: 初始步态，可以是名字或 :class:`GaitDefinition`。
        t0: 起始时刻，秒。
    """

    def __init__(self, gait: str | GaitDefinition = "trot", t0: float = 0.0) -> None:
        self.gait = get_gait(gait) if isinstance(gait, str) else gait
        self._t0 = float(t0)
        self._phase_at_t0 = 0.0
        self._pending: GaitDefinition | None = None

    # -- 相位 -----------------------------------------------------------------

    def global_phase(self, t: float) -> float:
        """整个步态循环的归一化相位，取值 [0, 1)。"""
        return float(((t - self._t0) / self.gait.period + self._phase_at_t0) % 1.0)

    def leg_phase(self, t: float, leg: str | None = None) -> np.ndarray | float:
        """每条腿在自己循环中的相位，取值 [0, 1)，0 表示刚落地。

        Args:
            t: 时刻，秒。
            leg: 指定腿则返回标量，否则返回按 :data:`kinematics.LEGS`
                排列的 (4,) 数组。
        """
        phase = (self.global_phase(t) - self.gait.offsets_array) % 1.0
        if leg is None:
            return phase
        return float(phase[LEGS.index(leg)])

    # -- 接触状态 -------------------------------------------------------------

    def contact(self, t: float) -> np.ndarray:
        """当前各腿是否触地，形状 (4,) 的布尔数组。"""
        return self.leg_phase(t) < self.gait.duty_factor

    def n_stance(self, t: float) -> int:
        """当前支撑腿数量。"""
        return int(self.contact(t).sum())

    def contact_schedule(self, t0: float, dt: float, horizon: int) -> np.ndarray:
        """未来 ``horizon`` 步的接触序列，形状 (horizon, 4)。

        **这是凸 MPC 的直接输入。** MPC 在每个预测步上只为支撑腿引入接触力
        决策变量，摆动腿的力被强制为零 —— 所以整个 QP 的结构由这张表决定。

        Args:
            t0: 预测时域起始时刻。
            dt: MPC 的离散步长（通常远大于控制周期，例如 0.03 s）。
            horizon: 预测步数。

        Returns:
            布尔数组，``[k, i]`` 表示第 ``k`` 个预测步上第 ``i`` 条腿是否触地。
        """
        if horizon <= 0:
            raise ValueError(f"预测步数必须为正，收到 {horizon}")
        times = t0 + np.arange(horizon) * dt
        phases = (
            ((times[:, None] - self._t0) / self.gait.period + self._phase_at_t0) % 1.0
            - self.gait.offsets_array[None, :]
        ) % 1.0
        return phases < self.gait.duty_factor

    # -- 相位内的归一化进度 ---------------------------------------------------

    def swing_phase(self, t: float) -> np.ndarray:
        """摆动进度，取值 [0, 1)；支撑相的腿返回 NaN。

        摆动腿规划器（M5）用它来在足端轨迹上取点：0 表示刚离地，
        1 表示即将落地。
        """
        phase = self.leg_phase(t)
        D = self.gait.duty_factor
        out = np.full(4, np.nan)
        swing = phase >= D
        if D < 1.0:
            out[swing] = (phase[swing] - D) / (1.0 - D)
        return out

    def stance_phase(self, t: float) -> np.ndarray:
        """支撑进度，取值 [0, 1)；摆动相的腿返回 NaN。"""
        phase = self.leg_phase(t)
        D = self.gait.duty_factor
        out = np.full(4, np.nan)
        stance = phase < D
        out[stance] = phase[stance] / D
        return out

    # -- 事件倒计时 -----------------------------------------------------------

    def time_to_liftoff(self, t: float) -> np.ndarray:
        """各腿距离下一次离地还有多久，秒；已在摆动相的腿返回 0。"""
        phase = self.leg_phase(t)
        D = self.gait.duty_factor
        return np.where(phase < D, (D - phase) * self.gait.period, 0.0)

    def time_to_touchdown(self, t: float) -> np.ndarray:
        """各腿距离下一次落地还有多久，秒；已在支撑相的腿返回 0。

        **这是落脚点规划器（M6）的核心输入。** Raibert 启发式需要知道
        "这只脚还有多久落地"，才能算出它应该落在哪里。
        """
        phase = self.leg_phase(t)
        D = self.gait.duty_factor
        return np.where(phase >= D, (1.0 - phase) * self.gait.period, 0.0)

    def next_touchdown_time(self, t: float) -> np.ndarray:
        """各腿下一次落地的绝对时刻，秒。

        支撑相的腿返回的是**本次支撑结束后再次落地**的时刻，也就是
        它在完成一次摆动之后的落地时刻。
        """
        phase = self.leg_phase(t)
        return t + (1.0 - phase) * self.gait.period

    # -- 步态切换 -------------------------------------------------------------

    def request_gait(self, gait: str | GaitDefinition) -> None:
        """请求切换步态，将在下一个周期边界生效。

        **不能立即切换。** 相位偏移一变，某些腿的接触状态会瞬间跳变 ——
        正踩着地的脚突然被判定为摆动腿，WBC 立刻撤掉它的接触约束，机器人
        会瞬间失去支撑。真实系统一律在周期边界或"所有腿都触地"的时刻切换。

        Args:
            gait: 目标步态。
        """
        self._pending = get_gait(gait) if isinstance(gait, str) else gait

    def update(self, t: float) -> bool:
        """在控制回路里每步调用，处理待生效的步态切换。

        Args:
            t: 当前时刻。

        Returns:
            本次调用是否真的完成了切换。
        """
        if self._pending is None:
            return False
        phase = self.global_phase(t)
        # 跨过周期边界（相位回绕）时切换
        if phase < getattr(self, "_last_phase", 0.0) or np.isclose(phase, 0.0, atol=1e-9):
            self.gait = self._pending
            self._pending = None
            self._t0 = t
            self._phase_at_t0 = 0.0
            self._last_phase = 0.0
            return True
        self._last_phase = phase
        return False

    def force_gait(self, gait: str | GaitDefinition, t: float) -> None:
        """立即切换步态并把相位归零。

        只应在机器人静止或四脚全部触地时使用 —— 见 :meth:`request_gait`
        中关于瞬时切换危险性的说明。
        """
        self.gait = get_gait(gait) if isinstance(gait, str) else gait
        self._pending = None
        self._t0 = float(t)
        self._phase_at_t0 = 0.0
        self._last_phase = 0.0
