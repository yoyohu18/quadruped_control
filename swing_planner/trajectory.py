"""摆动腿足端轨迹：从离地点到落地点的那条弧线。

看起来只是"插值一下"，但这条曲线的形状直接决定三件事：

1. **落地冲击**。足端在落地瞬间如果还有竖直速度，就会砸在地上。冲量
   :math:`m_{\\text{eff}} v_{td}` 瞬间传给机身，还会激发打滑 —— 而打滑
   在里程碑 3 里已经证明会造成**永久性**的位置估计误差。
2. **越障能力**。抬腿高度不够就绊倒；抬太高浪费能量、还可能超出工作空间。
3. **关节负担**。轨迹的二阶导决定关节加速度，进而决定所需力矩。一条
   位置上"看着挺顺"的曲线，可能在加速度上有尖峰。

本模块给出四种轨迹，并把上面三项指标**量化对比**出来 —— 不是罗列公式，
而是回答"到底该用哪一条、为什么"。

所有轨迹都用归一化摆动进度 :math:`s \\in [0, 1]` 参数化，正好对接里程碑 4
的 :meth:`~gait_scheduler.GaitScheduler.swing_phase`。速度与加速度需要
除以摆动时长换算到物理单位，这一步由各方法的 ``swing_duration`` 参数完成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np

__all__ = [
    "SwingTrajectory",
    "SineHeightSwing",
    "CycloidSwing",
    "BezierSwing",
    "QuinticSwing",
    "SWING_TRAJECTORIES",
]


class SwingTrajectory(ABC):
    """摆动腿足端轨迹的公共接口。

    Args:
        liftoff: 离地点，世界系，形状 (3,)。
        touchdown: 落地点，世界系，形状 (3,)。
        height: 相对离地/落地点连线的最大抬起高度，米。
    """

    def __init__(self, liftoff: np.ndarray, touchdown: np.ndarray, height: float) -> None:
        self.liftoff = np.asarray(liftoff, dtype=float).reshape(3).copy()
        self.touchdown = np.asarray(touchdown, dtype=float).reshape(3).copy()
        if height <= 0.0:
            raise ValueError(f"抬腿高度必须为正，收到 {height}")
        self.height = float(height)

    @abstractmethod
    def position(self, s: float | np.ndarray) -> np.ndarray:
        """摆动进度 ``s`` 处的足端位置，形状 (3,) 或 (n, 3)。"""

    def velocity(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        """足端速度，m/s。默认用中心差分，子类可覆盖为解析式。

        端点处把整个差分模板**整体移入区间内部**，而不是截断步长。截断会
        让中心差分退化成不等间距的三点公式，在端点给出完全错误的结果 ——
        评价"落地速度"这种端点量时这一步是致命的。
        """
        eps = 1e-6
        s = np.asarray(s, dtype=float)
        sc = np.clip(s, eps, 1.0 - eps)
        return (self.position(sc + eps) - self.position(sc - eps)) / (2 * eps * swing_duration)

    def acceleration(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        """足端加速度，m/s^2。默认用二阶中心差分，端点同样整体内移模板。"""
        eps = 1e-4
        s = np.asarray(s, dtype=float)
        sc = np.clip(s, eps, 1.0 - eps)
        p0 = self.position(sc - eps)
        p1 = self.position(sc)
        p2 = self.position(sc + eps)
        return (p2 - 2 * p1 + p0) / ((eps * swing_duration) ** 2)

    # -- 评价指标 -------------------------------------------------------------

    def touchdown_velocity(self, swing_duration: float) -> np.ndarray:
        """落地瞬间的足端速度，m/s。**竖直分量就是冲击的来源。**"""
        return self.velocity(1.0, swing_duration)

    def liftoff_velocity(self, swing_duration: float) -> np.ndarray:
        """离地瞬间的足端速度，m/s。不为零会拖拽地面，同样引发打滑。"""
        return self.velocity(0.0, swing_duration)

    def peak_acceleration(self, swing_duration: float, n: int = 2001) -> float:
        """整条轨迹上的最大加速度模长，m/s^2。它正比于所需的峰值关节力矩。"""
        s = np.linspace(0.0, 1.0, n)
        return float(np.max(np.linalg.norm(self.acceleration(s, swing_duration), axis=-1)))

    def clearance_profile(self, n: int = 501) -> tuple[np.ndarray, np.ndarray]:
        """足端相对离地-落地连线的离地高度随进度的变化。

        Returns:
            ``(s, clearance)``，``clearance`` 单位米。越障能力看的是这条
            曲线在障碍物所处水平位置上的取值，而不是它的最大值。
        """
        s = np.linspace(0.0, 1.0, n)
        p = self.position(s)
        baseline = self.liftoff[None, :] + s[:, None] * (self.touchdown - self.liftoff)[None, :]
        return s, p[:, 2] - baseline[:, 2]


class SineHeightSwing(SwingTrajectory):
    """水平三次平滑插值 + 竖直半正弦。**最常见的写法，也是最常见的坑。**

    .. math::

        x(s) = x_0 + (3s^2 - 2s^3)\\,\\Delta x, \\qquad
        z(s) = z_{\\text{base}}(s) + h\\sin(\\pi s)

    水平方向没问题：三次平滑插值在两端速度为零。**问题出在竖直方向** ——

    .. math::  \\left.\\frac{dz}{ds}\\right|_{s=1} = h\\pi\\cos(\\pi) = -h\\pi

    落地瞬间竖直速度是 :math:`-h\\pi / T_{\\text{swing}}`，抬得越高、摆动
    越快，砸得越狠。取 :math:`h = 6` cm、:math:`T = 0.2` s 时高达
    **0.94 m/s** —— 相当于从 4.5 cm 高自由落体砸下去。

    里程碑 3 里手写的数据生成器用的就是这条轨迹，当时只关心几何一致性，
    没考虑冲击。留在这里作为对照基线。
    """

    def position(self, s: float | np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        alpha = 3 * s**2 - 2 * s**3
        base = self.liftoff + alpha[..., None] * (self.touchdown - self.liftoff)
        out = base.copy()
        out[..., 2] += self.height * np.sin(np.pi * s)
        return out

    def velocity(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        d_alpha = 6 * s - 6 * s**2
        out = d_alpha[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] += self.height * np.pi * np.cos(np.pi * s)
        return out / swing_duration

    def acceleration(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        dd_alpha = 6 - 12 * s
        out = dd_alpha[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] -= self.height * np.pi**2 * np.sin(np.pi * s)
        return out / swing_duration**2


class CycloidSwing(SwingTrajectory):
    """摆线轨迹：两端**位置与速度**都连续，且速度为零。

    .. math::

        x(s) = x_0 + \\left(s - \\frac{\\sin 2\\pi s}{2\\pi}\\right)\\Delta x,
        \\qquad z(s) = z_{\\text{base}}(s) + \\frac{h}{2}\\left(1 - \\cos 2\\pi s\\right)

    验证两端：:math:`dx/ds = \\Delta x(1 - \\cos 2\\pi s)`，在 :math:`s=0,1`
    处为零；:math:`dz/ds = h\\pi\\sin 2\\pi s`，同样为零。

    **落地速度精确为零**，代价是加速度在两端不为零（有跳变），关节力矩
    会有台阶。适合大多数场合，实现也最简单。
    """

    def position(self, s: float | np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        alpha = s - np.sin(2 * np.pi * s) / (2 * np.pi)
        base = self.liftoff + alpha[..., None] * (self.touchdown - self.liftoff)
        out = base.copy()
        out[..., 2] += 0.5 * self.height * (1.0 - np.cos(2 * np.pi * s))
        return out

    def velocity(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        d_alpha = 1.0 - np.cos(2 * np.pi * s)
        out = d_alpha[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] += self.height * np.pi * np.sin(2 * np.pi * s)
        return out / swing_duration

    def acceleration(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        dd_alpha = 2 * np.pi * np.sin(2 * np.pi * s)
        out = dd_alpha[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] += 2 * self.height * np.pi**2 * np.cos(2 * np.pi * s)
        return out / swing_duration**2


class BezierSwing(SwingTrajectory):
    """贝塞尔曲线，端点重复次数可调 —— MIT Cheetah 系列的做法。

    贝塞尔曲线在端点的导数只取决于最靠近端点的几个控制点：

    .. math::  B'(0) = n(P_1 - P_0), \\qquad B''(0) = n(n-1)(P_2 - 2P_1 + P_0)

    所以**把端点控制点重复 k 次，就能让前 k-1 阶导数在该端点为零**。这是
    一个非常好用的技巧：不需要解方程，直接堆控制点即可。

    Args:
        liftoff: 离地点。
        touchdown: 落地点。
        height: 抬腿高度。
        endpoint_multiplicity: 端点重复次数。2 表示速度为零，
            3 表示速度与加速度都为零。
    """

    def __init__(
        self,
        liftoff: np.ndarray,
        touchdown: np.ndarray,
        height: float,
        endpoint_multiplicity: int = 3,
    ) -> None:
        super().__init__(liftoff, touchdown, height)
        if endpoint_multiplicity < 1:
            raise ValueError("端点重复次数至少为 1")
        self.endpoint_multiplicity = int(endpoint_multiplicity)
        self.control_points = self._build_control_points()

    def _build_control_points(self) -> np.ndarray:
        """构造控制点，并把顶点抬高到使**实际**离地高度等于 ``height``。

        贝塞尔曲线**不穿过中间控制点**：顶点控制点在 s=0.5 处的权重只有
        :math:`\binom{2k}{k}/2^{2k}`（k=3 时是 20/64 = 0.3125）。直接把
        控制点设在目标高度，实际只能抬起三分之一。这里按权重反解出所需的
        控制点高度，让实际抬腿高度精确等于设定值。
        """
        k = self.endpoint_multiplicity
        n = 2 * k
        apex_weight = comb(n, k) * 0.5**n  # 顶点控制点在 s=0.5 处的权重
        mid = 0.5 * (self.liftoff + self.touchdown)
        apex = mid.copy()
        apex[2] = mid[2] + self.height / apex_weight
        pts = [self.liftoff] * k + [apex] + [self.touchdown] * k
        return np.array(pts)

    @staticmethod
    def _evaluate(points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """在参数 ``s`` 处求值任意一组控制点定义的贝塞尔曲线。"""
        n = len(points) - 1
        out = np.zeros(s.shape + (3,))
        for i, p in enumerate(points):
            w = comb(n, i) * s**i * (1.0 - s) ** (n - i)
            out = out + w[..., None] * p
        return out

    def position(self, s: float | np.ndarray) -> np.ndarray:
        return self._evaluate(self.control_points, np.asarray(s, dtype=float))

    def velocity(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        """解析导数：贝塞尔曲线的导数仍是贝塞尔曲线，控制点取一阶差分。"""
        s = np.asarray(s, dtype=float)
        n = len(self.control_points) - 1
        d_points = n * np.diff(self.control_points, axis=0)
        return self._evaluate(d_points, s) / swing_duration

    def acceleration(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        """二阶导数：控制点取二阶差分。"""
        s = np.asarray(s, dtype=float)
        n = len(self.control_points) - 1
        dd_points = n * (n - 1) * np.diff(self.control_points, n=2, axis=0)
        return self._evaluate(dd_points, s) / swing_duration**2


class QuinticSwing(SwingTrajectory):
    """分段五次多项式：两端与顶点的位置、速度、加速度全部指定。

    离地 -> 顶点、顶点 -> 落地各用一段五次多项式，边界条件为

    * 两端：速度为零、加速度为零；
    * 顶点：竖直速度为零（真正的最高点）。

    五次多项式恰好有 6 个自由度，对应两端各 3 个约束（位置、速度、加速度），
    是能同时满足这些条件的**最低阶**多项式。

    **加速度在两端连续且为零**，因此关节力矩没有台阶 —— 这是它相对摆线的
    优势。代价是峰值加速度更大（要在更短时间内完成同样的位移）。
    """

    @staticmethod
    def _smooth(u):
        """五次插值基函数：0->1，两端一、二阶导均为零。"""
        return 6 * u**5 - 15 * u**4 + 10 * u**3

    @staticmethod
    def _d_smooth(u):
        return 30 * u**4 - 60 * u**3 + 30 * u**2

    @staticmethod
    def _dd_smooth(u):
        return 120 * u**3 - 180 * u**2 + 60 * u

    def _vertical_bump(self, s, order=0):
        """竖直方向的 0 -> h -> 0 隆起及其导数（对 s 求导）。"""
        u_up = np.clip(2 * s, 0.0, 1.0)
        u_down = np.clip(2 * s - 1.0, 0.0, 1.0)
        up = s <= 0.5
        if order == 0:
            return np.where(up, self.height * self._smooth(u_up),
                            self.height * (1.0 - self._smooth(u_down)))
        if order == 1:
            return np.where(up, 2 * self.height * self._d_smooth(u_up),
                            -2 * self.height * self._d_smooth(u_down))
        return np.where(up, 4 * self.height * self._dd_smooth(u_up),
                        -4 * self.height * self._dd_smooth(u_down))

    def position(self, s: float | np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = self.liftoff + self._smooth(s)[..., None] * (self.touchdown - self.liftoff)
        base_z = self.liftoff[2] + s * (self.touchdown[2] - self.liftoff[2])
        out[..., 2] = base_z + self._vertical_bump(s, 0)
        return out

    def velocity(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = self._d_smooth(s)[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] = (self.touchdown[2] - self.liftoff[2]) + self._vertical_bump(s, 1)
        return out / swing_duration

    def acceleration(self, s: float | np.ndarray, swing_duration: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = self._dd_smooth(s)[..., None] * (self.touchdown - self.liftoff)
        out[..., 2] = self._vertical_bump(s, 2)
        return out / swing_duration**2


#: 可供选择的轨迹类型，便于统一对比。
SWING_TRAJECTORIES = {
    "sine": SineHeightSwing,
    "cycloid": CycloidSwing,
    "bezier": BezierSwing,
    "quintic": QuinticSwing,
}
