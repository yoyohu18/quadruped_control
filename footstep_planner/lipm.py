"""线性倒立摆模型（LIPM）与捕获点：动态平衡的数学基础。

里程碑 4 算出一个尴尬的结论：**trot 的静态稳定裕度恒为 −0.84 cm**。
按静态标准，机器人一直在翻倒。可它明明能走。本模块解释为什么。

答案是：**四足根本不追求静态稳定，它追求的是"总能把脚迈到该去的地方"。**
就像人跑步时，任何一个瞬间都处在要摔倒的状态 —— 但只要下一步落对位置，
就永远摔不下去。

## 模型

把机器人简化成一个质心 + 一根无质量的腿撑在地面接触点 :math:`p` 上。
假设质心高度 :math:`h` 恒定（这就是"线性"的来源），则水平方向：

.. math::  \\ddot{x} = \\omega^2 (x - p), \\qquad \\omega = \\sqrt{g/h}

这是一个**不稳定**的一阶不稳定极点系统：质心离支撑点越远，加速度越大，
离得更远。:math:`\\omega` 是这台机器"摔倒得多快"的固有频率。

Go2 站立高度 0.30 m 时 :math:`\\omega = 5.72` rad/s，时间常数
:math:`1/\\omega = 0.175` s —— **和 trot 的支撑相时长 0.2 s 是同一量级**。
这个巧合不是巧合：步态周期必须比"摔倒时间常数"更快，否则来不及救。

## 捕获点

上面的二阶系统可以解耦成两个一阶系统。定义**发散分量**（也叫捕获点、DCM）

.. math::  \\xi = x + \\frac{\\dot{x}}{\\omega}

则

.. math::  \\dot{\\xi} = \\omega(\\xi - p), \\qquad \\dot{x} = -\\omega(x - \\xi)

第二式是**稳定**的：质心总在追捕获点。第一式是**不稳定**的：捕获点会
从支撑点逃走。所以全部的控制问题被压缩成一句话：

    **把脚落在捕获点上，捕获点就不动了，机器人随之停下。**

这就是"脚该落在哪"的理论答案，也是 Raibert 启发式背后真正的物理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LIPMParams", "capture_point", "lipm_step", "simulate_lipm", "time_to_boundary"]


@dataclass(frozen=True)
class LIPMParams:
    """线性倒立摆参数。

    Attributes:
        height: 质心高度，米。它决定了整台机器"摔得多快"。
        gravity: 重力加速度，m/s^2。
    """

    height: float = 0.30
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.height <= 0.0:
            raise ValueError(f"质心高度必须为正，收到 {self.height}")

    @property
    def omega(self) -> float:
        """固有频率 :math:`\\omega = \\sqrt{g/h}`，rad/s。"""
        return float(np.sqrt(self.gravity / self.height))

    @property
    def time_constant(self) -> float:
        """摔倒的时间常数 :math:`1/\\omega`，秒。

        **它是选择步态周期的物理依据**：支撑相时长必须与之同量级或更短，
        否则一步没迈完人已经倒了。
        """
        return 1.0 / self.omega


def capture_point(com: np.ndarray, com_velocity: np.ndarray, params: LIPMParams) -> np.ndarray:
    """捕获点（发散分量）:math:`\\xi = x + \\dot{x}/\\omega`。

    把脚落在这里，机器人会**渐近停下**。落在它前面则减速甚至后退，
    落在它后面则继续加速前冲。

    Args:
        com: 质心水平位置，形状 (2,) 或 (3,)（只用前两个分量）。
        com_velocity: 质心水平速度，同上。
        params: 倒立摆参数。

    Returns:
        捕获点的水平位置，形状 (2,)。
    """
    x = np.asarray(com, dtype=float).reshape(-1)[:2]
    v = np.asarray(com_velocity, dtype=float).reshape(-1)[:2]
    return x + v / params.omega


def lipm_step(
    com: np.ndarray,
    com_velocity: np.ndarray,
    foot: np.ndarray,
    dt: float,
    params: LIPMParams,
) -> tuple[np.ndarray, np.ndarray]:
    """用**解析解**推进线性倒立摆一步。

    :math:`\\ddot{x} = \\omega^2(x-p)` 是线性常系数方程，有闭式解：

    .. math::

        x(t) = p + (x_0 - p)\\cosh(\\omega t) + \\frac{\\dot{x}_0}{\\omega}\\sinh(\\omega t)

    用解析解而非数值积分，是因为这个系统**本身就是发散的**：欧拉法的
    局部误差会被指数放大，几步之后就分不清"物理发散"和"数值发散"。

    Args:
        com: 质心水平位置，形状 (2,)。
        com_velocity: 质心水平速度，形状 (2,)。
        foot: 支撑点水平位置，形状 (2,)。
        dt: 时间步长，秒。
        params: 倒立摆参数。

    Returns:
        ``(com, com_velocity)`` 推进后的状态。
    """
    x = np.asarray(com, dtype=float).reshape(-1)[:2]
    v = np.asarray(com_velocity, dtype=float).reshape(-1)[:2]
    p = np.asarray(foot, dtype=float).reshape(-1)[:2]
    w = params.omega
    ch, sh = np.cosh(w * dt), np.sinh(w * dt)

    x_new = p + (x - p) * ch + (v / w) * sh
    v_new = (x - p) * w * sh + v * ch
    return x_new, v_new


def simulate_lipm(
    com0: np.ndarray,
    velocity0: np.ndarray,
    foot_sequence: np.ndarray,
    step_duration: float,
    params: LIPMParams,
    substeps: int = 40,
) -> dict:
    """按给定的落脚点序列推进倒立摆，返回整条轨迹。

    Args:
        com0: 初始质心水平位置，形状 (2,)。
        velocity0: 初始质心水平速度，形状 (2,)。
        foot_sequence: 每一步的支撑点，形状 (n_steps, 2)。
        step_duration: 每一步的支撑时长，秒。
        params: 倒立摆参数。
        substeps: 每步内部的采样数，只影响输出分辨率，不影响精度
            （解析解逐段精确）。

    Returns:
        含 ``t``、``com``、``velocity``、``capture_point``、``foot`` 的字典。
    """
    x = np.asarray(com0, dtype=float).reshape(-1)[:2].copy()
    v = np.asarray(velocity0, dtype=float).reshape(-1)[:2].copy()
    feet = np.asarray(foot_sequence, dtype=float).reshape(-1, 2)

    dt = step_duration / substeps
    times, coms, vels, caps, foots = [], [], [], [], []
    t = 0.0
    for p in feet:
        for _ in range(substeps):
            times.append(t)
            coms.append(x.copy())
            vels.append(v.copy())
            caps.append(capture_point(x, v, params))
            foots.append(p.copy())
            x, v = lipm_step(x, v, p, dt, params)
            t += dt

    return {
        "t": np.array(times),
        "com": np.array(coms),
        "velocity": np.array(vels),
        "capture_point": np.array(caps),
        "foot": np.array(foots),
    }


def time_to_boundary(
    com: np.ndarray, com_velocity: np.ndarray, boundary: float, params: LIPMParams
) -> float:
    """捕获点跑到给定边界所需的时间，秒。

    这是"还剩多久必须迈步"的量化指标。边界通常取工作空间或支撑多边形的
    边缘 —— 捕获点一旦跑出可达范围，就再也追不回来了。

    Args:
        com: 质心水平位置（这里只处理一维，取第一个分量）。
        com_velocity: 质心水平速度。
        boundary: 边界位置（相对同一原点）。
        params: 倒立摆参数。

    Returns:
        剩余时间，秒。捕获点已在边界之外时返回 0；永远到不了则返回 inf。
    """
    xi = float(capture_point(com, com_velocity, params)[0])
    p = 0.0  # 以当前支撑点为原点
    w = params.omega
    if abs(xi - p) < 1e-12:
        return np.inf  # 恰在支撑点上，不会发散
    ratio = (boundary - p) / (xi - p)
    if ratio <= 0.0:
        return np.inf  # 朝反方向发散，这个边界够不着
    if ratio <= 1.0:
        return 0.0  # 已经越界
    return float(np.log(ratio) / w)
