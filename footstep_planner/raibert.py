"""落脚点策略：Raibert 启发式与基于捕获点的放置。

## Raibert 启发式（1986）

Raibert 在《Legged Robots That Balance》里给出的公式，四十年后仍然是
工业界的默认做法：

.. math::

    p = p_{\\text{hip}} + \\frac{T_{\\text{st}}}{2}\\,v
        + k\\,(v - v_{\\text{cmd}})

三项各司其职：

============================  ==================================================
项                            作用
============================  ==================================================
:math:`p_{\\text{hip}}`        髋关节在地面的投影 —— 站着不动时脚就该在这儿
:math:`\\frac{T_{st}}{2} v`    **前馈**：让支撑相相对髋部前后对称
:math:`k(v - v_{cmd})`         **反馈**：速度快了就多迈一点，把速度压回来
============================  ==================================================

中间那项最值得琢磨。支撑相时长 :math:`T_{st}` 内躯干前进 :math:`T_{st} v`，
把落脚点放在前方半个身位处，脚就会在支撑相的中点恰好位于髋下 —— 前后
受力对称，不产生净的俯仰力矩。

## 它和捕获点是什么关系

捕获点给出的答案是 :math:`p = x + v/\\omega`（见 :mod:`~footstep_planner.lipm`）。
Raibert 的前馈项是 :math:`T_{st}v/2`。两者形式完全一样，只是系数不同：

.. math::  \\frac{T_{\\text{st}}}{2} \\quad \\text{vs} \\quad \\frac{1}{\\omega}

Go2 的数字：:math:`T_{st}/2 = 0.10` s，:math:`1/\\omega = 0.175` s。
**同一个量级，差 1.75 倍。**

所以 Raibert 启发式**不是拍脑袋想出来的经验公式，它是捕获点的一个欠调
近似**。系数偏小意味着它"迈得不够远"，单靠前馈会残留速度误差 —— 这正是
第三项速度反馈存在的理由。

理解了这一层，就知道 :math:`k` 该怎么调：它补的是前馈项欠掉的那部分，
理论上的最优值在 :math:`1/\\omega - T_{st}/2` 附近。
"""

from __future__ import annotations

import numpy as np

from .lipm import LIPMParams, capture_point

__all__ = [
    "raibert_footstep",
    "capture_point_footstep",
    "exact_stride_coefficient",
    "exact_footstep",
    "optimal_feedback_gain",
    "deadbeat_feedback_gain",
    "steady_state_velocity_ratio",
]


def raibert_footstep(
    hip_position: np.ndarray,
    velocity: np.ndarray,
    velocity_command: np.ndarray,
    stance_duration: float,
    feedback_gain: float = 0.03,
) -> np.ndarray:
    """Raibert 启发式落脚点。

    Args:
        hip_position: 髋关节在地面的投影位置，形状 (2,)。
        velocity: 当前躯干水平速度，形状 (2,)。来自状态估计器（M3）。
        velocity_command: 期望水平速度，形状 (2,)。
        stance_duration: 支撑相时长，秒。来自步态调度器（M4）。
        feedback_gain: 速度反馈增益 :math:`k`，秒。

    Returns:
        落脚点的水平位置，形状 (2,)。
    """
    hip = np.asarray(hip_position, dtype=float).reshape(-1)[:2]
    v = np.asarray(velocity, dtype=float).reshape(-1)[:2]
    v_cmd = np.asarray(velocity_command, dtype=float).reshape(-1)[:2]
    return hip + 0.5 * stance_duration * v + feedback_gain * (v - v_cmd)


def capture_point_footstep(
    hip_position: np.ndarray,
    com: np.ndarray,
    velocity: np.ndarray,
    velocity_command: np.ndarray,
    params: LIPMParams,
    stance_duration: float,
) -> np.ndarray:
    """基于捕获点的落脚点。

    纯捕获点 :math:`x + v/\\omega` 会让机器人**停下来**。要维持指令速度
    :math:`v_{\\text{cmd}}`，就得在捕获点的基础上把"想保留的那部分速度"
    再往前推一个支撑相：

    .. math::  p = \\xi(x, v - v_{\\text{cmd}}) + p_{\\text{hip}} - x
                   + \\frac{T_{\\text{st}}}{2} v_{\\text{cmd}}

    直观理解：先用捕获点把**速度误差**吃掉，再用前馈维持指令速度。

    Args:
        hip_position: 髋关节地面投影，形状 (2,)。
        com: 质心水平位置，形状 (2,)。
        velocity: 当前水平速度，形状 (2,)。
        velocity_command: 期望水平速度，形状 (2,)。
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。

    Returns:
        落脚点的水平位置，形状 (2,)。
    """
    hip = np.asarray(hip_position, dtype=float).reshape(-1)[:2]
    x = np.asarray(com, dtype=float).reshape(-1)[:2]
    v = np.asarray(velocity, dtype=float).reshape(-1)[:2]
    v_cmd = np.asarray(velocity_command, dtype=float).reshape(-1)[:2]

    # 相对髋部：先吃掉速度误差（捕获点），再维持指令速度（前馈）
    error_capture = capture_point(np.zeros(2), v - v_cmd, params)
    return hip + error_capture + 0.5 * stance_duration * v_cmd


def optimal_feedback_gain(params: LIPMParams, stance_duration: float) -> float:
    """Raibert 反馈增益的理论参考值 :math:`1/\\omega - T_{st}/2`。

    它把前馈项欠掉的那部分补齐，使 Raibert 启发式在速度误差方向上与
    捕获点一致。实机上通常还要再打个折扣（0.5~0.8），因为模型忽略了
    腿的质量、接触柔性与执行器带宽。

    Args:
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。

    Returns:
        增益参考值，秒。
    """
    return float(params.time_constant - 0.5 * stance_duration)


def exact_stride_coefficient(params: LIPMParams, stance_duration: float) -> float:
    """倒立摆极限环的**精确**半步系数 :math:`\\tanh(\\omega T_{st}/2)/\\omega`。

    它统一了 Raibert 启发式与捕获点：前者是 :math:`\\omega T \\to 0` 的极限，
    后者是 :math:`\\omega T \\to \\infty` 的极限。用它做前馈可以**精确**跟踪
    指令速度，而两个启发式都会留下稳态误差。

    Args:
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。

    Returns:
        系数，单位秒。
    """
    w = params.omega
    return float(np.tanh(0.5 * w * stance_duration) / w)


def exact_footstep(
    hip_position: np.ndarray,
    velocity: np.ndarray,
    velocity_command: np.ndarray,
    params: LIPMParams,
    stance_duration: float,
    feedback_gain: float | None = None,
) -> np.ndarray:
    """用精确极限环系数的落脚点，形式与 Raibert 相同但系数取精确值。

    .. math::  p = p_{\\text{hip}} + \\frac{\\tanh(\\omega T_{st}/2)}{\\omega}\\,v_{\\text{cmd}}
                   + k\\,(v - v_{\\text{cmd}})

    注意前馈项用的是 **指令速度** 而不是当前速度 —— 这样前馈只负责"维持
    指令"，反馈只负责"消除误差"，两者职责分离，稳态误差为零。

    Args:
        hip_position: 髋关节地面投影，形状 (2,)。
        velocity: 当前水平速度，形状 (2,)。
        velocity_command: 期望水平速度，形状 (2,)。
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。
        feedback_gain: 速度反馈增益；缺省取 :func:`deadbeat_feedback_gain`。

            .. warning::
               **不能用** :func:`optimal_feedback_gain`。前馈项在这里用的是
               指令速度（常量），不再随当前速度变化，因此**不贡献闭环稳定性**；
               全部稳定裕度都压在 :math:`k` 上，稳定条件变成 :math:`k > c^*`。
               而 :math:`1/\\omega - T_{st}/2 = 0.0749 < c^{*} = 0.0904`，用它会发散。

    Returns:
        落脚点的水平位置，形状 (2,)。
    """
    hip = np.asarray(hip_position, dtype=float).reshape(-1)[:2]
    v = np.asarray(velocity, dtype=float).reshape(-1)[:2]
    v_cmd = np.asarray(velocity_command, dtype=float).reshape(-1)[:2]
    k = deadbeat_feedback_gain(params, stance_duration) if feedback_gain is None else feedback_gain
    c = exact_stride_coefficient(params, stance_duration)
    return hip + c * v_cmd + k * (v - v_cmd)


def steady_state_velocity_ratio(
    feedforward_coefficient: float,
    feedback_gain: float,
    params: LIPMParams,
    stance_duration: float,
) -> float:
    """闭环稳态速度与指令速度之比，用于**预测**跟踪误差。

    针对 Raibert 形式 :math:`p = c\\,v + k\\,(v - v_{cmd})`（前馈用当前速度），
    令一步之后速度不变，解得

    .. math::  \\frac{v_{ss}}{v_{cmd}} = \\frac{k}{c + k - c^{*}},
               \\qquad c^{*} = \\frac{\\tanh(\\omega T_{st}/2)}{\\omega}

    取 Go2 trot 的数值（:math:`c = 0.1`, :math:`k = 0.0749`）得 0.8859 ——
    与仿真吻合到小数点后四位。**这说明 Raibert 启发式的稳态误差是结构性的，
    不是调参问题。**

    Args:
        feedforward_coefficient: 前馈系数 :math:`c`，秒。
        feedback_gain: 反馈增益 :math:`k`，秒。
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。

    Returns:
        :math:`v_{ss}/v_{cmd}`。分母为零（闭环无稳态解）时返回 inf。
    """
    c_star = exact_stride_coefficient(params, stance_duration)
    denom = feedforward_coefficient + feedback_gain - c_star
    if abs(denom) < 1e-15:
        return float("inf")
    return float(feedback_gain / denom)


def deadbeat_feedback_gain(params: LIPMParams, stance_duration: float) -> float:
    """一步收敛（死拍）的反馈增益 :math:`\\coth(\\omega T_{st})/\\omega`。

    把闭环特征值直接打到零：

    .. math::  v_{n+1} = \\big[\\cosh\\omega T - k\\,\\omega\\sinh\\omega T\\big] v_n
                          + (\\cdots) v_{\\text{cmd}}

    令方括号为零即得 :math:`k = \\coth(\\omega T)/\\omega`。Go2 trot 下是
    0.2144 s，实测**一步**就把速度从 0 拉到指令值。

    实机上不要直接用这个值：死拍对模型误差极其敏感，而 LIPM 忽略了腿的
    质量、接触柔性、执行器带宽。工程上取 0.5~0.7 倍，牺牲收敛速度换鲁棒性。

    Args:
        params: 倒立摆参数。
        stance_duration: 支撑相时长，秒。

    Returns:
        增益，单位秒。
    """
    w = params.omega
    wT = w * stance_duration
    return float(np.cosh(wT) / (w * np.sinh(wT)))
