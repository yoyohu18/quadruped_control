"""四足单腿（3 自由度：侧摆 / 髋俯仰 / 膝）的闭式运动学。

腿在自己的**髋系（hip frame）**里描述：原点落在侧摆关节轴上，坐标轴与
机器人躯干系平行（x 向前，y 向左，z 向上）。

关节约定（与宇树 Go2 的 URDF 一致）：

    q0  侧摆 abduction（URDF 里叫 "hip"）  绕 +x
    q1  髋俯仰 hip pitch（URDF 里叫 "thigh"）绕 +y
    q2  膝 knee（URDF 里叫 "calf"）        绕 +y

连杆参数：

    l0  侧摆轴到髋俯仰轴沿 y 的**带符号**偏置（左腿为正，右腿为负）
    l1  大腿长度
    l2  小腿长度

本模块只用纯 NumPy：不碰 Pinocchio，不碰 URDF，不碰 ROS。这是刻意的 ——
这段代码是将来要在真机上以 1 kHz 运行的那条路径，同时也是用来校验
``kinematics.robot_model`` 里那套基于 Pinocchio 的通用代码的参考实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LegGeometry", "forward_kinematics", "leg_jacobian", "inverse_kinematics"]


@dataclass(frozen=True)
class LegGeometry:
    """单腿的连杆长度，在其髋系下表达。

    Attributes:
        l0: 沿 +y 的带符号侧摆偏置（正表示左腿）。
        l1: 大腿长度（髋俯仰轴到膝轴）。
        l2: 小腿长度（膝轴到足端点）。
    """

    l0: float
    l1: float
    l2: float

    @property
    def reach_max(self) -> float:
        """足端能达到的、距髋俯仰轴的最大距离。"""
        return self.l1 + self.l2

    @property
    def reach_min(self) -> float:
        """足端能达到的、距髋俯仰轴的最小距离。"""
        return abs(self.l1 - self.l2)


def forward_kinematics(q: np.ndarray, geom: LegGeometry) -> np.ndarray:
    """髋系下的足端位置。

    Args:
        q: 关节角 ``[q_abad, q_hip, q_knee]``，单位弧度，形状 (3,)。
        geom: 腿的连杆长度。

    Returns:
        髋系下的足端位置 ``[x, y, z]``，形状 (3,)。
    """
    q = np.asarray(q, dtype=float).reshape(3)
    l0, l1, l2 = geom.l0, geom.l1, geom.l2

    s0, c0 = np.sin(q[0]), np.cos(q[0])
    s1, c1 = np.sin(q[1]), np.cos(q[1])
    s12, c12 = np.sin(q[1] + q[2]), np.cos(q[1] + q[2])

    # 先解矢状面，再把整个平面绕 +x 转过 q0。
    x = -l1 * s1 - l2 * s12  # 前向偏移（不受侧摆影响）
    leg_len = l1 * c1 + l2 * c12  # 矢状面内的向下伸展量

    return np.array(
        [
            x,
            l0 * c0 + leg_len * s0,
            l0 * s0 - leg_len * c0,
        ]
    )


def leg_jacobian(q: np.ndarray, geom: LegGeometry) -> np.ndarray:
    """髋系下的足端解析雅可比 ``d(足端位置) / dq``。

    Args:
        q: 关节角 ``[q_abad, q_hip, q_knee]``，单位弧度，形状 (3,)。
        geom: 腿的连杆长度。

    Returns:
        形状 (3, 3) 的雅可比；第 ``i`` 列是关节 ``i`` 以单位速度运动时
        产生的足端速度。
    """
    q = np.asarray(q, dtype=float).reshape(3)
    l0, l1, l2 = geom.l0, geom.l1, geom.l2

    s0, c0 = np.sin(q[0]), np.cos(q[0])
    s1, c1 = np.sin(q[1]), np.cos(q[1])
    s12, c12 = np.sin(q[1] + q[2]), np.cos(q[1] + q[2])

    x = -l1 * s1 - l2 * s12
    leg_len = l1 * c1 + l2 * c12
    y = l0 * c0 + leg_len * s0
    z = l0 * s0 - leg_len * c0

    # 第 0 列：侧摆只是让足端绕过髋原点的 +x 轴做纯旋转，
    # 所以这一列恰好就是 x_hat × p = [0, -z, y]。
    return np.array(
        [
            [0.0, -leg_len, -l2 * c12],
            [-z, s0 * x, -s0 * l2 * s12],
            [y, -c0 * x, c0 * l2 * s12],
        ]
    )


def inverse_kinematics(
    p: np.ndarray,
    geom: LegGeometry,
    knee_backward: bool = True,
    clamp: bool = False,
) -> np.ndarray:
    """闭式逆运动学：足端位置 -> 关节角。

    一条 3 自由度腿对给定足端位置最多有四个解。本函数**永远**返回其中
    对运动控制真正有用的那一个：

    * **膝解支** —— 由 ``knee_backward`` 选定；
    * **腿向下解支** —— 面内伸展量 ``l1*cos(q1) + l2*cos(q1+q2)`` 取非负，
      即足端落在侧摆轴**下方**，而不是折叠到上方。

    因此，把 :func:`forward_kinematics` 的输出喂回本函数，只有当输入构型
    本身就位于上述解支上时才能还原出原来的关节角；否则返回的是镜像解，
    而该解的正运动学**仍然精确等于**你请求的位置。

    Args:
        p: 髋系下期望的足端位置 ``[x, y, z]``，形状 (3,)。
        geom: 腿的连杆长度。
        knee_backward: 解支选择。``True`` 给出 ``q_knee <= 0``，这是
            Go2 膝关节限位内唯一可行的解支。
        clamp: 若为 ``True``，对不可达目标静默投影到工作空间边界，
            而不是抛异常。这在控制回路里很有用 —— 一个过于激进的落脚点
            规划不应该把机器人搞崩。

    Returns:
        关节角 ``[q_abad, q_hip, q_knee]``，单位弧度，形状 (3,)。

    Raises:
        ValueError: 目标超出可达工作空间且 ``clamp`` 为 ``False``。
    """
    p = np.asarray(p, dtype=float).reshape(3)
    x, y, z = p
    l0, l1, l2 = geom.l0, geom.l1, geom.l2

    # --- 第一步：侧摆角 ------------------------------------------------------
    # y 和 z 只依赖 q0 和面内伸展量 leg_len：
    #     y = l0*cos(q0) + leg_len*sin(q0)
    #     z = l0*sin(q0) - leg_len*cos(q0)
    # 于是 y^2 + z^2 = l0^2 + leg_len^2，直接定出 leg_len。
    radial_sq = y * y + z * z - l0 * l0
    if radial_sq < 0.0:
        if not clamp:
            raise ValueError(
                f"目标 {p} 落在侧摆圆柱内部 "
                f"(sqrt(y^2+z^2)={np.hypot(y, z):.4f} < |l0|={abs(l0):.4f})，"
                "无实数解 (unreachable: inside abduction cylinder)。"
            )
        radial_sq = 0.0
    leg_len = np.sqrt(radial_sq)

    # 记 beta = atan2(leg_len, l0)，则 [y, z] = A * [cos(q0 - beta), sin(q0 - beta)]。
    q0 = np.arctan2(z, y) + np.arctan2(leg_len, l0)

    # --- 第二步：膝关节角 ----------------------------------------------------
    # 在矢状面内足端相对髋俯仰轴位于 (x, -leg_len)，
    # 对这条 2R 链用余弦定理即得 q2。
    d_sq = x * x + leg_len * leg_len
    cos_knee = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= cos_knee <= 1.0:
        if not clamp:
            raise ValueError(
                f"目标 {p} 不可达 (unreachable)：到髋俯仰轴的距离为 "
                f"{np.sqrt(d_sq):.4f} m，超出 "
                f"[{geom.reach_min:.4f}, {geom.reach_max:.4f}] m。"
            )
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
    q2 = -np.arccos(cos_knee) if knee_backward else np.arccos(cos_knee)

    # --- 第三步：髋俯仰角 ----------------------------------------------------
    # 把两连杆折叠成一根等效连杆 (a, b)，
    # 再从足端方向里减去这根等效连杆的固定相位。
    a = l1 + l2 * np.cos(q2)
    b = l2 * np.sin(q2)
    q1 = np.arctan2(-x, leg_len) - np.arctan2(b, a)

    return np.array([q0, q1, q2])
