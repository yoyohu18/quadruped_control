"""支撑多边形与静态稳定裕度：步态选择背后的物理。

**静态稳定**的定义：把质心竖直投影到地面，如果落在支撑多边形内部，
机器人即使完全静止也不会翻倒。

由此立刻推出一个结论：**至少需要三条腿触地**。两条腿的"多边形"退化成
一条线段，质心投影几乎不可能恰好落在线上，所以 trot、pace、bound 全都是
**静态不稳定**的 —— 它们靠运动本身维持平衡，就像人跑步时任何一个瞬间
都是要摔倒的状态。

这就是四条腿均匀分布时 :math:`D \\ge 0.75` 成为分水岭的原因：占空比
低于 0.75，就无法保证任意时刻都有三条腿在地上。

.. note::
   静态稳定裕度只对慢速步态有意义。trot 起来之后它恒为负，此时该看的是
   **动态**稳定性指标 —— 捕获点、零力矩点（ZMP）、发散分量（DCM）。
   那些是里程碑 6 的内容。
"""

from __future__ import annotations

import numpy as np

__all__ = ["convex_hull_2d", "support_polygon", "static_stability_margin", "point_in_polygon"]


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """二维凸包，Andrew 单调链算法，逆时针输出。

    这里不用 ``scipy.spatial.ConvexHull``，因为它在退化输入（少于 3 点、
    共线）时会抛异常，而支撑多边形**经常**是退化的 —— trot 只有两个
    接触点。手写版本对退化情况优雅降级。

    Args:
        points: 形状 (n, 2) 的点集。

    Returns:
        凸包顶点，逆时针顺序，形状 (m, 2)，``m <= n``。
        点数少于 3 或共线时，原样返回去重后的极值点。
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return hull if len(hull) >= 3 else pts


def support_polygon(foot_positions: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """支撑多边形：所有支撑脚在水平面内投影的凸包。

    Args:
        foot_positions: 四只脚的世界位置，形状 (4, 3)。
        contact: 接触标志，形状 (4,)。

    Returns:
        凸包顶点，形状 (m, 2)。没有支撑腿时返回空数组 (0, 2)。
    """
    feet = np.asarray(foot_positions, dtype=float).reshape(4, 3)
    contact = np.asarray(contact, dtype=bool)
    stance = feet[contact][:, :2]
    if len(stance) == 0:
        return np.zeros((0, 2))
    return convex_hull_2d(stance)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """点是否落在凸多边形内部（含边界）。多边形须为逆时针顺序。"""
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return False
    p = np.asarray(point, dtype=float).reshape(2)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if np.cross(b - a, p - a) < -1e-12:
            return False
    return True


def static_stability_margin(
    com_xy: np.ndarray, foot_positions: np.ndarray, contact: np.ndarray
) -> float:
    """静态稳定裕度：质心投影到支撑多边形边界的带符号距离，单位米。

    正值表示质心在多边形内部，数值就是"还能被推多远才翻倒"。负值表示
    已经在外面，静态意义上正在翻倒 —— 这不代表机器人一定会摔，只说明
    它必须靠动态效应（惯性、摆动腿反作用）维持平衡。

    Args:
        com_xy: 质心的水平投影，形状 (2,)。
        foot_positions: 四只脚的世界位置，形状 (4, 3)。
        contact: 接触标志，形状 (4,)。

    Returns:
        带符号距离，米。支撑腿少于 3 条时恒为非正 —— 这是几何决定的，
        不是数值问题。
    """
    poly = support_polygon(foot_positions, contact)
    p = np.asarray(com_xy, dtype=float).reshape(2)

    if len(poly) == 0:
        return -np.inf  # 腾空相，无支撑可言
    if len(poly) == 1:
        return -float(np.linalg.norm(p - poly[0]))
    if len(poly) == 2:
        # 退化成线段：裕度是到该线段的距离取负（永远静态不稳定）
        return -_distance_to_segment(p, poly[0], poly[1])

    inside = point_in_polygon(p, poly)
    dists = [
        _distance_to_segment(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))
    ]
    d = float(min(dists))
    return d if inside else -d


def _distance_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点到线段的欧氏距离。"""
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-18:
        return float(np.linalg.norm(p - a))
    s = float(np.clip((p - a) @ ab / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))
