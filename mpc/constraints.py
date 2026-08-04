"""接触力约束：把物理限制写成线性不等式。

MPC 之所以能是**凸**的，不只是因为动力学对力线性，还因为**接触的物理
限制也能写成线性不等式**。这一节把三条物理事实翻译成矩阵：

1. **地面只能推不能拉**：:math:`f_z \\ge 0`；
2. **摩擦有限**：:math:`\\|f_{xy}\\| \\le \\mu f_z`，这是一个**二次锥**；
3. **电机有上限**：:math:`f_z \\le f_{\\max}`。

第二条是唯一的麻烦：二次锥约束会把问题变成 SOCP（二阶锥规划）。虽然
SOCP 仍是凸的，但求解器比 QP 慢一个量级，1 kHz 下不划算。

**工业界的做法是把圆锥换成金字塔**：

.. math::  |f_x| \\le \\mu f_z, \\qquad |f_y| \\le \\mu f_z

四个线性不等式，问题退回 QP。

## 一个很多人搞反的方向

这个金字塔是**外接**于圆锥的，不是内接。正方形的半宽是 :math:`\\mu f_z`，
而内切圆半径也是 :math:`\\mu f_z` —— 所以正方形**包住**了圆。沿对角方向，
金字塔允许的切向力可达

.. math::  \\sqrt{(\\mu f_z)^2 + (\\mu f_z)^2} = \\sqrt{2}\\,\\mu f_z

**比真实摩擦极限大 41.4%。** 也就是说，这个常见写法不是保守而是**乐观**的：
求解器可以合法地开出一组会打滑的力。

真正保守（内接）的写法要把系数缩到 :math:`\\mu/\\sqrt 2`：

.. math::  |f_x| \\le \\frac{\\mu}{\\sqrt 2} f_z, \\qquad
           |f_y| \\le \\frac{\\mu}{\\sqrt 2} f_z

工程上两种做法都有人用：外接版配一个打过折的 :math:`\\mu`（比如实测 0.8
只填 0.5），内接版直接用实测值。**本模块默认内接**，因为"约束写出来就是
物理上成立的"比"靠调参数偷偷补回来"更容易验证 —— 测试里可以直接断言
求解结果落在**真实圆锥**内。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FrictionConstraints", "friction_pyramid_matrix", "build_force_constraints"]


@dataclass(frozen=True)
class FrictionConstraints:
    """接触力的物理限制。

    Attributes:
        mu: 摩擦系数。橡胶足端对混凝土约 0.6~0.8，冰面 0.1 以下。
            MPC 里通常取实际值的 70%~80% 留裕量。
        f_min: 支撑腿的最小法向力，N。取一个小正数而非零，可以避免支撑腿
            "若即若离"导致接触状态抖动。
        f_max: 单腿最大法向力，N。由电机力矩上限经雅可比换算而来。
        inscribed: 金字塔取内接（``True``，保守，默认）还是外接
            （``False``，即常见的 :math:`|f_x|\\le\\mu f_z` 写法，乐观 41%）。
    """

    mu: float = 0.6
    f_min: float = 5.0
    f_max: float = 250.0
    inscribed: bool = True

    def __post_init__(self) -> None:
        if self.mu <= 0.0:
            raise ValueError(f"摩擦系数必须为正，收到 {self.mu}")
        if not 0.0 <= self.f_min < self.f_max:
            raise ValueError(f"需要 0 <= f_min < f_max，收到 {self.f_min}, {self.f_max}")

    @property
    def pyramid_mu(self) -> float:
        """写进线性约束里的实际系数。

        内接时是 :math:`\\mu/\\sqrt 2`，外接时就是 :math:`\\mu` 本身。
        """
        return float(self.mu / np.sqrt(2.0)) if self.inscribed else float(self.mu)

    @property
    def worst_case_utilisation(self) -> float:
        """沿对角方向，金字塔边界相对真实圆锥边界的比例。

        内接：:math:`1/\\sqrt 2 \\approx 0.707`，即最坏方向上只用到 70.7%
        的摩擦，是**安全**的一侧。
        外接：:math:`\\sqrt 2 \\approx 1.414`，即最坏方向上超出真实极限
        41.4%，是**危险**的一侧。
        """
        return float(1.0 / np.sqrt(2.0)) if self.inscribed else float(np.sqrt(2.0))


def friction_pyramid_matrix(mu: float) -> np.ndarray:
    """单只脚的摩擦金字塔约束矩阵，形状 (4, 3)，满足 ``C f <= 0``。

    四行分别是

    .. math::

        f_x - \\mu f_z \\le 0, \\quad -f_x - \\mu f_z \\le 0, \\quad
        f_y - \\mu f_z \\le 0, \\quad -f_y - \\mu f_z \\le 0

    Args:
        mu: 摩擦系数。

    Returns:
        形状 (4, 3) 的矩阵。
    """
    return np.array(
        [
            [1.0, 0.0, -mu],
            [-1.0, 0.0, -mu],
            [0.0, 1.0, -mu],
            [0.0, -1.0, -mu],
        ]
    )


def build_force_constraints(
    contact: np.ndarray, constraints: FrictionConstraints
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造某一个预测步上全部四条腿的力约束 ``lb <= C f <= ub``。

    决策变量是 12 维（四条腿各 3 个力分量），**摆动腿的力被硬性钉为零**
    而不是从决策变量里删掉 —— 这样 QP 的维度在整个预测时域内保持不变，
    矩阵结构固定，可以预分配、可以热启动。这是实时实现的关键技巧。

    Args:
        contact: 该预测步上四条腿的接触标志，形状 (4,)。
        constraints: 物理限制。

    Returns:
        ``(C, lb, ub)``，形状分别为 (20, 12)、(20,)、(20,)。
        每条腿贡献 5 行：4 行摩擦金字塔 + 1 行法向力上下界。
    """
    contact = np.asarray(contact, dtype=bool).reshape(4)
    pyramid = friction_pyramid_matrix(constraints.pyramid_mu)

    rows_C, rows_lb, rows_ub = [], [], []
    for i in range(4):
        block = np.zeros((5, 12))
        block[:4, 3 * i : 3 * i + 3] = pyramid
        block[4, 3 * i + 2] = 1.0  # 法向力本身

        if contact[i]:
            lb = np.array([-np.inf] * 4 + [constraints.f_min])
            ub = np.array([0.0] * 4 + [constraints.f_max])
        else:
            # 摆动腿：法向力钉为零，摩擦约束随之退化为 f_x = f_y = 0
            lb = np.array([0.0] * 4 + [0.0])
            ub = np.array([0.0] * 4 + [0.0])

        rows_C.append(block)
        rows_lb.append(lb)
        rows_ub.append(ub)

    return np.vstack(rows_C), np.concatenate(rows_lb), np.concatenate(rows_ub)


def check_friction_cone(force: np.ndarray, mu: float, tol: float = 1e-6) -> bool:
    """检查一个接触力是否落在**真实的圆锥**（而不是金字塔）内。

    用来验证求解结果。**只有内接金字塔才能保证"金字塔可行 => 圆锥可行"**；
    外接写法下这个断言会失败，因为它本来就允许超出圆锥的力。

    Args:
        force: 接触力，形状 (3,)。
        mu: 摩擦系数。
        tol: 容差。

    Returns:
        是否满足 :math:`\\|f_{xy}\\| \\le \\mu f_z` 且 :math:`f_z \\ge 0`。
    """
    f = np.asarray(force, dtype=float).reshape(3)
    if f[2] < -tol:
        return False
    return bool(np.linalg.norm(f[:2]) <= mu * f[2] + tol)
