"""两条运动学路径各自有多快？

这个数字决定了 1 kHz 控制回路里到底能放什么。四足控制器每个周期都要为
四条腿算 FK 和雅可比；1 kHz 下总预算只有 1000 us，而运动学只能占其中
百分之几 —— 剩下的要留给 MPC 和 WBC。

运行::

    python scripts/benchmark_kinematics.py
"""

from __future__ import annotations

import math
import sys
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinematics import (  # noqa: E402
    LEG_GEOMETRY,
    LEGS,
    STANDING_JOINT_ANGLES,
    forward_kinematics,
    inverse_kinematics,
    leg_jacobian,
    load_go2,
)

CONTROL_PERIOD_US = 1000.0  # 1 kHz 控制周期


def scalar_fk(q0: float, q1: float, q2: float, l0: float, l1: float, l2: float):
    """同一套闭式 FK，用纯 float 标量而不是 NumPy 数组写成。

    放在这里的唯一目的，是把*算术本身*的开销与分配小 NumPy 数组的开销
    分离开来。库代码并不使用它。
    """
    s0, c0 = math.sin(q0), math.cos(q0)
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    x = -l1 * s1 - l2 * s12
    leg_len = l1 * c1 + l2 * c12
    return x, l0 * c0 + leg_len * s0, l0 * s0 - leg_len * c0


def bench(label: str, fn, number: int = 2000) -> float:
    """给 ``fn`` 计时，并报告每次调用的微秒数。"""
    seconds = timeit.timeit(fn, number=number) / number
    us = seconds * 1e6
    print(f"  {label:<44s} {us:9.2f} us    占 1 kHz 周期 {100 * us / CONTROL_PERIOD_US:6.2f}%")
    return us


def main() -> None:
    model = load_go2()
    q_all = STANDING_JOINT_ANGLES.copy()
    q_leg = q_all[:3]
    geom = LEG_GEOMETRY["FL"]
    target = forward_kinematics(q_leg, geom)
    target_base = model.foot_position(q_all, "FL")

    print("单腿、单次调用")
    t_fk = bench("闭式 FK", lambda: forward_kinematics(q_leg, geom))
    a, b, c = float(q_leg[0]), float(q_leg[1]), float(q_leg[2])
    t_sfk = bench("闭式 FK，纯 float 标量（不用 NumPy）", lambda: scalar_fk(a, b, c, geom.l0, geom.l1, geom.l2))
    t_jac = bench("闭式雅可比", lambda: leg_jacobian(q_leg, geom))
    t_ik = bench("闭式 IK", lambda: inverse_kinematics(target, geom))
    t_pfk = bench("Pinocchio FK + updateFramePlacements", lambda: model.foot_position(q_all, "FL"))
    t_pjac = bench("Pinocchio 帧雅可比", lambda: model.foot_jacobian(q_all, "FL"))
    t_nik = bench(
        "Pinocchio 阻尼最小二乘 IK",
        lambda: model.inverse_kinematics_numeric(target_base, "FL"),
        number=200,
    )

    print("\n整机、一个控制周期（四条腿：FK + 雅可比 + IK）")
    def analytic_tick():
        for i, leg in enumerate(LEGS):
            g = LEG_GEOMETRY[leg]
            q = q_all[3 * i : 3 * i + 3]
            forward_kinematics(q, g)
            leg_jacobian(q, g)
            inverse_kinematics(forward_kinematics(q, g), g)

    def pinocchio_tick():
        model.foot_positions(q_all)
        for leg in LEGS:
            model.foot_jacobian(q_all, leg)

    t_a = bench("闭式解，四条腿", analytic_tick)
    t_p = bench("Pinocchio，四条腿（不含 IK）", pinocchio_tick)

    print("\n小结")
    print(f"  闭式 FK 相对 Pinocchio FK        : {t_pfk / t_fk:6.2f}x")
    print(f"  标量 FK 相对 NumPy 闭式 FK       : 快 {t_fk / t_sfk:6.2f} 倍")
    print(f"  闭式 IK 相对阻尼最小二乘 IK      : 快 {t_nik / t_ik:6.2f} 倍")
    print(f"  闭式解为 MPC 与 WBC 留下 {CONTROL_PERIOD_US - t_a:7.1f} us（1 kHz 周期内）")
    print(f"  Pinocchio 为 MPC 与 WBC 留下 {CONTROL_PERIOD_US - t_p:7.1f} us（1 kHz 周期内）")
    print(
        "\n这几个数字要仔细读 —— 它们和你想当然的结论并不一致。\n"
        "\n1. 在 Python 里，闭式 FK 并不比 Pinocchio 快，甚至更慢。Pinocchio 的"
        "\n   算术跑在 C++ 里；闭式解每次调用都要付 NumPy 的数组分配开销。"
        "\n   纯 float 标量那一行精确隔离了这一点：同样的算术不用 NumPy 会快"
        "\n   好几倍，这个差距是纯开销，不是算法差异。"
        "\n"
        "\n2. 跨任何语言都成立的闭式优势是 IK。解析 IK 只是一遍三角运算；"
        "\n   阻尼最小二乘是迭代求解，既慢得多，又没有硬实时上界 —— 它的"
        "\n   耗时取决于初值。这一点在控制回路里是致命的，所以量产四足"
        "\n   机器人一律使用解析腿部 IK。"
        "\n"
        "\n3. 因此，手工推导腿部运动学的真正理由并不是 Python 里的裸速度，"
        "\n   而是：精确且耗时有界的 IK、一个可以再求导给 MPC 用的显式雅可比、"
        "\n   以及 1 kHz 线程里不依赖任何机器人库。MIT Cheetah 和宇树在 C++"
        "\n   里做的正是这件事。"
    )


if __name__ == "__main__":
    main()
