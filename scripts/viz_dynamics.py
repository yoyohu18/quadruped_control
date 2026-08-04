"""里程碑 2 的可视化验证：动力学到底长什么样，近似到底有多贵？

生成 ``docs/figures/m2_dynamics.png``，包含六组图：

1. 质量矩阵 M(q) 的结构 —— 基座块、腿块、以及二者之间的耦合；
2. 质心复合惯量 vs 躯干自身惯量 —— 凸 MPC 参数最容易取错的地方；
3. 自由飞行中的能量守恒 —— 积分器与动力学的整体体检；
4. 支撑力分配：等分 vs 优化解，量化"拍脑袋均分"的代价；
5. 单刚体近似的误差随腿摆动速度的增长；
6. 各项动力学量的计算耗时，对照 1 kHz 预算。

运行::

    python scripts/viz_dynamics.py
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import (  # noqa: E402
    FLOATING_BASE_DOF,
    load_go2_dynamics,
    nominal_configuration,
    srbd_acceleration,
    srbd_params_from_model,
)
from kinematics import LEGS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m2_dynamics.png"
CONTROL_PERIOD_US = 1000.0


def panel_mass_matrix(fig, ax, rbd, q) -> None:
    """质量矩阵的结构：块状分布本身就说明了耦合关系。"""
    M = rbd.mass_matrix(q)
    im = ax.imshow(np.abs(M), cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=np.abs(M).max()))
    fig.colorbar(im, ax=ax, label="|M| (对数刻度)")
    for k in (6, 9, 12, 15):
        ax.axhline(k - 0.5, color="w", lw=0.8, alpha=0.6)
        ax.axvline(k - 0.5, color="w", lw=0.8, alpha=0.6)
    ax.axhline(5.5, color="r", lw=1.6)
    ax.axvline(5.5, color="r", lw=1.6)
    ax.set_title("质量矩阵 M(q) 结构\n红线内为浮动基座 6 自由度（无电机）")
    ax.set_xlabel("速度分量下标")
    ax.set_ylabel("速度分量下标")
    ax.set_xticks([3, 8, 11, 14, 17])
    ax.set_xticklabels(["base", "FL", "FR", "RL", "RR"], fontsize=8)
    ax.set_yticks([3, 8, 11, 14, 17])
    ax.set_yticklabels(["base", "FL", "FR", "RL", "RR"], fontsize=8)


def panel_inertia(ax, rbd, q) -> None:
    """复合惯量 vs 躯干惯量 —— 差 5 到 7 倍。"""
    Ig = np.diag(rbd.centroidal_inertia(q))
    It = np.diag(np.array(rbd._m.inertias[1].inertia))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, It, w, label="躯干连杆自身惯量（错）", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, Ig, w, label="整机质心复合惯量（对）", color="#2171b5", alpha=0.9)
    for i, (a, b) in enumerate(zip(It, Ig)):
        ax.annotate(f"×{b / a:.1f}", (i + w / 2, b), ha="center", va="bottom", fontsize=10, color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ixx (横滚)", "Iyy (俯仰)", "Izz (偏航)"])
    ax.set_ylabel("转动惯量 [kg·m²]")
    ax.set_title("凸 MPC 该用哪个惯量？\n用错会让姿态响应差好几倍")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_energy(ax, rbd) -> None:
    """自由飞行中的机械能守恒。"""
    rng = np.random.default_rng(0)
    q = nominal_configuration(rbd)
    q[2] = 3.0
    v = np.zeros(rbd.nv)
    v[FLOATING_BASE_DOF:] = rng.normal(size=12) * 1.2
    v[:3] = np.array([0.5, -0.3, 2.0])

    dt = 1e-4
    n = 3000
    t = np.arange(n) * dt
    ke, pe = np.zeros(n), np.zeros(n)
    for k in range(n):
        ke[k] = rbd.kinetic_energy(q, v)
        pe[k] = rbd.potential_energy(q)
        a = rbd.forward_dynamics(q, v, np.zeros(rbd.nv))
        v = v + dt * a
        q = pin.integrate(rbd._m, q, dt * v)

    total = ke + pe
    ax.plot(t, ke, label="动能", lw=1.6)
    ax.plot(t, pe, label="势能", lw=1.6)
    ax.plot(t, total, "k-", label="机械能总和", lw=2)
    drift = (total.max() - total.min()) / abs(total[0])
    ax.set_title(f"自由飞行中的能量守恒\n相对漂移 {drift:.2e}（半隐式欧拉, dt=0.1 ms）")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("能量 [J]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_load_sharing(ax, rbd, q) -> None:
    """等分体重 vs 优化分配。"""
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    f_equal = np.tile([0.0, 0.0, params.mass * params.gravity / 4], (4, 1))
    f_opt = rbd.gravity_compensation_torque(q, LEGS)[0].reshape(4, 3)

    _, ang_equal = srbd_acceleration(com, np.eye(3), np.zeros(3), f_equal, feet, params)
    _, ang_opt = srbd_acceleration(com, np.eye(3), np.zeros(3), f_opt, feet, params)

    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, f_equal[:, 2], w, label="等分体重", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, f_opt[:, 2], w, label="最小二范数解", color="#2171b5", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(LEGS)
    ax.set_ylabel("足端垂直力 $f_z$ [N]")
    ax.set_title(
        f"支撑力分配\n等分 → 俯仰角加速度 {ang_equal[1]:+.2f} rad/s²，优化 → {ang_opt[1]:+.2e}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_srbd_error(ax, rbd) -> None:
    """腿摆得越快，「腿无质量」假设越站不住。

    对照组是**关节锁死**的整机模型（``a_joints = 0``），也就是"腿被 WBC 牢牢
    伺服住"这一理想情形。它与单刚体的差别只剩一项：腿运动带来的科氏效应。
    这才是 SRBD 真正丢掉的东西 —— 用质心动量去比是循环论证，恒等于零。
    """
    rng = np.random.default_rng(1)
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    J = rbd.contact_jacobian(q, LEGS)
    M = rbd.mass_matrix(q)

    speeds = np.linspace(0.0, 8.0, 12)
    med, hi = [], []
    for s in speeds:
        errs = []
        for _ in range(60):
            v = np.zeros(rbd.nv)
            v[FLOATING_BASE_DOF:] = rng.normal(size=12) * s
            forces = np.column_stack(
                [rng.normal(size=4) * 8, rng.normal(size=4) * 8, rng.uniform(20, 80, 4)]
            )
            tau_ext = J.T @ forces.reshape(-1)
            nle = rbd.nonlinear_effects(q, v)
            # 关节锁死后只剩基座那 6 行
            a_base = np.linalg.solve(M[:6, :6], tau_ext[:6] - nle[:6])
            ang_full = a_base[3:6]
            _, ang_srbd = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
            errs.append(np.linalg.norm(ang_srbd - ang_full) / max(np.linalg.norm(ang_full), 1e-9))
        med.append(np.median(errs))
        hi.append(np.percentile(errs, 90))

    ax.plot(speeds, 100 * np.array(med), "o-", lw=2, label="中位数")
    ax.plot(speeds, 100 * np.array(hi), "s--", lw=1.4, alpha=0.7, label="90 分位")
    ax.axhline(10, color="r", ls=":", lw=1)
    ax.annotate("10% 误差线", (0.2, 11.5), color="r", fontsize=8)
    ax.set_xlabel("关节速度幅值 [rad/s]")
    ax.set_ylabel("躯干角加速度相对误差 [%]")
    ax.set_title("「腿无质量」假设的代价\n静止时误差为 0（惯量取对了），腿越快越离谱")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_timing(ax, rbd, q) -> None:
    """各项动力学量的耗时，对照 1 kHz 预算。"""
    rng = np.random.default_rng(2)
    v = rng.normal(size=rbd.nv) * 0.3
    a = rng.normal(size=rbd.nv) * 0.3
    tau = rbd.inverse_dynamics(q, v, a)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    forces = np.tile([0.0, 0.0, 40.0], (4, 1))

    items = [
        ("RNEA 逆动力学", lambda: rbd.inverse_dynamics(q, v, a)),
        ("ABA 正动力学", lambda: rbd.forward_dynamics(q, v, tau)),
        ("CRBA 质量矩阵", lambda: rbd.mass_matrix(q)),
        ("科氏矩阵 C", lambda: rbd.coriolis_matrix(q, v)),
        ("质心动量矩阵 Ag", lambda: rbd.centroidal_momentum_matrix(q, v)),
        ("接触约束动力学", lambda: rbd.constrained_forward_dynamics(q, v, tau, LEGS)),
        ("单刚体 SRBD", lambda: srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)),
    ]
    names, times = [], []
    for name, fn in items:
        t = timeit.timeit(fn, number=500) / 500 * 1e6
        names.append(name)
        times.append(t)
        print(f"  {name:<22s} {t:8.2f} us   占 1 kHz 周期 {100 * t / CONTROL_PERIOD_US:5.2f}%")

    colors = ["#2171b5" if t < 50 else "#d62728" for t in times]
    y = np.arange(len(names))
    ax.barh(y, times, color=colors, alpha=0.9)
    for i, t in enumerate(times):
        ax.annotate(f"{t:.1f} µs", (t, i), va="center", xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("单次调用耗时 [µs]")
    ax.set_xscale("log")
    ax.set_title("动力学各项耗时（Python 绑定）\n1 kHz 周期总预算 1000 µs")
    ax.grid(alpha=0.25, axis="x")


def main() -> None:
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    params = srbd_params_from_model(rbd)

    trunk_mass = rbd._m.inertias[1].mass
    print("Go2 整机动力学：")
    print(f"  总质量        : {rbd.total_mass:.3f} kg")
    print(f"  躯干连杆      : {trunk_mass:.3f} kg（{100 * trunk_mass / rbd.total_mass:.1f}%）")
    print(f"  四条腿合计    : {rbd.total_mass - trunk_mass:.3f} kg"
          f"（{100 * (rbd.total_mass - trunk_mass) / rbd.total_mass:.1f}%）")
    print(f"  质心（躯干系）: {rbd.center_of_mass(q) - q[:3]}")
    print(f"  复合惯量对角  : {np.diag(params.inertia_body)}")
    print(f"  躯干惯量对角  : {np.diag(np.array(rbd._m.inertias[1].inertia))}")
    print(f"  静态每足垂直力: {rbd.total_mass * rbd.gravity / 4:.2f} N")
    print("\n耗时：")

    fig = plt.figure(figsize=(16.5, 10))
    panel_mass_matrix(fig, fig.add_subplot(2, 3, 1), rbd, q)
    panel_inertia(fig.add_subplot(2, 3, 2), rbd, q)
    panel_energy(fig.add_subplot(2, 3, 3), rbd)
    panel_load_sharing(fig.add_subplot(2, 3, 4), rbd, q)
    panel_srbd_error(fig.add_subplot(2, 3, 5), rbd)
    panel_timing(fig.add_subplot(2, 3, 6), rbd, q)

    fig.suptitle("四足运动控制 —— 里程碑 2：Go2 刚体动力学", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
