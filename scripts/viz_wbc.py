"""里程碑 8 的可视化：完整模型到底值多少。

生成 ``docs/figures/m8_wbc.png``，包含八组图：

1. 一个 trot 周期内 MPC → WBC 串起来的关节力矩；
2. 完整动力学残差 —— 不可讨价还价的判据；
3. WBC vs 朴素 tau = -J^T f，误差随关节速度增长；
4. 力矩限幅收紧时，MPC 的力如何让步（软任务的意义）；
5. 力跟踪权重的影响；
6. 求解耗时 vs 支撑腿数，对照 1 kHz 预算；
7. 摆动腿任务的跟踪；
8. 关键数字汇总。

运行::

    python scripts/viz_wbc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import (  # noqa: E402
    FLOATING_BASE_DOF,
    load_go2_dynamics,
    nominal_configuration,
    srbd_params_from_model,
)
from gait_scheduler import GaitScheduler  # noqa: E402
from kinematics import LEGS  # noqa: E402
from mpc import ConvexMPC, MPCConfig  # noqa: E402
from whole_body_controller import WBCConfig, WholeBodyController  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m8_wbc.png"
LEG_COLOR = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#9467bd"}


def build_scene():
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    return rbd, q, v, params, com, feet


def mpc_forces(params, com, feet, contact_row, velocity=(0.3, 0.0)):
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.array([velocity[0], velocity[1], 0.0]))
    ref = mpc.make_reference(x0, np.array(velocity))
    contact = np.tile(contact_row, (cfg.horizon, 1))
    r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)), np.tile(com, (cfg.horizon, 1)))
    return r.current_forces


def panel_stack_torques(ax, rbd, q, v, params, com, feet):
    """一个 trot 周期内，MPC -> WBC 串起来的关节力矩。"""
    sched = GaitScheduler("trot")
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    wbc = WholeBodyController(rbd)
    times = np.linspace(0.0, 2 * sched.gait.period, 70)
    knee = np.zeros((len(times), 4))

    for j, t in enumerate(times):
        contact = sched.contact_schedule(t, cfg.dt, cfg.horizon)
        x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.array([0.3, 0.0, 0.0]))
        ref = mpc.make_reference(x0, np.array([0.3, 0.0]))
        res = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                        np.tile(com, (cfg.horizon, 1)))
        stance = tuple(leg for i, leg in enumerate(LEGS) if contact[0, i])
        sf = np.array([res.current_forces[LEGS.index(leg)] for leg in stance])
        r = wbc.solve(q, v, stance, sf)
        knee[j] = r.torque.reshape(4, 3)[:, 2]

    for i, leg in enumerate(LEGS):
        ax.plot(times, knee[:, i], color=LEG_COLOR[leg], lw=2, label=leg)
    for k in range(1, 5):
        ax.axvline(k * sched.gait.period / 2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("膝关节力矩 [N·m]")
    ax.set_title("完整链路 M2→M4→M7→M8\n一个 trot 周期内的膝关节力矩")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)


def panel_dynamics_residual(ax, rbd, q, params, com, feet):
    """完整动力学残差：不可讨价还价的判据。"""
    wbc = WholeBodyController(rbd)
    rng = np.random.default_rng(0)
    residuals = []
    for _ in range(60):
        v = np.zeros(rbd.nv)
        v[FLOATING_BASE_DOF:] = rng.normal(size=12) * rng.uniform(0, 6)
        contact_row = np.array([True, False, False, True])
        f = mpc_forces(params, com, feet, contact_row)
        stance = tuple(leg for i, leg in enumerate(LEGS) if contact_row[i])
        sf = np.array([f[LEGS.index(leg)] for leg in stance])
        r = wbc.solve(q, v, stance, sf)
        if not r.success:
            continue
        M = rbd.mass_matrix(q)
        h = rbd.nonlinear_effects(q, v)
        Jc = rbd.contact_jacobian(q, stance)
        tau_full = np.zeros(rbd.nv)
        tau_full[rbd.actuated_dofs] = r.torque
        residuals.append(
            np.abs(M @ r.acceleration + h - tau_full - Jc.T @ r.contact_force.reshape(-1)).max()
        )

    residuals = np.array(residuals)
    ax.semilogy(residuals, "o", ms=4, color="#2171b5")
    ax.axhline(1e-8, color="r", ls="--", lw=1.5)
    ax.annotate("测试容差 1e-8", (2, 1.4e-8), color="r", fontsize=8)
    ax.set_xlabel("随机测试样本")
    ax.set_ylabel(r"$\|Ma + h - S^\top\tau - J^\top f\|_\infty$")
    ax.set_title(f"完整动力学残差（中位 {np.median(residuals):.1e}）\n这是物理定律，不是优化目标")
    ax.grid(alpha=0.25, which="both")


def panel_naive_comparison(ax, rbd, q, params, com, feet):
    """WBC vs 朴素 tau = -J^T f。"""
    wbc = WholeBodyController(rbd)
    rng = np.random.default_rng(1)
    contact_row = np.ones(4, dtype=bool)
    f = mpc_forces(params, com, feet, contact_row, velocity=(0.0, 0.0))

    speeds = np.linspace(0.0, 10.0, 12)
    med, hi = [], []
    for s in speeds:
        vals = []
        for _ in range(25):
            v = np.zeros(rbd.nv)
            v[FLOATING_BASE_DOF:] = rng.normal(size=12) * s
            r = wbc.solve(q, v, LEGS, f)
            if not r.success:
                continue
            naive = wbc.naive_torque(q, LEGS, f)
            vals.append(np.abs(r.torque - naive).max())
        med.append(np.median(vals))
        hi.append(np.percentile(vals, 90))

    ax.plot(speeds, med, "o-", color="#2171b5", lw=2, label="中位数")
    ax.plot(speeds, hi, "s--", color="#ff7f0e", lw=1.5, alpha=0.8, label="90 分位")
    ax.axhline(med[0], color="g", ls=":", lw=1.5)
    ax.annotate(f"静态差异 {med[0]:.2f} N·m\n（腿自身的重力）", (0.3, med[0] + 1.5),
                color="g", fontsize=8)
    ax.set_xlabel("关节速度幅值 [rad/s]")
    ax.set_ylabel(r"$\|\tau_{WBC} - \tau_{naive}\|_\infty$ [N·m]")
    ax.set_title("完整模型值多少\n朴素做法漏掉：腿惯量 + 科氏力 + 腿重力")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return med


def panel_torque_limit(ax, rbd, q, v, params, com, feet):
    """力矩限幅收紧时 MPC 的力如何让步 —— 软任务的意义。"""
    f = mpc_forces(params, com, feet, np.ones(4, dtype=bool), velocity=(0.0, 0.0))
    limits = np.linspace(1.5, 12.0, 22)
    dev, peak = [], []
    for lim in limits:
        r = WholeBodyController(rbd, WBCConfig(torque_limit=lim)).solve(q, v, LEGS, f)
        if not r.success:
            dev.append(np.nan)
            peak.append(np.nan)
            continue
        dev.append(np.abs(r.contact_force - f).max())
        peak.append(np.abs(r.torque).max())

    ax.plot(limits, dev, "o-", color="#d62728", lw=2, label="接触力偏离 MPC 期望 [N]")
    ax2 = ax.twinx()
    ax2.plot(limits, peak, "s--", color="#2171b5", lw=1.5, label="实际峰值力矩")
    ax2.plot(limits, limits, "k:", lw=1, label="限幅值")
    ax2.set_ylabel("力矩 [N·m]", color="#2171b5")
    ax.set_xlabel("力矩限幅 [N·m]")
    ax.set_ylabel("接触力偏离 [N]", color="#d62728")
    ax.set_title("软任务的意义\n限幅收紧时，MPC 的力主动让步而不是让 QP 崩掉")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)


def panel_weight(ax, rbd, q, v, params, com, feet):
    """力跟踪权重的影响。"""
    f = mpc_forces(params, com, feet, np.ones(4, dtype=bool), velocity=(0.0, 0.0))
    target = f + np.array([8.0, -5.0, 0.0])
    weights = np.logspace(-3, 3, 25)
    dev, base_acc = [], []
    for w in weights:
        r = WholeBodyController(rbd, WBCConfig(force_tracking_weight=w)).solve(q, v, LEGS, target)
        dev.append(np.abs(r.contact_force - target).max())
        base_acc.append(np.linalg.norm(r.acceleration[:FLOATING_BASE_DOF]))
    ax.loglog(weights, dev, "o-", color="#2171b5", lw=2)
    ax.set_xlabel("力跟踪权重")
    ax.set_ylabel("接触力偏离期望 [N]")
    ax.set_title("权重越大越贴近 MPC\n但永远是软的 —— 物理约束优先")
    ax.grid(alpha=0.25, which="both")


def panel_solve_time(ax, rbd, q, v, params, com, feet):
    wbc = WholeBodyController(rbd)
    f_all = mpc_forces(params, com, feet, np.ones(4, dtype=bool), velocity=(0.0, 0.0))
    labels, med, worst, nvar = [], [], [], []
    for stance in [(), ("FL",), ("FL", "RR"), ("FL", "FR", "RR"), tuple(LEGS)]:
        sf = np.array([f_all[LEGS.index(leg)] for leg in stance]) if stance else None
        ts = []
        for _ in range(40):
            r = wbc.solve(q, v, stance, sf)
            ts.append(r.solve_time)
        labels.append(f"{len(stance)} 条")
        med.append(np.median(ts) * 1000)
        worst.append(np.max(ts) * 1000)
        nvar.append(rbd.nv + 3 * len(stance))

    x = np.arange(len(labels))
    ax.bar(x - 0.2, med, 0.4, color="#2171b5", label="中位耗时")
    ax.bar(x + 0.2, worst, 0.4, color="#ff7f0e", label="最差耗时")
    ax.axhline(1.0, color="r", lw=2)
    ax.annotate("1 kHz 预算 1 ms", (0.0, 1.05), color="r", fontsize=9)
    for i, (m, n) in enumerate(zip(med, nvar)):
        ax.annotate(f"{n} 变量", (i, m + 0.02), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("支撑腿数量")
    ax.set_ylabel("QP 求解耗时 [ms]")
    ax.set_ylim(0, 1.25)
    ax.set_title("求解耗时 vs 接触数\n远低于 1 kHz 预算")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return med, worst, nvar


def panel_swing_task(ax, rbd, q, v, params, com, feet):
    """摆动腿任务的跟踪：权重越高跟得越准。"""
    contact_row = np.array([True, False, False, True])
    f = mpc_forces(params, com, feet, contact_row)
    stance = ("FL", "RR")
    sf = np.array([f[LEGS.index(leg)] for leg in stance])
    wbc = WholeBodyController(rbd)
    desired = np.array([2.0, 0.0, 5.0])

    weights = np.logspace(0, 4, 20)
    errs = []
    for w in weights:
        task = wbc.swing_foot_task(q, v, "FR", desired, weight=w)
        r = wbc.solve(q, v, stance, sf, tasks=[task])
        J = rbd.model.full_foot_jacobian(q, "FR")
        dJv = rbd.contact_jacobian_dot_v(q, v, ("FR",))
        errs.append(np.linalg.norm(J @ r.acceleration + dJv - desired))
    ax.loglog(weights, errs, "o-", color="#2ca02c", lw=2)
    ax.set_xlabel("摆动任务权重")
    ax.set_ylabel("足端加速度误差 [m/s²]")
    ax.set_title("摆动腿任务跟踪\n期望 [2.0, 0, 5.0] m/s²")
    ax.grid(alpha=0.25, which="both")


def main() -> None:
    rbd, q, v, params, com, feet = build_scene()
    wbc = WholeBodyController(rbd)
    f = mpc_forces(params, com, feet, np.ones(4, dtype=bool), velocity=(0.0, 0.0))
    r = wbc.solve(q, v, LEGS, f)

    print("全身控制（Go2，四脚站立）：")
    print(f"  求解器      : {wbc.cfg.solver}")
    print(f"  决策变量    : {r.n_variables}  = 18 加速度 + 12 接触力")
    print(f"  求解耗时    : {r.solve_time*1000:.3f} ms  (1 kHz 预算 1 ms)")
    print(f"  关节力矩    : max {np.abs(r.torque).max():.2f} N·m")
    print(f"  接触力偏离 MPC: {np.abs(r.contact_force - f).max():.2e} N")
    print(f"  基座加速度  : {np.abs(r.acceleration[:6]).max():.2e}")

    M = rbd.mass_matrix(q)
    h = rbd.nonlinear_effects(q, v)
    Jc = rbd.contact_jacobian(q, LEGS)
    tau_full = np.zeros(rbd.nv)
    tau_full[rbd.actuated_dofs] = r.torque
    resid = np.abs(M @ r.acceleration + h - tau_full - Jc.T @ r.contact_force.reshape(-1)).max()
    print(f"  完整动力学残差: {resid:.2e}")

    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.34)

    panel_stack_torques(fig.add_subplot(gs[0, 0]), rbd, q, v, params, com, feet)
    panel_dynamics_residual(fig.add_subplot(gs[0, 1]), rbd, q, params, com, feet)
    naive_med = panel_naive_comparison(fig.add_subplot(gs[0, 2]), rbd, q, params, com, feet)
    panel_torque_limit(fig.add_subplot(gs[1, 0]), rbd, q, v, params, com, feet)
    panel_weight(fig.add_subplot(gs[1, 1]), rbd, q, v, params, com, feet)
    times = panel_solve_time(fig.add_subplot(gs[1, 2]), rbd, q, v, params, com, feet)
    panel_swing_task(fig.add_subplot(gs[2, 0]), rbd, q, v, params, com, feet)

    print(f"\n  朴素做法误差：静态 {naive_med[0]:.2f} N·m -> 10 rad/s 时 {naive_med[-1]:.2f} N·m")
    print(f"  求解耗时（0/1/2/3/4 条支撑腿）中位：{[f'{t:.3f}' for t in times[0]]} ms")

    ax = fig.add_subplot(gs[2, 1:3])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "\n".join([
            "里程碑 8 —— 全身控制（经典技术栈的最后一块）",
            "",
            "里程碑 2 那个方程终于被完整用上：",
            "    M(q)a + C(q,v)v + g(q) = S^T tau + sum_i J_i^T f_i",
            "",
            "决策变量 z = [a (18); f (3n_c)]，tau 由后 12 行唯一确定后消去。",
            f"四脚站立时 {r.n_variables} 个变量，求解 {r.solve_time*1000:.3f} ms —— 1 kHz 预算 1 ms。",
            "",
            "硬约束（物理定律，不可妥协）：",
            "  浮动基座 6 行  M_b a + h_b = J_b^T f   右端没有 tau",
            "    —— 从 M2 一路说到现在的「S 前 6 列全为零」在这里生效",
            "  支撑脚不动    J_c a + dJ_c v = 0",
            "  摩擦金字塔、法向力上下界、关节力矩限幅",
            "",
            "软任务（可以让步）：",
            "  跟踪 MPC 的接触力、摆动腿加速度、躯干姿态",
            "",
            "为什么 MPC 的力必须是软的：M2 量化过，关节速度 8 rad/s 时单刚体",
            "模型的角加速度误差中位数 43%。MPC 的力本身就是近似的。当成硬约束",
            "会让摆动腿任务被牺牲、甚至 QP 直接不可行。",
            "MPC 负责「大方向对」，WBC 负责「物理上真的能做到」。",
            "两个模型不一致时，让完整模型赢。",
            "",
            "完整模型值多少（vs 朴素 tau = -J^T f）：",
            f"  静止站立    差 {naive_med[0]:.2f} N·m —— 漏掉的是腿自身的重力",
            f"  关节 10 rad/s 差 {naive_med[-1]:.2f} N·m —— 再加上腿惯量与科氏力",
            "",
            f"完整动力学残差 {resid:.1e} —— 这是判据，不是目标。",
        ]),
        va="top", fontsize=9.5, linespacing=1.42,
    )

    fig.suptitle("四足运动控制 —— 里程碑 8：全身控制（完整动力学 QP）", fontsize=16, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
