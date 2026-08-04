"""里程碑 7 的可视化：凸 MPC 在解什么，解得多快。

生成 ``docs/figures/m7_mpc.png``，包含八组图：

1. 摩擦锥 vs 内接/外接金字塔 —— 一个很多人搞反的方向；
2. QP 的结构：条件化矩阵的稀疏模式；
3. trot 一个周期内的接触力，看清力如何在对角腿之间交接；
4. 求解耗时 vs 预测时域，对照 50 Hz 预算；
5. 闭环速度跟踪（把 MPC 的力喂回单刚体模型）；
6. 摩擦系数对解的影响；
7. 姿态扰动下的回复力矩；
8. 关键数字汇总。

运行::

    python scripts/viz_mpc.py
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
    load_go2_dynamics,
    nominal_configuration,
    srbd_acceleration,
    srbd_params_from_model,
)
from gait_scheduler import GaitScheduler  # noqa: E402
from kinematics import LEGS  # noqa: E402
from mpc import ConvexMPC, FrictionConstraints, MPCConfig, check_friction_cone  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m7_mpc.png"
LEG_COLOR = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#9467bd"}


def panel_friction(ax) -> None:
    """摩擦锥的水平截面：圆 vs 内接/外接正方形。"""
    mu, fz = 0.6, 100.0
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(mu * fz * np.cos(theta), mu * fz * np.sin(theta), "k-", lw=2.5,
            label=f"真实摩擦圆锥 $\\mu f_z$ = {mu*fz:.0f} N")

    m_out = mu
    ax.plot([-m_out * fz, m_out * fz, m_out * fz, -m_out * fz, -m_out * fz],
            [-m_out * fz, -m_out * fz, m_out * fz, m_out * fz, -m_out * fz],
            color="#d62728", ls="--", lw=2, label="外接（常见写法）—— 乐观 41%")
    m_in = mu / np.sqrt(2)
    ax.plot([-m_in * fz, m_in * fz, m_in * fz, -m_in * fz, -m_in * fz],
            [-m_in * fz, -m_in * fz, m_in * fz, m_in * fz, -m_in * fz],
            color="#2171b5", lw=2.5, label="内接（本模块默认）—— 保守 29%")

    ax.scatter([m_out * fz], [m_out * fz], s=110, color="#d62728", zorder=5)
    ax.annotate(f"{np.sqrt(2)*mu*fz:.0f} N\n会打滑", (m_out * fz, m_out * fz),
                xytext=(8, -18), textcoords="offset points", color="#d62728", fontsize=8)
    ax.set_xlabel("$f_x$ [N]")
    ax.set_ylabel("$f_y$ [N]")
    ax.set_aspect("equal")
    ax.set_title(f"摩擦锥的线性化（$f_z$ = {fz:.0f} N）\n常见的金字塔写法是外接，不是内接")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)


def panel_sparsity(ax, mpc, feet, com) -> None:
    """条件化后的 B_qp 稀疏模式：下三角块结构。"""
    N = mpc.cfg.horizon
    ft = np.tile(feet, (N, 1, 1))
    ct = np.tile(com, (N, 1))
    _, B_qp = mpc.build_prediction_matrices(0.0, ft, ct)
    ax.imshow(np.abs(B_qp) > 1e-12, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"决策变量（{B_qp.shape[1]} 个接触力分量）")
    ax.set_ylabel(f"预测状态（{B_qp.shape[0]} = 13 × {N}）")
    ax.set_title("条件化矩阵 $B_{qp}$ 的结构\n下三角块 = 因果性：未来的力不影响过去")


def panel_trot_forces(ax, mpc, params, com, feet, sched) -> None:
    """trot 一个周期内的接触力交接。"""
    cfg = mpc.cfg
    period = sched.gait.period
    times = np.linspace(0.0, 2 * period, 60)
    fz = np.zeros((len(times), 4))
    for j, t in enumerate(times):
        x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.array([0.4, 0.0, 0.0]))
        ref = mpc.make_reference(x0, np.array([0.4, 0.0]))
        contact = sched.contact_schedule(t, cfg.dt, cfg.horizon)
        r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                      np.tile(com, (cfg.horizon, 1)))
        fz[j] = r.current_forces[:, 2]

    for i, leg in enumerate(LEGS):
        ax.plot(times, fz[:, i], color=LEG_COLOR[leg], lw=2, label=leg)
    ax.axhline(params.mass * params.gravity, color="k", ls="--", lw=1.2, label="体重")
    ax.plot(times, fz.sum(axis=1), "k-", lw=1.5, alpha=0.6, label="合计")
    for k in range(1, 5):
        ax.axvline(k * period / 2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("足端法向力 $f_z$ [N]")
    ax.set_title("trot 一个周期内的接触力\n对角腿交替承担全部体重")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)


def panel_solve_time(ax, params, com, feet, sched) -> None:
    horizons = [4, 6, 8, 10, 12, 14, 16, 20]
    med, worst, nvar = [], [], []
    for N in horizons:
        cfg = MPCConfig(horizon=N, dt=0.03)
        mpc = ConvexMPC(params, cfg)
        x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))
        ref = mpc.make_reference(x0, np.array([0.3, 0.0]))
        contact = sched.contact_schedule(0.0, cfg.dt, N)
        ft, ct = np.tile(feet, (N, 1, 1)), np.tile(com, (N, 1))
        ts = [mpc.solve(x0, ref, contact, ft, ct).solve_time for _ in range(12)]
        med.append(np.median(ts) * 1000)
        worst.append(np.max(ts) * 1000)
        nvar.append(12 * N)

    ax.plot(horizons, med, "o-", color="#2171b5", lw=2, label="中位耗时")
    ax.plot(horizons, worst, "s--", color="#ff7f0e", lw=1.5, label="最差耗时")
    ax.axhline(20, color="r", lw=2)
    ax.annotate("50 Hz 预算 20 ms", (12.5, 20.6), color="r", fontsize=9)
    ax.axvline(10, color="g", ls=":", lw=1.5)
    ax.annotate("本项目默认\nN=10", (10.4, 12), color="g", fontsize=8)
    ax.set_xlabel("预测步数 N")
    ax.set_ylabel("QP 求解耗时 [ms]")
    ax.set_title(f"求解耗时 vs 预测时域\n稠密 QP：变量数 = 12N，N=10 时 120 个")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_closed_loop(ax, params, com, feet):
    """闭环：把 MPC 的力喂回单刚体模型。"""
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    dt = 0.01
    contact = np.ones((cfg.horizon, 4), dtype=bool)

    results = {}
    for v_target, color in [(0.2, "#2ca02c"), (0.4, "#2171b5"), (0.8, "#d62728")]:
        pos, vel = com.copy(), np.zeros(3)
        hist = []
        for _ in range(200):
            feet_now = feet + (pos - com)
            feet_now[:, 2] = 0.0
            x0 = ConvexMPC.make_state(np.zeros(3), pos, np.zeros(3), vel)
            ref = mpc.make_reference(x0, np.array([v_target, 0.0]), height=com[2])
            r = mpc.solve(x0, ref, contact, np.tile(feet_now, (cfg.horizon, 1, 1)),
                          np.tile(pos, (cfg.horizon, 1)))
            lin, _ = srbd_acceleration(pos, np.eye(3), np.zeros(3), r.current_forces,
                                       feet_now, params)
            vel = vel + dt * lin
            pos = pos + dt * vel
            hist.append(vel[0])
        results[v_target] = np.array(hist)
        t = np.arange(len(hist)) * dt
        ax.plot(t, hist, color=color, lw=2, label=f"指令 {v_target} m/s → 稳态 {hist[-1]:.3f}")
        ax.axhline(v_target, color=color, ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("质心前进速度 [m/s]")
    ax.set_title("闭环速度跟踪\nMPC 的力喂回单刚体模型")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return results


def panel_friction_effect(ax, params, com, feet):
    """摩擦系数越小，可用的水平力越少。"""
    mus = np.linspace(0.1, 1.0, 12)
    ratios, fx_total = [], []
    for mu in mus:
        cfg = MPCConfig(horizon=10, dt=0.03, friction=FrictionConstraints(mu=mu))
        mpc = ConvexMPC(params, cfg)
        contact = np.ones((cfg.horizon, 4), dtype=bool)
        x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))
        ref = mpc.make_reference(x0, np.array([1.2, 0.0]))
        r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                      np.tile(com, (cfg.horizon, 1)))
        f = r.current_forces
        ratios.append(np.max(np.linalg.norm(f[:, :2], axis=1) / np.maximum(f[:, 2], 1e-9)))
        fx_total.append(f[:, 0].sum())

    ax.plot(mus, ratios, "o-", color="#2171b5", lw=2, label="实际 $|f_{xy}|/f_z$")
    ax.plot(mus, mus, "k--", lw=1.5, label="真实圆锥上限 $\\mu$")
    ax.plot(mus, mus / np.sqrt(2), color="#2ca02c", ls=":", lw=1.5,
            label="内接金字塔上限 $\\mu/\\sqrt{2}$")
    ax2 = ax.twinx()
    ax2.plot(mus, fx_total, "s--", color="#ff7f0e", lw=1.5, ms=4)
    ax2.set_ylabel("可用前向合力 [N]", color="#ff7f0e")
    ax.set_xlabel("摩擦系数 $\\mu$")
    ax.set_ylabel("切向/法向力比")
    ax.set_title("摩擦决定能推多猛\n（指令 1.2 m/s，全力加速）")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)


def panel_attitude(ax, params, com, feet):
    """姿态扰动下 MPC 产生的回复力矩。"""
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    from dynamics import rpy_to_matrix

    angles = np.linspace(-0.25, 0.25, 21)
    torques = {"roll": [], "pitch": []}
    for a in angles:
        for axis, name in [(0, "roll"), (1, "pitch")]:
            rpy = np.zeros(3)
            rpy[axis] = a
            x0 = ConvexMPC.make_state(rpy, com, np.zeros(3), np.zeros(3))
            ref = mpc.make_reference(x0, np.zeros(2))
            r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                          np.tile(com, (cfg.horizon, 1)))
            _, ang = srbd_acceleration(com, rpy_to_matrix(rpy), np.zeros(3),
                                       r.current_forces, feet, params)
            torques[name].append(ang[axis])

    ax.plot(np.degrees(angles), torques["roll"], "o-", color="#2171b5", lw=2, label="横滚")
    ax.plot(np.degrees(angles), torques["pitch"], "s-", color="#d62728", lw=2, label="俯仰")
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("姿态偏差 [度]")
    ax.set_ylabel("角加速度 [rad/s²]")
    ax.set_title("姿态回复：负反馈斜率\n偏一边就往回扳")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def main() -> None:
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    sched = GaitScheduler("trot")

    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))
    ref = mpc.make_reference(x0, np.array([0.4, 0.0]))
    contact = sched.contact_schedule(0.0, cfg.dt, cfg.horizon)
    r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                  np.tile(com, (cfg.horizon, 1)))

    print("凸 MPC（Go2，trot，N=10，dt=0.03 s）：")
    print(f"  求解器        : {cfg.solver}")
    print(f"  决策变量      : {r.n_variables}  (12 × N)")
    print(f"  不等式约束    : {r.n_constraints}")
    print(f"  求解耗时      : {r.solve_time*1000:.2f} ms  (50 Hz 预算 20 ms)")
    print(f"  求解成功      : {r.success}")
    print(f"  垂直力之和    : {r.current_forces[:,2].sum():.2f} N  (体重 {params.mass*params.gravity:.2f} N)")
    print(f"  全部落在真实圆锥内: {all(check_friction_cone(f, cfg.friction.mu) for f in r.current_forces)}")
    print(f"\n  摩擦金字塔    : mu={cfg.friction.mu}, 内接={cfg.friction.inscribed}, "
          f"写进约束的系数={cfg.friction.pyramid_mu:.4f}")

    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.34)

    panel_friction(fig.add_subplot(gs[0, 0]))
    panel_sparsity(fig.add_subplot(gs[0, 1]), mpc, feet, com)
    panel_trot_forces(fig.add_subplot(gs[0, 2]), mpc, params, com, feet, sched)
    panel_solve_time(fig.add_subplot(gs[1, 0]), params, com, feet, sched)
    loop = panel_closed_loop(fig.add_subplot(gs[1, 1]), params, com, feet)
    panel_friction_effect(fig.add_subplot(gs[1, 2]), params, com, feet)
    panel_attitude(fig.add_subplot(gs[2, 0]), params, com, feet)

    print("\n闭环稳态速度：")
    for k, v in loop.items():
        print(f"  指令 {k} m/s -> 稳态 {v[-1]:.4f} m/s   误差 {abs(v[-1]-k)*1000:.1f} mm/s")

    ax = fig.add_subplot(gs[2, 1:3])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "\n".join([
            "里程碑 7 —— 凸模型预测控制",
            "",
            "与 NMPC 的唯一区别，但它决定一切：四足的 MPC 是凸的。",
            "  凸性来源一：足端位置已知时，单刚体动力学对接触力线性（M2）",
            "  凸性来源二：摩擦锥可以内接成金字塔 —— 线性不等式",
            "  于是问题退化成 QP：全局最优、多项式时间、求解时间可预测",
            "",
            f"问题规模（N = {cfg.horizon}，dt = {cfg.dt} s）：",
            f"  决策变量 {r.n_variables} 个（12 × N），不等式约束 {r.n_constraints} 条",
            f"  求解器 {cfg.solver}，耗时 {r.solve_time*1000:.2f} ms —— 50 Hz 预算 20 ms",
            "  条件化后是稠密 QP，所以稀疏求解器（OSQP）在这里没有优势",
            "",
            "三个输入全部来自前面的里程碑：",
            "  M2 单刚体模型 -> A、B 矩阵（已交叉验证一致）",
            "  M4 接触序列   -> 每步哪几条腿有力决策变量",
            "  M6 足端位置   -> B 矩阵里的力臂 r_i × f_i",
            "",
            "一个很多人搞反的方向：",
            "  常见写法 |fx| <= mu*fz 的金字塔是「外接」于圆锥的，",
            "  沿对角方向允许 sqrt(2)*mu*fz —— 比真实极限大 41.4%。",
            "  它不是保守而是「乐观」：求解器可以合法开出会打滑的力。",
            f"  本模块默认内接（系数 mu/sqrt(2) = {cfg.friction.pyramid_mu:.4f}），",
            "  测试可以直接断言解落在「真实圆锥」内。",
            "",
            "工程细节：摆动腿的力被钉为零而不是从变量里删掉 ——",
            "保持 QP 维度恒定，矩阵结构固定，可预分配、可热启动。",
        ]),
        va="top", fontsize=9.5, linespacing=1.45,
    )

    fig.suptitle("四足运动控制 —— 里程碑 7：凸 MPC（单刚体模型 + 摩擦锥 QP）",
                 fontsize=16, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
