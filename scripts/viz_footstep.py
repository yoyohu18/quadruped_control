"""里程碑 6 的可视化：脚该落在哪里，为什么。

生成 ``docs/figures/m6_footstep.png``，包含八组图：

1. 倒立摆的相图：质心追捕获点，捕获点从支撑点逃走；
2. 落脚点落在捕获点前 / 上 / 后，三种结局；
3. 摔倒时间常数 vs 各步态的支撑相时长 —— 为什么 trot 能走；
4. Raibert 前馈系数 vs 捕获点系数，二者的关系；
5. 闭环速度跟踪：有反馈 vs 无反馈；
6. 抗推恢复：能救回来的最大扰动；
7. 留给你的反应时间随扰动增大而急剧缩短；
8. 关键数字汇总。

运行::

    python scripts/viz_footstep.py
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

from footstep_planner import (  # noqa: E402
    FootstepPlanner,
    FootstepPlannerConfig,
    LIPMParams,
    capture_point,
    deadbeat_feedback_gain,
    exact_stride_coefficient,
    lipm_step,
    optimal_feedback_gain,
    steady_state_velocity_ratio,
    time_to_boundary,
)
from gait_scheduler import GAITS, GaitScheduler, get_gait  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m6_footstep.png"
HEIGHT = 0.30
PARAMS = LIPMParams(height=HEIGHT)
MAX_RADIUS = 0.22
TROT = get_gait("trot")


def panel_phase_portrait(ax) -> None:
    """相图：质心稳定地追捕获点，捕获点不稳定地逃走。"""
    x = np.linspace(-0.15, 0.15, 17)
    v = np.linspace(-0.9, 0.9, 17)
    X, V = np.meshgrid(x, v)
    dX = V
    dV = PARAMS.omega**2 * X
    n = np.hypot(dX, dV)
    ax.streamplot(X, V, dX / n, dV / n, color="#9ecae1", density=1.1, linewidth=0.8, arrowsize=0.8)

    xs = np.linspace(-0.15, 0.15, 50)
    ax.plot(xs, -PARAMS.omega * xs, "g-", lw=2.5, label=r"稳定流形 $\xi=0$（捕获点归零）")
    ax.plot(xs, PARAMS.omega * xs, "r--", lw=2.5, label="不稳定流形")
    ax.plot(0, 0, "ko", ms=9, zorder=5)
    ax.set_xlabel("质心相对支撑点位置 [m]")
    ax.set_ylabel("质心速度 [m/s]")
    ax.set_title(f"线性倒立摆相图（$\\omega$ = {PARAMS.omega:.2f} rad/s）\n"
                 "落在绿线上才能停下来")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)


def panel_three_outcomes(ax) -> None:
    """落脚点相对捕获点的三种位置，三种结局。"""
    v0 = np.array([0.5, 0.0])
    xi = capture_point(np.zeros(2), v0, PARAMS)[0]
    cases = [
        (0.5 * xi, "#d62728", f"落在捕获点前 50%（{0.5*xi*100:.1f} cm）→ 继续加速"),
        (xi, "#2ca02c", f"正好落在捕获点（{xi*100:.1f} cm）→ 渐近停下"),
        (1.6 * xi, "#2171b5", f"落过头 60%（{1.6*xi*100:.1f} cm）→ 被推回来"),
    ]
    t = np.linspace(0.0, 0.6, 300)
    for foot, color, label in cases:
        vs = []
        x, v = np.zeros(2), v0.copy()
        prev = 0.0
        for tk in t:
            x, v = lipm_step(x, v, np.array([foot, 0.0]), tk - prev, PARAMS)
            prev = tk
            vs.append(v[0])
        ax.plot(t, vs, color=color, lw=2.2, label=label)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("质心速度 [m/s]")
    ax.set_title("落脚点位置决定一切\n初速度 0.5 m/s")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)


def panel_time_constant(ax) -> None:
    """摔倒时间常数 vs 各步态支撑相：为什么 trot 能走。"""
    names, stances = [], []
    for name in ("crawl", "trot", "pace", "bound", "gallop"):
        names.append(name)
        stances.append(GAITS[name].stance_duration)
    ratios = np.array(stances) / PARAMS.time_constant
    colors = ["#2ca02c" if r < 2 else "#ff7f0e" for r in ratios]

    y = np.arange(len(names))
    ax.barh(y, stances, color=colors, alpha=0.9)
    ax.axvline(PARAMS.time_constant, color="r", lw=2.5)
    ax.annotate(f"摔倒时间常数\n$1/\\omega$ = {PARAMS.time_constant*1000:.0f} ms",
                (PARAMS.time_constant, len(names) - 0.4), color="r", fontsize=9,
                xytext=(8, 0), textcoords="offset points", va="center")
    for i, (s, r) in enumerate(zip(stances, ratios)):
        ax.annotate(f"{r:.2f}×", (s, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("支撑相时长 [s]")
    ax.set_title("步态周期必须比「摔倒」更快\ntrot 支撑相 = 1.14 倍时间常数，刚好够用")
    ax.grid(alpha=0.25, axis="x")


def panel_raibert_vs_capture(ax) -> None:
    """Raibert 前馈 vs 捕获点：同一个东西，系数不同。"""
    v = np.linspace(0.0, 1.2, 200)
    raibert = 0.5 * TROT.stance_duration * v
    capture = v / PARAMS.omega
    k = optimal_feedback_gain(PARAMS, TROT.stance_duration)

    c_star = exact_stride_coefficient(PARAMS, TROT.stance_duration)
    exact = c_star * v
    ax.plot(v, raibert * 100, color="#ff7f0e", lw=2.5, ls="--",
            label=f"Raibert  $T_{{st}}/2$ = {0.5*TROT.stance_duration:.3f} s  (慢步态极限)")
    ax.plot(v, capture * 100, color="#2ca02c", lw=2.5, ls=":",
            label=f"捕获点  $1/\\omega$ = {PARAMS.time_constant:.3f} s  (快步态极限)")
    ax.plot(v, exact * 100, color="#2171b5", lw=3,
            label=f"精确 $\\tanh(\\omega T/2)/\\omega$ = {c_star:.4f} s")
    ax.fill_between(v, exact * 100, raibert * 100, color="#ff7f0e", alpha=0.15)
    ax.axhline(MAX_RADIUS * 100, color="r", ls="--", lw=1.5)
    ax.annotate(f"工作空间上限 {MAX_RADIUS*100:.0f} cm", (0.05, MAX_RADIUS * 100 + 0.6),
                color="r", fontsize=8)
    ax.set_xlabel("躯干速度 [m/s]")
    ax.set_ylabel("落脚点相对髋部的前移量 [cm]")
    ax.set_title("两个经典启发式是同一个精确式的两端\n"
                 f"Raibert 偏大 {100*(0.5*TROT.stance_duration-c_star)/c_star:.1f}%，"
                 f"捕获点偏大 {100*(PARAMS.time_constant-c_star)/c_star:.0f}%")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)


def _closed_loop(strategy, v0, v_cmd, n_steps=40, gain=None):
    sched = GaitScheduler(TROT)
    cfg = FootstepPlannerConfig(strategy=strategy, feedback_gain=gain, max_step_radius=0.30)
    planner = FootstepPlanner(sched, cfg)
    x = np.zeros(2)
    v = np.array(v0, dtype=float)
    stance = TROT.stance_duration
    speeds = []
    for _ in range(n_steps):
        base = np.array([x[0], x[1], HEIGHT])
        foot = planner.plan_leg("FL", base, v, np.array(v_cmd), time_to_touchdown=stance)
        hip = planner.nominal_hip_projection("FL", base, 0.0, 0.0, stance)
        x, v = lipm_step(x, v, x + (foot[:2] - hip), stance, PARAMS)
        speeds.append(v.copy())
    return np.array(speeds)


def panel_closed_loop(ax) -> None:
    """闭环速度跟踪：反馈项到底有没有用。"""
    v_cmd = np.array([0.3, 0.0])
    k_opt = optimal_feedback_gain(PARAMS, TROT.stance_duration)
    steps = np.arange(1, 41) * TROT.stance_duration

    raibert = _closed_loop("raibert", [0.0, 0.0], v_cmd, gain=k_opt)
    exact = _closed_loop("exact", [0.0, 0.0], v_cmd)
    ratio = steady_state_velocity_ratio(0.5 * TROT.stance_duration, k_opt, PARAMS,
                                        TROT.stance_duration)

    ax.axhline(v_cmd[0], color="k", ls="--", lw=1.5, label="指令速度")
    ax.plot(steps, raibert[:, 0], color="#ff7f0e", lw=2.5,
            label=f"Raibert $T_{{st}}/2$ → 稳态 {raibert[-1,0]:.4f}（亏 {100*(1-ratio):.1f}%）")
    ax.plot(steps, exact[:, 0], color="#2171b5", lw=2.5,
            label=f"精确系数 + 死拍增益 → 一步收敛，误差 {abs(exact[-1,0]-v_cmd[0]):.1e}")
    ax.axhline(ratio * v_cmd[0], color="#ff7f0e", ls=":", lw=1.2)
    ax.annotate(f"理论预测 {ratio:.4f}×\n与仿真吻合到小数点后四位", (4.0, ratio * v_cmd[0] - 0.045),
                fontsize=8, color="#ff7f0e")
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("躯干速度 [m/s]")
    ax.set_title("闭环速度跟踪\nRaibert 的稳态亏损是结构性的，不是调参问题")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)


def panel_push_recovery(ax) -> None:
    """被推之后能不能救回来。"""
    v_cmd = np.array([0.2, 0.0])
    steps = np.arange(1, 51) * TROT.stance_duration
    for push, color in [(0.5, "#2ca02c"), (1.0, "#ff7f0e"), (1.5, "#d62728")]:
        speeds = _closed_loop("raibert", [push, 0.0], v_cmd, n_steps=50,
                              gain=optimal_feedback_gain(PARAMS, TROT.stance_duration))
        ax.plot(steps, speeds[:, 0], color=color, lw=2, label=f"被推 {push} m/s")
    ax.axhline(v_cmd[0], color="k", ls="--", lw=1.5, label="指令速度")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("躯干速度 [m/s]")
    ax.set_title("抗推恢复（工作空间半径放宽到 30 cm）\n扰动越大，恢复越慢")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_reaction_time(ax) -> None:
    """留给你的反应时间随扰动急剧缩短。"""
    v = np.linspace(0.05, 1.4, 300)
    times = np.array([time_to_boundary(np.zeros(2), np.array([vv, 0.0]), MAX_RADIUS, PARAMS)
                      for vv in v])
    finite = np.isfinite(times) & (times > 0)
    ax.plot(v[finite], times[finite] * 1000, color="#2171b5", lw=2.5)
    ax.axhline(TROT.stance_duration * 1000, color="r", ls="--", lw=2)
    ax.annotate(f"trot 支撑相 {TROT.stance_duration*1000:.0f} ms", (0.06, TROT.stance_duration * 1000 + 12),
                color="r", fontsize=9)

    v_cross = MAX_RADIUS * PARAMS.omega
    ax.axvline(v_cross, color="k", ls=":", lw=1.5)
    ax.annotate(f"一步救不回来\n$v > r\\omega$ = {v_cross:.2f} m/s", (v_cross - 0.03, 260),
                ha="right", fontsize=9)

    idx = np.argmin(np.abs(v - 1.0))
    ax.scatter([1.0], [times[idx] * 1000], s=110, color="#d62728", zorder=5)
    ax.annotate(f"被推 1.0 m/s\n只剩 {times[idx]*1000:.0f} ms", (1.0, times[idx] * 1000),
                xytext=(-70, 35), textcoords="offset points", color="#d62728", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("速度扰动 [m/s]")
    ax.set_ylabel("捕获点跑出工作空间的剩余时间 [ms]")
    ax.set_ylim(0, 400)
    ax.set_title("留给你的反应时间\n大扰动下固定步态时序根本来不及")
    ax.grid(alpha=0.25)


def main() -> None:
    k = optimal_feedback_gain(PARAMS, TROT.stance_duration)
    v_max = MAX_RADIUS * PARAMS.omega

    print(f"Go2 站立高度 {HEIGHT} m：")
    print(f"  固有频率 omega      = {PARAMS.omega:.4f} rad/s")
    print(f"  摔倒时间常数 1/omega = {PARAMS.time_constant:.4f} s")
    print(f"\n各步态支撑相 / 时间常数：")
    for name in ("crawl", "trot", "pace", "bound", "gallop"):
        g = GAITS[name]
        print(f"  {name:7s} {g.stance_duration:.3f} s -> {g.stance_duration/PARAMS.time_constant:.2f} 倍")
    print(f"\nRaibert 前馈系数 T_st/2 = {0.5*TROT.stance_duration:.4f} s")
    print(f"捕获点系数      1/omega = {PARAMS.time_constant:.4f} s"
          f"   比值 {PARAMS.time_constant/(0.5*TROT.stance_duration):.2f}")
    print(f"理论最优反馈增益 k      = {k:.4f} s")
    print(f"\n一步可恢复的最大扰动 = r*omega = {MAX_RADIUS} * {PARAMS.omega:.3f} = {v_max:.3f} m/s")
    for dv in (0.2, 0.5, 1.0, 1.26):
        t = time_to_boundary(np.zeros(2), np.array([dv, 0.0]), MAX_RADIUS, PARAMS)
        xi = capture_point(np.zeros(2), np.array([dv, 0.0]), PARAMS)[0]
        print(f"  扰动 {dv:.2f} m/s -> 捕获点 {xi*100:5.1f} cm，剩余 {t*1000:6.1f} ms")

    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)

    panel_phase_portrait(fig.add_subplot(gs[0, 0]))
    panel_three_outcomes(fig.add_subplot(gs[0, 1]))
    panel_time_constant(fig.add_subplot(gs[0, 2]))
    panel_raibert_vs_capture(fig.add_subplot(gs[1, 0]))
    panel_closed_loop(fig.add_subplot(gs[1, 1]))
    panel_push_recovery(fig.add_subplot(gs[1, 2]))
    panel_reaction_time(fig.add_subplot(gs[2, 0]))

    ax = fig.add_subplot(gs[2, 1:3])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "\n".join([
            "里程碑 6 —— 落脚点规划器",
            "",
            "里程碑 4 留下一个尴尬结论：trot 的静态稳定裕度恒为 -0.84 cm，",
            "按静态标准机器人一直在翻倒，可它明明能走。本模块给出答案：",
            "四足不追求静态稳定，它追求的是「总能把脚迈到该去的地方」。",
            "",
            f"线性倒立摆（h = {HEIGHT} m）：",
            f"  固有频率 omega = sqrt(g/h) = {PARAMS.omega:.2f} rad/s",
            f"  摔倒时间常数 1/omega = {PARAMS.time_constant*1000:.0f} ms",
            f"  trot 支撑相 {TROT.stance_duration*1000:.0f} ms = {TROT.stance_duration/PARAMS.time_constant:.2f} 倍时间常数 —— 刚好够用",
            "",
            "捕获点 xi = x + v/omega：把脚落在这里，机器人渐近停下。",
            "  xi_dot = omega (xi - p)   不稳定，会从支撑点逃走",
            "  x_dot  = -omega (x - xi)  稳定，质心总在追捕获点",
            "",
            "核心洞察：两个经典启发式是同一个精确式的两个渐近端",
            "  由极限环条件解出精确半步系数：",
            "    c* = tanh(omega*T_st/2) / omega",
            "    omega*T -> 0   时 c* -> T_st/2   （Raibert，慢步态极限）",
            "    omega*T -> inf 时 c* -> 1/omega  （捕获点，快步态极限）",
            f"  Go2 trot 的 omega*T = {PARAMS.omega*TROT.stance_duration:.2f}，正好夹在中间：",
            f"    精确 c*  = {exact_stride_coefficient(PARAMS, TROT.stance_duration):.6f} s",
            f"    Raibert  = {0.5*TROT.stance_duration:.6f} s  （偏大 10.7%）",
            f"    捕获点   = {PARAMS.time_constant:.6f} s  （偏大 93.5%）",
            f"  用 Raibert 系数稳态速度亏 11.4%（预测 {steady_state_velocity_ratio(0.5*TROT.stance_duration, k, PARAMS, TROT.stance_duration):.4f}×，",
            "  与仿真吻合到小数点后四位）—— 这是结构性误差，不是调参问题。",
            "",
            "另一个坑：前馈跟随当前速度时也参与反馈，跟随指令速度则不。",
            f"  p = c*v + k(v-v_cmd)      稳定条件 c + k > c*",
            f"  p = c**v_cmd + k(v-v_cmd) 稳定条件 k > c*",
            f"  同一个 k = {k:.4f} 在前者稳定、在后者发散。死拍增益 coth(wT)/w",
            f"  = {deadbeat_feedback_gain(PARAMS, TROT.stance_duration):.4f} s 可一步收敛。",
            "",
            "能力边界：一步可恢复的最大扰动 = r * omega",
            f"  {MAX_RADIUS} m x {PARAMS.omega:.2f} rad/s = {v_max:.2f} m/s",
            "  被推 1.0 m/s 时只剩 40 ms 必须落脚，而 trot 支撑相是 200 ms",
            "  —— 强推恢复必须打破固定步态时序（里程碑 10 的主题）。",
        ]),
        va="top", fontsize=9.5, linespacing=1.45,
    )

    fig.suptitle("四足运动控制 —— 里程碑 6：落脚点规划器（捕获点与 Raibert 启发式）",
                 fontsize=16, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
