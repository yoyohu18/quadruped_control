"""里程碑 4 的可视化：步态到底是什么，为什么选 trot？

生成 ``docs/figures/m4_gait.png``，包含六组图：

1. 步态图：六种步态的接触时序，四足研究里最经典的一张图；
2. 支撑腿数量随时间变化，看清哪些步态有腾空相；
3. Go2 真实几何下的静态稳定裕度 —— trot vs pace 差 17 倍；
4. 支撑多边形快照：为什么两条腿撑不住；
5. 凸 MPC 看到的接触序列矩阵；
6. 占空比与稳定性的权衡总览。

运行::

    python scripts/viz_gait.py
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

from dynamics import load_go2_dynamics, nominal_configuration  # noqa: E402
from gait_scheduler import (  # noqa: E402
    GAITS,
    GaitScheduler,
    get_gait,
    static_stability_margin,
    support_polygon,
)
from kinematics import LEGS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m4_gait.png"
SHOWCASE = ["stand", "crawl", "trot", "pace", "bound", "gallop"]
LEG_COLOR = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#9467bd"}


def panel_gait_diagram(ax) -> None:
    """经典步态图：横轴是归一化相位，黑条表示触地。"""
    row = 0
    yticks, ylabels = [], []
    phases = np.linspace(0.0, 2.0, 800)  # 画两个周期
    for name in SHOWCASE:
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        for leg in LEGS:
            i = LEGS.index(leg)
            contact = np.array([sched.contact(p * gait.period)[i] for p in phases])
            ax.fill_between(
                phases, row - 0.4, row + 0.4, where=contact, color=LEG_COLOR[leg], lw=0
            )
            yticks.append(row)
            ylabels.append(f"{leg}")
            row -= 1
        ax.axhline(row + 0.5, color="k", lw=0.8)
        ax.text(-0.85, row + 2.5, name, ha="left", va="center", fontweight="bold", fontsize=11)
        row -= 0.6

    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlim(-0.9, 2.0)
    ax.set_xlabel("归一化相位（两个周期）")
    ax.set_title("步态图：色块表示该腿触地\n六种步态的差别全部在相位偏移上")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(alpha=0.2, axis="x")


def panel_stance_count(ax) -> None:
    for name in SHOWCASE:
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        p = np.linspace(0.0, 1.0, 500, endpoint=False)
        n = np.array([sched.n_stance(x * gait.period) for x in p])
        ax.step(p, n, where="post", lw=2, label=f"{name} (D={gait.duty_factor:.2f})")
    ax.axhline(3, color="r", ls="--", lw=1.2)
    ax.annotate("静态稳定需要 ≥3 条腿", (0.02, 3.08), color="r", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("归一化相位")
    ax.set_ylabel("支撑腿数量")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title("支撑腿数量\n占空比越低，腾空相越明显")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)


def panel_stability(ax, com, feet) -> None:
    """Go2 真实几何下的静态稳定裕度。"""
    for name in ["stand", "crawl", "trot", "pace"]:
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        p = np.linspace(0.0, 1.0, 600, endpoint=False)
        m = np.array(
            [static_stability_margin(com[:2], feet, sched.contact(x * gait.period)) for x in p]
        )
        ax.plot(p, m * 100, lw=2, label=name)
    ax.axhline(0, color="k", lw=1.2)
    ax.fill_between([0, 1], -20, 0, color="red", alpha=0.06)
    ax.annotate("静态不稳定区", (0.5, -12), ha="center", color="#a00", fontsize=9)
    ax.set_xlabel("归一化相位")
    ax.set_ylabel("静态稳定裕度 [cm]")
    ax.set_ylim(-16, 16)
    ax.set_title("Go2 真实几何下的静态稳定裕度\ntrot −0.8 cm vs pace −14.2 cm：差 17 倍")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def _draw_stance(ax, com, feet, contact, title, color) -> None:
    for i, leg in enumerate(LEGS):
        marker = "o" if contact[i] else "x"
        size = 130 if contact[i] else 70
        ax.scatter(feet[i, 0], feet[i, 1], s=size, marker=marker, color=LEG_COLOR[leg],
                   zorder=3, linewidths=2)
        ax.annotate(leg, (feet[i, 0], feet[i, 1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8)
    poly = support_polygon(feet, contact)
    if len(poly) >= 3:
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.25, zorder=1)
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                color=color, lw=2, zorder=2)
    elif len(poly) == 2:
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=3, zorder=2)
    margin = static_stability_margin(com[:2], feet, contact)
    ax.scatter(com[0], com[1], s=180, marker="*", color="k", zorder=4, label="质心投影")
    ax.set_title(f"{title}\n裕度 {margin * 100:+.2f} cm", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.32, 0.32)
    ax.set_ylim(-0.24, 0.24)
    ax.grid(alpha=0.2)


def panel_polygons(fig, gs_positions, com, feet) -> None:
    """三个支撑多边形快照。"""
    cases = [
        (np.ones(4, dtype=bool), "四脚站立", "#2ca02c"),
        (np.array([1, 1, 1, 0], dtype=bool), "crawl：抬起 RR", "#ff7f0e"),
        (np.array([1, 0, 0, 1], dtype=bool), "trot：FL+RR 支撑", "#d62728"),
    ]
    for pos, (contact, title, color) in zip(gs_positions, cases):
        ax = fig.add_subplot(pos)
        _draw_stance(ax, com, feet, contact, title, color)
        if pos == gs_positions[0]:
            ax.legend(fontsize=7, loc="lower right")


def panel_mpc_schedule(ax) -> None:
    """凸 MPC 看到的东西：一张接触序列表。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    dt, n = 0.03, 24
    schedule = sched.contact_schedule(0.0, dt, n)
    ax.imshow(schedule.T.astype(float), aspect="auto", cmap="Blues", vmin=0, vmax=1.4,
              interpolation="nearest")
    for k in range(n):
        for i in range(4):
            ax.text(k, i, "1" if schedule[k, i] else "0", ha="center", va="center",
                    fontsize=6, color="white" if schedule[k, i] else "#555")
    ax.set_yticks(range(4))
    ax.set_yticklabels(LEGS)
    ax.set_xlabel(f"MPC 预测步（dt = {dt} s，共 {n} 步 = {n * dt:.2f} s）")
    ax.set_title("凸 MPC 的输入：接触序列表\n每一列决定该步有几个接触力决策变量")


def panel_summary(ax, com, feet) -> None:
    """占空比 vs 最差稳定裕度。"""
    names, duties, worst = [], [], []
    for name in SHOWCASE:
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        m = [
            static_stability_margin(com[:2], feet, sched.contact(x * gait.period))
            for x in np.linspace(0, 1, 300, endpoint=False)
        ]
        m = [x for x in m if np.isfinite(x)]
        names.append(name)
        duties.append(gait.duty_factor)
        worst.append(min(m) * 100 if m else -25.0)

    colors = ["#2ca02c" if w >= 0 else "#d62728" for w in worst]
    ax.scatter(duties, worst, s=160, c=colors, zorder=3, edgecolors="k", linewidths=1)
    for n, d, w in zip(names, duties, worst):
        ax.annotate(n, (d, w), xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="k", lw=1.2)
    ax.axvline(0.75, color="b", ls="--", lw=1.2)
    ax.annotate("D = 0.75\n三腿支撑分水岭", (0.755, -20), color="b", fontsize=8)
    ax.set_xlabel("占空比 D")
    ax.set_ylabel("最差静态稳定裕度 [cm]")
    ax.set_title("步态选择的本质权衡\n占空比越低越快，但静态稳定性越差")
    ax.grid(alpha=0.25)


def main() -> None:
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    print("Go2 标称站姿：")
    print(f"  质心投影 : {com[:2]}")
    print(f"  足端 xy  :\n{feet[:, :2]}")
    print("\n各步态的静态稳定裕度（Go2 真实几何）：")
    for name in SHOWCASE:
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        m = np.array(
            [
                static_stability_margin(com[:2], feet, sched.contact(x * gait.period))
                for x in np.linspace(0, 1, 400, endpoint=False)
            ]
        )
        finite = m[np.isfinite(m)]
        flight = "  含腾空相" if len(finite) < len(m) else ""
        print(
            f"  {name:7s} D={gait.duty_factor:.2f} T={gait.period:.2f}s  "
            f"最差 {finite.min() * 100:+7.2f} cm  最好 {finite.max() * 100:+7.2f} cm{flight}"
        )

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.0, 1.0], hspace=0.42, wspace=0.32)

    panel_gait_diagram(fig.add_subplot(gs[0, 0:2]))
    panel_stance_count(fig.add_subplot(gs[0, 2]))
    panel_stability(fig.add_subplot(gs[0, 3]), com, feet)
    panel_polygons(fig, [gs[1, 0], gs[1, 1], gs[1, 2]], com, feet)
    panel_summary(fig.add_subplot(gs[1, 3]), com, feet)
    panel_mpc_schedule(fig.add_subplot(gs[2, 0:2]))

    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "里程碑 4 —— 步态调度器\n\n"
        "一个周期性步态只需三个参数：\n"
        "  周期 T、占空比 D、相位偏移 phi_i\n"
        "六种步态的差别全部在相位偏移上：\n"
        "  trot 对角配对   pace 同侧配对\n"
        "  bound 前后配对  pronk 完全同步\n\n"
        "静态稳定需要 >= 3 条腿支撑，四腿均布时\n"
        "对应 D >= 0.75。低于此值只能靠动态平衡。\n\n"
        "Go2 真实几何下的关键数字：\n"
        "  站立  +14.2 cm    crawl  ±0.8 cm\n"
        "  trot   -0.8 cm    pace  -14.2 cm\n"
        "  trot 的对角支撑线几乎穿过质心，\n"
        "  pace 的同侧支撑线偏了 14 cm —— 差 17 倍。\n"
        "  这就是 trot 成为默认中速步态的原因。\n\n"
        "对 MPC 的意义：无人机的分配矩阵不随时间变，\n"
        "四足的接触集合每步都变，所以 MPC 必须提前\n"
        "知道整个预测时域的接触序列才能构造 QP。\n"
        "步态调度器因此是 MPC 的输入，而非辅助模块。",
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    fig.suptitle("四足运动控制 —— 里程碑 4：步态调度器", fontsize=16, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
