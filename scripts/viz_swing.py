"""里程碑 5 的可视化：四种摆动轨迹，到底该用哪一条？

生成 ``docs/figures/m5_swing.png``，包含八组图：

1. 四种轨迹的形状对比（矢状面）；
2. 竖直速度曲线 —— 落地冲击一眼可见；
3. 加速度模长 —— 消除冲击的代价；
4. 越障能力：离地高度沿水平位置的分布；
5. 摆动过程中的关节角；
6. 关节角速度，含奇异附近的放大效应；
7. 步长扫描：关节限位给出的硬上限；
8. 关键指标汇总。

运行::

    python scripts/viz_swing.py
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

from gait_scheduler import GaitScheduler  # noqa: E402
from kinematics import (  # noqa: E402
    HIP_OFFSETS,
    LEG_GEOMETRY,
    LEGS,
    inverse_kinematics,
    leg_jacobian,
    load_go2,
)
from swing_planner import SWING_TRAJECTORIES, SwingLegConfig, SwingLegController  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m5_swing.png"

LEG = "FR"
HIP_X = HIP_OFFSETS[LEG][0]
FOOT_Y = HIP_OFFSETS[LEG][1] + LEG_GEOMETRY[LEG].l0
STEP = 0.20
HEIGHT = 0.08
SWING_T = 0.2
BASE_P = np.array([0.0, 0.0, 0.30])
BASE_R = np.eye(3)
LIFTOFF = np.array([HIP_X - STEP / 2, FOOT_Y, 0.0])
TOUCHDOWN = np.array([HIP_X + STEP / 2, FOOT_Y, 0.0])

STYLE = {
    "sine": ("#d62728", "-", "半正弦（常见写法）"),
    "cycloid": ("#ff7f0e", "--", "摆线"),
    "bezier": ("#2171b5", "-", "贝塞尔（端点重复 3 次）"),
    "quintic": ("#2ca02c", "-.", "分段五次多项式"),
}


def trajectories():
    return {n: cls(LIFTOFF, TOUCHDOWN, HEIGHT) for n, cls in SWING_TRAJECTORIES.items()}


def panel_shapes(ax, trajs) -> None:
    s = np.linspace(0.0, 1.0, 400)
    for name, traj in trajs.items():
        color, ls, label = STYLE[name]
        p = traj.position(s)
        ax.plot(p[:, 0], p[:, 2] * 100, color=color, ls=ls, lw=2, label=label)
    ax.plot(*[[LIFTOFF[0], TOUCHDOWN[0]], [0, 0]], "ko", ms=8, zorder=5)
    ax.axhline(0, color="k", lw=1.5)
    ax.annotate("离地", (LIFTOFF[0], 1), fontsize=8, ha="center")
    ax.annotate("落地", (TOUCHDOWN[0], 1), fontsize=8, ha="center")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("离地高度 [cm]")
    ax.set_title(f"四种摆动轨迹（步长 {STEP*100:.0f} cm，抬腿 {HEIGHT*100:.0f} cm）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_vertical_velocity(ax, trajs) -> None:
    s = np.linspace(0.0, 1.0, 400)
    for name, traj in trajs.items():
        color, ls, label = STYLE[name]
        ax.plot(s, traj.velocity(s, SWING_T)[:, 2], color=color, ls=ls, lw=2, label=label)
    ax.axhline(0, color="k", lw=1)
    v_td = trajs["sine"].touchdown_velocity(SWING_T)[2]
    ax.scatter([1.0], [v_td], s=120, color="#d62728", zorder=5, marker="v")
    ax.annotate(f"{v_td:.2f} m/s\n砸地", (1.0, v_td), xytext=(-58, 8),
                textcoords="offset points", color="#d62728", fontsize=9)
    ax.set_xlabel("摆动进度 s")
    ax.set_ylabel("足端竖直速度 [m/s]")
    ax.set_title("落地冲击\n半正弦在 s=1 处速度为 $-h\\pi/T$，其余三种精确为零")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_acceleration(ax, trajs) -> None:
    s = np.linspace(0.0, 1.0, 800)
    for name, traj in trajs.items():
        color, ls, label = STYLE[name]
        a = np.linalg.norm(traj.acceleration(s, SWING_T), axis=-1)
        ax.plot(s, a, color=color, ls=ls, lw=2, label=f"{label}  峰值 {a.max():.0f}")
    ax.set_xlabel("摆动进度 s")
    ax.set_ylabel("足端加速度模长 [m/s²]")
    ax.set_title("消除冲击的代价\n峰值加速度正比于所需关节力矩")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_clearance(ax, trajs) -> None:
    for name, traj in trajs.items():
        color, ls, label = STYLE[name]
        s, c = traj.clearance_profile()
        x = traj.position(s)[:, 0]
        ax.plot(x, c * 100, color=color, ls=ls, lw=2, label=label)
    # 一块假想的障碍物
    obstacle_x, obstacle_h = HIP_X + 0.02, 5.0
    ax.bar([obstacle_x], [obstacle_h], width=0.02, color="gray", alpha=0.5, zorder=0)
    ax.annotate("5 cm 障碍", (obstacle_x, obstacle_h + 0.3), ha="center", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("离地高度 [cm]")
    ax.set_title("越障能力\n看的是障碍所在位置的高度，不是最大高度")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def _joint_traj(traj, n=400):
    """把足端轨迹解算成关节角与关节角速度。"""
    s = np.linspace(0.0, 1.0, n)
    hip_world = BASE_P + BASE_R @ HIP_OFFSETS[LEG]
    positions = traj.position(s)
    velocities = traj.velocity(s, SWING_T)
    q = np.zeros((n, 3))
    dq = np.zeros((n, 3))
    cond = np.zeros(n)
    for k in range(n):
        p_hip = BASE_R.T @ (positions[k] - hip_world)
        q[k] = inverse_kinematics(p_hip, LEG_GEOMETRY[LEG], clamp=True)
        J = leg_jacobian(q[k], LEG_GEOMETRY[LEG])
        dq[k] = np.linalg.lstsq(J, BASE_R.T @ velocities[k], rcond=None)[0]
        cond[k] = np.linalg.cond(J)
    return s, q, dq, cond


def panel_joint_angles(ax, trajs) -> None:
    s, q, _, _ = _joint_traj(trajs["bezier"])
    for k, name in enumerate(["q0 侧摆", "q1 髋俯仰", "q2 膝"]):
        ax.plot(s, q[:, k], lw=2, label=name)
    model = load_go2()
    i = LEGS.index(LEG) * 3
    lo = model.model.lowerPositionLimit[i : i + 3]
    hi = model.model.upperPositionLimit[i : i + 3]
    ax.axhline(hi[2], color="r", ls=":", lw=1.2)
    ax.annotate("膝关节上限", (0.02, hi[2] + 0.05), color="r", fontsize=8)
    ax.axhline(lo[1], color="m", ls=":", lw=1.2)
    ax.set_xlabel("摆动进度 s")
    ax.set_ylabel("关节角 [rad]")
    ax.set_title("摆动中的关节角（贝塞尔轨迹）\n这是里程碑 1 的 IK 第一次进控制回路")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_joint_speed(ax, trajs) -> None:
    for name, traj in trajs.items():
        color, ls, label = STYLE[name]
        s, _, dq, _ = _joint_traj(traj)
        speed = np.linalg.norm(dq, axis=1)
        ax.plot(s, speed, color=color, ls=ls, lw=2, label=f"{label}  峰值 {speed.max():.1f}")
    ax.set_xlabel("摆动进度 s")
    ax.set_ylabel("关节角速度模长 [rad/s]")
    ax.set_title("关节角速度\n由 $\\dot q = J^{-1}v$ 得到，奇异附近会被放大")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_step_limit(ax) -> None:
    """步长扫描：关节限位给出的硬上限。"""
    sched = GaitScheduler("trot")
    ctl = SwingLegController(sched, SwingLegConfig(swing_height=HEIGHT, trajectory="bezier"))
    model = load_go2()
    i = LEGS.index(LEG) * 3
    limits = (model.model.lowerPositionLimit[i : i + 3], model.model.upperPositionLimit[i : i + 3])

    lengths = np.linspace(0.05, 0.60, 40)
    margins, conds, speeds = [], [], []
    for L in lengths:
        p0 = np.array([HIP_X - L / 2, FOOT_Y, 0.0])
        p1 = np.array([HIP_X + L / 2, FOOT_Y, 0.0])
        r = ctl.check_swing_feasibility(LEG, p0, p1, BASE_P, BASE_R, limits, n=201)
        margins.append(r["worst_margin"])
        conds.append(r["max_condition_number"])
        speeds.append(r["max_joint_speed"])

    margins = np.array(margins)
    ax.plot(lengths * 100, margins, "o-", color="#2171b5", lw=2, ms=3, label="关节裕度 [rad]")
    ax.axhline(0, color="r", lw=1.5)
    crossing = lengths[np.argmax(margins < 0)] * 100 if np.any(margins < 0) else np.nan
    if np.isfinite(crossing):
        ax.axvline(crossing, color="r", ls="--", lw=1.2)
        ax.annotate(f"步长上限 ≈ {crossing:.0f} cm", (crossing - 1, 0.35), color="r",
                    fontsize=9, ha="right")
    ax2 = ax.twinx()
    ax2.plot(lengths * 100, speeds, "s--", color="#ff7f0e", lw=1.5, ms=3, label="峰值关节速度")
    ax2.set_ylabel("峰值关节角速度 [rad/s]", color="#ff7f0e")
    ax.set_xlabel("步长 [cm]")
    ax.set_ylabel("最差关节裕度 [rad]", color="#2171b5")
    ax.set_title("步长的硬上限来自关节限位\n而不是几何可达范围")
    ax.grid(alpha=0.25)


def main() -> None:
    trajs = trajectories()

    print(f"摆动参数：步长 {STEP} m，抬腿 {HEIGHT} m，摆动时长 {SWING_T} s\n")
    header = f"{'轨迹':<10} {'落地速度':>12} {'落地加速度':>12} {'峰值加速度':>12} {'峰值关节速度':>14}"
    print(header)
    print("-" * len(header))
    summary = {}
    for name, traj in trajs.items():
        _, _, dq, _ = _joint_traj(traj, n=400)
        summary[name] = {
            "v_td": abs(traj.touchdown_velocity(SWING_T)[2]),
            "a_td": np.linalg.norm(traj.acceleration(1.0, SWING_T)),
            "a_peak": traj.peak_acceleration(SWING_T),
            "dq_peak": np.linalg.norm(dq, axis=1).max(),
        }
        d = summary[name]
        print(
            f"{name:<10} {d['v_td']:12.4f} {d['a_td']:12.1f} {d['a_peak']:12.1f} {d['dq_peak']:14.2f}"
        )

    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30)

    panel_shapes(fig.add_subplot(gs[0, 0]), trajs)
    panel_vertical_velocity(fig.add_subplot(gs[0, 1]), trajs)
    panel_acceleration(fig.add_subplot(gs[0, 2]), trajs)
    panel_clearance(fig.add_subplot(gs[1, 0]), trajs)
    panel_joint_angles(fig.add_subplot(gs[1, 1]), trajs)
    panel_joint_speed(fig.add_subplot(gs[1, 2]), trajs)
    panel_step_limit(fig.add_subplot(gs[2, 0]))

    ax = fig.add_subplot(gs[2, 1:3])
    ax.axis("off")
    lines = [
        "里程碑 5 —— 摆动腿规划器",
        "",
        f"步长 {STEP*100:.0f} cm，抬腿 {HEIGHT*100:.0f} cm，摆动时长 {SWING_T} s",
        "",
        "三层结构，逐层变贵：",
        "",
        f"  半正弦      落地速度 {summary['sine']['v_td']:.2f} m/s —— 砸地，会激发打滑",
        f"              峰值加速度 {summary['sine']['a_peak']:.0f} m/s²（最低）",
        "",
        f"  摆线        落地速度 0，但落地加速度 {summary['cycloid']['a_td']:.0f} —— 接触力有阶跃",
        f"              峰值加速度 {summary['cycloid']['a_peak']:.0f} m/s²",
        "",
        "  贝塞尔/五次  落地速度与加速度都为零 —— 接触力连续",
        f"              峰值加速度 {summary['bezier']['a_peak']:.0f} / "
        f"{summary['quintic']['a_peak']:.0f} m/s²（高 60-70%）",
        "",
        "为什么在意落地冲击：里程碑 3 已经证明，打滑造成的",
        "位置估计误差是永久性的 —— 没有绝对参考就无法纠正。",
        "所以这里多花的关节力矩，买的是上游估计器的精度。",
        "",
        "本模块第一次把里程碑 1 的 IK 放进控制回路：",
        "  M4 相位 -> 足端轨迹 -> 世界系转髋系 -> IK  -> 关节角",
        "                                    -> J^-1 -> 关节角速度",
        "端到端绕一圈回到原点，误差 1e-9（见测试）。",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10, linespacing=1.45,
            family="sans-serif")

    fig.suptitle("四足运动控制 —— 里程碑 5：摆动腿规划器", fontsize=16, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
