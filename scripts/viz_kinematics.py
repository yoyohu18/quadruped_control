"""里程碑 1 的可视化验证：几何看起来真的像一台 Go2 吗？

生成 ``docs/figures/m1_kinematics.png``，包含四组图：

1. 由正运动学画出的三维站立姿态；
2. 单腿矢状面工作空间，并高亮关节限位内的可行区域 —— 这正是后续落脚点
   规划器真正受到的约束；
3. 该工作空间上的雅可比条件数，即腿接近奇异、力控性能变差的位置；
4. 用逆运动学跟踪的摆动腿轨迹，以及由此解出的关节角。

运行::

    python scripts/viz_kinematics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体：Noto Sans CJK 覆盖简体汉字；关闭 unicode 负号避免显示成方块
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinematics import (  # noqa: E402
    HIP_OFFSETS,
    LEG_GEOMETRY,
    LEGS,
    STANDING_JOINT_ANGLES,
    forward_kinematics,
    inverse_kinematics,
    leg_jacobian,
    load_go2,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m1_kinematics.png"
KNEE_LIMITS = (-2.7227, -0.83776)
THIGH_LIMITS = (-0.5236, 3.4907)  # 取更紧的后腿限位


def leg_link_points(leg: str, q: np.ndarray) -> np.ndarray:
    """躯干系下 [侧摆轴, 髋俯仰轴, 膝, 足端] 四个点的位置。"""
    geom = LEG_GEOMETRY[leg]
    hip = HIP_OFFSETS[leg]
    s0, c0 = np.sin(q[0]), np.cos(q[0])

    thigh = hip + np.array([0.0, geom.l0 * c0, geom.l0 * s0])
    # 把小腿长度置零，此时"足端"就是膝关节位置。
    knee_local = forward_kinematics(q, LEG_GEOMETRY[leg].__class__(geom.l0, geom.l1, 0.0))
    knee = hip + knee_local
    foot = hip + forward_kinematics(q, geom)
    return np.vstack([hip, thigh, knee, foot])


def panel_robot(ax) -> None:
    q_all = STANDING_JOINT_ANGLES
    colors = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#9467bd"}
    for i, leg in enumerate(LEGS):
        pts = leg_link_points(leg, q_all[3 * i : 3 * i + 3])
        ax.plot(*pts.T, "-o", color=colors[leg], lw=2.5, ms=4, label=leg)

    trunk = np.array([HIP_OFFSETS[l] for l in ("FL", "FR", "RR", "RL", "FL")])
    ax.plot(*trunk.T, "k-", lw=3, alpha=0.6)

    feet_z = min(leg_link_points(l, q_all[3 * i : 3 * i + 3])[-1, 2] for i, l in enumerate(LEGS))
    xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 2), np.linspace(-0.3, 0.3, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, feet_z), alpha=0.12, color="gray")

    ax.set_title(f"正运动学解出的站立姿态\n躯干高度 = {-feet_z:.3f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect([1.0, 0.75, 0.75])
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=18, azim=-125)


def sagittal_grid(geom, n=220):
    """髋系下的 (x, z) 网格，附带可达性与关节限位可行性掩码。"""
    x = np.linspace(-0.45, 0.45, n)
    z = np.linspace(-0.48, 0.20, n)
    X, Z = np.meshgrid(x, z)
    reach = np.full(X.shape, np.nan)
    feasible = np.zeros(X.shape, dtype=bool)
    cond = np.full(X.shape, np.nan)

    for i in range(n):
        for j in range(n):
            p = np.array([X[i, j], geom.l0, Z[i, j]])
            try:
                q = inverse_kinematics(p, geom)
            except ValueError:
                continue
            reach[i, j] = 1.0
            if KNEE_LIMITS[0] <= q[2] <= KNEE_LIMITS[1] and THIGH_LIMITS[0] <= q[1] <= THIGH_LIMITS[1]:
                feasible[i, j] = True
                cond[i, j] = np.linalg.cond(leg_jacobian(q, geom))
    return X, Z, reach, feasible, cond


def panel_workspace(ax, X, Z, reach, feasible) -> None:
    ax.contourf(X, Z, np.nan_to_num(reach), levels=[0.5, 1.5], colors=["#c6dbef"], alpha=0.9)
    ax.contourf(X, Z, feasible.astype(float), levels=[0.5, 1.5], colors=["#2171b5"], alpha=0.85)
    ax.plot(0, 0, "ko", ms=6)
    ax.annotate("髋俯仰轴", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=8)

    q_stand = STANDING_JOINT_ANGLES[:3]
    p_stand = forward_kinematics(q_stand, LEG_GEOMETRY["FL"])
    ax.plot(p_stand[0], p_stand[2], "r*", ms=14, label="标称站立点")

    ax.set_title("单腿矢状面工作空间\n浅色：纯几何可达    深色：关节限位内可行")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)


def panel_condition(fig, ax, X, Z, cond) -> None:
    im = ax.pcolormesh(X, Z, np.clip(cond, 1, 40), cmap="magma_r", shading="auto")
    fig.colorbar(im, ax=ax, label=r"cond($J$) 条件数")
    ax.contour(X, Z, np.nan_to_num(cond, nan=1e3), levels=[5, 10, 20], colors="w", linewidths=0.8)
    p_stand = forward_kinematics(STANDING_JOINT_ANGLES[:3], LEG_GEOMETRY["FL"])
    ax.plot(p_stand[0], p_stand[2], "c*", ms=14)
    ax.set_title("雅可比条件数\n数值越大越接近奇异，力控性能越差")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")


def panel_swing(ax_traj, ax_joint) -> None:
    """用闭式逆运动学跟踪的摆动相足端弧线。"""
    geom = LEG_GEOMETRY["FL"]
    p0 = np.array([-0.10, geom.l0, -0.30])
    p1 = np.array([0.10, geom.l0, -0.30])
    height = 0.08
    s = np.linspace(0.0, 1.0, 200)
    # x 方向用三次平滑插值，z 方向用半个正弦抬腿。
    alpha = 3 * s**2 - 2 * s**3
    traj = np.outer(1 - alpha, p0) + np.outer(alpha, p1)
    traj[:, 2] += height * np.sin(np.pi * s)

    q_traj = np.array([inverse_kinematics(p, geom) for p in traj])
    err = np.array([np.linalg.norm(forward_kinematics(q, geom) - p) for q, p in zip(q_traj, traj)])

    ax_traj.plot(traj[:, 0], traj[:, 2], "k--", lw=1.2, label="期望轨迹")
    fk = np.array([forward_kinematics(q, geom) for q in q_traj])
    ax_traj.plot(fk[:, 0], fk[:, 2], "r-", lw=2, alpha=0.6, label="FK(IK(期望))")
    for k in range(0, 200, 25):
        pts = leg_link_points("FL", q_traj[k]) - HIP_OFFSETS["FL"]
        ax_traj.plot(pts[:, 0], pts[:, 2], "-", color="gray", lw=1, alpha=0.5)
    ax_traj.set_title(f"逆运动学跟踪的摆动弧线\n最大跟踪误差 = {err.max():.2e} m")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_aspect("equal")
    ax_traj.legend(fontsize=8)
    ax_traj.grid(alpha=0.25)

    for k, name in enumerate(["q0 侧摆", "q1 髋俯仰", "q2 膝"]):
        ax_joint.plot(s, q_traj[:, k], lw=2, label=name)
    ax_joint.axhline(KNEE_LIMITS[1], color="r", ls=":", lw=1)
    ax_joint.annotate("膝关节限位", (0.02, KNEE_LIMITS[1] + 0.03), color="r", fontsize=8)
    ax_joint.set_title("摆动过程中的关节角")
    ax_joint.set_xlabel("摆动相位")
    ax_joint.set_ylabel("关节角 [rad]")
    ax_joint.legend(fontsize=8)
    ax_joint.grid(alpha=0.25)


def main() -> None:
    model = load_go2()
    feet = model.foot_positions(STANDING_JOINT_ANGLES)
    print("躯干系下的站立足端位置：")
    for leg in LEGS:
        print(f"  {leg}: {np.array2string(feet[leg], precision=4)}")
    print(f"躯干相对足端的高度：{-feet['FL'][2]:.4f} m")

    geom = LEG_GEOMETRY["FL"]
    print("正在计算工作空间图 ...")
    X, Z, reach, feasible, cond = sagittal_grid(geom)
    cell = (X[0, 1] - X[0, 0]) * (Z[1, 0] - Z[0, 0])
    print(f"  纯几何工作空间面积   : {np.nansum(reach) * cell:.4f} m^2")
    print(f"  关节限位内可行面积   : {feasible.sum() * cell:.4f} m^2 "
          f"（占纯几何的 {100 * feasible.sum() / np.nansum(reach):.1f}%）")

    fig = plt.figure(figsize=(16, 9.5))
    panel_robot(fig.add_subplot(2, 3, 1, projection="3d"))
    panel_workspace(fig.add_subplot(2, 3, 2), X, Z, reach, feasible)
    panel_condition(fig, fig.add_subplot(2, 3, 3), X, Z, cond)
    panel_swing(fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5))

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "里程碑 1 —— 运动学\n\n"
        f"模型      : Unitree Go2, {model.nq} 自由度（固定基座）\n"
        f"大腿/小腿 : {geom.l1:.3f} / {geom.l2:.3f} m\n"
        f"侧摆偏置  : {abs(geom.l0):.4f} m\n"
        f"最大伸展  : {geom.reach_max:.3f} m（纯几何）\n"
        f"           {np.sqrt(geom.l1**2 + geom.l2**2 + 2*geom.l1*geom.l2*np.cos(KNEE_LIMITS[1])):.3f}"
        " m（受膝限位）\n\n"
        "已与 Pinocchio 交叉验证至 1e-12：\n"
        "  正运动学\n"
        "  足端雅可比 (LOCAL_WORLD_ALIGNED)\n"
        "  浮动基座坐标变换\n\n"
        "闭式逆运动学精确求解，解支固定\n"
        "（膝向后弯、腿向下）。",
        va="top",
        # 系统没有中日韩等宽字体，这里退回 CJK 无衬线，避免汉字变成方块
        family="sans-serif",
        fontsize=10,
        linespacing=1.5,
    )

    fig.suptitle("四足运动控制 —— 里程碑 1：Go2 单腿运动学", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"已保存 {OUT}")


if __name__ == "__main__":
    main()
