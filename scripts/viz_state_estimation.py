"""里程碑 3 的可视化验证：估计器到底准不准，什么时候会失效？

生成 ``docs/figures/m3_state_estimation.png``，包含六组图：

1. 轨迹俯视图：真值 vs ESKF vs 纯 IMU 积分；
2. 位置误差随时间：融合 vs 开环，纵轴对数；
3. 零偏估计：陀螺零偏收敛，加速度计零偏不收敛；
4. 可观测性：横滚俯仰的不确定度收敛，偏航不收敛；
5. 打滑的破坏力：与编码器噪声对比，以及打滑指标；
6. 各误差源的量级对比。

运行::

    python scripts/viz_state_estimation.py
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

from dynamics import matrix_to_rpy  # noqa: E402
from state_estimator import (  # noqa: E402
    ErrorStateKF,
    ESKFState,
    LegOdometry,
    TrajectoryConfig,
    base_velocity_from_legs,
    generate_trot,
    simulate_imu,
)
from state_estimator.simulation import add_encoder_noise, inject_slip  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m3_state_estimation.png"
GRAVITY = 9.81
ACCEL_BIAS = np.array([0.08, -0.05, 0.10])
GYRO_BIAS = np.array([0.010, -0.008, 0.005])


def make_filter(gt):
    init = ESKFState(
        position=gt.base_position[0].copy(),
        velocity=gt.base_velocity_world[0].copy(),
        rotation=gt.base_rotation[0].copy(),
    )
    init.foot_position[:] = gt.foot_position_world[0]
    return ErrorStateKF(dt=gt.config.dt, initial_state=init)


def run(gt, joint_pos=None, accel=None, gyro=None, update=True):
    """跑一遍滤波器，返回轨迹与各类误差。"""
    joint_pos = gt.joint_position if joint_pos is None else joint_pos
    if accel is None:
        accel, gyro = simulate_imu(gt, 0.02, 0.002, ACCEL_BIAS, GYRO_BIAS, seed=0)
    kf = make_filter(gt)
    n = len(gt)
    out = {
        "pos": np.zeros((n, 3)),
        "pos_err": np.zeros((n, 3)),
        "rpy_err": np.zeros((n, 3)),
        "ba": np.zeros((n, 3)),
        "bg": np.zeros((n, 3)),
        "att_std": np.zeros((n, 3)),
    }
    for k in range(n):
        if update:
            kf.step(accel[k], gyro[k], joint_pos[k], gt.contact[k])
        else:
            kf.predict(accel[k], gyro[k], gt.contact[k])
        out["pos"][k] = kf.state.position
        out["pos_err"][k] = kf.state.position - gt.base_position[k]
        out["rpy_err"][k] = matrix_to_rpy(kf.state.rotation) - gt.base_rpy[k]
        out["ba"][k] = kf.state.accel_bias
        out["bg"][k] = kf.state.gyro_bias
        out["att_std"][k] = kf.attitude_std
    return out


def panel_trajectory(ax, gt, fused, openloop) -> None:
    ax.plot(gt.base_position[:, 0], gt.base_position[:, 1], "k-", lw=2.5, label="真值")
    ax.plot(fused["pos"][:, 0], fused["pos"][:, 1], "--", color="#2171b5", lw=2, label="ESKF 融合")
    ax.plot(openloop["pos"][:, 0], openloop["pos"][:, 1], ":", color="#d62728", lw=2, label="纯 IMU 积分")
    ax.plot(gt.base_position[0, 0], gt.base_position[0, 1], "go", ms=8, label="起点")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("轨迹俯视图\nIMU 单独用两秒就飘走了")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")


def panel_position_error(ax, gt, fused, openloop) -> None:
    ax.semilogy(gt.t, np.linalg.norm(openloop["pos_err"], axis=1) + 1e-9, color="#d62728", lw=2,
                label="纯 IMU 积分")
    ax.semilogy(gt.t, np.linalg.norm(fused["pos_err"], axis=1) + 1e-9, color="#2171b5", lw=2,
                label="ESKF 融合")
    ratio = np.linalg.norm(openloop["pos_err"][-1]) / max(np.linalg.norm(fused["pos_err"][-1]), 1e-12)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("位置误差 [m]")
    ax.set_title(f"位置误差（对数纵轴）\n末端相差 {ratio:.0f} 倍")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")


def panel_bias(ax, gt, fused) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, name in enumerate("xyz"):
        ax.plot(gt.t, fused["bg"][:, i], color=colors[i], lw=1.8, label=f"陀螺 b{name} 估计")
        ax.axhline(GYRO_BIAS[i], color=colors[i], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("陀螺零偏 [rad/s]")
    ax.set_title("陀螺零偏可观测，能收敛到真值\n（虚线为真值）")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(alpha=0.25)


def panel_accel_bias(ax, gt, fused) -> None:
    """加速度计零偏不可观测，被姿态倾角吸收。"""
    ax.plot(gt.t, np.degrees(fused["rpy_err"][:, 0]), color="#1f77b4", lw=1.8, label="横滚估计误差")
    ax.axhline(np.degrees(ACCEL_BIAS[1] / GRAVITY), color="#1f77b4", ls="--", lw=1.4,
               label=r"$b_y/g$ 预测值")
    ax.plot(gt.t, np.degrees(fused["rpy_err"][:, 1]), color="#ff7f0e", lw=1.8, label="俯仰估计误差")
    ax.axhline(np.degrees(-ACCEL_BIAS[0] / GRAVITY), color="#ff7f0e", ls="--", lw=1.4,
               label=r"$-b_x/g$ 预测值")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("姿态误差 [度]")
    ax.set_title("加速度计零偏不可观测\n它被姿态倾角吸收，量级恰好是 b/g")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_observability(ax, gt, fused) -> None:
    labels = ["横滚 roll", "俯仰 pitch", "偏航 yaw"]
    colors = ["#2171b5", "#2ca02c", "#d62728"]
    for i in range(3):
        ax.plot(gt.t, np.degrees(fused["att_std"][:, i]), color=colors[i], lw=2, label=labels[i])
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("姿态不确定度 $\\sigma$ [度]")
    ax.set_title("可观测性结构\n重力锚定横滚俯仰；偏航无绝对参考")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_slip(ax, gt) -> None:
    """打滑 vs 编码器噪声。"""
    accel, gyro = simulate_imu(gt, 0.02, 0.002, ACCEL_BIAS, GYRO_BIAS, seed=0)

    ideal = run(gt, accel=accel, gyro=gyro)
    jp1, _ = add_encoder_noise(gt, 1e-3, seed=0)
    enc = run(gt, joint_pos=jp1, accel=accel, gyro=gyro)
    slipped = inject_slip(gt, "FL", 1.0, 1.4, np.array([0.15, 0.0, 0.0]))
    slip = run(gt, joint_pos=slipped.joint_position, accel=accel, gyro=gyro)

    ax.plot(gt.t, np.linalg.norm(ideal["pos_err"], axis=1) * 1000, color="#2ca02c", lw=2, label="理想")
    ax.plot(gt.t, np.linalg.norm(enc["pos_err"], axis=1) * 1000, color="#2171b5", lw=2,
            label="编码器噪声 1 mrad")
    ax.plot(gt.t, np.linalg.norm(slip["pos_err"], axis=1) * 1000, color="#d62728", lw=2,
            label="FL 打滑 0.15 m/s")
    ax.axvspan(1.0, 1.4, color="red", alpha=0.12)
    ax.annotate("打滑窗口", (1.2, ax.get_ylim()[1] * 0.5), ha="center", color="#d62728", fontsize=8)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("位置误差 [mm]")
    ax.set_title("打滑是真正的杀手\n误差在打滑后永久保留，无法自我纠正")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return {
        "理想": np.linalg.norm(ideal["pos_err"][-1]),
        "编码器 1 mrad": np.linalg.norm(enc["pos_err"][-1]),
        "打滑 0.15 m/s": np.linalg.norm(slip["pos_err"][-1]),
    }


def panel_slip_indicator(ax, gt) -> None:
    """各支撑腿速度估计的离散度 —— 免费的打滑检测量。

    接触切换的前后若干毫秒必须屏蔽掉：那里关节速度靠差分得到，跨越
    落地/离地的不连续点时会产生虚假尖峰。真实控制器同样会屏蔽这些时刻，
    因为接触状态本身在那几毫秒里就是不可靠的。
    """
    slipped = inject_slip(gt, "FL", 1.0, 1.4, np.array([0.15, 0.0, 0.0]))

    # 标出接触切换点，前后各屏蔽 15 ms
    switch = np.zeros(len(gt), dtype=bool)
    changed = np.any(np.diff(gt.contact.astype(int), axis=0) != 0, axis=1)
    guard = int(0.015 / gt.config.dt)
    for k in np.flatnonzero(changed):
        switch[max(0, k - guard) : k + guard] = True

    ind_clean = np.full(len(gt), np.nan)
    ind_slip = np.full(len(gt), np.nan)
    for k in range(len(gt)):
        if switch[k]:
            continue
        _, pc = base_velocity_from_legs(
            gt.joint_position[k], gt.joint_velocity[k], gt.base_rotation[k], gt.omega_body[k], gt.contact[k]
        )
        _, ps = base_velocity_from_legs(
            slipped.joint_position[k], slipped.joint_velocity[k], gt.base_rotation[k],
            gt.omega_body[k], gt.contact[k],
        )
        ind_clean[k] = LegOdometry.slip_indicator(pc)
        ind_slip[k] = LegOdometry.slip_indicator(ps)

    ax.semilogy(gt.t, ind_clean, color="#2ca02c", lw=1.6, label="无打滑")
    ax.semilogy(gt.t, ind_slip, color="#d62728", lw=1.6, label="FL 打滑")
    ax.axvspan(1.0, 1.4, color="red", alpha=0.12)
    ratio = np.nanmax(ind_slip[(gt.t > 1.05) & (gt.t < 1.35)]) / np.nanmedian(ind_clean)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("各腿速度估计离散度 [m/s]")
    ax.set_title(f"打滑检测：不需要力传感器\n支撑腿之间的分歧就是信号（信噪比 {ratio:.0f}×）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")


def main() -> None:
    gt = generate_trot(TrajectoryConfig(duration=4.0))
    accel, gyro = simulate_imu(gt, 0.02, 0.002, ACCEL_BIAS, GYRO_BIAS, seed=0)

    fused = run(gt, accel=accel, gyro=gyro, update=True)
    openloop = run(gt, accel=accel, gyro=gyro, update=False)

    print("Go2 状态估计（4 秒 trot，含 IMU 噪声与零偏）：")
    print(f"  ESKF 末端位置误差   : {np.linalg.norm(fused['pos_err'][-1]) * 1000:8.2f} mm")
    print(f"  纯 IMU 末端位置误差 : {np.linalg.norm(openloop['pos_err'][-1]):8.2f} m")
    print(f"  陀螺零偏真值 / 估计 : {GYRO_BIAS} / {fused['bg'][-1]}")
    print(f"  加速度计零偏真值    : {ACCEL_BIAS}")
    print(f"  加速度计零偏估计    : {fused['ba'][-1]}   <- 没收敛")
    print(f"  由此产生的横滚误差  : {np.degrees(fused['rpy_err'][-1, 0]):.4f} 度"
          f"，b_y/g 预测 {np.degrees(ACCEL_BIAS[1] / GRAVITY):.4f} 度")
    print(f"  姿态不确定度 首/末  : 横滚 {np.degrees(fused['att_std'][0, 0]):.3f} -> "
          f"{np.degrees(fused['att_std'][-1, 0]):.3f} 度")
    print(f"                        偏航 {np.degrees(fused['att_std'][0, 2]):.3f} -> "
          f"{np.degrees(fused['att_std'][-1, 2]):.3f} 度")

    fig = plt.figure(figsize=(21, 10))
    panel_trajectory(fig.add_subplot(2, 4, 1), gt, fused, openloop)
    panel_position_error(fig.add_subplot(2, 4, 2), gt, fused, openloop)
    panel_bias(fig.add_subplot(2, 4, 3), gt, fused)
    panel_accel_bias(fig.add_subplot(2, 4, 4), gt, fused)
    panel_observability(fig.add_subplot(2, 4, 5), gt, fused)
    summary = panel_slip(fig.add_subplot(2, 4, 6), gt)
    panel_slip_indicator(fig.add_subplot(2, 4, 7), gt)

    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "里程碑 3 —— 状态估计\n\n"
        "融合方案：误差状态卡尔曼滤波 ESKF\n"
        "  IMU 1 kHz 推进 + 支撑腿运动学纠正\n"
        "  27 维误差状态，足端位置纳入状态\n\n"
        f"末端位置误差\n"
        f"  ESKF 融合   : {np.linalg.norm(fused['pos_err'][-1]) * 1000:.1f} mm\n"
        f"  纯 IMU 积分 : {np.linalg.norm(openloop['pos_err'][-1]):.2f} m\n"
        f"  相差        : {np.linalg.norm(openloop['pos_err'][-1]) / np.linalg.norm(fused['pos_err'][-1]):.0f} 倍\n\n"
        "可观测性结构（本里程碑核心结论）\n"
        "  可观测   : 横滚、俯仰、陀螺零偏\n"
        "  不可观测 : 绝对位置、偏航\n"
        "  不可分辨 : 加速度计水平零偏 <-> 倾角\n\n"
        "误差源量级对比（末端位置误差）\n"
        + "".join(f"  {k:<14s}: {v * 1000:6.1f} mm\n" for k, v in summary.items())
        + "\n打滑比编码器噪声危险得多，且误差\n永久保留 —— 只能靠外部传感器消除。",
        va="top",
        family="sans-serif",
        fontsize=10,
        linespacing=1.45,
    )

    print("\n各误差源的末端位置误差：")
    for k, v in summary.items():
        print(f"  {k:<16s} {v * 1000:8.2f} mm")

    fig.suptitle("四足运动控制 —— 里程碑 3：状态估计（IMU + 支撑腿融合）", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
