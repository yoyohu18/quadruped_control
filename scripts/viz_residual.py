"""里程碑 10 的可视化：名义控制器长什么样、残差在补什么。

生成 ``docs/figures/m10_residual.png``，八组图：

1. 名义足端轨迹（躯干系），支撑相与摆动相分色；
2. 一个周期内四条腿的接触时序与足端高度；
3. 落脚点如何随速度指令移动（M6 的 Raibert 在动）；
4. **调出来的 bug 一号**：落地瞬间足端目标的跳变（修复前后对比）；
5. **调出来的 bug 二号**：足端竖直基准用固定值 vs 用实测高度；
6. 残差的有界性：动作空间里名义点与可达集；
7. 批量吞吐：名义控制器在 GPU 上的耗时 vs 环境数；
8. 关键数字汇总。

全部在 CPU 上跑，不需要 Isaac Sim。

运行::

    python scripts/viz_residual.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.nominal_controller import (  # noqa: E402
    LEG_ORDER,
    NominalGaitConfig,
    NominalGaitController,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m10_residual.png"
LEG_COLOR = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#9467bd"}


# ---------------------------------------------------------------- 数值实验


def buggy_foot_x(ctrl: NominalGaitController, times, v_cmd, v_meas) -> np.ndarray:
    """复现"落地跳变"这个 bug：支撑起点用 v_cmd，摆动终点用 Raibert 的 v_meas。

    这是本里程碑第一版的写法。留在这里不是为了怀旧，而是因为**这张图是
    整个里程碑最有教学价值的一张**：两段各自都"看起来对"的轨迹，
    接在一起就产生了一个瞬移。
    """
    cfg = ctrl.cfg
    phase = ctrl.phase(times).numpy()
    nominal = ctrl._nominal_xy.numpy()

    touchdown = nominal[:, 0] + 0.5 * cfg.stance_duration * v_meas + cfg.raibert_gain * (v_meas - v_cmd)
    liftoff_wrong = nominal[:, 0] - 0.5 * cfg.stance_duration * v_cmd

    x = np.empty_like(phase)
    stance = phase < cfg.duty_factor
    s_st = np.clip(phase / cfg.duty_factor, 0, 1)
    s_sw = np.clip((phase - cfg.duty_factor) / (1 - cfg.duty_factor), 0, 1)
    smooth = s_sw**2 * (3 - 2 * s_sw)

    x_stance = nominal[:, 0] - cfg.stance_duration * (s_st - 0.5) * v_cmd
    x_swing = liftoff_wrong + (touchdown - liftoff_wrong) * smooth
    x[stance] = x_stance[stance]
    x[~stance] = x_swing[~stance]
    return x


def throughput_benchmark(sizes=(64, 256, 1024, 4096, 16384), repeats=7, iters=50) -> list[tuple[int, float]]:
    """名义控制器的吞吐：每秒能算多少个环境步。

    **取多轮的最小耗时**，不取平均：这台机器上可能同时跑着别的东西
    （第一次测的时候后台正好在跑 Isaac Sim，结果 256 环境比 64 环境还慢，
    数字自相矛盾）。最小值是"没有被打扰时"的耗时，才是我们想量的东西。
    这是性能基准的通用做法，里程碑 1 的 ``benchmark_kinematics.py`` 同理。
    """
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    out = []
    for n in sizes:
        t = torch.rand(n) * 0.4
        cmd = torch.randn(n, 3) * 0.5
        vel = torch.randn(n, 2) * 0.5
        h = torch.full((n,), 0.30)
        ctrl.compute(t, cmd, vel, h)  # 预热

        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                ctrl.compute(t, cmd, vel, h)
            best = min(best, time.perf_counter() - t0)
        out.append((n, iters * n / best))
    return out


# ---------------------------------------------------------------- 绘图


def main() -> None:
    cfg = NominalGaitConfig()
    ctrl = NominalGaitController(cfg, device="cpu")

    print("名义控制器吞吐基准……")
    bench = throughput_benchmark()
    for n, fps in bench:
        print(f"  {n:6d} 环境 → {fps / 1e6:.2f} M step/s")

    times = torch.linspace(0.0, cfg.period, 401)
    v_cmd, v_meas = 0.6, 0.45
    command = torch.tensor([[v_cmd, 0.0, 0.0]]).expand(401, 3)
    measured = torch.tensor([[v_meas, 0.0]]).expand(401, 2)
    height = torch.full((401,), 0.30)
    feet = ctrl.foot_targets(times, command, measured, height).numpy()
    phase = ctrl.phase(times).numpy()
    contact = ctrl.contact(times).numpy()

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.27)

    # -- 1. 足端轨迹 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    # 只画 FL 与 RL：左右腿在 x–z 投影里完全重叠（只差 y），画四条会互相盖住
    for i, leg in ((0, "FL"), (2, "RL")):
        st = contact[:, i]
        ax.plot(feet[st, i, 0], feet[st, i, 2], color=LEG_COLOR[leg], lw=3, alpha=0.85)
        ax.plot(feet[~st, i, 0], feet[~st, i, 2], color=LEG_COLOR[leg], lw=1.6, ls="--")
        ax.plot([], [], color=LEG_COLOR[leg], lw=2, label=leg)
    ax.set_xlabel("躯干系 x [m]")
    ax.set_ylabel("躯干系 z [m]")
    ax.set_title("① 名义足端轨迹（粗=支撑，虚=摆动）\n左右腿在此投影下重叠，只画左侧", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # -- 2. 接触时序 + 足端高度 -------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    t = times.numpy()
    for i, leg in enumerate(LEG_ORDER):
        ax.plot(t, feet[:, i, 2] + 0.30, color=LEG_COLOR[leg], lw=1.8, label=leg)
        ax.fill_between(t, -0.012 - 0.006 * i, -0.006 - 0.006 * i, where=contact[:, i],
                        color=LEG_COLOR[leg], alpha=0.75, step="mid")
    ax.axhline(cfg.swing_height, color="0.4", ls=":", lw=1.2)
    ax.text(0.005, cfg.swing_height + 0.002, f"抬腿 {cfg.swing_height} m", fontsize=8, color="0.35")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("足端离地高度 [m]")
    ax.set_title("② 一个 trot 周期：对角腿成对交替（M4）", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)

    # -- 3. 落脚点随速度移动 ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    speeds = np.linspace(-3.0, 3.0, 80)
    td = []
    for v in speeds:
        f = ctrl.foot_targets(
            torch.tensor([cfg.period * (1 - 1e-9)]),
            torch.tensor([[v, 0.0, 0.0]]),
            torch.tensor([[v, 0.0]]),
            torch.tensor([0.30]),
        )
        td.append(f[0, 0, 0].item())
    td = np.array(td)
    nominal_x = ctrl._nominal_xy[0, 0].item()
    ax.plot(speeds, td - nominal_x, color="#1f77b4", lw=2.2, label="名义控制器")
    ax.plot(speeds, 0.5 * cfg.stance_duration * speeds, "--", color="#d62728", lw=1.8,
            label=r"Raibert 前馈 $\frac{T_{st}}{2}v$")
    ax.axhline(cfg.max_stride, color="#2ca02c", ls=":", lw=1.5)
    ax.axhline(-cfg.max_stride, color="#2ca02c", ls=":", lw=1.5)
    ax.text(-2.9, cfg.max_stride * 1.08, f"工作空间限幅 ±{cfg.max_stride} m", fontsize=8, color="#2ca02c")
    ax.set_ylim(-0.42, 0.42)
    ax.set_xlabel("速度指令 = 实测速度 [m/s]")
    ax.set_ylabel("落脚点相对标称位置 [m]")
    ax.set_title("③ 落脚点跟着速度走，超出工作空间就限幅（M6）", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # -- 4. bug 一号：落地跳变 ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    x_bad = buggy_foot_x(ctrl, times, v_cmd, v_meas)
    ax.plot(t, x_bad[:, 0], color="#d62728", lw=2.2, label="修复前：支撑起点用 $v_{cmd}$")
    ax.plot(t, feet[:, 0, 0], color="#2ca02c", lw=2.2, label="修复后：由落脚点倒推离地点")
    # 不连续处表现为相邻采样点之间的一次跃变，量它才对。
    # 原先写成首尾两点之差 —— 而首尾恰好是同一个相位，永远得 0。
    jump = float(np.abs(np.diff(x_bad[:, 0])).max())
    ax.axvline(cfg.period * cfg.duty_factor, color="0.6", ls=":", lw=1.2)
    idx = int(np.argmax(np.abs(np.diff(x_bad[:, 0]))))
    ax.annotate(
        f"落地瞬间跳变 {jump * 100:.1f} cm",
        xy=(t[idx], x_bad[idx, 0]), xytext=(0.10, x_bad[:, 0].max() + 0.005),
        fontsize=8.5, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("FL 足端 x [m]")
    ax.set_title("④ bug 一号：$v_{meas}\\neq v_{cmd}$ 时足端目标不连续", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # -- 5. bug 二号：竖直基准 ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    sag = 0.26  # 纯位置控制下的实测躯干高度
    z_fixed = ctrl.foot_targets(times, command, measured, torch.full((401,), 0.30)).numpy()[:, 0, 2]
    z_meas = ctrl.foot_targets(times, command, measured, torch.full((401,), sag)).numpy()[:, 0, 2]
    ax.plot(t, z_fixed, color="#d62728", lw=2.2, label="基准 = 假定的 0.30 m")
    ax.plot(t, z_meas, color="#2ca02c", lw=2.2, label=f"基准 = 实测的 {sag:.2f} m")
    ax.axhline(-sag, color="0.35", ls="--", lw=1.5)
    ax.text(0.005, -sag + 0.004, "真实地面", fontsize=8.5, color="0.3")
    swing_mask = ~contact[:, 0]
    ax.fill_between(t, -0.34, -sag, where=(z_fixed < -sag) & swing_mask, color="#d62728", alpha=0.18)
    ax.text(0.24, -0.325, "摆动腿被埋在地下\n→ 提前触地 → 机身被推着倒退", fontsize=8, color="#d62728")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("FL 足端 z [m]")
    ax.set_title("⑤ bug 二号：竖直基准必须用实测高度", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # -- 6. 残差有界 -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    q_nom = ctrl.compute(times, command, measured, height).numpy()
    for alpha, color in [(0.02, "#2ca02c"), (0.10, "#1f77b4"), (0.25, "#d62728")]:
        ax.fill_between(t, q_nom[:, 4] - alpha * 3, q_nom[:, 4] + alpha * 3,
                        color=color, alpha=0.18)
        ax.plot([], [], color=color, lw=6, alpha=0.4, label=rf"$\alpha={alpha}$（±3σ 可达）")
    ax.plot(t, q_nom[:, 4], color="0.15", lw=2.2, label="名义 $q_{nom}$")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("FL 髋俯仰角 [rad]")
    ax.set_title("⑥ 残差有界：$\\|q^{des}-q_{nom}\\|\\leq\\alpha\\|a\\|$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # -- 7. 吞吐 -----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ns = [n for n, _ in bench]
    fps = [f / 1e6 for _, f in bench]
    ax.loglog(ns, fps, "o-", color="#1f77b4", lw=2, ms=6)
    ax.axhline(0.2, color="#d62728", ls="--", lw=1.5)
    ax.text(70, 0.22, "训练需要的 0.2 M step/s（4096 环境 @ 50 Hz）", fontsize=8, color="#d62728")
    ax.set_xlabel("并行环境数")
    ax.set_ylabel("吞吐 [M step/s]")
    ax.set_title("⑦ 名义控制器的开销（CPU，单线程）", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    # -- 8. 相位编码 -------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    t2 = torch.linspace(0.0, 2 * cfg.period, 400)
    ph = ctrl.phase(t2)[:, 0].numpy()
    ax.plot(t2.numpy(), ph, color="#d62728", lw=2, label=r"$\phi$ 本身（有跳变）")
    ax.plot(t2.numpy(), np.sin(2 * np.pi * ph), color="#1f77b4", lw=2, label=r"$\sin 2\pi\phi$")
    ax.plot(t2.numpy(), np.cos(2 * np.pi * ph), color="#2ca02c", lw=2, label=r"$\cos 2\pi\phi$")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("相位编码")
    ax.set_title("⑧ 相位是环形量，必须用 sin/cos 进观测", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # -- 9. 汇总 -----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "\n".join([
            "里程碑 10 的关键数字",
            "",
            "名义控制器 = M1 闭式 IK + M4 相位 + M5 摆动 + M6 落脚点，",
            "全部重写成批量 torch。M7 的 MPC 与 M8 的 WBC **进不来** ——",
            "它们要解 QP，无法批量化到 GPU 上跑 4096 份。",
            "",
            f"吞吐（CPU 单线程）：4096 环境 {bench[3][1] / 1e6:.2f} M step/s，",
            "训练只需要 0.2 M step/s —— 名义控制器不是瓶颈。",
            "",
            "交叉验证（4 条独立路径）：",
            "  批量 IK        vs  M1 的闭式 numpy IK（1e-12）",
            "  相位/接触序列  vs  M4 的 GaitScheduler（逐元素相等）",
            "  落脚点         vs  M6 的 raibert_footstep（1e-9）",
            "  零指令站姿     vs  M1 的 STANDING_JOINT_ANGLES",
            "",
            "调出来的三个真 bug（都由测试或实测抓到）：",
            f"  ① 落地瞬间足端目标跳变 {jump * 100:.1f} cm",
            "     —— 支撑起点用 v_cmd，摆动终点用 Raibert 的 v_meas",
            "  ② 竖直基准写成常数 —— 摆动腿提前触地，零指令下倒退 0.35 m/s；",
            "     但改成纯实测又会丢掉高度控制权（塌到 0.11 m）。正解是比例伺服。",
            "  ③ 推力课程的降级分支比对了模式名而非事件名，",
            "     整条课程被**静默**关掉，训练照跑、日志全 0。",
            "",
            "Isaac Lab 实测（平地 300 轮，与里程碑 9 逐项对齐）：",
            "  最终回报    30.8 → 34.7      样本效率 2.0~2.3 倍",
            "  对角腿相关  0.712 → 0.978    运输成本 0.664 → 0.457",
            "  动作变化率  1.48 → 0.747     —— 上真机友好得多",
            "",
            "抗推（50% 存活对应的推力）：",
            "  纯 RL 1.72 m/s，残差 1.57 m/s，残差+推力课程 **3.0 m/s**",
            "",
            "残差的可证明性质：|q_des − q_nom| ≤ α·|a|。",
            "这是纯 RL 给不了的界。",
        ]),
        va="top", fontsize=9.3, linespacing=1.4,
    )

    fig.suptitle("四足运动控制 —— 里程碑 10：残差强化学习", fontsize=16, y=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
