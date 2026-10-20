"""里程碑 11 评估：MPC + WBC 在 MuJoCo 全身物理仿真里到底能做到什么。

四组实验，全部可复现（无随机性，同一台机器上逐位一致）：

1. **速度跟踪**：指令 0 → 1.5 m/s 扫描，两种模型参数（与控制器一致的 URDF
   惯量 / 宇树官方 MJCF，轻 5.5%）。报稳态均速、线速度 RMS、运输成本。
2. **抗推**：0.5 m/s 行走中施加 0.1 s 水平推力，**遍历 4 个步态相位**，
   报所有相位都能恢复的最大冲量（最差情况，不挑相位）。
3. **实时性**：MPC、WBC、整个控制周期的墙钟时间。
4. **足端力估计**：广义动量观测器 vs MuJoCo 接触力真值；接触检测 vs
   步态调度器的计划接触。

用法::

    conda activate env_isaaclab_recovered
    python scripts/eval_mujoco.py            # 全部，约 3 分钟（24 核）
    python scripts/eval_mujoco.py --quick    # 缩小扫描，用于冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamics import NOMINAL_BASE_HEIGHT, load_go2_dynamics, nominal_configuration  # noqa: E402
from locomotion import LocomotionController  # noqa: E402
from sim import MujocoGo2, Push, SimConfig, run_episode  # noqa: E402
from state_estimator import ContactDetector, MomentumObserver  # noqa: E402

RAMP = 0.5  # 前 0.5 s 零指令站稳
SETTLE = 2.0  # 稳态统计窗口起点
FOOT_RADIUS = 0.022


def initial_q(rbd):
    q = nominal_configuration(rbd)
    q[2] = NOMINAL_BASE_HEIGHT + FOOT_RADIUS
    return q


def command(vx):
    return lambda t: np.array([vx if t >= RAMP else 0.0, 0.0, 0.0])


# -- 1. 速度跟踪 ------------------------------------------------------------------


def speed_trial(args):
    vx, inertia, duration = args
    rbd = load_go2_dynamics()
    log = run_episode(MujocoGo2(SimConfig(inertia=inertia)), LocomotionController(rbd), initial_q(rbd),
                      duration, command(vx))
    if log.fell:
        return dict(vx=vx, inertia=inertia, fell=True, fall_time=log.fall_time)
    w = log.window(SETTLE, duration)
    vb = log.body_velocity()[w]
    return dict(
        vx=vx, inertia=inertia, fell=False,
        mean_vx=float(vb[:, 0].mean()), mean_vy=float(vb[:, 1].mean()),
        rms=log.velocity_rms(SETTLE, duration),
        cot=log.cost_of_transport(SETTLE, duration) if vx > 0 else None,
        peak_torque=float(np.abs(log.torque[w]).max()),
        height_std=float(log.base_pos[w, 2].std()),
        mpc_ms=(log.mpc_time * 1e3).tolist(), wbc_ms=(log.wbc_time * 1e3).tolist(),
        loop_ms=(log.loop_time * 1e3).tolist(),
        trace=dict(t=log.t[::10].tolist(), vx=log.body_velocity()[::10, 0].tolist()),
    )


# -- 2. 抗推 ----------------------------------------------------------------------


def push_trial(args):
    direction, impulse, phase_offset = args
    rbd = load_go2_dynamics()
    duration = 0.1
    force = impulse / duration * (np.array([0, 1.0, 0]) if direction == "lateral" else np.array([1.0, 0, 0]))
    push = Push(SETTLE + phase_offset, duration, force)
    log = run_episode(MujocoGo2(SimConfig()), LocomotionController(rbd), initial_q(rbd),
                      SETTLE + phase_offset + 2.5, command(0.5), [push])
    return dict(direction=direction, impulse=impulse, phase=phase_offset, fell=log.fell)


# -- 4. 足端力估计 ------------------------------------------------------------------


def estimation_trial(args):
    vx, gain, armature, duration = args
    rbd = load_go2_dynamics()
    sim = MujocoGo2(SimConfig())
    log = run_episode(sim, LocomotionController(rbd), initial_q(rbd), duration, command(vx))
    arm = sim.model.dof_armature[6:].copy() if armature else 0.0
    obs = MomentumObserver(rbd, gain=gain, joint_armature=arm)
    det = ContactDetector()
    n = len(log.t)
    est = np.zeros((n, 4, 3))
    detected = np.ones((n, 4), dtype=bool)
    obs.reset(log.q[0], log.v[0])
    for k in range(n - 1):
        # 第 k 步的力矩作用到第 k+1 步的状态上
        obs.update(log.q[k + 1], log.v[k + 1], log.torque[k], sim.cfg.dt)
        est[k + 1] = obs.foot_forces(log.q[k + 1])
        detected[k + 1] = det.update(est[k + 1, :, 2])

    truth = log.contact_force_true[:, :, 2]
    # 同带宽低通后的真值：观测器是一阶低通，拿它跟原始冲击比没有意义。
    lp = np.zeros_like(truth)
    a = gain * sim.cfg.dt
    for k in range(1, n):
        lp[k] = lp[k - 1] + a * (truth[k] - lp[k - 1])
    w = log.window(SETTLE, duration)
    in_contact = truth > 1.0
    err_lp = est[w, :, 2] - lp[w]
    err_raw = est[w, :, 2] - truth[w]

    # 触地/离地检测延迟：检测器的边沿相对真值边沿的**带符号**时间差（正=滞后）。
    # 只找"真值之后第一次置位"会把提前置位算成 0，看不出误检。
    def edge_delays(sign):
        out = []
        for leg in range(4):
            tr = np.where(np.diff(in_contact[:, leg].astype(int)) == sign)[0] + 1
            de = np.where(np.diff(detected[:, leg].astype(int)) == sign)[0] + 1
            for r in tr:
                if log.t[r] < SETTLE or not len(de):
                    continue
                d = de[np.argmin(np.abs(de - r))]
                if abs(d - r) * sim.cfg.dt < 0.05:
                    out.append((d - r) * sim.cfg.dt * 1e3)
        return out

    td, lo = edge_delays(+1), edge_delays(-1)
    swing_w = ~in_contact[w]
    false_pos = float(np.mean(detected[w][swing_w])) if swing_w.any() else 0.0
    stance_mean = float(truth[w][in_contact[w]].mean())
    return dict(
        vx=vx, gain=gain, armature=armature,
        rms_vs_lowpass=float(np.sqrt(np.mean(err_lp**2))),
        bias=float(err_lp.mean()),
        rms_vs_raw_stance=float(np.sqrt(np.mean(err_raw[in_contact[w]] ** 2))),
        stance_mean_fz=stance_mean,
        peak_truth=float(truth[w].max()),
        detect_accuracy=float(np.mean(detected[w] == in_contact[w])),
        schedule_accuracy=float(np.mean(log.stance[w] == in_contact[w])),
        touchdown_delay_ms=float(np.median(td)) if td else None,
        liftoff_delay_ms=float(np.median(lo)) if lo else None,
        false_positive_swing=false_pos,
        trace=dict(
            t=log.t[w][:600].tolist(), truth=truth[w][:600, 0].tolist(),
            lowpass=lp[w][:600, 0].tolist(), est=est[w][:600, 0, 2].tolist(),
        ) if (armature and gain == 100.0) else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "m11_mujoco_eval.json")
    args = ap.parse_args()

    speeds = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    impulses = [10, 20, 30] if args.quick else [5, 10, 15, 20, 25, 30, 35, 40, 45]
    phases = [0.0, 0.2] if args.quick else [0.0, 0.1, 0.2, 0.3]  # trot 周期 0.4 s
    duration = 4.0 if args.quick else 6.0

    jobs_speed = [(v, m, duration) for m in ("urdf", "mjcf") for v in speeds]
    jobs_push = [(d, J, p) for d in ("lateral", "forward") for J in impulses for p in phases]
    jobs_est = [(0.5, K, arm, duration) for K in (50.0, 100.0, 200.0) for arm in (False, True)]
    jobs_est += [(1.0, 100.0, True, duration)]

    with ProcessPoolExecutor() as ex:
        speed = list(ex.map(speed_trial, jobs_speed))
        push = list(ex.map(push_trial, jobs_push))
        est = list(ex.map(estimation_trial, jobs_est))

    # --- 汇总 -------------------------------------------------------------------
    print("\n== 1. 速度跟踪（稳态窗口 %.1f–%.1f s）" % (SETTLE, duration))
    print(f"{'参数':6s} {'指令':>5s} {'均速':>6s} {'RMS':>6s} {'CoT':>6s} {'峰值力矩':>8s}")
    for r in speed:
        if r["fell"]:
            print(f"{r['inertia']:6s} {r['vx']:5.2f}  摔倒 @ {r['fall_time']:.2f}s")
            continue
        cot = f"{r['cot']:.3f}" if r["cot"] is not None else "  —"
        print(f"{r['inertia']:6s} {r['vx']:5.2f} {r['mean_vx']:6.3f} {r['rms']:6.3f} {cot:>6s} {r['peak_torque']:8.1f}")

    print("\n== 2. 抗推（0.5 m/s 行走，0.1 s 推力；所有相位都存活才算）")
    summary_push = {}
    for d in ("lateral", "forward"):
        ok = [J for J in impulses if all(not r["fell"] for r in push if r["direction"] == d and r["impulse"] == J)]
        # 取"从最小冲量起连续存活"的上界，避免偶然存活的大冲量
        best = 0
        for J in impulses:
            if J in ok:
                best = J
            else:
                break
        summary_push[d] = best
        per_phase = {p: [r["impulse"] for r in push if r["direction"] == d and r["phase"] == p and not r["fell"]]
                     for p in phases}
        print(f"{d:8s} 最差相位下最大可恢复冲量 {best} N·s；各相位存活冲量 {per_phase}")

    # 计时必须单进程单独跑：并行池里 24 个进程抢 CPU，墙钟时间会被放大数倍。
    timed = [speed_trial((v, "urdf", duration)) for v in (0.5, 1.0)]
    mpc = np.concatenate([r["mpc_ms"] for r in timed])
    wbc = np.concatenate([r["wbc_ms"] for r in timed])
    loop = np.concatenate([r["loop_ms"] for r in timed])
    timing = {k: dict(median=float(np.median(x)), p99=float(np.percentile(x, 99))) for k, x in
              (("mpc", mpc), ("wbc", wbc), ("loop", loop))}
    print("\n== 3. 实时性（ms，单进程，0.5 与 1.0 m/s 各 %.0f s）" % duration)
    for k, s in timing.items():
        print(f"{k:5s} 中位 {s['median']:.3f}  p99 {s['p99']:.3f}")

    print("\n== 4. 足端力估计")
    for r in est:
        print(f"v={r['vx']} K={r['gain']:.0f} 转子惯量={'补' if r['armature'] else '不补'}: "
              f"vs 低通真值 RMS {r['rms_vs_lowpass']:.1f} N (偏置 {r['bias']:+.1f})，"
              f"vs 原始真值(支撑) {r['rms_vs_raw_stance']:.1f} N；支撑均值 {r['stance_mean_fz']:.0f} N，"
              f"真值峰值 {r['peak_truth']:.0f} N；接触检测 {r['detect_accuracy']*100:.1f}% "
              f"(调度器 {r['schedule_accuracy']*100:.1f}%)，触地/离地延迟中位 {r['touchdown_delay_ms']}/{r['liftoff_delay_ms']} ms，"
              f"摆动相误检 {r['false_positive_swing']*100:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for r in speed:
        for k in ("mpc_ms", "wbc_ms", "loop_ms"):
            r.pop(k, None)
    args.out.write_text(json.dumps(dict(speed=speed, push=push, push_summary=summary_push, timing=timing,
                                        estimation=est), ensure_ascii=False))
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
