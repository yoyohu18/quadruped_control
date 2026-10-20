"""跑一段闭环仿真并记录所有评估要用的量。

评估脚本和测试共用这一个循环，保证"测试里验证的"和"报告里写的"是同一件事。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from kinematics import LEGS

from .mujoco_go2 import MujocoGo2

__all__ = ["EpisodeLog", "Push", "run_episode"]


@dataclass
class Push:
    """对躯干施加一次水平推力。"""

    t_start: float
    duration: float
    force: np.ndarray  # 世界系 (3,)

    @property
    def impulse(self) -> float:
        return float(np.linalg.norm(self.force) * self.duration)


@dataclass
class EpisodeLog:
    t: np.ndarray
    base_pos: np.ndarray  # (T, 3)
    base_vel_world: np.ndarray  # (T, 3)
    yaw_rate: np.ndarray
    rpy: np.ndarray  # (T, 3)
    command: np.ndarray  # (T, 3) vx, vy, yaw_rate（机体系速度指令）
    torque: np.ndarray  # (T, 12) 实际施加的力矩（已按执行器上限饱和）
    q: np.ndarray  # (T, 19) Pinocchio 约定
    v: np.ndarray  # (T, 18)
    joint_vel: np.ndarray  # (T, 12)
    contact_force_true: np.ndarray  # (T, 4, 3) MuJoCo 真值
    stance: np.ndarray  # (T, 4) 调度器给的支撑相
    mpc_time: np.ndarray  # 只含真正求解了 MPC 的那些周期
    wbc_time: np.ndarray
    loop_time: np.ndarray  # 控制器整个 step 的墙钟时间
    fell: bool
    fall_time: float | None
    mass: float

    def window(self, t0: float, t1: float) -> np.ndarray:
        return (self.t >= t0) & (self.t < t1)

    def body_velocity(self) -> np.ndarray:
        """躯干水平速度转到**当前偏航**坐标系（与指令同系）。"""
        yaw = self.rpy[:, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        vx = c * self.base_vel_world[:, 0] + s * self.base_vel_world[:, 1]
        vy = -s * self.base_vel_world[:, 0] + c * self.base_vel_world[:, 1]
        return np.stack([vx, vy], axis=1)

    def velocity_rms(self, t0: float, t1: float) -> float:
        """线速度跟踪 RMS，m/s —— 与 M9/M10 的指标同一定义（水平 2 维）。"""
        w = self.window(t0, t1)
        err = self.body_velocity()[w] - self.command[w, :2]
        return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

    def cost_of_transport(self, t0: float, t1: float) -> float:
        """机械运输成本 ``Σ|τ·q̇| / (m g |v|)``，与 M9/M10 同一定义。"""
        w = self.window(t0, t1)
        power = np.sum(np.abs(self.torque[w] * self.joint_vel[w]), axis=1)
        speed = np.linalg.norm(self.base_vel_world[w, :2], axis=1)
        return float(np.mean(power) / (self.mass * 9.81 * max(np.mean(speed), 1e-3)))


def run_episode(
    sim: MujocoGo2,
    controller,
    q0: np.ndarray,
    duration: float,
    command: Callable[[float], np.ndarray] | np.ndarray = np.zeros(3),
    pushes: list[Push] | None = None,
    fall_height: float = 0.15,
) -> EpisodeLog:
    """闭环跑 ``duration`` 秒。控制器与物理同频（每个物理步调一次 WBC）。

    Args:
        command: 常值 ``[vx, vy, yaw_rate]``，或 ``t -> 指令`` 的函数。
    """
    cmd_fn = command if callable(command) else (lambda t, c=np.asarray(command, float): c)
    pushes = pushes or []
    dt = sim.cfg.dt
    n = int(round(duration / dt))

    s = sim.reset(q0)
    controller.reset(s.t, s.q)

    rec = {k: [] for k in ("t", "p", "vw", "rpy", "cmd", "tau", "dq", "f", "st", "mpc", "wbc", "loop", "wz", "ran", "q", "v")}
    fell, fall_time = False, None
    for _ in range(n):
        t = s.t
        cmd = np.asarray(cmd_fn(t), dtype=float)
        push = sum((p.force for p in pushes if p.t_start <= t < p.t_start + p.duration), np.zeros(3))
        sim.apply_push(push)

        t0 = time.perf_counter()
        out = controller.step(t, s.q, s.v, cmd[:2], float(cmd[2]), dt=dt)
        loop = time.perf_counter() - t0

        R = sim.base_rotation()
        rpy = np.array(
            [np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])]
        )
        rec["t"].append(t)
        rec["p"].append(s.q[:3].copy())
        rec["vw"].append(R @ s.v[:3])
        rec["rpy"].append(rpy)
        rec["cmd"].append(cmd)
        rec["tau"].append(np.clip(out.torque, -sim.torque_limit, sim.torque_limit))
        rec["q"].append(s.q.copy())
        rec["v"].append(s.v.copy())
        rec["dq"].append(s.v[6:].copy())
        rec["f"].append(np.array([s.contact_force[leg] for leg in LEGS]))
        rec["st"].append([leg in out.stance for leg in LEGS])
        rec["mpc"].append(out.mpc_time)
        rec["wbc"].append(out.wbc_time)
        rec["loop"].append(loop)
        rec["wz"].append((R @ s.v[3:6])[2])
        rec["ran"].append(out.mpc_ran)

        s = sim.step(out.torque)
        if s.q[2] < fall_height or R[2, 2] < 0.5:
            fell, fall_time = True, s.t
            break
    sim.apply_push(np.zeros(3))

    A = {k: np.array(v) for k, v in rec.items()}
    return EpisodeLog(
        t=A["t"], base_pos=A["p"], base_vel_world=A["vw"], yaw_rate=A["wz"],
        rpy=A["rpy"], command=A["cmd"], torque=A["tau"], q=A["q"], v=A["v"], joint_vel=A["dq"],
        contact_force_true=A["f"], stance=A["st"].astype(bool), mpc_time=A["mpc"][A["ran"].astype(bool)],
        wbc_time=A["wbc"], loop_time=A["loop"], fell=fell, fall_time=fall_time,
        mass=sim.total_mass(),
    )
