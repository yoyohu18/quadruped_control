"""把 M2–M8 串成一个真正能在物理仿真里走路的控制器。

前八个里程碑各自都被单独验证过，但**从来没有一起闭环跑过**。本模块就是
那根线：

::

    状态 (q, v) ──┬── 步态调度 M4 ── 接触序列 ──┐
                  ├── 落脚点 M6 ── 摆动轨迹 M5 ─┤
                  │                              ├── 凸 MPC M7 (50 Hz) ── 接触力
                  │                              │                          │
                  └──────────── 全身 QP M8 (1 kHz) ◄─────────────────────────┘
                                     │
                                  12 个关节力矩

**两个频率。** MPC 50 Hz，WBC 1 kHz。两次 MPC 之间用上一次的力计划；但只要
接触集合一变（某只脚刚落地或刚抬起），立即重解一次 MPC —— 否则刚落地的
那只脚在最多 20 ms 内拿不到力，刚抬起的脚反而还被要求出力。

**坐标系。** 全部用世界系，除了一处：Pinocchio 浮动基座的广义加速度
``a[0:6]`` 是**机体系**的。躯干任务因此要自己构造，见
:meth:`LocomotionController._body_task`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dynamics import NOMINAL_BASE_HEIGHT, RigidBodyDynamics, matrix_to_rpy, nominal_configuration, srbd_params_from_model
from footstep_planner import FootstepPlanner, FootstepPlannerConfig
from gait_scheduler import GaitScheduler
from kinematics import LEGS
from mpc import ConvexMPC, MPCConfig
from swing_planner import BezierSwing
from whole_body_controller import Task, WBCConfig, WholeBodyController, orientation_error

__all__ = ["LocomotionConfig", "LocomotionController", "ControlOutput"]


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


#: 闭环用的 MPC 状态权重。相对 M7 的默认值，把水平位置 5 -> 20、水平速度
#: 12 -> 40：M7 只在单刚体模型上闭环，模型误差为零；放进全身物理仿真后，
#: 腿部动量等未建模项要靠更硬的速度反馈压住。扫描见 docs/11 第 5 节。
_MPC_WEIGHTS = np.array(
    [80.0, 80.0, 20.0, 20.0, 20.0, 250.0, 1.0, 1.0, 8.0, 40.0, 40.0, 6.0, 0.0]
)


@dataclass
class LocomotionConfig:
    """闭环控制器参数。

    Attributes:
        gait: 步态名。
        mpc_period: MPC 求解周期，秒。
        foot_radius: 足端球半径。Pinocchio 的足端坐标系在**球心**，所以
            球心在平地上的高度是它，而不是 0。
        body_height: 躯干期望高度（世界系 z）。缺省取标称站高 + 足端半径。
        swing_height: 抬腿高度。
        swing_kp, swing_kd: 摆动腿的笛卡尔 PD，单位 1/s², 1/s。
        body_kp, body_kd: 躯干位置 PD。
        orient_kp, orient_kd: 躯干姿态 PD。
        body_weight, orient_weight, swing_weight: WBC 任务权重。
        torque_limit: 逐关节力矩上限（Pinocchio 顺序）。
    """

    gait: str = "trot"
    mpc_period: float = 0.02
    mpc: MPCConfig = field(default_factory=lambda: MPCConfig(state_weights=_MPC_WEIGHTS.copy()))
    wbc: WBCConfig = field(default_factory=WBCConfig)
    foot_radius: float = 0.022
    body_height: float | None = None
    swing_height: float = 0.08
    swing_kp: float = 900.0
    swing_kd: float = 60.0
    body_kp: float = 100.0
    body_kd: float = 20.0
    orient_kp: float = 200.0
    orient_kd: float = 28.0
    body_weight: float = 1.0
    orient_weight: float = 1.0
    swing_weight: float = 100.0
    torque_limit: np.ndarray = field(default_factory=lambda: np.array([23.7, 23.7, 45.43] * 4))


@dataclass
class ControlOutput:
    """一个控制周期的输出，外加诊断量。"""

    torque: np.ndarray
    stance: tuple[str, ...]
    mpc_forces: np.ndarray  # (4, 3) 世界系，摆动腿为零
    wbc_forces: dict[str, np.ndarray]
    mpc_solved: bool
    mpc_ran: bool  # 本周期是否重新求解了 MPC
    mpc_time: float
    wbc_time: float
    wbc_success: bool
    foot_targets: dict[str, np.ndarray]


class LocomotionController:
    """MPC + WBC 的分层闭环控制器。

    Args:
        rbd: 浮动基座整机动力学（控制器的模型，与仿真器无关）。
        config: 参数。
    """

    def __init__(self, rbd: RigidBodyDynamics, config: LocomotionConfig | None = None) -> None:
        self.rbd = rbd
        self.cfg = cfg = config or LocomotionConfig()
        cfg.wbc.torque_limit = np.asarray(cfg.torque_limit, dtype=float)
        self.params = srbd_params_from_model(rbd)
        self.mpc = ConvexMPC(self.params, cfg.mpc)
        self.wbc = WholeBodyController(rbd, cfg.wbc)
        self.scheduler = GaitScheduler(cfg.gait)
        self.planner = FootstepPlanner(
            self.scheduler, FootstepPlannerConfig(nominal_height=NOMINAL_BASE_HEIGHT)
        )

        q_nom = nominal_configuration(rbd)
        self.body_height = (
            cfg.body_height if cfg.body_height is not None else NOMINAL_BASE_HEIGHT + cfg.foot_radius
        )
        # MPC 的状态是质心，不是躯干原点；两者在标称位形下的高度差是常数。
        self.com_offset_z = rbd.center_of_mass(q_nom)[2] - q_nom[2]

        self._mpc_plan: np.ndarray | None = None  # (N, 4, 3)
        self._mpc_t = -np.inf
        self._mpc_contact: np.ndarray | None = None
        self._last_contact = np.ones(4, dtype=bool)
        self._liftoff: dict[str, np.ndarray] = {}
        self._mpc_stats = (False, 0.0)
        self.body_xy_ref: np.ndarray | None = None
        self.yaw_ref = 0.0

    # -- 状态 -----------------------------------------------------------------

    def _base_state(self, q: np.ndarray, v: np.ndarray):
        quat = q[3:7]  # xyzw
        x, y, z, w = quat
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return R, R @ v[0:3], R @ v[3:6]

    # -- 主入口 ---------------------------------------------------------------

    def reset(self, t: float, q: np.ndarray) -> None:
        self.scheduler.force_gait(self.cfg.gait, t)
        self.body_xy_ref = q[:2].copy()
        R, _, _ = self._base_state(q, np.zeros(18))
        self.yaw_ref = float(matrix_to_rpy(R)[2])
        self._mpc_plan = None
        self._last_contact = np.ones(4, dtype=bool)
        self._liftoff.clear()

    def step(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        velocity_command: np.ndarray = np.zeros(2),
        yaw_rate_command: float = 0.0,
        dt: float = 0.001,
    ) -> ControlOutput:
        """一个 WBC 周期（1 kHz）。"""
        cfg = self.cfg
        if self.body_xy_ref is None:
            self.reset(t, q)
        v_cmd = np.asarray(velocity_command, dtype=float).reshape(2)

        R, v_world, w_world = self._base_state(q, v)
        rpy = matrix_to_rpy(R)
        feet = self.rbd.model.foot_positions(q)

        # 参考：按指令速度积分出躯干水平位置与偏航的参考。只积分参考，
        # 不从实测回拉 —— 否则就退化成纯速度控制，漂移无法被纠正。
        c, s = np.cos(self.yaw_ref), np.sin(self.yaw_ref)
        v_cmd_world = np.array([c * v_cmd[0] - s * v_cmd[1], s * v_cmd[0] + c * v_cmd[1]])
        self.body_xy_ref = self.body_xy_ref + v_cmd_world * dt
        self.yaw_ref += yaw_rate_command * dt
        # 参考不能离实测太远，否则一次大扰动后 PD 会要求一个不合理的回拉。
        err = self.body_xy_ref - q[:2]
        if np.linalg.norm(err) > 0.1:
            self.body_xy_ref = q[:2] + err * (0.1 / np.linalg.norm(err))

        contact = self.scheduler.contact(t)
        stance = tuple(leg for i, leg in enumerate(LEGS) if contact[i])

        # 刚离地的腿：记下离地点，作为摆动轨迹起点。
        for i, leg in enumerate(LEGS):
            if self._last_contact[i] and not contact[i]:
                self._liftoff[leg] = feet[leg].copy()
        touchdown = self.planner.plan(
            t, q[:3], v_world[:2], v_cmd_world, yaw=rpy[2], yaw_rate=w_world[2],
            terrain_height={leg: cfg.foot_radius for leg in LEGS},
        )

        # ---- MPC（50 Hz，接触集合变化时立即重解）--------------------------
        contact_changed = bool(np.any(contact != self._last_contact))
        mpc_ran = self._mpc_plan is None or t - self._mpc_t >= cfg.mpc_period - 1e-9 or contact_changed
        if mpc_ran:
            self._solve_mpc(t, q, v, R, rpy, v_world, w_world, feet, touchdown, v_cmd_world, yaw_rate_command)
        k = min(int((t - self._mpc_t) / cfg.mpc.dt), cfg.mpc.horizon - 1)
        f_plan = self._mpc_plan[k].copy()
        f_plan[~contact] = 0.0
        self._last_contact = contact.copy()

        # ---- WBC ----------------------------------------------------------
        tasks = [self._body_task(q, v, R, v_world, w_world, v_cmd_world, yaw_rate_command)]
        tasks += self._orientation_task(R, v, w_world, yaw_rate_command)
        swing_phase = self.scheduler.swing_phase(t)
        T_sw = self.scheduler.gait.swing_duration
        for i, leg in enumerate(LEGS):
            if contact[i]:
                continue
            p0 = self._liftoff.get(leg, feet[leg])
            traj = BezierSwing(p0, touchdown[leg], cfg.swing_height)
            sph = float(np.clip(swing_phase[i], 0.0, 1.0))
            p_des = traj.position(sph)
            v_des = traj.velocity(sph, T_sw)
            a_ff = traj.acceleration(sph, T_sw)
            J = self.rbd.model.full_foot_jacobian(q, leg)
            v_foot = J @ v
            a_des = a_ff + cfg.swing_kp * (p_des - feet[leg]) + cfg.swing_kd * (v_des - v_foot)
            tasks.append(self.wbc.swing_foot_task(q, v, leg, a_des, weight=cfg.swing_weight))

        f_des = np.array([f_plan[LEGS.index(leg)] for leg in stance]) if stance else None
        res = self.wbc.solve(q, v, stance, f_des, tasks)
        torque = res.torque if res.success else np.zeros(12)

        return ControlOutput(
            torque=torque,
            stance=stance,
            mpc_forces=f_plan,
            wbc_forces={leg: res.contact_force[j] for j, leg in enumerate(stance)},
            mpc_solved=self._mpc_stats[0],
            mpc_ran=mpc_ran,
            mpc_time=self._mpc_stats[1],
            wbc_time=res.solve_time,
            wbc_success=res.success,
            foot_targets=touchdown,
        )

    # -- MPC ------------------------------------------------------------------

    def _solve_mpc(self, t, q, v, R, rpy, v_world, w_world, feet, touchdown, v_cmd_world, yaw_rate_cmd):
        cfg = self.cfg
        N, dt = cfg.mpc.horizon, cfg.mpc.dt
        com = self.rbd.center_of_mass(q)
        vcom = self.rbd.com_velocity(q, v)
        x0 = ConvexMPC.make_state(rpy, com, w_world, vcom)

        # 参考：从"躯干参考位置"出发外推，而不是从实测质心出发。
        ref = np.zeros((N, 13))
        com_ref0 = np.array([*self.body_xy_ref + (com[:2] - q[:2]), self.body_height + self.com_offset_z])
        for k in range(N):
            tk = (k + 1) * dt
            ref[k, 2] = self.yaw_ref + yaw_rate_cmd * tk
            ref[k, 3:5] = com_ref0[:2] + v_cmd_world * tk
            ref[k, 5] = com_ref0[2]
            ref[k, 8] = yaw_rate_cmd
            ref[k, 9:11] = v_cmd_world
            ref[k, 12] = -9.81
        # 偏航参考要和实测在同一分支上，避免 ±π 处跳变。
        ref[:, 2] = rpy[2] + np.angle(np.exp(1j * (ref[:, 2] - rpy[2])))

        contact_sched = self.scheduler.contact_schedule(t, dt, N)
        foot_traj = np.zeros((N, 4, 3))
        for k in range(N):
            for i, leg in enumerate(LEGS):
                # 当前支撑且预测步仍支撑：脚不动。当前摆动：将落在规划点。
                foot_traj[k, i] = feet[leg] if leg not in touchdown else touchdown[leg]
        com_traj = np.tile(com, (N, 1))
        com_traj[:, :2] += v_cmd_world * (np.arange(1, N + 1) * dt)[:, None]

        t0 = time.perf_counter()
        res = self.mpc.solve(x0, ref, contact_sched, foot_traj, com_traj, yaw=rpy[2])
        wall = time.perf_counter() - t0
        if res.success or self._mpc_plan is None:
            self._mpc_plan = res.forces
            self._mpc_t = t
            self._mpc_contact = contact_sched
        self._mpc_stats = (res.success, wall)

    # -- 躯干任务 -------------------------------------------------------------

    def _body_task(self, q, v, R, v_world, w_world, v_cmd_world, yaw_rate_cmd) -> Task:
        """躯干线加速度任务，**世界系**。

        Pinocchio 的 ``a[0:3]`` 是机体系线速度的导数。世界系下躯干原点的
        经典加速度是

        .. math::  \\ddot p = R\\,(a_{0:3} + \\omega_b \\times v_b)

        所以雅可比是 ``[R, 0, ...]``，漂移是 ``R (ω_b × v_b)``。
        直接把 ``a[0:3]`` 当世界系加速度，转弯时就会出现莫名其妙的侧滑。
        """
        cfg = self.cfg
        p_ref = np.array([*self.body_xy_ref, self.body_height])
        v_ref = np.array([*v_cmd_world, 0.0])
        a_des = cfg.body_kp * (p_ref - q[:3]) + cfg.body_kd * (v_ref - v_world)
        J = np.zeros((3, self.rbd.nv))
        J[:, 0:3] = R
        drift = R @ np.cross(v[3:6], v[0:3])
        return Task("body_pos", J, drift, a_des, cfg.body_weight)

    def _orientation_task(self, R, v, w_world, yaw_rate_cmd) -> list[Task]:
        """躯干角加速度任务，世界系：``ω̇_w = R a_{3:6}``（漂移为零）。"""
        cfg = self.cfg
        cy, sy = np.cos(self.yaw_ref), np.sin(self.yaw_ref)
        R_des = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        e = orientation_error(R_des, R)
        a_des = cfg.orient_kp * e + cfg.orient_kd * (np.array([0, 0, yaw_rate_cmd]) - w_world)
        J = np.zeros((3, self.rbd.nv))
        J[:, 3:6] = R
        return [Task("body_orient", J, np.zeros(3), a_des, cfg.orient_weight)]
