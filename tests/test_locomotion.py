"""里程碑 11：MPC + WBC 在 MuJoCo 全身物理仿真里的闭环，以及足端力估计。

M7 只在单刚体模型上闭环过（被控对象 = 预测模型，误差天然为零），M8 只做过
单步物理判据。这里的每一条都是**在另一个物理引擎里**才成立的结论。

运行::

    pytest tests/test_locomotion.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

pytest.importorskip("mujoco")

from dynamics import NOMINAL_BASE_HEIGHT, load_go2_dynamics, nominal_configuration  # noqa: E402
from kinematics import LEGS  # noqa: E402
from locomotion import LocomotionConfig, LocomotionController  # noqa: E402
from sim import GO2_MJCF, MujocoGo2, Push, SimConfig, run_episode  # noqa: E402
from state_estimator import ContactDetector, ContactDetectorConfig, MomentumObserver  # noqa: E402

pytestmark = pytest.mark.skipif(not GO2_MJCF.is_file(), reason=f"找不到 {GO2_MJCF}")


@pytest.fixture(scope="module")
def rbd():
    return load_go2_dynamics(floating_base=True)


@pytest.fixture(scope="module")
def q0(rbd):
    q = nominal_configuration(rbd)
    q[2] = NOMINAL_BASE_HEIGHT + 0.022
    return q


def walk(rbd, q0, vx, duration, inertia="urdf", gait="trot", pushes=None):
    return run_episode(
        MujocoGo2(SimConfig(inertia=inertia)),
        LocomotionController(rbd, LocomotionConfig(gait=gait)),
        q0, duration, lambda t: np.array([vx if t >= 0.5 else 0.0, 0.0, 0.0]), pushes,
    )


@pytest.fixture(scope="module")
def trot_log(rbd, q0):
    return walk(rbd, q0, 0.5, 4.0, inertia="mjcf")


# -- 坐标系 -----------------------------------------------------------------------


def test_body_task_is_world_frame_classical_acceleration(rbd):
    """躯干任务 ``R (a_lin + ω×v)`` 必须等于 Pinocchio 算的世界系经典加速度。"""
    ctl = LocomotionController(rbd)
    pm, pd = rbd.model.model, rbd.model.data
    rng = np.random.default_rng(0)
    base = pm.getFrameId("base") if pm.existFrame("base") else pm.getFrameId("base_link")
    for _ in range(10):
        q = pin.randomConfiguration(pm, -np.ones(pm.nq), np.ones(pm.nq))
        q[3:7] /= np.linalg.norm(q[3:7])
        v, a = rng.normal(size=18), rng.normal(size=18)
        R, v_world, w_world = ctl._base_state(q, v)
        ctl.body_xy_ref = q[:2].copy()
        task = ctl._body_task(q, v, R, v_world, w_world, np.zeros(2), 0.0)
        pin.forwardKinematics(pm, pd, q, v, a)
        pin.updateFramePlacements(pm, pd)
        ref = pin.getFrameClassicalAcceleration(pm, pd, base, pin.LOCAL_WORLD_ALIGNED).linear
        np.testing.assert_allclose(task.jacobian @ a + task.drift, ref, atol=1e-10)


# -- 闭环 -------------------------------------------------------------------------


def test_stands_still_with_full_stack(rbd, q0):
    log = walk(rbd, q0, 0.0, 2.0, gait="stand")
    assert not log.fell
    assert np.abs(log.base_pos[-1, :2] - q0[:2]).max() < 0.01
    assert abs(log.base_pos[-1, 2] - q0[2]) < 0.01


def test_standing_forces_carry_body_weight(rbd, q0):
    log = walk(rbd, q0, 0.0, 1.5, gait="stand")
    total = log.contact_force_true[-200:].sum(axis=1)[:, 2].mean()
    assert total == pytest.approx(log.mass * 9.81, rel=0.01)


def test_trots_forward_under_model_mismatch(trot_log):
    """官方 MJCF 比控制器模型轻 5.5%，0.5 m/s 仍要跟住。"""
    assert not trot_log.fell
    w = trot_log.window(2.0, 4.0)
    assert trot_log.body_velocity()[w, 0].mean() == pytest.approx(0.5, abs=0.05)
    assert trot_log.velocity_rms(2.0, 4.0) < 0.08
    assert trot_log.base_pos[w, 2].std() < 0.01


def test_mpc_and_wbc_meet_their_budgets(trot_log):
    assert np.median(trot_log.mpc_time) < 0.02  # 50 Hz
    assert np.median(trot_log.wbc_time) < 0.001  # 1 kHz


def test_torques_respect_actuator_limits(trot_log):
    assert np.all(np.abs(trot_log.torque) <= np.array([23.7, 23.7, 45.43] * 4) + 1e-9)


def test_recovers_from_lateral_push(rbd, q0):
    log = walk(rbd, q0, 0.5, 4.0, pushes=[Push(2.0, 0.1, np.array([0.0, 100.0, 0.0]))])  # 10 N·s
    assert not log.fell


# -- 足端力估计 -------------------------------------------------------------------


def replay(rbd, log, **kw):
    obs = MomentumObserver(rbd, **kw)
    obs.reset(log.q[0], log.v[0])
    est = np.zeros((len(log.t), 4, 3))
    for k in range(len(log.t) - 1):
        obs.update(log.q[k + 1], log.v[k + 1], log.torque[k], 0.001)
        est[k + 1] = obs.foot_forces(log.q[k + 1])
    return est


def test_observer_residual_is_zero_in_free_fall(rbd):
    """没有外力时残差必须为零 —— 否则就是模型写错了。"""
    q = nominal_configuration(rbd)
    q[2] = 2.0
    v = np.zeros(18)
    v[6:] = np.random.default_rng(0).normal(size=12)
    obs = MomentumObserver(rbd, gain=100.0)
    obs.reset(q, v)
    tau = np.zeros(12)
    for _ in range(200):  # 用 Pinocchio 自己积分，排除仿真器差异
        a = rbd.forward_dynamics(q, v, np.concatenate([np.zeros(6), tau]))
        v = v + a * 0.001
        q = pin.integrate(rbd.model.model, q, v * 0.001)
        r = obs.update(q, v, tau, 0.001)
    assert np.abs(r).max() < 0.5  # 一阶积分误差量级


def test_observer_recovers_body_weight_when_standing(rbd, q0):
    log = walk(rbd, q0, 0.0, 1.5, gait="stand")
    est = replay(rbd, log, gain=100.0)
    tail = slice(-300, None)
    assert est[tail, :, 2].sum(axis=1).mean() == pytest.approx(log.mass * 9.81, rel=0.02)
    np.testing.assert_allclose(est[tail, :, 2].mean(axis=0), log.contact_force_true[tail, :, 2].mean(axis=0),
                               atol=3.0)


def test_armature_compensation_reduces_error(rbd, trot_log):
    """转子惯量是电机手册参数；不补，摆动腿减速会被读成接触力。"""
    truth = trot_log.contact_force_true[:, :, 2]
    lp = np.zeros_like(truth)
    for k in range(1, len(truth)):
        lp[k] = lp[k - 1] + 0.1 * (truth[k] - lp[k - 1])
    w = trot_log.window(1.0, 4.0)
    errs = {}
    for arm in (0.0, 0.01):
        est = replay(rbd, trot_log, gain=100.0, joint_armature=arm)
        errs[arm] = np.sqrt(np.mean((est[w, :, 2] - lp[w]) ** 2))
    assert errs[0.01] < 0.8 * errs[0.0]
    assert errs[0.01] < 8.0


def test_estimated_contact_beats_the_schedule(rbd, trot_log):
    """估计接触比步态计划更接近真实接触 —— 这才是估计器存在的理由。"""
    est = replay(rbd, trot_log, gain=100.0, joint_armature=0.01)
    det = ContactDetector()
    detected = np.array([det.update(f[:, 2]) for f in est])
    truth = trot_log.contact_force_true[:, :, 2] > 1.0
    w = trot_log.window(1.0, 4.0)
    acc_est = np.mean(detected[w] == truth[w])
    acc_sched = np.mean(trot_log.stance[w] == truth[w])
    assert acc_est > 0.95
    assert acc_est > acc_sched


def test_contact_detector_hysteresis():
    det = ContactDetector(ContactDetectorConfig(on=25.0, off=10.0))
    det.state[:] = False
    seq = [5, 20, 30, 20, 12, 8, 20]
    out = [bool(det.update(np.full(4, f))[0]) for f in seq]
    assert out == [False, False, True, True, True, False, False]
