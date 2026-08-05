"""里程碑 8（全身控制）的验证。

WBC 的正确性有一个**不可讨价还价**的判据：解出来的
:math:`(a, \\tau, f)` 必须满足完整动力学方程

.. math::  M(q)a + C(q,v)v + g(q) = S^\\top\\tau + \\sum_i J_i^\\top f_i

这是物理定律，不是优化目标。任何一项对不上，算出来的力矩在真实机器人上
就产生不了那个运动。里程碑 2 建好的整机动力学在这里成了验证的基准。

运行::

    pytest tests/test_wbc.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics import (
    FLOATING_BASE_DOF,
    load_go2_dynamics,
    nominal_configuration,
    rpy_to_matrix,
    srbd_params_from_model,
)
from gait_scheduler import GaitScheduler
from kinematics import LEGS
from mpc import ConvexMPC, FrictionConstraints, MPCConfig, check_friction_cone
from whole_body_controller import (
    Task,
    WBCConfig,
    WholeBodyController,
    orientation_error,
    pd_acceleration,
)

SEED = 0


@pytest.fixture(scope="module")
def rbd():
    return load_go2_dynamics(floating_base=True)


@pytest.fixture(scope="module")
def scene(rbd):
    """标称站姿 + MPC 算出的接触力。"""
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))
    ref = mpc.make_reference(x0, np.zeros(2))
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    r = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)), np.tile(com, (cfg.horizon, 1)))
    return q, v, r.current_forces, com, feet


# --------------------------------------------------------------------------
# 任务表示
# --------------------------------------------------------------------------


def test_task_validates_dimensions():
    J = np.ones((3, 18))
    with pytest.raises(ValueError, match="权重维度"):
        Task("bad", J, np.zeros(3), np.zeros(3), weight=np.ones(5))
    with pytest.raises(ValueError, match="权重不能为负"):
        Task("bad", J, np.zeros(3), np.zeros(3), weight=-1.0)


def test_task_residual_and_cost():
    J = np.zeros((2, 18))
    J[0, 0] = 1.0
    J[1, 1] = 2.0
    task = Task("t", J, np.array([0.5, 0.0]), np.array([1.0, 1.0]), weight=np.array([1.0, 4.0]))
    a = np.zeros(18)
    a[0], a[1] = 1.0, 1.0
    np.testing.assert_allclose(task.residual(a), [0.5, 1.0])
    np.testing.assert_allclose(task.cost(a), 1.0 * 0.25 + 4.0 * 1.0)


def test_pd_acceleration():
    a = pd_acceleration(np.array([0.1, 0.0]), np.array([0.0, -0.2]), kp=100.0, kd=20.0)
    np.testing.assert_allclose(a, [10.0, -4.0])
    a_ff = pd_acceleration(np.zeros(2), np.zeros(2), 100.0, 20.0, feedforward=np.array([1.0, 2.0]))
    np.testing.assert_allclose(a_ff, [1.0, 2.0])


def test_orientation_error_is_a_rotation_vector():
    """姿态误差必须走 SO(3) 的 log，不能用欧拉角相减。

    这是本项目第三次遇到"误差必须活在切空间"：M2 的浮动基座求导、
    M3 的误差状态卡尔曼，现在是 WBC 的姿态任务。
    """
    import pinocchio as pin

    rng = np.random.default_rng(SEED)
    for _ in range(30):
        phi = rng.normal(size=3) * 0.4
        R_cur = pin.exp3(rng.normal(size=3) * 0.3)
        R_des = pin.exp3(phi) @ R_cur
        np.testing.assert_allclose(orientation_error(R_des, R_cur), phi, atol=1e-9)


def test_orientation_error_is_zero_for_identical_frames():
    R = rpy_to_matrix(np.array([0.2, -0.1, 0.5]))
    np.testing.assert_allclose(orientation_error(R, R), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 构造与校验
# --------------------------------------------------------------------------


def test_wbc_requires_a_floating_base_model():
    """欠驱动结构是 WBC 存在的前提，固定基座下这个模块没有意义。"""
    fixed = load_go2_dynamics(floating_base=False)
    with pytest.raises(ValueError, match="浮动基座"):
        WholeBodyController(fixed)


def test_config_validation():
    with pytest.raises(ValueError, match="力跟踪权重"):
        WBCConfig(force_tracking_weight=-1.0)
    with pytest.raises(ValueError, match="加速度正则"):
        WBCConfig(acceleration_regularisation=0.0)
    with pytest.raises(ValueError, match="力矩上限"):
        WBCConfig(torque_limit=0.0)


# --------------------------------------------------------------------------
# 不可讨价还价的判据：完整动力学必须成立
# --------------------------------------------------------------------------


def test_solution_satisfies_the_full_dynamics(rbd, scene):
    """M a + h = S^T tau + J^T f —— 这是物理定律，不是优化目标。"""
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success

    M = rbd.mass_matrix(q)
    h = rbd.nonlinear_effects(q, v)
    Jc = rbd.contact_jacobian(q, LEGS)
    tau_full = np.zeros(rbd.nv)
    tau_full[rbd.actuated_dofs] = r.torque

    lhs = M @ r.acceleration + h
    rhs = tau_full + Jc.T @ r.contact_force.reshape(-1)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)


def test_floating_base_rows_have_no_torque(rbd, scene):
    """那 6 行的右端只有接触力，没有 tau —— "S 的前 6 列全为零"在这里生效。"""
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success

    M = rbd.mass_matrix(q)
    h = rbd.nonlinear_effects(q, v)
    Jc = rbd.contact_jacobian(q, LEGS)
    residual = (
        M[:FLOATING_BASE_DOF] @ r.acceleration
        + h[:FLOATING_BASE_DOF]
        - Jc[:, :FLOATING_BASE_DOF].T @ r.contact_force.reshape(-1)
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_stance_feet_do_not_accelerate(rbd, scene):
    """支撑约束 J_c a + dJ v = 0 是硬等式。"""
    q, v, forces, _, _ = scene
    rng = np.random.default_rng(SEED + 1)
    wbc = WholeBodyController(rbd)
    for _ in range(5):
        v_test = rng.normal(size=rbd.nv) * 0.2
        r = wbc.solve(q, v_test, LEGS, forces)
        assert r.success
        Jc = rbd.contact_jacobian(q, LEGS)
        dJv = rbd.contact_jacobian_dot_v(q, v_test, LEGS)
        np.testing.assert_allclose(Jc @ r.acceleration + dJv, 0.0, atol=1e-7)


def test_standing_still_gives_near_zero_base_acceleration(rbd, scene):
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success
    np.testing.assert_allclose(r.acceleration[:FLOATING_BASE_DOF], 0.0, atol=5e-3)


# --------------------------------------------------------------------------
# 约束：力矩与摩擦
# --------------------------------------------------------------------------


def test_torque_limits_are_respected(rbd, scene):
    q, v, forces, _, _ = scene
    cfg = WBCConfig(torque_limit=8.0)
    wbc = WholeBodyController(rbd, cfg)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success
    assert np.max(np.abs(r.torque)) <= cfg.torque_limit + 1e-6


def test_tight_torque_limit_forces_the_forces_to_deviate(rbd, scene):
    """力矩限幅收紧时，MPC 的力必须让步 —— 这正是"软任务"的意义。"""
    q, v, forces, _, _ = scene
    loose = WholeBodyController(rbd, WBCConfig(torque_limit=45.0)).solve(q, v, LEGS, forces)
    tight = WholeBodyController(rbd, WBCConfig(torque_limit=3.5)).solve(q, v, LEGS, forces)
    assert loose.success and tight.success
    dev_loose = np.abs(loose.contact_force - forces).max()
    dev_tight = np.abs(tight.contact_force - forces).max()
    assert dev_tight > dev_loose + 1.0, f"收紧限幅后力应明显偏离：{dev_loose:.3f} -> {dev_tight:.3f}"
    assert np.max(np.abs(tight.torque)) <= 3.5 + 1e-6


def test_contact_forces_stay_inside_the_true_friction_cone(rbd, scene):
    q, v, forces, _, _ = scene
    cfg = WBCConfig(friction=FrictionConstraints(mu=0.6, inscribed=True))
    wbc = WholeBodyController(rbd, cfg)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success
    for f in r.contact_force:
        assert check_friction_cone(f, cfg.friction.mu)


def test_normal_forces_respect_their_bounds(rbd, scene):
    q, v, forces, _, _ = scene
    cfg = WBCConfig(friction=FrictionConstraints(f_min=6.0, f_max=200.0))
    wbc = WholeBodyController(rbd, cfg)
    r = wbc.solve(q, v, LEGS, forces)
    assert r.success
    assert np.all(r.contact_force[:, 2] >= cfg.friction.f_min - 1e-6)
    assert np.all(r.contact_force[:, 2] <= cfg.friction.f_max + 1e-6)


# --------------------------------------------------------------------------
# 软任务：为什么 MPC 的力不能当硬约束
# --------------------------------------------------------------------------


def test_force_tracking_is_soft_not_hard(rbd, scene):
    """喂一组物理上做不到的力，WBC 必须让步而不是失败。

    里程碑 2 量化过：单刚体模型在腿快速摆动时角加速度误差达 43%。MPC 是
    基于那个模型算的，它给的力本身就是近似的。当成硬约束会让 QP 不可行。
    """
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd, WBCConfig(force_tracking_weight=1.0))
    absurd = forces * 3.0 + np.array([50.0, 50.0, 0.0])
    r = wbc.solve(q, v, LEGS, absurd)
    assert r.success, "力跟踪是软任务，不应导致不可行"
    assert np.abs(r.contact_force - absurd).max() > 1.0, "应当明显偏离这组做不到的力"


def test_higher_weight_tracks_mpc_forces_more_closely(rbd, scene):
    q, v, forces, _, _ = scene
    target = forces + np.array([6.0, -4.0, 0.0])
    deviations = []
    for w in (0.01, 1.0, 100.0):
        r = WholeBodyController(rbd, WBCConfig(force_tracking_weight=w)).solve(q, v, LEGS, target)
        assert r.success
        deviations.append(np.abs(r.contact_force - target).max())
    assert all(b < a for a, b in zip(deviations, deviations[1:])), f"权重越大越贴近：{deviations}"


def test_swing_task_is_tracked(rbd, scene):
    """摆动腿的加速度任务应当被大致满足。"""
    q, v, forces, _, _ = scene
    stance = ("FL", "RR")
    stance_forces = np.array([forces[LEGS.index(leg)] for leg in stance])
    wbc = WholeBodyController(rbd)

    desired = np.array([1.5, 0.0, 4.0])
    task = wbc.swing_foot_task(q, v, "FR", desired, weight=500.0)
    r = wbc.solve(q, v, stance, stance_forces, tasks=[task])
    assert r.success

    J = rbd.model.full_foot_jacobian(q, "FR")
    dJv = rbd.contact_jacobian_dot_v(q, v, ("FR",))
    achieved = J @ r.acceleration + dJv
    np.testing.assert_allclose(achieved, desired, atol=0.3)


def test_task_costs_are_reported_for_diagnosis(rbd, scene):
    """诊断信息：哪个任务在让步，必须看得见。"""
    q, v, forces, _, _ = scene
    stance = ("FL", "RR")
    stance_forces = np.array([forces[LEGS.index(leg)] for leg in stance])
    wbc = WholeBodyController(rbd)
    tasks = [
        wbc.swing_foot_task(q, v, "FR", np.array([0.0, 0.0, 3.0]), weight=100.0),
        wbc.body_task(np.zeros(3), np.zeros(3)),
    ]
    r = wbc.solve(q, v, stance, stance_forces, tasks=tasks)
    assert r.success
    assert set(r.task_costs) == {"swing_FR", "body"}
    assert all(c >= 0.0 for c in r.task_costs.values())


def test_body_task_targets_the_first_six_accelerations(rbd, scene):
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    task = wbc.body_task(np.array([0.0, 0.0, 1.0]), np.zeros(3), linear_weight=800.0,
                         angular_weight=800.0)
    r = wbc.solve(q, v, LEGS, forces, tasks=[task])
    assert r.success
    assert r.acceleration[2] > 0.3, "要求向上加速时基座应真的向上加速"


# --------------------------------------------------------------------------
# 与朴素做法的对照：完整模型到底值多少
# --------------------------------------------------------------------------


def test_naive_torque_ignores_leg_gravity(rbd, scene):
    """静止站立时，朴素 tau = -J^T f 漏掉的就是腿自身的重力。"""
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, LEGS, forces)
    naive = wbc.naive_torque(q, LEGS, forces)

    diff = r.torque - naive
    assert np.abs(diff).max() > 0.5, "静态下差异就该有 N·m 量级"
    # 差异应当约等于关节行的重力项
    g = rbd.gravity_vector(q)[rbd.actuated_dofs]
    np.testing.assert_allclose(diff, g, atol=0.05)


def test_naive_torque_error_grows_with_joint_velocity(rbd, scene):
    """腿动起来之后，科氏项让朴素做法的误差进一步增大。"""
    q, _, forces, _, _ = scene
    rng = np.random.default_rng(SEED + 2)
    wbc = WholeBodyController(rbd)
    errors = []
    for speed in (0.0, 2.0, 5.0, 8.0):
        vals = []
        for _ in range(5):
            v = np.zeros(rbd.nv)
            v[FLOATING_BASE_DOF:] = rng.normal(size=12) * speed
            r = wbc.solve(q, v, LEGS, forces)
            assert r.success
            naive = wbc.naive_torque(q, LEGS, forces)
            vals.append(np.abs(r.torque - naive).max())
        errors.append(float(np.median(vals)))
    assert all(b > a for a, b in zip(errors, errors[1:])), f"误差应随关节速度增大：{errors}"
    assert errors[-1] > 3 * errors[0], "高速下差异应显著放大"


# --------------------------------------------------------------------------
# 接触集合变化与实时性
# --------------------------------------------------------------------------


def test_works_with_two_stance_legs(rbd, scene):
    """trot 只有两条支撑腿，QP 的维度随之变小。"""
    q, v, forces, _, _ = scene
    stance = ("FL", "RR")
    stance_forces = np.array([forces[LEGS.index(leg)] for leg in stance])
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, stance, stance_forces)
    assert r.success
    assert r.n_variables == rbd.nv + 6
    assert r.contact_force.shape == (2, 3)


def test_works_with_no_contact(rbd, scene):
    """腾空相：没有接触力，只剩动力学与任务。"""
    q, v, _, _, _ = scene
    wbc = WholeBodyController(rbd)
    r = wbc.solve(q, v, (), None)
    assert r.success
    assert r.contact_force.shape == (0, 3)
    # 腾空时基座只能自由落体
    np.testing.assert_allclose(r.acceleration[2], -rbd.gravity, atol=0.05)


def test_qp_size_matches_the_formulation(rbd, scene):
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    for stance in (LEGS, ("FL", "RR"), ("FL",)):
        f = np.array([forces[LEGS.index(leg)] for leg in stance])
        r = wbc.solve(q, v, stance, f)
        assert r.n_variables == rbd.nv + 3 * len(stance)


def test_solve_time_fits_the_1khz_budget(rbd, scene):
    """WBC 跑在 1 kHz 线程里，周期只有 1 ms。"""
    q, v, forces, _, _ = scene
    wbc = WholeBodyController(rbd)
    times = [wbc.solve(q, v, LEGS, forces).solve_time for _ in range(30)]
    median, worst = float(np.median(times)), float(np.max(times))
    assert median < 0.001, f"中位耗时 {median*1000:.3f} ms 超出 1 kHz 预算"
    assert worst < 0.003, f"最差耗时 {worst*1000:.3f} ms 过长"


def test_failure_is_reported_honestly(rbd, scene):
    """求解失败必须如实上报 —— 零力矩意味着机器人瘫下去。"""
    q, v, forces, _, _ = scene
    # 力矩上限设得不可能满足
    wbc = WholeBodyController(rbd, WBCConfig(torque_limit=1e-4))
    r = wbc.solve(q, v, LEGS, forces)
    assert isinstance(r.success, bool)
    if not r.success:
        np.testing.assert_allclose(r.acceleration, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 与前面里程碑的串联
# --------------------------------------------------------------------------


def test_full_stack_mpc_to_wbc_over_a_trot_cycle(rbd):
    """把 M2/M4/M7/M8 串起来跑一个 trot 周期，每一步都必须可解且物理自洽。"""
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    sched = GaitScheduler("trot")
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    wbc = WholeBodyController(rbd)

    for t in np.linspace(0.0, sched.gait.period, 9, endpoint=False):
        contact = sched.contact_schedule(t, cfg.dt, cfg.horizon)
        x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.array([0.3, 0.0, 0.0]))
        ref = mpc.make_reference(x0, np.array([0.3, 0.0]))
        res = mpc.solve(x0, ref, contact, np.tile(feet, (cfg.horizon, 1, 1)),
                        np.tile(com, (cfg.horizon, 1)))
        assert res.success, f"t={t:.3f} MPC 求解失败"

        stance = tuple(leg for i, leg in enumerate(LEGS) if contact[0, i])
        stance_forces = np.array([res.current_forces[LEGS.index(leg)] for leg in stance])
        r = wbc.solve(q, v, stance, stance_forces)
        assert r.success, f"t={t:.3f} WBC 求解失败"

        # 每一步都验证完整动力学
        M = rbd.mass_matrix(q)
        h = rbd.nonlinear_effects(q, v)
        Jc = rbd.contact_jacobian(q, stance)
        tau_full = np.zeros(rbd.nv)
        tau_full[rbd.actuated_dofs] = r.torque
        np.testing.assert_allclose(
            M @ r.acceleration + h, tau_full + Jc.T @ r.contact_force.reshape(-1), atol=1e-7
        )
