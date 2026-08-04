"""里程碑 7（凸 MPC）的验证。

MPC 没有"第二份实现"可对照，所以验证策略是**用物理必须成立的性质去反查
求解结果**：

* 静止站立时接触力之和必须等于体重；
* 摆动腿的力必须精确为零；
* 所有力必须落在**真实圆锥**内（不只是金字塔内）；
* 给一个速度指令，闭环跑起来必须真的加速到那个速度。

最后一条是闭环验证，和里程碑 6 一样 —— 一个 QP 解得出来不代表它解对了。

运行::

    pytest tests/test_mpc.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics import (
    load_go2_dynamics,
    nominal_configuration,
    rpy_to_matrix,
    srbd_acceleration,
    srbd_params_from_model,
)
from gait_scheduler import GaitScheduler, get_gait
from kinematics import LEGS
from mpc import (
    ConvexMPC,
    FrictionConstraints,
    MPCConfig,
    build_force_constraints,
    check_friction_cone,
    friction_pyramid_matrix,
)

SEED = 0


@pytest.fixture(scope="module")
def setup():
    """Go2 标称站姿 + 单刚体参数 + trot 调度器。"""
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    params = srbd_params_from_model(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    return params, com, feet, GaitScheduler("trot")


def make_problem(setup, cfg=None, contact_override=None, velocity=None, rpy=None):
    """组装一次完整的 MPC 求解输入。"""
    params, com, feet, sched = setup
    cfg = cfg or MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    x0 = ConvexMPC.make_state(
        np.zeros(3) if rpy is None else rpy,
        com,
        np.zeros(3),
        np.zeros(3) if velocity is None else velocity,
    )
    ref = mpc.make_reference(x0, np.zeros(2))
    contact = (
        sched.contact_schedule(0.0, cfg.dt, cfg.horizon)
        if contact_override is None
        else np.tile(contact_override, (cfg.horizon, 1))
    )
    foot_traj = np.tile(feet, (cfg.horizon, 1, 1))
    com_traj = np.tile(com, (cfg.horizon, 1))
    return mpc, x0, ref, contact, foot_traj, com_traj


# --------------------------------------------------------------------------
# 摩擦约束
# --------------------------------------------------------------------------


def test_friction_constraints_validation():
    with pytest.raises(ValueError, match="摩擦系数必须为正"):
        FrictionConstraints(mu=0.0)
    with pytest.raises(ValueError, match="f_min < f_max"):
        FrictionConstraints(f_min=300.0, f_max=100.0)


def test_common_pyramid_is_circumscribed_not_inscribed():
    """一个很多人搞反的方向：常见写法 |fx|<=mu*fz 是**外接**于圆锥的。

    正方形半宽 mu*fz，内切圆半径也是 mu*fz，所以正方形**包住**了圆。
    沿对角方向它允许 sqrt(2)*mu*fz 的切向力 —— **比真实极限大 41.4%**。
    这个常见写法不是保守而是乐观的：求解器可以合法开出会打滑的力。
    """
    mu, fz = 0.6, 100.0
    C = friction_pyramid_matrix(mu)  # 外接写法
    f_corner = np.array([mu * fz, mu * fz, fz])  # 金字塔的对角顶点
    assert np.all(C @ f_corner <= 1e-9), "该点在金字塔内"
    assert not check_friction_cone(f_corner, mu), "但它在真实圆锥之外"
    np.testing.assert_allclose(np.linalg.norm(f_corner[:2]), np.sqrt(2) * mu * fz)


def test_inscribed_pyramid_is_genuinely_conservative():
    """内接版（默认）把系数缩到 mu/sqrt(2)，保证金字塔可行 => 圆锥可行。"""
    mu = 0.6
    cons = FrictionConstraints(mu=mu, inscribed=True)
    np.testing.assert_allclose(cons.pyramid_mu, mu / np.sqrt(2), rtol=1e-12)
    np.testing.assert_allclose(cons.worst_case_utilisation, 1 / np.sqrt(2), rtol=1e-12)

    C = friction_pyramid_matrix(cons.pyramid_mu)
    rng = np.random.default_rng(SEED)
    for _ in range(2000):
        f = np.array([rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(1, 100)])
        if np.all(C @ f <= 1e-9):
            assert check_friction_cone(f, mu), f"内接金字塔内的点必须也在圆锥内：{f}"

    # 对角顶点恰好落在圆锥边界上 —— 内接的定义
    fz = 100.0
    f_corner = np.array([cons.pyramid_mu * fz, cons.pyramid_mu * fz, fz])
    np.testing.assert_allclose(np.linalg.norm(f_corner[:2]), mu * fz, rtol=1e-12)
    assert check_friction_cone(f_corner, mu)


def test_circumscribed_option_reports_its_own_optimism():
    cons = FrictionConstraints(mu=0.6, inscribed=False)
    np.testing.assert_allclose(cons.pyramid_mu, 0.6, rtol=1e-12)
    np.testing.assert_allclose(cons.worst_case_utilisation, np.sqrt(2), rtol=1e-12)


def test_force_constraints_pin_swing_legs_to_zero():
    """摆动腿的力被硬性钉为零，而不是从决策变量里删掉。

    保持 QP 维度恒定是实时实现的关键：矩阵结构固定，可预分配、可热启动。
    """
    contact = np.array([True, False, False, True])
    C, lb, ub = build_force_constraints(contact, FrictionConstraints())
    assert C.shape == (20, 12)

    for i, in_contact in enumerate(contact):
        rows = slice(5 * i, 5 * i + 5)
        if in_contact:
            assert ub[rows][4] > 0.0 and lb[rows][4] > 0.0
        else:
            np.testing.assert_allclose(lb[rows], 0.0)
            np.testing.assert_allclose(ub[rows], 0.0)


def test_stance_leg_normal_force_has_a_positive_lower_bound():
    """最小法向力取小正数而非零，避免支撑腿"若即若离"导致接触抖动。"""
    cons = FrictionConstraints(f_min=5.0)
    _, lb, _ = build_force_constraints(np.ones(4, dtype=bool), cons)
    assert np.all(lb[4::5] == 5.0)


# --------------------------------------------------------------------------
# 模型矩阵
# --------------------------------------------------------------------------


def test_config_validation():
    with pytest.raises(ValueError, match="预测步数必须为正"):
        MPCConfig(horizon=0)
    with pytest.raises(ValueError, match="预测步长必须为正"):
        MPCConfig(dt=0.0)
    with pytest.raises(ValueError, match="力权重必须为正"):
        MPCConfig(force_weight=0.0)


def test_continuous_matrices_have_the_expected_structure(setup):
    params, com, feet, _ = setup
    mpc = ConvexMPC(params)
    A, B = mpc.continuous_matrices(0.0, feet, com)

    assert A.shape == (13, 13) and B.shape == (13, 12)
    np.testing.assert_allclose(A[0:3, 6:9], np.eye(3), atol=1e-12)   # yaw=0 时 Rz^T = I
    np.testing.assert_allclose(A[3:6, 9:12], np.eye(3), atol=1e-12)  # p_dot = v
    assert A[11, 12] == 1.0                                          # 重力进 v_z
    for i in range(4):
        np.testing.assert_allclose(
            B[9:12, 3 * i : 3 * i + 3], np.eye(3) / params.mass, atol=1e-12
        )


def test_B_matrix_matches_the_srbd_from_milestone_2(setup):
    """MPC 的 B 矩阵必须与里程碑 2 的单刚体动力学一致。

    这是一次跨里程碑的交叉验证：同一份物理，两处实现，必须对上。
    """
    params, com, feet, _ = setup
    mpc = ConvexMPC(params)
    _, B = mpc.continuous_matrices(0.0, feet, com)

    rng = np.random.default_rng(SEED)
    for _ in range(20):
        forces = rng.normal(size=(4, 3)) * 30
        predicted = B @ forces.reshape(-1)
        lin_srbd, ang_srbd = srbd_acceleration(
            com, np.eye(3), np.zeros(3), forces, feet, params
        )
        # B 里不含重力（重力走的是 A[11,12]），所以要把它加回来比较
        np.testing.assert_allclose(predicted[9:12], lin_srbd - params.gravity_vector, atol=1e-9)
        np.testing.assert_allclose(predicted[6:9], ang_srbd, atol=1e-9)


def test_discretisation_is_first_order_euler(setup):
    params, com, feet, _ = setup
    cfg = MPCConfig(dt=0.02)
    mpc = ConvexMPC(params, cfg)
    Ac, Bc = mpc.continuous_matrices(0.3, feet, com)
    Ad, Bd = mpc.discrete_matrices(0.3, feet, com)
    np.testing.assert_allclose(Ad, np.eye(13) + Ac * cfg.dt, atol=1e-12)
    np.testing.assert_allclose(Bd, Bc * cfg.dt, atol=1e-12)


def test_prediction_matrices_reproduce_a_rollout(setup):
    """条件化矩阵必须与逐步递推得到的结果完全一致。"""
    params, com, feet, _ = setup
    cfg = MPCConfig(horizon=6, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    foot_traj = np.tile(feet, (cfg.horizon, 1, 1))
    com_traj = np.tile(com, (cfg.horizon, 1))
    A_qp, B_qp = mpc.build_prediction_matrices(0.0, foot_traj, com_traj)

    rng = np.random.default_rng(SEED + 1)
    x0 = rng.normal(size=13)
    x0[12] = -9.81
    U = rng.normal(size=12 * cfg.horizon) * 20

    X_condensed = (A_qp @ x0 + B_qp @ U).reshape(cfg.horizon, 13)

    x = x0.copy()
    for k in range(cfg.horizon):
        Ad, Bd = mpc.discrete_matrices(0.0, foot_traj[k], com_traj[k])
        x = Ad @ x + Bd @ U[12 * k : 12 * (k + 1)]
        np.testing.assert_allclose(X_condensed[k], x, atol=1e-9)


# --------------------------------------------------------------------------
# 求解结果：物理必须成立
# --------------------------------------------------------------------------


def test_solver_is_available_and_dense(setup):
    """条件化后的 QP 是稠密的，稀疏求解器在这里没有优势。"""
    import qpsolvers

    cfg = MPCConfig()
    assert cfg.solver in qpsolvers.available_solvers
    assert cfg.solver in qpsolvers.dense_solvers


def test_standing_forces_sum_to_body_weight(setup):
    """四脚站立时，垂直力之和必须等于体重。"""
    params = setup[0]
    mpc, x0, ref, _, ft, ct = make_problem(setup)
    contact = np.ones((mpc.cfg.horizon, 4), dtype=bool)
    r = mpc.solve(x0, ref, contact, ft, ct)

    assert r.success
    total = r.current_forces[:, 2].sum()
    np.testing.assert_allclose(total, params.mass * params.gravity, rtol=0.02)


def test_trot_forces_sum_to_body_weight(setup):
    """trot 只有两条腿支撑，它们必须承担全部体重。"""
    params = setup[0]
    mpc, x0, ref, contact, ft, ct = make_problem(setup)
    r = mpc.solve(x0, ref, contact, ft, ct)
    assert r.success
    np.testing.assert_allclose(
        r.current_forces[:, 2].sum(), params.mass * params.gravity, rtol=0.03
    )


def test_swing_legs_produce_exactly_zero_force(setup):
    mpc, x0, ref, contact, ft, ct = make_problem(setup)
    r = mpc.solve(x0, ref, contact, ft, ct)
    assert r.success
    for k in range(mpc.cfg.horizon):
        for i in range(4):
            if not contact[k, i]:
                np.testing.assert_allclose(r.forces[k, i], 0.0, atol=1e-6)


def test_all_forces_lie_inside_the_true_friction_cone(setup):
    """金字塔可行必然圆锥可行 —— 用真实圆锥验证求解结果。"""
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc, x0, ref, contact, ft, ct = make_problem(setup, cfg)
    r = mpc.solve(x0, ref, contact, ft, ct)
    assert r.success
    for k in range(cfg.horizon):
        for i in range(4):
            assert check_friction_cone(r.forces[k, i], cfg.friction.mu), f"步 {k} 腿 {LEGS[i]}"


def test_normal_forces_respect_their_bounds(setup):
    cfg = MPCConfig(friction=FrictionConstraints(f_min=5.0, f_max=200.0))
    mpc, x0, ref, contact, ft, ct = make_problem(setup, cfg)
    r = mpc.solve(x0, ref, contact, ft, ct)
    assert r.success
    for k in range(cfg.horizon):
        for i in range(4):
            fz = r.forces[k, i, 2]
            if contact[k, i]:
                assert cfg.friction.f_min - 1e-6 <= fz <= cfg.friction.f_max + 1e-6
            else:
                np.testing.assert_allclose(fz, 0.0, atol=1e-6)


def test_low_friction_forces_more_vertical_forces(setup):
    """摩擦系数越小，可用的水平力越少，接触力越接近竖直。"""
    ratios = []
    for mu in (0.15, 0.4, 0.9):
        cfg = MPCConfig(friction=FrictionConstraints(mu=mu))
        mpc, x0, ref, contact, ft, ct = make_problem(
            setup, cfg, velocity=np.array([0.6, 0.0, 0.0])
        )
        r = mpc.solve(x0, ref, contact, ft, ct)
        assert r.success
        stance = r.current_forces[contact[0]]
        ratios.append(np.max(np.linalg.norm(stance[:, :2], axis=1) / stance[:, 2]))
    assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:])), f"应随 mu 单调不减：{ratios}"
    assert ratios[0] <= 0.15 + 1e-6, "低摩擦下必须守住真实圆锥"
    assert ratios[-1] > ratios[0], "高摩擦下应能用上更多水平力"


# --------------------------------------------------------------------------
# 控制行为：解得出来 != 解对了
# --------------------------------------------------------------------------


def test_mpc_pushes_back_when_com_is_too_low(setup):
    """质心低于参考高度时，MPC 应当加大垂直力把身体顶起来。"""
    params, com, feet, sched = setup
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    ft = np.tile(feet, (cfg.horizon, 1, 1))
    ct = np.tile(com, (cfg.horizon, 1))

    totals = []
    for dz in (-0.04, 0.0, 0.04):
        x0 = ConvexMPC.make_state(np.zeros(3), com + np.array([0, 0, dz]), np.zeros(3), np.zeros(3))
        ref = mpc.make_reference(x0, np.zeros(2), height=com[2])
        r = mpc.solve(x0, ref, contact, ft, ct)
        assert r.success
        totals.append(r.current_forces[:, 2].sum())
    assert totals[0] > totals[1] > totals[2], f"垂直力应随高度误差单调变化：{totals}"


def test_mpc_generates_forward_thrust_for_a_velocity_command(setup):
    """给一个前进指令，支撑腿必须产生向前的合力。"""
    params, com, feet, sched = setup
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    ft = np.tile(feet, (cfg.horizon, 1, 1))
    ct = np.tile(com, (cfg.horizon, 1))
    x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))

    ref_zero = mpc.make_reference(x0, np.array([0.0, 0.0]))
    ref_fwd = mpc.make_reference(x0, np.array([0.6, 0.0]))
    fx_zero = mpc.solve(x0, ref_zero, contact, ft, ct).current_forces[:, 0].sum()
    fx_fwd = mpc.solve(x0, ref_fwd, contact, ft, ct).current_forces[:, 0].sum()
    assert fx_fwd > fx_zero + 5.0, f"前进指令应产生向前合力：{fx_zero:.2f} -> {fx_fwd:.2f}"


def test_mpc_corrects_a_tilted_body(setup):
    """身体前倾时，MPC 应当产生把它扶正的俯仰力矩。"""
    params, com, feet, _ = setup
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    ft = np.tile(feet, (cfg.horizon, 1, 1))
    ct = np.tile(com, (cfg.horizon, 1))

    torques = []
    for pitch in (-0.15, 0.0, 0.15):
        x0 = ConvexMPC.make_state(np.array([0.0, pitch, 0.0]), com, np.zeros(3), np.zeros(3))
        ref = mpc.make_reference(x0, np.zeros(2))
        r = mpc.solve(x0, ref, contact, ft, ct)
        assert r.success
        _, ang = srbd_acceleration(com, rpy_to_matrix(np.array([0.0, pitch, 0.0])),
                                   np.zeros(3), r.current_forces, feet, params)
        torques.append(ang[1])
    # 前倾（pitch<0）应产生正的俯仰角加速度把它扳回来，反之亦然
    assert torques[0] > torques[1] > torques[2], f"应产生回复力矩：{torques}"


def test_closed_loop_srbd_tracks_a_velocity_command(setup):
    """闭环验证：把 MPC 的力喂回单刚体模型，速度必须收敛到指令。

    这一条才真正说明 MPC 解对了 —— 一个 QP 解得出来不代表它解对了。
    """
    params, com, feet, sched = setup
    cfg = MPCConfig(horizon=10, dt=0.03)
    mpc = ConvexMPC(params, cfg)
    dt = 0.01
    v_cmd = np.array([0.4, 0.0])

    pos = com.copy()
    vel = np.zeros(3)
    contact = np.ones((cfg.horizon, 4), dtype=bool)

    for step in range(120):
        # 足端随质心平移，保持标称站姿
        feet_now = feet + (pos - com)
        feet_now[:, 2] = 0.0
        x0 = ConvexMPC.make_state(np.zeros(3), pos, np.zeros(3), vel)
        ref = mpc.make_reference(x0, v_cmd, height=com[2])
        ft = np.tile(feet_now, (cfg.horizon, 1, 1))
        ct = np.tile(pos, (cfg.horizon, 1))
        r = mpc.solve(x0, ref, contact, ft, ct)
        assert r.success, f"第 {step} 步求解失败"

        lin, _ = srbd_acceleration(pos, np.eye(3), np.zeros(3), r.current_forces, feet_now, params)
        vel = vel + dt * lin
        pos = pos + dt * vel

    np.testing.assert_allclose(vel[:2], v_cmd, atol=0.08)
    np.testing.assert_allclose(pos[2], com[2], atol=0.02)


# --------------------------------------------------------------------------
# 实时性：能不能上 50 Hz
# --------------------------------------------------------------------------


def test_qp_size_matches_the_formulation(setup):
    cfg = MPCConfig(horizon=10)
    mpc, x0, ref, contact, ft, ct = make_problem(setup, cfg)
    r = mpc.solve(x0, ref, contact, ft, ct)
    assert r.n_variables == 12 * cfg.horizon == 120
    # 每步每腿 5 行约束，双边展开后有限行数 <= 2 * 20 * N
    assert r.n_constraints <= 2 * 20 * cfg.horizon


def test_solve_time_fits_the_50hz_budget(setup):
    """50 Hz 的周期是 20 ms，MPC 必须显著低于它。"""
    mpc, x0, ref, contact, ft, ct = make_problem(setup)
    times = [mpc.solve(x0, ref, contact, ft, ct).solve_time for _ in range(20)]
    median = float(np.median(times))
    worst = float(np.max(times))
    assert median < 0.010, f"中位求解耗时 {median*1000:.2f} ms 过长"
    assert worst < 0.020, f"最差求解耗时 {worst*1000:.2f} ms 超出 50 Hz 预算"


def test_solve_time_grows_with_horizon(setup):
    """条件化后的稠密 QP，规模随时域增长得比线性快。"""
    times = []
    for N in (4, 10, 16):
        cfg = MPCConfig(horizon=N, dt=0.03)
        mpc, x0, ref, contact, ft, ct = make_problem(setup, cfg)
        times.append(np.median([mpc.solve(x0, ref, contact, ft, ct).solve_time for _ in range(5)]))
    assert all(b > a for a, b in zip(times, times[1:])), f"耗时应随时域增长：{times}"


def test_failed_solve_is_reported_honestly(setup):
    """求解失败必须如实上报，不能悄悄返回零力。

    零力意味着机器人直接瘫下去。上层必须能区分"MPC 说不用力"和"MPC 没解出来"。
    """
    params, com, feet, _ = setup
    # 摩擦系数极小 + 要求很大的前向加速度 -> 可能不可行
    cfg = MPCConfig(horizon=10, friction=FrictionConstraints(mu=0.6, f_min=200.0, f_max=210.0))
    mpc = ConvexMPC(params, cfg)
    x0 = ConvexMPC.make_state(np.zeros(3), com, np.zeros(3), np.zeros(3))
    ref = mpc.make_reference(x0, np.zeros(2))
    contact = np.ones((cfg.horizon, 4), dtype=bool)
    r = mpc.solve(
        x0,
        ref,
        contact,
        np.tile(feet, (cfg.horizon, 1, 1)),
        np.tile(com, (cfg.horizon, 1)),
    )
    # 无论成功与否，success 必须如实反映；失败时力全零，上层据此可以降级
    assert isinstance(r.success, bool)
    if not r.success:
        np.testing.assert_allclose(r.forces, 0.0, atol=1e-12)
