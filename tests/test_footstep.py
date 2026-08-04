"""里程碑 6（落脚点规划器）的验证。

倒立摆是解析可解的，所以又回到"两条互不相干的路径必须一致"：解析解 vs
数值积分、捕获点定义 vs 闭环收敛行为。

此外还有一类前几个里程碑没有的测试：**闭环行为验证**。落脚点策略的正确
与否，最终要看"把它放进倒立摆里跑，机器人会不会真的收敛到指令速度"。

运行::

    pytest tests/test_footstep.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from footstep_planner import (
    FootstepPlanner,
    FootstepPlannerConfig,
    LIPMParams,
    capture_point,
    capture_point_footstep,
    lipm_step,
    optimal_feedback_gain,
    raibert_footstep,
    simulate_lipm,
    time_to_boundary,
)
from gait_scheduler import GaitScheduler, get_gait
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS

HEIGHT = 0.30
PARAMS = LIPMParams(height=HEIGHT)
GRAVITY = 9.81


# --------------------------------------------------------------------------
# 线性倒立摆
# --------------------------------------------------------------------------


def test_omega_and_time_constant():
    np.testing.assert_allclose(PARAMS.omega, np.sqrt(GRAVITY / HEIGHT), rtol=1e-12)
    np.testing.assert_allclose(PARAMS.time_constant, 1.0 / PARAMS.omega, rtol=1e-12)
    np.testing.assert_allclose(PARAMS.omega, 5.718, atol=1e-3)
    np.testing.assert_allclose(PARAMS.time_constant, 0.1749, atol=1e-4)


def test_lipm_rejects_nonpositive_height():
    with pytest.raises(ValueError, match="质心高度必须为正"):
        LIPMParams(height=0.0)


def test_taller_robots_fall_more_slowly():
    """时间常数正比于 sqrt(h) —— 个子高的机器人有更多时间反应。"""
    constants = [LIPMParams(height=h).time_constant for h in (0.2, 0.3, 0.5, 0.8)]
    assert all(b > a for a, b in zip(constants, constants[1:]))
    np.testing.assert_allclose(
        LIPMParams(height=0.8).time_constant / LIPMParams(height=0.2).time_constant, 2.0, rtol=1e-12
    )


def test_analytic_step_matches_numerical_integration():
    """解析解 vs 精细数值积分 —— 老套路。

    这个系统本身发散，欧拉法的局部误差会被指数放大，所以必须用解析解。
    这个测试同时说明了"为什么"：数值积分要非常小的步长才能追上。
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        x0 = rng.normal(size=2) * 0.05
        v0 = rng.normal(size=2) * 0.3
        p = rng.normal(size=2) * 0.05
        dt = 0.05

        x_ana, v_ana = lipm_step(x0, v0, p, dt, PARAMS)

        # 极细步长的显式积分作为对照
        x, v = x0.copy(), v0.copy()
        n = 200000
        h = dt / n
        for _ in range(n):
            a = PARAMS.omega**2 * (x - p)
            v = v + h * a
            x = x + h * v
        np.testing.assert_allclose(x_ana, x, atol=1e-5)
        np.testing.assert_allclose(v_ana, v, atol=1e-4)


def test_capture_point_definition():
    x = np.array([0.1, -0.05])
    v = np.array([0.4, 0.2])
    np.testing.assert_allclose(capture_point(x, v, PARAMS), x + v / PARAMS.omega, rtol=1e-12)


def test_capture_point_dynamics_are_first_order_unstable():
    """捕获点满足 xi_dot = omega (xi - p)，是一个纯粹的一阶发散。"""
    x = np.array([0.02, 0.0])
    v = np.array([0.3, 0.0])
    p = np.array([0.0, 0.0])
    dt = 1e-4

    # 中心差分：解析解对负步长同样成立，所以可以往回推半步。
    # 前向差分的截断误差是 O(dt)，不足以验证这条恒等式。
    x_plus, v_plus = lipm_step(x, v, p, dt / 2, PARAMS)
    x_minus, v_minus = lipm_step(x, v, p, -dt / 2, PARAMS)
    xi_dot = (capture_point(x_plus, v_plus, PARAMS) - capture_point(x_minus, v_minus, PARAMS)) / dt

    xi0 = capture_point(x, v, PARAMS)
    # 中心差分的截断误差是 O(dt^2)，dt=1e-4 时约 1.4e-8，容差按此设定。
    np.testing.assert_allclose(xi_dot, PARAMS.omega * (xi0 - p), rtol=1e-6)


def test_com_dynamics_are_first_order_stable():
    """质心满足 x_dot = -omega (x - xi)：它总在追捕获点，且是稳定的。"""
    x = np.array([0.02, 0.01])
    v = np.array([0.3, -0.1])
    xi = capture_point(x, v, PARAMS)
    np.testing.assert_allclose(v, -PARAMS.omega * (x - xi), rtol=1e-12)


def test_stepping_on_the_capture_point_brings_the_robot_to_rest():
    """把脚落在捕获点上，机器人渐近停下 —— 捕获点的定义性质。"""
    x = np.array([0.0, 0.0])
    v = np.array([0.5, 0.0])
    foot = capture_point(x, v, PARAMS)

    for _ in range(60):
        x, v = lipm_step(x, v, foot, 0.02, PARAMS)
    np.testing.assert_allclose(v, 0.0, atol=1e-3)
    np.testing.assert_allclose(x, foot, atol=1e-3)


def test_stepping_short_of_the_capture_point_keeps_accelerating():
    """落脚点不够远，机器人继续前冲 —— 这就是"迈不动腿就摔"的数学表述。"""
    x = np.array([0.0, 0.0])
    v = np.array([0.5, 0.0])
    foot = capture_point(x, v, PARAMS) * 0.5  # 只迈到一半

    x_end, v_end = x.copy(), v.copy()
    for _ in range(30):
        x_end, v_end = lipm_step(x_end, v_end, foot, 0.02, PARAMS)
    assert v_end[0] > v[0], "落脚不足时速度应继续增大"


def test_stepping_beyond_the_capture_point_reverses_motion():
    x = np.array([0.0, 0.0])
    v = np.array([0.5, 0.0])
    foot = capture_point(x, v, PARAMS) * 2.0  # 迈过头

    x_end, v_end = x.copy(), v.copy()
    for _ in range(40):
        x_end, v_end = lipm_step(x_end, v_end, foot, 0.02, PARAMS)
    assert v_end[0] < 0.0, "迈过头会把机器人推回去"


def test_time_to_boundary_matches_analytic_divergence():
    """捕获点按 exp(omega t) 发散，到达边界的时间应为 log(比值)/omega。"""
    v = np.array([0.5, 0.0])
    boundary = 0.22
    t = time_to_boundary(np.zeros(2), v, boundary, PARAMS)
    xi0 = capture_point(np.zeros(2), v, PARAMS)[0]
    np.testing.assert_allclose(t, np.log(boundary / xi0) / PARAMS.omega, rtol=1e-12)

    # 用倒立摆推进到该时刻，捕获点应恰好抵达边界
    x, vv = lipm_step(np.zeros(2), v, np.zeros(2), t, PARAMS)
    np.testing.assert_allclose(capture_point(x, vv, PARAMS)[0], boundary, rtol=1e-6)


def test_time_to_boundary_edge_cases():
    assert time_to_boundary(np.zeros(2), np.zeros(2), 0.22, PARAMS) == np.inf
    assert time_to_boundary(np.zeros(2), np.array([-0.5, 0.0]), 0.22, PARAMS) == np.inf
    assert time_to_boundary(np.zeros(2), np.array([2.0, 0.0]), 0.22, PARAMS) == 0.0


def test_simulate_lipm_returns_consistent_trajectory():
    feet = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    out = simulate_lipm(np.zeros(2), np.array([0.3, 0.0]), feet, 0.2, PARAMS, substeps=25)
    assert out["com"].shape == (75, 2)
    np.testing.assert_allclose(
        out["capture_point"], out["com"] + out["velocity"] / PARAMS.omega, rtol=1e-12
    )


# --------------------------------------------------------------------------
# Raibert 启发式与捕获点的关系：本里程碑的核心洞察
# --------------------------------------------------------------------------


def test_raibert_reduces_to_hip_projection_at_rest():
    """静止且指令为零时，脚就该落在髋下。"""
    hip = np.array([0.19, 0.14])
    p = raibert_footstep(hip, np.zeros(2), np.zeros(2), 0.2, 0.03)
    np.testing.assert_allclose(p, hip, atol=1e-12)


def test_raibert_feedforward_is_half_stance_times_velocity():
    """前馈项让支撑相相对髋部前后对称。"""
    hip = np.array([0.0, 0.0])
    v = np.array([0.4, -0.1])
    stance = 0.2
    p = raibert_footstep(hip, v, v, stance, feedback_gain=0.03)  # 速度=指令，反馈为零
    np.testing.assert_allclose(p, 0.5 * stance * v, atol=1e-12)


def test_raibert_feedback_steps_further_when_too_fast():
    """速度超过指令时应当迈得更远，把速度压回来。"""
    hip = np.zeros(2)
    v_cmd = np.array([0.3, 0.0])
    slow = raibert_footstep(hip, v_cmd, v_cmd, 0.2, 0.08)
    fast = raibert_footstep(hip, v_cmd + np.array([0.3, 0.0]), v_cmd, 0.2, 0.08)
    assert fast[0] > slow[0]


def test_raibert_is_an_underdamped_approximation_of_the_capture_point():
    """本里程碑最重要的洞察：Raibert 前馈项就是捕获点，只是系数偏小。

    Raibert:  T_st / 2 = 0.100 s
    捕获点:   1 / omega = 0.175 s
    比值 1.75 —— 同一量级，形式完全相同。

    所以 Raibert 启发式不是拍脑袋的经验公式，而是捕获点的一个**欠调**
    近似；第三项速度反馈补的正是欠掉的那部分。
    """
    stance = get_gait("trot").stance_duration
    raibert_coeff = 0.5 * stance
    capture_coeff = PARAMS.time_constant

    np.testing.assert_allclose(raibert_coeff, 0.100, atol=1e-9)
    np.testing.assert_allclose(capture_coeff, 0.1749, atol=1e-4)
    np.testing.assert_allclose(capture_coeff / raibert_coeff, 1.75, atol=0.01)
    assert raibert_coeff < capture_coeff, "Raibert 的前馈是欠调的"


def test_optimal_gain_closes_the_gap_to_the_capture_point():
    """理论最优增益恰好补齐 1/omega 与 T_st/2 之差。"""
    stance = get_gait("trot").stance_duration
    k = optimal_feedback_gain(PARAMS, stance)
    np.testing.assert_allclose(k, PARAMS.time_constant - 0.5 * stance, rtol=1e-12)
    np.testing.assert_allclose(k, 0.0749, atol=1e-4)
    assert k > 0.0, "前馈欠调，所以最优增益为正"


def test_raibert_with_optimal_gain_matches_capture_point_on_velocity_error():
    """用最优增益时，Raibert 对**速度误差**的响应与捕获点策略一致。"""
    stance = get_gait("trot").stance_duration
    k = optimal_feedback_gain(PARAMS, stance)
    hip = np.array([0.19, 0.14])
    v_cmd = np.array([0.3, 0.0])
    v = v_cmd + np.array([0.4, -0.2])  # 存在速度误差

    p_raibert = raibert_footstep(hip, v, v_cmd, stance, k)
    p_capture = capture_point_footstep(hip, np.zeros(2), v, v_cmd, PARAMS, stance)
    np.testing.assert_allclose(p_raibert, p_capture, atol=1e-12)


def test_capture_point_strategy_returns_hip_at_rest():
    hip = np.array([0.19, 0.14])
    p = capture_point_footstep(hip, np.zeros(2), np.zeros(2), np.zeros(2), PARAMS, 0.2)
    np.testing.assert_allclose(p, hip, atol=1e-12)


# --------------------------------------------------------------------------
# 推倒恢复：能力边界
# --------------------------------------------------------------------------


def test_maximum_recoverable_push_is_set_by_workspace():
    """一步能救回来的最大扰动 = 工作空间半径 × omega。

    Go2 的落脚点半径上限 0.22 m（里程碑 5 实测步长上限 0.50 m 的一半再留
    余量），对应最大可恢复速度 0.22 × 5.718 = 1.26 m/s。超过这个数，
    一步之内无论落在哪里都救不回来，只能连迈几步。
    """
    max_radius = 0.22
    v_max = max_radius * PARAMS.omega
    np.testing.assert_allclose(v_max, 1.258, atol=0.01)

    # 恰好在边界内：捕获点可达
    xi = capture_point(np.zeros(2), np.array([v_max * 0.99, 0.0]), PARAMS)
    assert np.linalg.norm(xi) < max_radius
    # 超出边界：够不到
    xi = capture_point(np.zeros(2), np.array([v_max * 1.2, 0.0]), PARAMS)
    assert np.linalg.norm(xi) > max_radius


def test_large_pushes_leave_less_time_than_the_gait_period():
    """大扰动下留给你的时间比步态周期还短 —— 固定时序救不了。

    1.0 m/s 的扰动只剩 40 ms 必须迈步，而 trot 的支撑相是 200 ms。这意味着
    强推恢复必须**打破固定步态时序**，提前落脚。这正是抗推恢复难做的原因，
    也是里程碑 10 的主题。
    """
    stance = get_gait("trot").stance_duration
    t_small = time_to_boundary(np.zeros(2), np.array([0.2, 0.0]), 0.22, PARAMS)
    t_large = time_to_boundary(np.zeros(2), np.array([1.0, 0.0]), 0.22, PARAMS)

    assert t_small > stance, "小扰动可以等到下一次计划落脚"
    assert t_large < 0.25 * stance, "大扰动必须提前落脚"
    np.testing.assert_allclose(t_large, 0.0401, atol=2e-3)


# --------------------------------------------------------------------------
# 规划器：与步态、转弯、工作空间的整合
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planner():
    return FootstepPlanner(GaitScheduler("trot"), FootstepPlannerConfig(strategy="raibert"))


def test_config_validation():
    with pytest.raises(ValueError, match="未知策略"):
        FootstepPlannerConfig(strategy="magic")
    with pytest.raises(ValueError, match="最大步长半径"):
        FootstepPlannerConfig(max_step_radius=0.0)


def test_default_gain_is_the_theoretical_optimum():
    p = FootstepPlanner(GaitScheduler("trot"))
    np.testing.assert_allclose(
        p.cfg.feedback_gain, optimal_feedback_gain(LIPMParams(0.30), 0.2), rtol=1e-12
    )


def test_hip_projection_matches_nominal_stance(planner):
    """零偏航时，髋部投影就是标称站姿的足端水平位置。"""
    base = np.array([1.0, 2.0, 0.30])
    for leg in LEGS:
        hip = planner.nominal_hip_projection(leg, base, yaw=0.0)
        expected = base[:2] + np.array(
            [HIP_OFFSETS[leg][0], HIP_OFFSETS[leg][1] + LEG_GEOMETRY[leg].l0]
        )
        np.testing.assert_allclose(hip, expected, atol=1e-12)


def test_yaw_rotates_the_hip_projection(planner):
    """偏航 90 度后，前腿的髋投影应转到侧向。"""
    base = np.zeros(3)
    hip = planner.nominal_hip_projection("FL", base, yaw=np.pi / 2)
    nominal = np.array([HIP_OFFSETS["FL"][0], HIP_OFFSETS["FL"][1] + LEG_GEOMETRY["FL"].l0])
    expected = np.array([-nominal[1], nominal[0]])
    np.testing.assert_allclose(hip, expected, atol=1e-12)


def test_yaw_rate_uses_the_future_hip_position(planner):
    """转弯时必须按**落地时刻**的偏航角算髋位置，而不是当前时刻。"""
    base = np.zeros(3)
    now = planner.nominal_hip_projection("FL", base, yaw=0.0, yaw_rate=1.0, time_ahead=0.0)
    later = planner.nominal_hip_projection("FL", base, yaw=0.0, yaw_rate=1.0, time_ahead=0.1)
    assert np.linalg.norm(later - now) > 0.01
    direct = planner.nominal_hip_projection("FL", base, yaw=0.1)
    np.testing.assert_allclose(later, direct, atol=1e-12)


def test_workspace_clamping_limits_step_radius():
    """规划层就该拒绝够不到的目标，而不是丢给逆运动学去救。"""
    cfg = FootstepPlannerConfig(max_step_radius=0.15)
    p = FootstepPlanner(GaitScheduler("trot"), cfg)
    base = np.zeros(3)
    target = p.plan_leg("FL", base, np.array([3.0, 0.0]), np.zeros(2), time_to_touchdown=0.1)
    hip = p.nominal_hip_projection("FL", base, 0.0, 0.0, 0.1)
    np.testing.assert_allclose(np.linalg.norm(target[:2] - hip), cfg.max_step_radius, atol=1e-9)
    assert p.is_clamped(target, hip)


def test_clamping_preserves_direction():
    """被钳制后方向仍然正确，只是幅度受限。"""
    cfg = FootstepPlannerConfig(max_step_radius=0.10)
    p = FootstepPlanner(GaitScheduler("trot"), cfg)
    base = np.zeros(3)
    v = np.array([2.0, 1.0])
    target = p.plan_leg("FL", base, v, np.zeros(2), time_to_touchdown=0.1)
    hip = p.nominal_hip_projection("FL", base, 0.0, 0.0, 0.1)
    delta = target[:2] - hip
    np.testing.assert_allclose(delta / np.linalg.norm(delta), v / np.linalg.norm(v), atol=1e-9)


def test_plan_only_returns_swing_legs(planner):
    base = np.array([0.0, 0.0, 0.30])
    targets = planner.plan(0.05, base, np.array([0.3, 0.0]), np.array([0.3, 0.0]))
    contact = planner.scheduler.contact(0.05)
    swing_legs = {leg for i, leg in enumerate(LEGS) if not contact[i]}
    assert set(targets) == swing_legs
    assert len(targets) == 2  # trot 恒有两条摆动腿


def test_terrain_height_is_applied(planner):
    base = np.array([0.0, 0.0, 0.30])
    targets = planner.plan(
        0.05, base, np.zeros(2), np.zeros(2), terrain_height={leg: 0.07 for leg in LEGS}
    )
    for p in targets.values():
        np.testing.assert_allclose(p[2], 0.07, atol=1e-12)


def test_capture_point_diagnostic(planner):
    base = np.array([0.5, 0.2, 0.30])
    v = np.array([0.4, -0.1])
    np.testing.assert_allclose(
        planner.capture_point_world(base, v), base[:2] + v / PARAMS.omega, rtol=1e-12
    )


# --------------------------------------------------------------------------
# 闭环行为：策略对不对，最终看跑起来收不收敛
# --------------------------------------------------------------------------


def _closed_loop(strategy, v0, v_cmd, n_steps=40, gain=None):
    """把落脚点策略放进倒立摆里跑，返回每步结束时的速度。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    cfg = FootstepPlannerConfig(strategy=strategy, feedback_gain=gain, max_step_radius=0.30)
    planner = FootstepPlanner(sched, cfg)

    x = np.zeros(2)
    v = np.array(v0, dtype=float)
    stance = gait.stance_duration
    speeds = []
    for _ in range(n_steps):
        base = np.array([x[0], x[1], HEIGHT])
        foot = planner.plan_leg("FL", base, v, np.array(v_cmd), time_to_touchdown=stance)
        # 落脚点相对髋部的偏移即为实际支撑点（去掉标称站姿偏置）
        hip = planner.nominal_hip_projection("FL", base, 0.0, 0.0, stance)
        support = x + (foot[:2] - hip)
        x, v = lipm_step(x, v, support, stance, PARAMS)
        speeds.append(v.copy())
    return np.array(speeds)


def test_raibert_closed_loop_converges_to_commanded_velocity():
    """闭环验证：Raibert 策略能把速度收敛到指令值。"""
    v_cmd = np.array([0.3, 0.0])
    speeds = _closed_loop("raibert", [0.0, 0.0], v_cmd)
    np.testing.assert_allclose(speeds[-1], v_cmd, atol=0.05)


def test_capture_point_closed_loop_converges_too():
    v_cmd = np.array([0.25, 0.0])
    speeds = _closed_loop("capture_point", [0.6, 0.0], v_cmd)
    np.testing.assert_allclose(speeds[-1], v_cmd, atol=0.05)


def test_closed_loop_rejects_a_push():
    """被推之后应当收敛回指令速度。"""
    v_cmd = np.array([0.2, 0.0])
    speeds = _closed_loop("raibert", [1.0, 0.4], v_cmd, n_steps=50)
    np.testing.assert_allclose(speeds[-1], v_cmd, atol=0.05)
    # 误差应单调下降（允许少量非单调）
    err = np.linalg.norm(speeds - v_cmd, axis=1)
    assert err[-1] < 0.1 * err[0]


def test_zero_feedback_gain_fails_to_track():
    """去掉反馈项只剩前馈，速度会稳态偏离指令 —— 前馈欠调的直接后果。"""
    v_cmd = np.array([0.3, 0.0])
    with_fb = _closed_loop("raibert", [0.0, 0.0], v_cmd, gain=optimal_feedback_gain(PARAMS, 0.2))
    without_fb = _closed_loop("raibert", [0.0, 0.0], v_cmd, gain=0.0)
    err_with = np.linalg.norm(with_fb[-1] - v_cmd)
    err_without = np.linalg.norm(without_fb[-1] - v_cmd)
    assert err_with < 0.2 * err_without, f"有反馈 {err_with:.4f} vs 无反馈 {err_without:.4f}"


# --------------------------------------------------------------------------
# 精确极限环系数：统一 Raibert 与捕获点
# --------------------------------------------------------------------------


def test_exact_coefficient_matches_the_limit_cycle_condition():
    """精确系数由"一步之后速度不变"直接解出，应满足解析关系。"""
    from footstep_planner import exact_stride_coefficient

    stance = get_gait("trot").stance_duration
    c = exact_stride_coefficient(PARAMS, stance)
    w = PARAMS.omega
    np.testing.assert_allclose(c, np.tanh(0.5 * w * stance) / w, rtol=1e-12)
    np.testing.assert_allclose(c, 0.090359, atol=1e-6)

    # 直接验证极限环：以该偏移落脚，一步后速度不变
    v0 = np.array([0.35, 0.0])
    _, v1 = lipm_step(np.zeros(2), v0, c * v0, stance, PARAMS)
    np.testing.assert_allclose(v1, v0, rtol=1e-12)


def test_raibert_and_capture_point_are_the_two_asymptotic_limits():
    """本里程碑最重要的结论：两个经典启发式是同一个式子的两端。

    tanh(wT/2)/w  ->  T/2      当 wT -> 0   （Raibert，慢步态极限）
                  ->  1/omega  当 wT -> inf （捕获点，快步态极限）
    """
    from footstep_planner import exact_stride_coefficient

    w = PARAMS.omega

    # 慢步态极限：趋近 Raibert
    for stance in (1e-4, 1e-3, 1e-2):
        c = exact_stride_coefficient(PARAMS, stance)
        np.testing.assert_allclose(c, 0.5 * stance, rtol=1e-3)

    # 快步态极限：趋近捕获点
    for stance in (2.0, 5.0, 10.0):
        c = exact_stride_coefficient(PARAMS, stance)
        np.testing.assert_allclose(c, 1.0 / w, rtol=1e-3)

    # Go2 trot 夹在中间，两个近似都不准
    stance = get_gait("trot").stance_duration
    c = exact_stride_coefficient(PARAMS, stance)
    assert 0.5 * stance > c, "Raibert 系数偏大"
    assert 1.0 / w > c, "捕获点系数偏大得更多"
    np.testing.assert_allclose((0.5 * stance - c) / c, 0.107, atol=0.005)


def test_steady_state_ratio_predicts_the_simulated_tracking_error():
    """解析预测 vs 闭环仿真 —— 又一次"两条路径必须一致"。

    Raibert 的稳态速度亏损是**结构性**的，不是调参问题：预测 0.8859，
    仿真 0.8859，吻合到小数点后四位。
    """
    from footstep_planner import steady_state_velocity_ratio

    stance = get_gait("trot").stance_duration
    k = optimal_feedback_gain(PARAMS, stance)
    predicted = steady_state_velocity_ratio(0.5 * stance, k, PARAMS, stance)
    np.testing.assert_allclose(predicted, 0.8859, atol=1e-3)

    v_cmd = np.array([0.3, 0.0])
    speeds = _closed_loop("raibert", [0.0, 0.0], v_cmd, n_steps=200, gain=k)
    np.testing.assert_allclose(speeds[-1, 0] / v_cmd[0], predicted, rtol=1e-3)


def test_exact_strategy_tracks_the_command_without_steady_state_error():
    """换成精确系数后稳态误差归零。"""
    from footstep_planner import exact_stride_coefficient, steady_state_velocity_ratio

    stance = get_gait("trot").stance_duration
    k = optimal_feedback_gain(PARAMS, stance)
    c = exact_stride_coefficient(PARAMS, stance)
    np.testing.assert_allclose(
        steady_state_velocity_ratio(c, k, PARAMS, stance), 1.0, rtol=1e-9
    )

    for v_cmd in ([0.2, 0.0], [0.5, 0.1], [-0.3, 0.0]):
        speeds = _closed_loop("exact", [0.0, 0.0], v_cmd, n_steps=120)
        np.testing.assert_allclose(speeds[-1], v_cmd, atol=2e-3)


def test_exact_strategy_beats_raibert_on_tracking():
    """各用各的默认增益时，精确系数的稳态误差小两个数量级以上。

    **不能用同一个增益去比。** 两种写法的稳定条件不同（见下一个测试）：
    Raibert 形式要求 c + k > c*，精确形式要求 k > c*。共用增益会让其中
    一个发散，比较就失去意义了。
    """
    v_cmd = np.array([0.4, 0.0])
    err_raibert = abs(_closed_loop("raibert", [0.0, 0.0], v_cmd, 150)[-1, 0] - v_cmd[0])
    err_exact = abs(_closed_loop("exact", [0.0, 0.0], v_cmd, 150)[-1, 0] - v_cmd[0])
    assert err_raibert > 0.03, "Raibert 应有明显稳态误差"
    assert err_exact < 0.01 * err_raibert, f"精确 {err_exact:.6f} vs Raibert {err_raibert:.5f}"


def test_feedforward_on_current_velocity_also_provides_stability():
    """结构性事实：前馈跟随当前速度时**也参与反馈**，跟随指令速度则不。

    p = c*v + k(v-v_cmd)      -> 特征值 cosh - w*sinh*(c+k)，条件 c+k > c*
    p = c**v_cmd + k(v-v_cmd) -> 特征值 cosh - w*sinh*k    ，条件 k   > c*

    后果很实在：k = 1/w - T/2 = 0.0749 在第一种写法下稳定，在第二种写法下
    发散 —— 换了个"更精确"的系数，机器人反而飞了。这是实现时踩到的坑。
    """
    from footstep_planner import deadbeat_feedback_gain, exact_stride_coefficient

    stance = get_gait("trot").stance_duration
    w = PARAMS.omega
    ch, sh = np.cosh(w * stance), np.sinh(w * stance)
    c_star = exact_stride_coefficient(PARAMS, stance)
    k_opt = optimal_feedback_gain(PARAMS, stance)

    # Raibert 形式：c + k = 0.1749 > c* = 0.0904，稳定
    assert abs(ch - w * sh * (0.5 * stance + k_opt)) < 1.0
    # 精确形式配同一增益：k = 0.0749 < c*，发散
    assert abs(ch - w * sh * k_opt) > 1.0
    assert k_opt < c_star

    # 死拍增益把特征值打到零
    k_db = deadbeat_feedback_gain(PARAMS, stance)
    np.testing.assert_allclose(ch - w * sh * k_db, 0.0, atol=1e-12)
    np.testing.assert_allclose(k_db, 0.2144, atol=1e-3)
    assert k_db > c_star


def test_deadbeat_gain_converges_in_one_step():
    """死拍增益：一步把速度从零拉到指令值。"""
    v_cmd = np.array([0.3, 0.0])
    speeds = _closed_loop("exact", [0.0, 0.0], v_cmd, n_steps=5)
    np.testing.assert_allclose(speeds[0], v_cmd, atol=1e-9)


def test_planner_accepts_the_exact_strategy():
    p = FootstepPlanner(GaitScheduler("trot"), FootstepPlannerConfig(strategy="exact"))
    base = np.array([0.0, 0.0, HEIGHT])
    targets = p.plan(0.05, base, np.array([0.3, 0.0]), np.array([0.3, 0.0]))
    assert len(targets) == 2
    for target in targets.values():
        assert np.all(np.isfinite(target))
