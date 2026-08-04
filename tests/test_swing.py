"""里程碑 5（摆动腿规划器）的验证。

轨迹是解析式，所以验证策略回到里程碑 1、2 的老路：**用解析性质与数值
微分互相对照**。此外还要验证一件前几个里程碑没做过的事 —— 整条链路的
端到端一致性：

    步态相位(M4) -> 足端轨迹(本模块) -> 逆运动学(M1) -> 正运动学(M1)

绕一圈回来，必须精确等于最初的足端目标。

运行::

    pytest tests/test_swing.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from gait_scheduler import GaitScheduler
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS, forward_kinematics, load_go2
from swing_planner import (
    SWING_TRAJECTORIES,
    BezierSwing,
    CycloidSwing,
    QuinticSwing,
    SineHeightSwing,
    SwingLegConfig,
    SwingLegController,
)

SEED = 0
LIFTOFF = np.array([-0.10, 0.142, 0.0])
TOUCHDOWN = np.array([0.10, 0.142, 0.0])
HEIGHT = 0.08
SWING_T = 0.2

#: 落地速度精确为零的那几种轨迹。
ZERO_IMPACT = ("cycloid", "bezier", "quintic")


def make(name, liftoff=LIFTOFF, touchdown=TOUCHDOWN, height=HEIGHT):
    return SWING_TRAJECTORIES[name](liftoff, touchdown, height)


# --------------------------------------------------------------------------
# 端点条件：轨迹必须真的从离地点出发、落到落地点
# --------------------------------------------------------------------------


def test_all_trajectories_hit_their_endpoints():
    for name in SWING_TRAJECTORIES:
        traj = make(name)
        np.testing.assert_allclose(traj.position(0.0), LIFTOFF, atol=1e-12), name
        np.testing.assert_allclose(traj.position(1.0), TOUCHDOWN, atol=1e-12), name


def test_trajectory_rejects_nonpositive_height():
    for name, cls in SWING_TRAJECTORIES.items():
        with pytest.raises(ValueError, match="抬腿高度必须为正"):
            cls(LIFTOFF, TOUCHDOWN, 0.0)


def test_actual_clearance_equals_requested_height():
    """贝塞尔曲线不穿过中间控制点，顶点必须按权重反解 —— 否则只抬三分之一。"""
    for name in SWING_TRAJECTORIES:
        _, clearance = make(name).clearance_profile()
        np.testing.assert_allclose(clearance.max(), HEIGHT, rtol=2e-3), name
        assert clearance.min() >= -1e-12, f"{name}: 足端不应低于离地-落地连线"


def test_horizontal_progress_is_monotonic():
    """水平方向必须单调前进，来回摆会撞到障碍物也浪费能量。"""
    s = np.linspace(0.0, 1.0, 1001)
    for name in SWING_TRAJECTORIES:
        x = make(name).position(s)[:, 0]
        assert np.all(np.diff(x) >= -1e-12), f"{name}: 水平位移应单调"


# --------------------------------------------------------------------------
# 落地冲击：本里程碑的核心结论
# --------------------------------------------------------------------------


def test_sine_trajectory_slams_into_the_ground():
    """半正弦抬腿在落地瞬间竖直速度为 -h*pi/T_swing —— 一个经典的坑。"""
    traj = make("sine")
    v = traj.touchdown_velocity(SWING_T)
    expected = -HEIGHT * np.pi / SWING_T
    np.testing.assert_allclose(v[2], expected, rtol=1e-4)
    assert abs(v[2]) > 1.0, "抬腿 8 cm、摆动 0.2 s 时落地速度超过 1 m/s"


def test_zero_impact_trajectories_land_softly():
    """解析导数下落地速度是**精确**为零，不是"接近零"。"""
    for name in ZERO_IMPACT:
        v = make(name).touchdown_velocity(SWING_T)
        np.testing.assert_allclose(v, 0.0, atol=1e-12), name


def test_zero_impact_trajectories_also_lift_off_softly():
    """离地速度不为零会拖拽地面，同样引发打滑。"""
    for name in ZERO_IMPACT:
        v = make(name).liftoff_velocity(SWING_T)
        np.testing.assert_allclose(v, 0.0, atol=1e-12), name


def test_cycloid_still_has_an_acceleration_step_at_touchdown():
    """摆线的落地**速度**为零，但**加速度**不为零 —— 接触力会有台阶。

    这是"零冲击"三层结构的中间一层：
      sine    落地速度不为零，砸地
      cycloid 速度为零、加速度不为零，力有阶跃
      bezier / quintic  速度与加速度都为零，力连续
    """
    assert np.linalg.norm(make("cycloid").acceleration(1.0, SWING_T)) > 10.0
    for name in ("bezier", "quintic"):
        np.testing.assert_allclose(make(name).acceleration(1.0, SWING_T), 0.0, atol=1e-9), name


def test_touchdown_impact_scales_with_height_and_inverse_duration():
    """冲击速度正比于抬腿高度、反比于摆动时长 —— 高速步态尤其危险。"""
    for h in (0.04, 0.08, 0.12):
        for T in (0.1, 0.2, 0.4):
            traj = make("sine", height=h)
            v = traj.touchdown_velocity(T)[2]
            np.testing.assert_allclose(v, -h * np.pi / T, rtol=1e-4)


def test_bezier_endpoint_multiplicity_controls_derivative_order():
    """端点控制点重复 k 次，前 k-1 阶导数在该端点为零。"""
    double = BezierSwing(LIFTOFF, TOUCHDOWN, HEIGHT, endpoint_multiplicity=2)
    triple = BezierSwing(LIFTOFF, TOUCHDOWN, HEIGHT, endpoint_multiplicity=3)

    # 重复 2 次：速度为零，加速度一般不为零
    np.testing.assert_allclose(double.touchdown_velocity(SWING_T), 0.0, atol=1e-12)
    assert np.linalg.norm(double.acceleration(1.0, SWING_T)) > 1.0

    # 重复 3 次：速度与加速度都为零
    np.testing.assert_allclose(triple.touchdown_velocity(SWING_T), 0.0, atol=1e-12)
    np.testing.assert_allclose(triple.acceleration(1.0, SWING_T), 0.0, atol=1e-9)


def test_bezier_rejects_invalid_multiplicity():
    with pytest.raises(ValueError, match="端点重复次数"):
        BezierSwing(LIFTOFF, TOUCHDOWN, HEIGHT, endpoint_multiplicity=0)


def test_soft_landing_costs_peak_acceleration():
    """本里程碑的定量权衡：消除落地冲击的代价是峰值加速度翻倍。

    半正弦轨迹落地砸得狠，但它的加速度分布最平缓。零冲击轨迹必须在两端
    把速度压到零，中段就得加速得更猛 —— 峰值加速度正比于所需关节力矩。
    天下没有免费的午餐。
    """
    sine_peak = make("sine").peak_acceleration(SWING_T)
    peaks = {name: make(name).peak_acceleration(SWING_T) for name in ZERO_IMPACT}
    for name, peak in peaks.items():
        assert peak > sine_peak, f"{name}: 峰值加速度应高于半正弦"
    # 连力也要连续的那两种（落地加速度为零），代价更大
    for name in ("bezier", "quintic"):
        assert peaks[name] > 1.5 * sine_peak, f"{name}: 应付出 1.5 倍以上的峰值加速度"


# --------------------------------------------------------------------------
# 导数的正确性：解析性质 vs 数值微分
# --------------------------------------------------------------------------


def test_velocity_matches_numerical_derivative_of_position():
    """解析导数 vs 数值微分 —— 老套路：两条互不相干的路径必须一致。"""
    s = np.linspace(0.02, 0.98, 60)
    ds = 1e-6
    for name in SWING_TRAJECTORIES:
        traj = make(name)
        v_num = (traj.position(s + ds) - traj.position(s - ds)) / (2 * ds * SWING_T)
        np.testing.assert_allclose(traj.velocity(s, SWING_T), v_num, atol=1e-6), name


def test_acceleration_matches_numerical_second_derivative():
    s = np.linspace(0.02, 0.98, 60)
    ds = 1e-4
    for name in SWING_TRAJECTORIES:
        traj = make(name)
        a_num = (traj.position(s + ds) - 2 * traj.position(s) + traj.position(s - ds)) / (
            (ds * SWING_T) ** 2
        )
        np.testing.assert_allclose(traj.acceleration(s, SWING_T), a_num, atol=1e-4), name


def test_endpoint_derivatives_are_exact():
    """落地速度是本模块的头号指标，必须解析求出而非差分逼近。

    用差分的话，模板落在 s = 1 - eps 上，摆线会给出 ~8e-6 m/s 的假残留，
    看起来"接近零"其实是数值偏差。解析导数下它精确为零。
    """
    for name in ZERO_IMPACT:
        assert abs(make(name).touchdown_velocity(SWING_T)[2]) < 1e-15, name
    # 半正弦的落地速度也应精确等于解析值
    np.testing.assert_allclose(
        make("sine").touchdown_velocity(SWING_T)[2], -HEIGHT * np.pi / SWING_T, rtol=1e-14
    )


def test_acceleration_is_finite_everywhere():
    s = np.linspace(0.0, 1.0, 501)
    for name in SWING_TRAJECTORIES:
        a = make(name).acceleration(s, SWING_T)
        assert np.all(np.isfinite(a)), name


# --------------------------------------------------------------------------
# 摆动腿控制器
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    """一个可复用的场景：trot 步态 + 标称站姿。"""
    sched = GaitScheduler("trot")
    base_position = np.array([0.0, 0.0, 0.30])
    base_rotation = np.eye(3)
    feet = np.array(
        [[HIP_OFFSETS[leg][0], HIP_OFFSETS[leg][1] + LEG_GEOMETRY[leg].l0, 0.0] for leg in LEGS]
    )
    targets = {leg: feet[i] + np.array([0.10, 0.0, 0.0]) for i, leg in enumerate(LEGS)}
    return sched, base_position, base_rotation, feet, targets


def test_controller_rejects_unknown_trajectory(setup):
    sched = setup[0]
    with pytest.raises(ValueError, match="未知轨迹类型"):
        SwingLegController(sched, SwingLegConfig(trajectory="spiral"))


def test_stance_leg_target_is_its_current_position(setup):
    """支撑腿的目标必须是它当前所在的位置。

    开机瞬间尚未摆动过的腿如果拿到零向量，逆运动学会解出一个荒唐的姿态 ——
    这是实现时真实踩到的坑。
    """
    sched, base_p, base_R, feet, targets = setup
    ctl = SwingLegController(sched)
    ctl.update(0.0, feet, targets)
    contact = sched.contact(0.0)
    for i, leg in enumerate(LEGS):
        if contact[i]:
            np.testing.assert_allclose(ctl.foot_target(0.0, leg), feet[i], atol=1e-12)


def test_stance_leg_velocity_is_zero(setup):
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched)
    ctl.update(0.05, feet, targets)
    for i, leg in enumerate(LEGS):
        if sched.contact(0.05)[i]:
            np.testing.assert_allclose(ctl.foot_velocity(0.05, leg), 0.0, atol=1e-12)


def test_liftoff_point_is_latched_not_tracked(setup):
    """轨迹起点是**离地那一刻**的位置，不是当前位置。

    每周期实时读当前足端位置会让轨迹自我追逐，足端在原地打转。
    """
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched)
    i = LEGS.index("FR")  # FR 在 t=0 处于摆动相

    ctl.update(0.0, feet, targets)
    latched = ctl.foot_target(0.0, "FR").copy()
    np.testing.assert_allclose(latched, feet[i], atol=1e-12)

    # 后续喂进去完全不同的"当前位置"，轨迹起点不应改变
    moved = feet.copy()
    moved[i] += np.array([0.5, 0.5, 0.5])
    ctl.update(0.05, moved, targets)
    assert ctl._legs["FR"].liftoff is not None
    np.testing.assert_allclose(ctl._legs["FR"].liftoff, feet[i], atol=1e-12)


def test_touchdown_target_can_be_updated_mid_swing(setup):
    """落脚点规划器每周期都在重算目标，轨迹必须跟着变。"""
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched)
    ctl.update(0.0, feet, targets)

    new_targets = {leg: p + np.array([0.05, 0.0, 0.0]) for leg, p in targets.items()}
    ctl.update(0.05, feet, new_targets)
    np.testing.assert_allclose(ctl._legs["FR"].touchdown, new_targets["FR"], atol=1e-12)
    # 轨迹终点随之改变
    np.testing.assert_allclose(
        ctl._legs["FR"].trajectory.position(1.0), new_targets["FR"], atol=1e-12
    )


def test_swing_trajectory_follows_scheduler_phase(setup):
    """摆动进度从 0 到 1，足端应从离地点走到落地点。"""
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched)
    i = LEGS.index("FR")
    swing_duration = sched.gait.swing_duration

    ctl.update(0.0, feet, targets)
    start = ctl.foot_target(0.0, "FR")
    ctl.update(swing_duration * 0.999, feet, targets)
    end = ctl.foot_target(swing_duration * 0.999, "FR")

    np.testing.assert_allclose(start, feet[i], atol=1e-9)
    np.testing.assert_allclose(end, targets["FR"], atol=2e-3)


def test_foot_rises_above_ground_mid_swing(setup):
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched, SwingLegConfig(swing_height=0.08))
    t_mid = sched.gait.swing_duration * 0.5
    ctl.update(t_mid, feet, targets)
    assert ctl.foot_target(t_mid, "FR")[2] > 0.06


# --------------------------------------------------------------------------
# 迟落地下探
# --------------------------------------------------------------------------


def test_probe_descends_and_saturates(setup):
    """地面比预期低时必须继续下探，但不能无限探下去。"""
    sched, _, _, feet, targets = setup
    cfg = SwingLegConfig(probe_velocity=0.15, max_probe_depth=0.06)
    ctl = SwingLegController(sched, cfg)
    ctl.update(0.0, feet, targets)

    dt = 1e-3
    total = 0.0
    for _ in range(1000):
        total += -ctl.probe("FR", dt)[2]
    np.testing.assert_allclose(total, cfg.max_probe_depth, atol=1e-9)
    np.testing.assert_allclose(ctl.probe_depth("FR"), cfg.max_probe_depth, atol=1e-9)
    np.testing.assert_allclose(ctl.probe("FR", dt), 0.0, atol=1e-12)


def test_probe_resets_when_leg_returns_to_stance(setup):
    sched, _, _, feet, targets = setup
    ctl = SwingLegController(sched)
    ctl.update(0.0, feet, targets)
    for _ in range(50):
        ctl.probe("FR", 1e-3)
    assert ctl.probe_depth("FR") > 0.0
    ctl.update(sched.gait.swing_duration + 1e-3, feet, targets)  # FR 回到支撑相
    assert ctl.probe_depth("FR") == 0.0


# --------------------------------------------------------------------------
# 端到端：M4 相位 -> 轨迹 -> M1 逆运动学 -> M1 正运动学
# --------------------------------------------------------------------------


def test_joint_command_round_trips_through_kinematics(setup):
    """整条链路绕一圈必须精确回到原点。

    这一条同时验证了里程碑 1 的 IK/FK、髋部偏置的处理，以及本模块的
    世界系到髋系变换。任何一处符号错都会在这里暴露。
    """
    sched, base_p, base_R, feet, targets = setup
    ctl = SwingLegController(sched)
    for k in range(0, 200, 7):
        t = k * 1e-3
        ctl.update(t, feet, targets)
        for leg in LEGS:
            q, _ = ctl.joint_command(t, leg, base_p, base_R)
            p_hip = forward_kinematics(q, LEG_GEOMETRY[leg])
            p_world = base_p + base_R @ (HIP_OFFSETS[leg] + p_hip)
            np.testing.assert_allclose(p_world, ctl.foot_target(t, leg), atol=1e-9)


def test_joint_command_works_with_rotated_base(setup):
    """躯干有姿态时链路依然自洽 —— 世界系与髋系的变换必须对。"""
    from dynamics import rpy_to_matrix

    sched, base_p, _, feet, targets = setup
    ctl = SwingLegController(sched)
    R = rpy_to_matrix(np.array([0.08, -0.05, 0.3]))
    ctl.update(0.05, feet, targets)
    for leg in LEGS:
        q, _ = ctl.joint_command(0.05, leg, base_p, R)
        p_world = base_p + R @ (HIP_OFFSETS[leg] + forward_kinematics(q, LEG_GEOMETRY[leg]))
        np.testing.assert_allclose(p_world, ctl.foot_target(0.05, leg), atol=1e-9)


def test_joint_velocity_matches_foot_velocity(setup):
    """dq = J^-1 v：再乘回 J 必须还原足端速度。"""
    sched, base_p, base_R, feet, targets = setup
    from kinematics import leg_jacobian

    ctl = SwingLegController(sched)
    for k in range(10, 190, 13):
        t = k * 1e-3
        ctl.update(t, feet, targets)
        q, dq = ctl.joint_command(t, "FR", base_p, base_R)
        v_hip = leg_jacobian(q, LEG_GEOMETRY["FR"]) @ dq
        np.testing.assert_allclose(v_hip, base_R.T @ ctl.foot_velocity(t, "FR"), atol=1e-6)


# --------------------------------------------------------------------------
# 可行性：抬腿高度与步长的实际上限
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def joint_limits():
    model = load_go2()
    i = LEGS.index("FR") * 3
    return (
        model.model.lowerPositionLimit[i : i + 3].copy(),
        model.model.upperPositionLimit[i : i + 3].copy(),
    )


def test_nominal_swing_is_feasible(setup, joint_limits):
    sched, base_p, base_R, _, _ = setup
    ctl = SwingLegController(sched, SwingLegConfig(swing_height=0.08))
    p0 = np.array([0.1934 - 0.05, -0.142, 0.0])
    p1 = np.array([0.1934 + 0.05, -0.142, 0.0])
    r = ctl.check_swing_feasibility("FR", p0, p1, base_p, base_R, joint_limits)
    assert r["feasible"]
    assert r["worst_margin"] > 0.3
    assert r["max_condition_number"] < 5.0


def test_step_length_has_a_hard_limit(setup, joint_limits):
    """步长不能无限加大 —— 限制来自**关节限位**，而不是几何可达范围。

    Go2 的几何最大伸展是 0.426 m，但膝限位把它压到 0.389 m（里程碑 1），
    再叠加 0.30 m 的站立高度，步长在 0.5 m 附近就撞上关节限位了。
    """
    sched, base_p, base_R, _, _ = setup
    ctl = SwingLegController(sched, SwingLegConfig(swing_height=0.08))
    results = {}
    for length in (0.10, 0.30, 0.50):
        p0 = np.array([0.1934 - length / 2, -0.142, 0.0])
        p1 = np.array([0.1934 + length / 2, -0.142, 0.0])
        results[length] = ctl.check_swing_feasibility("FR", p0, p1, base_p, base_R, joint_limits)

    assert results[0.10]["feasible"]
    assert results[0.30]["feasible"]
    assert not results[0.50]["feasible"], "0.5 m 步长应当超出关节限位"
    margins = [results[l]["worst_margin"] for l in (0.10, 0.30, 0.50)]
    assert all(b < a for a, b in zip(margins, margins[1:])), "裕度应随步长单调下降"


def test_joint_speed_grows_with_step_length(setup, joint_limits):
    """步长越大，同样时间内要走完的距离越长，关节速度越高。"""
    sched, base_p, base_R, _, _ = setup
    ctl = SwingLegController(sched, SwingLegConfig(swing_height=0.08))
    speeds = []
    for length in (0.10, 0.20, 0.30, 0.40):
        p0 = np.array([0.1934 - length / 2, -0.142, 0.0])
        p1 = np.array([0.1934 + length / 2, -0.142, 0.0])
        speeds.append(
            ctl.check_swing_feasibility("FR", p0, p1, base_p, base_R, joint_limits)["max_joint_speed"]
        )
    assert all(b > a for a, b in zip(speeds, speeds[1:])), f"关节速度应随步长增长：{speeds}"


def test_feasibility_reports_trajectory_specific_metrics(setup, joint_limits):
    """不同轨迹的关节裕度相同（几何一样），但冲击与加速度不同。"""
    sched, base_p, base_R, _, _ = setup
    p0 = np.array([0.1934 - 0.05, -0.142, 0.0])
    p1 = np.array([0.1934 + 0.05, -0.142, 0.0])

    results = {}
    for name in SWING_TRAJECTORIES:
        ctl = SwingLegController(sched, SwingLegConfig(swing_height=0.08, trajectory=name))
        results[name] = ctl.check_swing_feasibility("FR", p0, p1, base_p, base_R, joint_limits)

    margins = [r["worst_margin"] for r in results.values()]
    np.testing.assert_allclose(margins, margins[0], atol=1e-9)  # 几何相同
    assert results["sine"]["touchdown_speed"] > 1.0
    for name in ZERO_IMPACT:
        assert results[name]["touchdown_speed"] < 1e-12
        assert results[name]["peak_foot_acceleration"] > results["sine"]["peak_foot_acceleration"]
