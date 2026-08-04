"""里程碑 4（步态调度器）的验证。

步态调度器是纯逻辑模块，没有"第二份实现"可以对照。所以验证策略换成
**用步态的定义性质去反查实现**：

* trot 的定义就是对角腿同相 —— 那就断言 FL 与 RR 的接触状态恒等；
* 占空比的定义就是支撑时间占比 —— 那就统计出来核对；
* 静态稳定的定义是质心投影落在支撑多边形内 —— 那就用真实 Go2 几何算。

外加一条：调度器给下游的所有查询（接触序列、摆动进度、落地倒计时）
必须**互相自洽**，任意两个之间的矛盾都会被抓出来。

运行::

    pytest tests/test_gait.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics import load_go2_dynamics, nominal_configuration
from gait_scheduler import (
    GAITS,
    GaitDefinition,
    GaitScheduler,
    convex_hull_2d,
    get_gait,
    point_in_polygon,
    static_stability_margin,
    support_polygon,
)
from kinematics import LEGS

SEED = 0


@pytest.fixture(scope="module")
def go2_stance():
    """Go2 标称站姿下的质心与四足位置 —— 稳定性分析的真实几何。"""
    rbd = load_go2_dynamics(floating_base=True)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    return com, feet


# --------------------------------------------------------------------------
# 步态定义
# --------------------------------------------------------------------------


def test_gait_library_is_well_formed():
    for name, gait in GAITS.items():
        assert gait.name == name
        assert set(gait.phase_offsets) == set(LEGS)
        assert 0.0 < gait.duty_factor <= 1.0
        np.testing.assert_allclose(
            gait.stance_duration + gait.swing_duration, gait.period, atol=1e-12
        )


def test_gait_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="周期必须为正"):
        GaitDefinition("bad", -1.0, 0.5, {leg: 0.0 for leg in LEGS})
    with pytest.raises(ValueError, match="占空比"):
        GaitDefinition("bad", 1.0, 1.5, {leg: 0.0 for leg in LEGS})
    with pytest.raises(ValueError, match="缺少"):
        GaitDefinition("bad", 1.0, 0.5, {"FL": 0.0})
    with pytest.raises(ValueError, match="相位偏移必须"):
        GaitDefinition("bad", 1.0, 0.5, {leg: 1.5 for leg in LEGS})


def test_get_gait_rejects_unknown_name():
    with pytest.raises(KeyError, match="未知步态"):
        get_gait("moonwalk")


def test_only_high_duty_gaits_can_be_statically_stable():
    """D >= 0.75 是能否保证三腿支撑的分水岭。"""
    assert get_gait("stand").is_statically_stable_capable()
    assert get_gait("crawl").is_statically_stable_capable()
    for name in ("trot", "pace", "bound", "gallop"):
        assert not get_gait(name).is_statically_stable_capable()


# --------------------------------------------------------------------------
# 各步态的定义性质
# --------------------------------------------------------------------------


def _sample(gait, n=997):
    """在一个周期内密采样，997 是质数，避开与相位偏移的公约数。"""
    sched = GaitScheduler(gait)
    times = np.linspace(0.0, gait.period, n, endpoint=False)
    return sched, times, np.array([sched.contact(t) for t in times])


def test_trot_moves_diagonal_pairs_together():
    """trot 的定义：FL+RR 一组，FR+RL 一组，两组严格互补。"""
    _, _, contact = _sample(get_gait("trot"))
    fl, fr, rl, rr = (LEGS.index(x) for x in ("FL", "FR", "RL", "RR"))
    np.testing.assert_array_equal(contact[:, fl], contact[:, rr])
    np.testing.assert_array_equal(contact[:, fr], contact[:, rl])
    assert np.all(contact[:, fl] != contact[:, fr])
    assert set(np.unique(contact.sum(axis=1))) == {2}


def test_pace_moves_lateral_pairs_together():
    """pace 的定义：同侧腿一组。与 trot 只差配对方式。"""
    _, _, contact = _sample(get_gait("pace"))
    fl, fr, rl, rr = (LEGS.index(x) for x in ("FL", "FR", "RL", "RR"))
    np.testing.assert_array_equal(contact[:, fl], contact[:, rl])
    np.testing.assert_array_equal(contact[:, fr], contact[:, rr])
    assert set(np.unique(contact.sum(axis=1))) == {2}


def test_bound_moves_front_and_rear_pairs_together():
    _, _, contact = _sample(get_gait("bound"))
    fl, fr, rl, rr = (LEGS.index(x) for x in ("FL", "FR", "RL", "RR"))
    np.testing.assert_array_equal(contact[:, fl], contact[:, fr])
    np.testing.assert_array_equal(contact[:, rl], contact[:, rr])


def test_crawl_always_has_exactly_three_stance_legs():
    """crawl 的存在意义：任意时刻三条腿支撑，因而可能静态稳定。"""
    _, _, contact = _sample(get_gait("crawl"))
    assert set(np.unique(contact.sum(axis=1))) == {3}


def test_stand_keeps_all_feet_down():
    _, _, contact = _sample(get_gait("stand"))
    assert np.all(contact)


def test_low_duty_gaits_have_flight_phases():
    """占空比低于 0.25 时必然出现四脚离地的腾空相。"""
    for name in ("bound", "gallop"):
        _, _, contact = _sample(get_gait(name))
        assert contact.sum(axis=1).min() == 0, f"{name} 应当存在腾空相"


def test_duty_factor_matches_measured_stance_fraction():
    """占空比的定义就是支撑时间占比，直接统计核对。"""
    for name, gait in GAITS.items():
        _, _, contact = _sample(gait, n=10007)
        measured = contact.mean()
        np.testing.assert_allclose(measured, gait.duty_factor, atol=2e-3), name


# --------------------------------------------------------------------------
# 调度器的自洽性：下游模块依赖这些查询互不矛盾
# --------------------------------------------------------------------------


def test_leg_phase_wraps_within_unit_interval():
    sched = GaitScheduler("trot")
    for t in np.linspace(0.0, 3.0, 500):
        phase = sched.leg_phase(t)
        assert np.all((phase >= 0.0) & (phase < 1.0))


def test_contact_agrees_with_leg_phase_and_duty():
    for gait in GAITS.values():
        sched = GaitScheduler(gait)
        for t in np.linspace(0.0, 2 * gait.period, 311):
            np.testing.assert_array_equal(
                sched.contact(t), sched.leg_phase(t) < gait.duty_factor
            )


def test_swing_and_stance_phase_are_complementary():
    """每条腿在任一时刻只能处于其中一个相，另一个必须是 NaN。"""
    for gait in GAITS.values():
        sched = GaitScheduler(gait)
        for t in np.linspace(0.0, 2 * gait.period, 211):
            sw, st = sched.swing_phase(t), sched.stance_phase(t)
            assert np.all(np.isnan(sw) != np.isnan(st))
            valid_sw = sw[~np.isnan(sw)]
            valid_st = st[~np.isnan(st)]
            assert np.all((valid_sw >= 0.0) & (valid_sw < 1.0))
            assert np.all((valid_st >= 0.0) & (valid_st < 1.0))


def test_swing_phase_progresses_monotonically_within_a_swing():
    """摆动进度必须从 0 单调增到 1，否则 M5 的足端轨迹会来回抖。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    i = LEGS.index("FR")
    times = np.linspace(0.0, gait.period, 2000, endpoint=False)
    sw = np.array([sched.swing_phase(t)[i] for t in times])
    valid = ~np.isnan(sw)
    seq = sw[valid]
    assert np.all(np.diff(seq) > 0), "同一次摆动内进度应严格递增"
    assert seq[0] < 1e-2 and seq[-1] > 0.99


def test_time_to_touchdown_is_consistent_with_contact():
    """支撑腿的落地倒计时为 0；摆动腿的倒计时必须真的对应落地时刻。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    for t in np.linspace(0.0, 2 * gait.period, 173):
        contact = sched.contact(t)
        ttd = sched.time_to_touchdown(t)
        np.testing.assert_allclose(ttd[contact], 0.0, atol=1e-12)
        for i in np.flatnonzero(~contact):
            assert ttd[i] > 0.0
            # 倒计时结束的那一刻必须已经落地
            assert sched.contact(t + ttd[i] + 1e-9)[i]
            # 提前一点点则还没落地
            assert not sched.contact(t + ttd[i] - 1e-4)[i]


def test_time_to_liftoff_is_consistent_with_contact():
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    for t in np.linspace(0.0, 2 * gait.period, 173):
        contact = sched.contact(t)
        ttl = sched.time_to_liftoff(t)
        np.testing.assert_allclose(ttl[~contact], 0.0, atol=1e-12)
        for i in np.flatnonzero(contact):
            assert not sched.contact(t + ttl[i] + 1e-9)[i]
            assert sched.contact(t + ttl[i] - 1e-4)[i]


def test_next_touchdown_time_matches_time_to_touchdown():
    sched = GaitScheduler("trot")
    for t in np.linspace(0.0, 1.0, 97):
        swing = ~sched.contact(t)
        expected = t + sched.time_to_touchdown(t)
        np.testing.assert_allclose(
            sched.next_touchdown_time(t)[swing], expected[swing], atol=1e-12
        )


# --------------------------------------------------------------------------
# MPC 接口：接触序列
# --------------------------------------------------------------------------


def test_contact_schedule_matches_pointwise_contact():
    """批量查询与逐点查询必须给出完全相同的结果。"""
    for gait in GAITS.values():
        sched = GaitScheduler(gait)
        dt, n = gait.period / 7.3, 20
        schedule = sched.contact_schedule(0.31, dt, n)
        for k in range(n):
            np.testing.assert_array_equal(schedule[k], sched.contact(0.31 + k * dt))


def test_contact_schedule_shape_and_validation():
    sched = GaitScheduler("trot")
    assert sched.contact_schedule(0.0, 0.03, 12).shape == (12, 4)
    with pytest.raises(ValueError, match="预测步数必须为正"):
        sched.contact_schedule(0.0, 0.03, 0)


def test_contact_schedule_covers_a_full_cycle():
    """预测时域跨过整个周期时，每条腿都应既出现支撑也出现摆动。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    schedule = sched.contact_schedule(0.0, gait.period / 12, 24)
    assert np.all(schedule.any(axis=0)), "每条腿都应有支撑步"
    assert np.all(~schedule.all(axis=0)), "每条腿都应有摆动步"


# --------------------------------------------------------------------------
# 步态切换
# --------------------------------------------------------------------------


def test_requested_gait_switch_waits_for_cycle_boundary():
    """切换必须等到周期边界 —— 中途切换会让支撑脚瞬间失去约束。"""
    gait = get_gait("trot")
    sched = GaitScheduler(gait)
    sched.update(0.0)
    sched.request_gait("crawl")

    dt = 1e-3
    switched_at = None
    for k in range(1, int(2 * gait.period / dt)):
        t = k * dt
        if sched.update(t):
            switched_at = t
            break
    assert switched_at is not None, "两个周期内应完成切换"
    assert switched_at > 0.5 * gait.period, "不应在周期中途切换"
    assert sched.gait.name == "crawl"


def test_no_switch_requested_means_no_change():
    sched = GaitScheduler("trot")
    for t in np.linspace(0.0, 2.0, 500):
        assert not sched.update(t)
    assert sched.gait.name == "trot"


def test_force_gait_resets_phase():
    sched = GaitScheduler("trot")
    sched.force_gait("bound", t=1.234)
    assert sched.gait.name == "bound"
    np.testing.assert_allclose(sched.global_phase(1.234), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 凸包与支撑多边形
# --------------------------------------------------------------------------


def test_convex_hull_of_a_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    hull = convex_hull_2d(pts)
    assert len(hull) == 4
    assert not any(np.allclose(h, [0.5, 0.5]) for h in hull), "内点不应出现在凸包上"


def test_convex_hull_handles_degenerate_input():
    """支撑多边形经常退化 —— trot 只有两个接触点，不能抛异常。"""
    assert convex_hull_2d(np.zeros((0, 2))).shape[0] == 0
    assert len(convex_hull_2d(np.array([[1.0, 2.0]]))) == 1
    assert len(convex_hull_2d(np.array([[0.0, 0.0], [1.0, 1.0]]))) == 2
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert len(convex_hull_2d(collinear)) <= 3


def test_point_in_polygon():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert point_in_polygon(np.array([0.5, 0.5]), square)
    assert not point_in_polygon(np.array([1.5, 0.5]), square)
    assert not point_in_polygon(np.array([0.5, 0.5]), square[:2])  # 退化


def test_support_polygon_tracks_contact_set(go2_stance):
    _, feet = go2_stance
    assert len(support_polygon(feet, np.ones(4, dtype=bool))) == 4
    assert len(support_polygon(feet, np.array([1, 0, 0, 1], dtype=bool))) == 2
    assert support_polygon(feet, np.zeros(4, dtype=bool)).shape == (0, 2)


# --------------------------------------------------------------------------
# 静态稳定裕度：步态选择背后的物理
# --------------------------------------------------------------------------


def test_standing_is_comfortably_stable(go2_stance):
    com, feet = go2_stance
    margin = static_stability_margin(com[:2], feet, np.ones(4, dtype=bool))
    np.testing.assert_allclose(margin, 0.142, atol=0.005)


def test_flight_phase_has_no_support(go2_stance):
    com, feet = go2_stance
    assert static_stability_margin(com[:2], feet, np.zeros(4, dtype=bool)) == -np.inf


def test_two_contact_gaits_are_never_statically_stable(go2_stance):
    """两条腿的支撑多边形退化成线段，裕度恒为非正 —— 几何决定，非数值问题。"""
    com, feet = go2_stance
    for name in ("trot", "pace"):
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        margins = [
            static_stability_margin(com[:2], feet, sched.contact(t))
            for t in np.linspace(0.0, gait.period, 200, endpoint=False)
        ]
        assert max(margins) <= 1e-12, f"{name} 不应出现正裕度"


def test_pace_is_far_less_stable_than_trot(go2_stance):
    """本里程碑的核心定量结论：trot 的对角支撑线几乎穿过质心，pace 差 17 倍。

    这就是 trot 成为四足默认中速步态、而 pace 会让机器人明显左右晃的原因。
    """
    com, feet = go2_stance

    def worst(name):
        gait = get_gait(name)
        sched = GaitScheduler(gait)
        return min(
            static_stability_margin(com[:2], feet, sched.contact(t))
            for t in np.linspace(0.0, gait.period, 200, endpoint=False)
        )

    trot_margin, pace_margin = worst("trot"), worst("pace")
    np.testing.assert_allclose(trot_margin, -0.0084, atol=0.002)
    np.testing.assert_allclose(pace_margin, -0.142, atol=0.005)
    assert pace_margin < 10 * trot_margin, "pace 应比 trot 差一个数量级以上"


def test_crawl_margin_hovers_around_zero_at_centred_com(go2_stance):
    """质心位于几何中心时，crawl 的三腿支撑三角形的一条边恰好经过它。

    这不是数值巧合：矩形的对角线就是去掉一个角之后三角形的一条边。
    真实的 crawl 步态必须**把质心朝支撑三角形一侧挪**，这正是四足慢走
    时左右摇摆的原因。
    """
    com, feet = go2_stance
    gait = get_gait("crawl")
    sched = GaitScheduler(gait)
    margins = np.array(
        [
            static_stability_margin(com[:2], feet, sched.contact(t))
            for t in np.linspace(0.0, gait.period, 400, endpoint=False)
        ]
    )
    assert margins.min() < 0.0, "居中质心下 crawl 有一半时间是临界不稳的"
    assert abs(margins).max() < 0.02, "但偏离量很小，稍微挪一点质心就能救回来"


def test_shifting_com_into_the_triangle_restores_stability(go2_stance):
    """把质心朝支撑三角形的形心方向挪，裕度立刻转正 —— 慢走摇摆的原理。"""
    com, feet = go2_stance
    gait = get_gait("crawl")
    sched = GaitScheduler(gait)
    improved = 0
    total = 0
    for t in np.linspace(0.0, gait.period, 200, endpoint=False):
        contact = sched.contact(t)
        poly = support_polygon(feet, contact)
        centroid = poly.mean(axis=0)
        base = static_stability_margin(com[:2], feet, contact)
        # 朝形心挪 3 cm
        direction = centroid - com[:2]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        shifted = static_stability_margin(com[:2] + 0.03 * direction, feet, contact)
        total += 1
        if shifted > base:
            improved += 1
    assert improved == total, "朝形心挪动应当总是改善裕度"
