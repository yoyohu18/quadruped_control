"""里程碑 1（运动学）的验证。

本仓库自始至终的策略是*在两份独立实现之间做交叉验证*：一份是手工推导的
闭式代码，一份是由 URDF 驱动的 Pinocchio 代码。符号写反、连杆长度用错，
都不可能同时在两边存活，因为它们是从不同来源写出来的。

运行::

    pytest tests/test_kinematics.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from kinematics import (
    HIP_OFFSETS,
    ISAAC_JOINT_ORDER,
    LEG_GEOMETRY,
    LEGS,
    PINOCCHIO_JOINT_ORDER,
    forward_kinematics,
    inverse_kinematics,
    isaac_to_pinocchio,
    leg_jacobian,
    load_go2,
    pinocchio_to_isaac,
)
from kinematics.go2 import ABAD_OFFSET_Y, CALF_LENGTH, THIGH_LENGTH

N_SAMPLES = 300
SEED = 0


@pytest.fixture(scope="module")
def model():
    return load_go2(floating_base=False)


@pytest.fixture(scope="module")
def float_model():
    return load_go2(floating_base=True)


@pytest.fixture(scope="module")
def samples(model):
    """URDF 限位内的随机关节构型。"""
    rng = np.random.default_rng(SEED)
    return np.array([model.random_joint_configuration(rng) for _ in range(N_SAMPLES)])


def leg_slice(q_all: np.ndarray, leg: str) -> np.ndarray:
    """从 Pinocchio 顺序的 12 维向量中取出 ``leg`` 的三个关节角。"""
    i = LEGS.index(leg) * 3
    return q_all[i : i + 3]


def in_plane_extension(q: np.ndarray, geom) -> float:
    """``l1*cos(q1) + l2*cos(q1+q2)`` —— 足端在髋下方时为正。"""
    return geom.l1 * np.cos(q[1]) + geom.l2 * np.cos(q[1] + q[2])


def wrap(a: np.ndarray) -> np.ndarray:
    """把角度折算到 (-pi, pi] 区间。"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# URDF 是唯一真值来源：硬编码常数必须与之一致。
# --------------------------------------------------------------------------


def test_model_dimensions(model, float_model):
    assert (model.nq, model.nv) == (12, 12)
    assert (float_model.nq, float_model.nv) == (19, 18)


def test_joint_order_matches_urdf(model):
    assert tuple(model.model.names[1:]) == PINOCCHIO_JOINT_ORDER


def test_hardcoded_geometry_matches_urdf(model):
    """防止常数与它们所来源的 URDF 悄悄脱节。"""
    m = model.model
    for leg in LEGS:
        abad = m.jointPlacements[m.getJointId(f"{leg}_hip_joint")].translation
        thigh = m.jointPlacements[m.getJointId(f"{leg}_thigh_joint")].translation
        calf = m.jointPlacements[m.getJointId(f"{leg}_calf_joint")].translation
        np.testing.assert_allclose(abad, HIP_OFFSETS[leg], atol=1e-12)
        np.testing.assert_allclose(thigh[1], LEG_GEOMETRY[leg].l0, atol=1e-12)
        np.testing.assert_allclose(abs(thigh[1]), ABAD_OFFSET_Y, atol=1e-12)
        np.testing.assert_allclose(-calf[2], THIGH_LENGTH, atol=1e-12)

    foot = m.frames[m.getFrameId("FL_foot")].placement.translation
    np.testing.assert_allclose(-foot[2], CALF_LENGTH, atol=1e-12)


# --------------------------------------------------------------------------
# 正运动学
# --------------------------------------------------------------------------


def test_analytic_fk_matches_pinocchio(model, samples):
    """闭式 FK == Pinocchio FK，四条腿，覆盖整个关节量程。"""
    for q_all in samples:
        feet = model.foot_positions(q_all)
        for leg in LEGS:
            p_analytic = forward_kinematics(leg_slice(q_all, leg), LEG_GEOMETRY[leg])
            p_pin = feet[leg] - HIP_OFFSETS[leg]  # Pinocchio 给的是躯干系下的结果
            np.testing.assert_allclose(p_analytic, p_pin, atol=1e-12)


def test_fk_at_zero_configuration(model):
    """q = 0 时腿完全伸直垂下 —— 这是任何人都能手算核对的数值。"""
    for leg in LEGS:
        p = forward_kinematics(np.zeros(3), LEG_GEOMETRY[leg])
        expected = np.array([0.0, LEG_GEOMETRY[leg].l0, -(THIGH_LENGTH + CALF_LENGTH)])
        np.testing.assert_allclose(p, expected, atol=1e-12)


def test_abduction_preserves_distance_to_hip_axis():
    """q0 是绕 +x 的旋转，因此不可能改变 x，也不可能改变 y-z 平面内的半径。"""
    geom = LEG_GEOMETRY["FL"]
    base = forward_kinematics(np.array([0.0, 0.7, -1.4]), geom)
    for q0 in np.linspace(-1.0, 1.0, 21):
        p = forward_kinematics(np.array([q0, 0.7, -1.4]), geom)
        np.testing.assert_allclose(p[0], base[0], atol=1e-12)
        np.testing.assert_allclose(np.hypot(*p[1:]), np.hypot(*base[1:]), atol=1e-12)


# --------------------------------------------------------------------------
# 雅可比
# --------------------------------------------------------------------------


def test_analytic_jacobian_matches_finite_differences(samples):
    """解析雅可比确实是解析 FK 的导数。"""
    eps = 1e-6
    for q_all in samples[:50]:
        for leg in LEGS:
            geom = LEG_GEOMETRY[leg]
            q = leg_slice(q_all, leg)
            J = leg_jacobian(q, geom)
            J_fd = np.zeros((3, 3))
            for i in range(3):
                dq = np.zeros(3)
                dq[i] = eps
                J_fd[:, i] = (forward_kinematics(q + dq, geom) - forward_kinematics(q - dq, geom)) / (2 * eps)
            np.testing.assert_allclose(J, J_fd, atol=1e-7)


def test_analytic_jacobian_matches_pinocchio(model, samples):
    """闭式雅可比 == Pinocchio 的 LOCAL_WORLD_ALIGNED 帧雅可比。"""
    for q_all in samples:
        for leg in LEGS:
            J = leg_jacobian(leg_slice(q_all, leg), LEG_GEOMETRY[leg])
            J_pin = model.foot_jacobian(q_all, leg)
            np.testing.assert_allclose(J, J_pin, atol=1e-12)


def test_jacobian_maps_joint_velocity_to_foot_velocity(model, samples):
    """用恒定 qdot 推进 q，检查足端是否按预测运动。"""
    rng = np.random.default_rng(SEED + 1)
    dt = 1e-6
    for q_all in samples[:20]:
        for leg in LEGS:
            geom = LEG_GEOMETRY[leg]
            q = leg_slice(q_all, leg)
            qdot = rng.normal(size=3)
            v_predicted = leg_jacobian(q, geom) @ qdot
            v_actual = (
                forward_kinematics(q + dt * qdot, geom) - forward_kinematics(q - dt * qdot, geom)
            ) / (2 * dt)
            np.testing.assert_allclose(v_predicted, v_actual, atol=1e-8)


def test_jacobian_is_singular_when_leg_is_fully_extended():
    """膝完全伸直是经典的腿部奇异位形：秩降为 2。"""
    geom = LEG_GEOMETRY["FL"]
    J = leg_jacobian(np.array([0.0, 0.0, 0.0]), geom)  # 膝角为 0 即完全伸直
    assert abs(np.linalg.det(J)) < 1e-12
    assert np.linalg.matrix_rank(J, tol=1e-9) == 2


def test_jacobian_conditioning_degrades_towards_extension():
    """膝越伸直，条件数必须单调增大。"""
    geom = LEG_GEOMETRY["FL"]
    conds = [np.linalg.cond(leg_jacobian(np.array([0.0, 0.8, k]), geom)) for k in np.linspace(-1.6, -0.2, 8)]
    assert all(b > a for a, b in zip(conds, conds[1:]))


# --------------------------------------------------------------------------
# 逆运动学
# --------------------------------------------------------------------------


def test_ik_round_trip(samples):
    """在"腿向下、膝向后"解支上，q -> FK -> IK 应还原出原来的 q。"""
    checked = 0
    for q_all in samples:
        for leg in LEGS:
            geom = LEG_GEOMETRY[leg]
            q = leg_slice(q_all, leg)
            # 只取本身就落在求解器目标解支上的构型。
            if q[2] > -1e-6 or in_plane_extension(q, geom) <= 1e-6:
                continue
            q_rec = inverse_kinematics(forward_kinematics(q, geom), geom)
            np.testing.assert_allclose(wrap(q_rec), wrap(q), atol=1e-9)
            checked += 1
    assert checked > 100, f"只有 {checked} 个样本参与了往返检验，覆盖不足"


def test_ik_returns_mirror_branch_for_folded_configurations(samples):
    """解支之外的输入会返回不同的 q，但该 q 的 FK 仍精确命中同一点。

    这才是真正重要的性质：即使求解器无法还原你最初的关节角，它也绝不会
    在足端位置上说谎。这一点是被写进文档的行为，而不是 bug。
    """
    found_mirror = False
    for q_all in samples:
        for leg in LEGS:
            geom = LEG_GEOMETRY[leg]
            q = leg_slice(q_all, leg)
            # 避开解支边界 —— 在那里两个解会合并到一起。
            if in_plane_extension(q, geom) >= -0.01:
                continue  # 足端折叠到侧摆轴上方
            p = forward_kinematics(q, geom)
            q_rec = inverse_kinematics(p, geom)
            np.testing.assert_allclose(forward_kinematics(q_rec, geom), p, atol=1e-9)
            assert np.linalg.norm(wrap(q_rec - q)) > 1e-3
            found_mirror = True
    assert found_mirror, "没有采到折叠向上的样本，本测试形同虚设"


def test_ik_then_fk_reproduces_target(model):
    """直接采样足端位置，检查 IK 是否精确落到这些点上。"""
    rng = np.random.default_rng(SEED + 2)
    for leg in LEGS:
        geom = LEG_GEOMETRY[leg]
        for _ in range(200):
            target = np.array(
                [
                    rng.uniform(-0.12, 0.12),
                    geom.l0 + rng.uniform(-0.06, 0.06),
                    rng.uniform(-0.36, -0.18),
                ]
            )
            assert np.linalg.norm(target) < geom.reach_max, "采样目标必须可达"
            q = inverse_kinematics(target, geom)
            np.testing.assert_allclose(forward_kinematics(q, geom), target, atol=1e-9)


def test_ik_solution_respects_urdf_joint_limits(model):
    """规划器真正会请求的姿态必须落在关节限位之内。

    下面这个矩形范围是可用的站姿工作空间。它明显小于纯几何可达范围：
    Go2 的膝限位 -0.838 rad 把腿的伸展量封顶在约 0.389 m，
    远小于 l1 + l2 = 0.426 m。
    """
    lo, hi = model.model.lowerPositionLimit, model.model.upperPositionLimit
    for leg in LEGS:
        geom = LEG_GEOMETRY[leg]
        i = LEGS.index(leg) * 3
        for z in np.linspace(-0.36, -0.20, 10):
            for x in np.linspace(-0.10, 0.10, 7):
                q = inverse_kinematics(np.array([x, geom.l0, z]), geom)
                assert np.all(q >= lo[i : i + 3] - 1e-9), f"{leg} 低于下限：{q}"
                assert np.all(q <= hi[i : i + 3] + 1e-9), f"{leg} 高于上限：{q}"


def test_knee_limit_caps_reach_below_geometric_maximum():
    """把上一个测试所依赖的那个差距定量化。"""
    geom = LEG_GEOMETRY["FL"]
    knee_limit = -0.83776
    reach_limited = np.sqrt(geom.l1**2 + geom.l2**2 + 2 * geom.l1 * geom.l2 * np.cos(knee_limit))
    assert reach_limited < geom.reach_max
    np.testing.assert_allclose(reach_limited, 0.3892, atol=1e-4)


def test_ik_rejects_unreachable_target():
    geom = LEG_GEOMETRY["FL"]
    with pytest.raises(ValueError, match="unreachable"):
        inverse_kinematics(np.array([0.0, geom.l0, -1.0]), geom)  # 远在地面以下
    with pytest.raises(ValueError, match="abduction cylinder"):
        inverse_kinematics(np.array([0.0, 0.0, 0.0]), geom)  # 正好落在侧摆轴上


def test_ik_clamp_never_raises():
    """clamp=True 是控制回路的安全网：永远返回一个可用的 q。"""
    geom = LEG_GEOMETRY["FL"]
    for target in ([0.0, geom.l0, -1.0], [0.0, 0.0, 0.0], [2.0, 2.0, -2.0]):
        q = inverse_kinematics(np.array(target), geom, clamp=True)
        assert np.all(np.isfinite(q))


def test_numeric_ik_agrees_with_closed_form(model):
    """阻尼最小二乘应收敛到与解析 IK 相同的关节角。"""
    rng = np.random.default_rng(SEED + 3)
    for leg in LEGS:
        geom = LEG_GEOMETRY[leg]
        for _ in range(10):
            q_true = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.4, 1.2), rng.uniform(-2.2, -0.9)])
            target_base = forward_kinematics(q_true, geom) + HIP_OFFSETS[leg]
            q_init = model.neutral()
            q_init[LEGS.index(leg) * 3 : LEGS.index(leg) * 3 + 3] = [0.0, 0.8, -1.5]
            q_num, converged = model.inverse_kinematics_numeric(target_base, leg, q_init=q_init)
            assert converged, f"{leg}：阻尼最小二乘 IK 未收敛"
            np.testing.assert_allclose(leg_slice(q_num, leg), q_true, atol=1e-6)


# --------------------------------------------------------------------------
# 浮动基座与坐标系约定
# --------------------------------------------------------------------------


def test_floating_base_foot_position_transforms_correctly(float_model, samples):
    """在非平凡的躯干位姿下，世界系与躯干系之间的足端位置变换必须自洽。"""
    rng = np.random.default_rng(SEED + 4)
    for q_joints in samples[:20]:
        pos = rng.uniform(-1.0, 1.0, 3)
        quat = pin.Quaternion(pin.exp3(rng.normal(size=3)))
        q = float_model.make_configuration(
            q_joints, base_position=pos, base_quaternion_xyzw=quat.coeffs()
        )
        base_T = pin.SE3(quat, pos)
        for leg in LEGS:
            p_world = float_model.foot_position(q, leg)
            p_base = float_model.foot_position_in_base(q, leg)
            np.testing.assert_allclose(p_world, base_T.act(p_base), atol=1e-12)
            # 而且躯干系下的位置必须等于固定基座模型给出的答案。
            p_analytic = forward_kinematics(leg_slice(q_joints, leg), LEG_GEOMETRY[leg])
            np.testing.assert_allclose(p_base, p_analytic + HIP_OFFSETS[leg], atol=1e-12)


def test_full_jacobian_contains_leg_jacobian(float_model, samples):
    """整机雅可比中属于该腿的分块，必须等于闭式腿部雅可比。"""
    for q_joints in samples[:10]:
        q = float_model.make_configuration(q_joints)
        for leg in LEGS:
            J_full = float_model.full_foot_jacobian(q, leg)
            J_leg = J_full[:, float_model.leg_v_index[leg]]
            np.testing.assert_allclose(J_leg, leg_jacobian(leg_slice(q_joints, leg), LEG_GEOMETRY[leg]), atol=1e-12)
        # 浮动基座模型的基座列绝不可能全为零。
        assert np.linalg.norm(float_model.full_foot_jacobian(q, "FL")[:, :6]) > 0.1


# --------------------------------------------------------------------------
# Pinocchio 与 Isaac Lab 之间的关节顺序
# --------------------------------------------------------------------------


def test_joint_order_maps_are_inverse():
    q = np.arange(12.0)
    np.testing.assert_array_equal(pinocchio_to_isaac(isaac_to_pinocchio(q)), q)
    np.testing.assert_array_equal(isaac_to_pinocchio(pinocchio_to_isaac(q)), q)


def test_joint_order_map_matches_names():
    """对关节**名字**做重排，检查映射是否把它们对齐。"""
    names_isaac = np.array(ISAAC_JOINT_ORDER)
    assert tuple(isaac_to_pinocchio(names_isaac)) == PINOCCHIO_JOINT_ORDER
    names_pin = np.array(PINOCCHIO_JOINT_ORDER)
    assert tuple(pinocchio_to_isaac(names_pin)) == ISAAC_JOINT_ORDER


def test_orderings_are_actually_different():
    """一旦两种顺序变得相同，重排映射就成了死代码，这个测试必须报警。"""
    assert ISAAC_JOINT_ORDER != PINOCCHIO_JOINT_ORDER
