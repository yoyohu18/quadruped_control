"""里程碑 2（刚体动力学）的验证。

延续里程碑 1 的策略，但这里"两份独立实现"的含义更丰富：完整动力学的
每一项都由**另一条互不相干的路径**去校验 ——

* 质量矩阵 M   <- 动能的二次型
* 重力项 g     <- 势能在李群上的梯度
* RNEA         <- M a + C v + g
* ABA          <- RNEA 的逆
* 科氏矩阵 C   <- Mdot - 2C 的反对称性
* 质心动量     <- 数值微分
* 单刚体模型   <- 整机模型的质心动力学

运行::

    pytest tests/test_dynamics.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from dynamics import (
    FLOATING_BASE_DOF,
    RigidBodyDynamics,
    SRBDParams,
    contact_wrench,
    load_go2_dynamics,
    matrix_to_rpy,
    nominal_configuration,
    rpy_rates_from_angular_velocity,
    rpy_to_matrix,
    skew,
    srbd_acceleration,
    srbd_acceleration_mpc,
    srbd_params_from_model,
)
from kinematics import LEGS

SEED = 0
N_SAMPLES = 40


@pytest.fixture(scope="module")
def rbd():
    return load_go2_dynamics(floating_base=True)


@pytest.fixture(scope="module")
def states(rbd):
    """随机的 (q, v, a) 三元组，躯干姿态取一般位形而非近似水平。"""
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_SAMPLES):
        q = pin.randomConfiguration(rbd._m)
        q[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        q[3:7] = quat / np.linalg.norm(quat)
        v = rng.normal(size=rbd.nv) * 0.8
        a = rng.normal(size=rbd.nv) * 0.8
        out.append((q, v, a))
    return out


# --------------------------------------------------------------------------
# 模型基本属性
# --------------------------------------------------------------------------


def test_model_dimensions(rbd):
    assert (rbd.nq, rbd.nv) == (19, 18)
    assert len(rbd.actuated_dofs) == 12
    assert len(rbd.underactuated_dofs) == FLOATING_BASE_DOF


def test_total_mass_is_sum_of_link_masses(rbd):
    link_sum = sum(rbd._m.inertias[i].mass for i in range(1, rbd._m.njoints))
    np.testing.assert_allclose(rbd.total_mass, link_sum, atol=1e-12)
    assert 15.0 < rbd.total_mass < 17.0, "Go2 整机质量应在 16 kg 附近"


def test_legs_carry_most_of_the_mass(rbd):
    """腿占了一半以上质量 —— 这正是"腿无质量"假设代价高昂的原因。"""
    trunk = rbd._m.inertias[1].mass
    legs = rbd.total_mass - trunk
    assert legs > trunk, "Go2 四条腿总质量应超过躯干"
    np.testing.assert_allclose(legs / rbd.total_mass, 0.548, atol=0.01)


# --------------------------------------------------------------------------
# 质量矩阵
# --------------------------------------------------------------------------


def test_mass_matrix_is_symmetric_positive_definite(rbd, states):
    """CRBA 只填上三角；忘记补全对称是极常见的 bug。"""
    for q, _, _ in states:
        M = rbd.mass_matrix(q)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(M)) > 0.0


def test_mass_matrix_reproduces_kinetic_energy(rbd, states):
    """M 的定义就是动能的二次型：KE = 0.5 v' M v。"""
    for q, v, _ in states:
        M = rbd.mass_matrix(q)
        np.testing.assert_allclose(0.5 * v @ M @ v, rbd.kinetic_energy(q, v), rtol=1e-12)


def test_mass_matrix_top_left_block_is_the_locked_inertia(rbd):
    """M 的左上 3x3 块必须是 m*I —— 所有关节冻住时，躯干平动只感受到总质量。"""
    q = nominal_configuration(rbd)
    M = rbd.mass_matrix(q)
    np.testing.assert_allclose(M[:3, :3], rbd.total_mass * np.eye(3), atol=1e-10)


def test_mass_matrix_is_independent_of_base_pose(rbd):
    """把机器人整体平移或绕 z 转动，惯量属性不应改变。"""
    q0 = nominal_configuration(rbd)
    M0 = rbd.mass_matrix(q0)
    q1 = q0.copy()
    q1[:3] += np.array([3.0, -2.0, 1.5])  # 纯平移
    np.testing.assert_allclose(rbd.mass_matrix(q1), M0, atol=1e-12)


# --------------------------------------------------------------------------
# 重力项：与势能梯度对照
# --------------------------------------------------------------------------


def test_gravity_is_gradient_of_potential_energy(rbd, states):
    """g(q) = dU/dq。因为浮动基座在李群上，必须用 pin.integrate 做扰动。"""
    eps = 1e-6
    for q, _, _ in states[:10]:
        g = rbd.gravity_vector(q)
        g_num = np.zeros(rbd.nv)
        for i in range(rbd.nv):
            dv = np.zeros(rbd.nv)
            dv[i] = eps
            u_plus = rbd.potential_energy(pin.integrate(rbd._m, q, dv))
            u_minus = rbd.potential_energy(pin.integrate(rbd._m, q, -dv))
            g_num[i] = (u_plus - u_minus) / (2 * eps)
        np.testing.assert_allclose(g, g_num, atol=1e-6)


def test_gravity_vertical_component_equals_weight(rbd):
    """基座竖直方向的重力广义力就是整机重量。"""
    q = nominal_configuration(rbd)
    g = rbd.gravity_vector(q)
    np.testing.assert_allclose(g[2], rbd.total_mass * rbd.gravity, atol=1e-10)


# --------------------------------------------------------------------------
# RNEA / ABA / 科氏项
# --------------------------------------------------------------------------


def test_rnea_equals_mass_matrix_form(rbd, states):
    """RNEA(q,v,a) == M a + C v + g，这是全部控制方程的基础恒等式。"""
    for q, v, a in states:
        tau = rbd.inverse_dynamics(q, v, a)
        M = rbd.mass_matrix(q)
        C = rbd.coriolis_matrix(q, v)
        g = rbd.gravity_vector(q)
        np.testing.assert_allclose(tau, M @ a + C @ v + g, atol=1e-10)


def test_nonlinear_effects_match_coriolis_plus_gravity(rbd, states):
    for q, v, _ in states:
        C = rbd.coriolis_matrix(q, v)
        g = rbd.gravity_vector(q)
        np.testing.assert_allclose(rbd.nonlinear_effects(q, v), C @ v + g, atol=1e-10)


def test_aba_inverts_rnea(rbd, states):
    """正动力学与逆动力学互为逆运算。"""
    for q, v, a in states:
        tau = rbd.inverse_dynamics(q, v, a)
        np.testing.assert_allclose(rbd.forward_dynamics(q, v, tau), a, atol=1e-10)


def test_rnea_at_rest_is_gravity(rbd, states):
    for q, _, _ in states:
        zero = np.zeros(rbd.nv)
        np.testing.assert_allclose(rbd.inverse_dynamics(q, zero, zero), rbd.gravity_vector(q), atol=1e-12)


def test_mdot_minus_2c_is_skew_symmetric(rbd, states):
    """经典性质，也是无源性 / Lyapunov 稳定性证明的基石，必考。"""
    dt = 1e-6
    for q, v, _ in states[:10]:
        C = rbd.coriolis_matrix(q, v)
        M_plus = rbd.mass_matrix(pin.integrate(rbd._m, q, dt * v))
        M_minus = rbd.mass_matrix(pin.integrate(rbd._m, q, -dt * v))
        M_dot = (M_plus - M_minus) / (2 * dt)
        N = M_dot - 2 * C
        np.testing.assert_allclose(N, -N.T, atol=1e-6)


def test_coriolis_vanishes_at_zero_velocity(rbd, states):
    for q, _, _ in states[:10]:
        np.testing.assert_allclose(rbd.coriolis_matrix(q, np.zeros(rbd.nv)), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 欠驱动：整个 WBC 与 MPC 存在的理由
# --------------------------------------------------------------------------


def test_selection_matrix_has_zero_base_columns(rbd):
    """S 的前 6 列全为零 —— 躯干的 6 个自由度上没有电机。"""
    S = rbd.selection_matrix()
    assert S.shape == (12, 18)
    np.testing.assert_allclose(S[:, :FLOATING_BASE_DOF], 0.0, atol=1e-15)
    np.testing.assert_allclose(S[:, FLOATING_BASE_DOF:], np.eye(12), atol=1e-15)


def test_robot_cannot_accelerate_its_com_without_contact(rbd):
    """悬空时，无论电机怎么转，质心加速度只能是重力加速度。

    这是欠驱动最直观的表述，也是"必须靠脚蹬地"的数学证明。
    """
    rng = np.random.default_rng(SEED + 1)
    q = nominal_configuration(rbd)
    q[2] = 2.0  # 悬在空中
    for _ in range(20):
        v = rng.normal(size=rbd.nv) * 0.5
        tau = np.zeros(rbd.nv)
        tau[rbd.actuated_dofs] = rng.normal(size=12) * 30.0  # 关节力矩随便给
        a = rbd.forward_dynamics(q, v, tau)

        # 由质心动量定理，质心线加速度必须恰好是 -g。
        Ag = rbd.centroidal_momentum_matrix(q, v)
        dAg_v = _centroidal_momentum_rate_drift(rbd, q, v)
        h_dot = Ag @ a + dAg_v
        com_acc = h_dot[:3] / rbd.total_mass
        np.testing.assert_allclose(com_acc, [0.0, 0.0, -rbd.gravity], atol=1e-8)


def _centroidal_momentum_rate_drift(rbd, q, v):
    """质心动量变化率中的漂移项 dAg @ v。"""
    return pin.dccrba(rbd._m, rbd._d, q, v) @ v


def test_angular_momentum_is_conserved_in_free_flight(rbd):
    """悬空且无外力时，关于质心的角动量守恒 —— 猫落地翻身的物理基础。"""
    rng = np.random.default_rng(SEED + 2)
    q = nominal_configuration(rbd)
    q[2] = 5.0
    v = np.zeros(rbd.nv)
    v[FLOATING_BASE_DOF:] = rng.normal(size=12) * 1.5  # 只有腿在动

    dt = 1e-4
    h0 = rbd.centroidal_momentum(q, v)[3:].copy()
    for _ in range(200):
        a = rbd.forward_dynamics(q, v, np.zeros(rbd.nv))
        v = v + dt * a
        q = pin.integrate(rbd._m, q, dt * v)
    h1 = rbd.centroidal_momentum(q, v)[3:]
    np.testing.assert_allclose(h1, h0, atol=2e-3)


# --------------------------------------------------------------------------
# 质心量
# --------------------------------------------------------------------------


def test_centroidal_momentum_matrix_maps_velocity(rbd, states):
    for q, v, _ in states:
        Ag = rbd.centroidal_momentum_matrix(q, v)
        np.testing.assert_allclose(Ag @ v, rbd.centroidal_momentum(q, v), atol=1e-12)


def test_linear_momentum_equals_mass_times_com_velocity(rbd, states):
    for q, v, _ in states:
        h = rbd.centroidal_momentum(q, v)
        np.testing.assert_allclose(h[:3], rbd.total_mass * rbd.com_velocity(q, v), atol=1e-12)


def test_com_velocity_is_derivative_of_com(rbd, states):
    dt = 1e-6
    for q, v, _ in states[:10]:
        v_num = (
            rbd.center_of_mass(pin.integrate(rbd._m, q, dt * v))
            - rbd.center_of_mass(pin.integrate(rbd._m, q, -dt * v))
        ) / (2 * dt)
        np.testing.assert_allclose(rbd.com_velocity(q, v), v_num, atol=1e-8)


def test_centroidal_inertia_is_much_larger_than_trunk_inertia(rbd):
    """复合惯量 vs 躯干自身惯量 —— MPC 参数最常见的踩坑点。"""
    q = nominal_configuration(rbd)
    Ig = rbd.centroidal_inertia(q)
    I_trunk = np.array(rbd._m.inertias[1].inertia)
    ratios = np.diag(Ig) / np.diag(I_trunk)
    assert ratios.min() > 4.0, f"复合惯量应显著大于躯干惯量，实测比值 {ratios}"
    np.testing.assert_allclose(Ig, Ig.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(Ig)) > 0.0


# --------------------------------------------------------------------------
# 接触与静力学
# --------------------------------------------------------------------------


def test_contact_jacobian_shape_and_content(rbd):
    q = nominal_configuration(rbd)
    J = rbd.contact_jacobian(q, ("FL", "RR"))
    assert J.shape == (6, 18)
    np.testing.assert_allclose(J[:3], rbd.model.full_foot_jacobian(q, "FL"), atol=1e-12)
    np.testing.assert_allclose(J[3:], rbd.model.full_foot_jacobian(q, "RR"), atol=1e-12)
    assert rbd.contact_jacobian(q, ()).shape == (0, 18)


def test_constrained_dynamics_keeps_stance_feet_still(rbd):
    """支撑约束的定义就是 J a + Jdot v = 0，即足端加速度为零。"""
    rng = np.random.default_rng(SEED + 3)
    q = nominal_configuration(rbd)
    for _ in range(10):
        v = rng.normal(size=rbd.nv) * 0.1
        tau = np.zeros(rbd.nv)
        tau[rbd.actuated_dofs] = rng.normal(size=12) * 10.0
        a, f = rbd.constrained_forward_dynamics(q, v, tau, LEGS)
        J = rbd.contact_jacobian(q, LEGS)
        dJv = rbd.contact_jacobian_dot_v(q, v, LEGS)
        np.testing.assert_allclose(J @ a + dJv, 0.0, atol=1e-7)
        assert f.shape == (12,)


def test_standing_still_requires_body_weight_of_contact_force(rbd):
    """静止站立时，四足垂直力之和必须等于体重。"""
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    tau = np.zeros(rbd.nv)
    tau[rbd.actuated_dofs] = rbd.gravity_compensation_torque(q, LEGS)[1]
    a, f = rbd.constrained_forward_dynamics(q, v, tau, LEGS)
    np.testing.assert_allclose(a, 0.0, atol=1e-7)
    fz_total = f.reshape(4, 3)[:, 2].sum()
    np.testing.assert_allclose(fz_total, rbd.total_mass * rbd.gravity, rtol=1e-8)


def test_gravity_compensation_balances_the_base(rbd):
    """求出的接触力必须恰好平衡掉基座那 6 行的重力广义力。"""
    q = nominal_configuration(rbd)
    f, tau_act = rbd.gravity_compensation_torque(q, LEGS)
    g = rbd.gravity_vector(q)
    J = rbd.contact_jacobian(q, LEGS)
    np.testing.assert_allclose(J[:, :FLOATING_BASE_DOF].T @ f, g[:FLOATING_BASE_DOF], atol=1e-8)
    np.testing.assert_allclose(f.reshape(4, 3)[:, 2].sum(), rbd.total_mass * rbd.gravity, rtol=1e-9)
    assert tau_act.shape == (12,)


def test_symmetric_stance_distributes_load_evenly(rbd):
    """对称站姿下四条腿的垂直力应接近相等（质心略偏后，允许一点差异）。"""
    q = nominal_configuration(rbd)
    f, _ = rbd.gravity_compensation_torque(q, LEGS)
    fz = f.reshape(4, 3)[:, 2]
    expected = rbd.total_mass * rbd.gravity / 4
    np.testing.assert_allclose(fz, expected, rtol=0.12)


# --------------------------------------------------------------------------
# 姿态工具函数
# --------------------------------------------------------------------------


def test_skew_matches_cross_product():
    rng = np.random.default_rng(SEED + 4)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)
        np.testing.assert_allclose(skew(a), -skew(a).T, atol=1e-14)


def test_rpy_matches_pinocchio():
    """自己写的 rpy 约定必须与 Pinocchio 一致，否则和整机模型对不上。"""
    rng = np.random.default_rng(SEED + 5)
    for _ in range(50):
        rpy = rng.uniform(-1.2, 1.2, 3)
        np.testing.assert_allclose(rpy_to_matrix(rpy), pin.rpy.rpyToMatrix(rpy), atol=1e-12)
        np.testing.assert_allclose(matrix_to_rpy(rpy_to_matrix(rpy)), rpy, atol=1e-12)


def test_rpy_rates_match_finite_differences():
    """欧拉角速率 != 角速度。这个测试就是用来钉死这一点的。"""
    rng = np.random.default_rng(SEED + 6)
    dt = 1e-7
    for _ in range(30):
        rpy = rng.uniform(-1.0, 1.0, 3)
        omega = rng.normal(size=3)
        rpy_dot = rpy_rates_from_angular_velocity(rpy, omega)

        R = rpy_to_matrix(rpy)
        R_next = pin.exp3(omega * dt) @ R  # 世界系角速度作用在左侧
        rpy_dot_num = (matrix_to_rpy(R_next) - rpy) / dt
        np.testing.assert_allclose(rpy_dot, rpy_dot_num, atol=1e-5)


def test_angular_velocity_differs_from_rpy_rates():
    """确认两者确实不同 —— 否则上一个测试形同虚设。"""
    rpy = np.array([0.3, 0.5, 0.8])
    omega = np.array([1.0, 0.0, 0.0])
    assert np.linalg.norm(rpy_rates_from_angular_velocity(rpy, omega) - omega) > 0.1


def test_rpy_rates_raise_at_gimbal_lock():
    with pytest.raises(ValueError, match="万向锁"):
        rpy_rates_from_angular_velocity(np.array([0.0, np.pi / 2, 0.0]), np.ones(3))


# --------------------------------------------------------------------------
# 单刚体模型（SRBD）—— 凸 MPC 的模型
# --------------------------------------------------------------------------


def test_srbd_params_rejects_invalid_inertia():
    with pytest.raises(ValueError, match="对称"):
        SRBDParams(mass=1.0, inertia_body=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="正定"):
        SRBDParams(mass=1.0, inertia_body=np.diag([1.0, 1.0, -1.0]))


def test_srbd_params_extracted_from_model(rbd):
    params = srbd_params_from_model(rbd)
    np.testing.assert_allclose(params.mass, rbd.total_mass, atol=1e-12)
    q = nominal_configuration(rbd)
    np.testing.assert_allclose(params.inertia_body, rbd.centroidal_inertia(q), atol=1e-12)


def test_contact_wrench_sums_correctly():
    forces = np.array([[0.0, 0.0, 40.0], [0.0, 0.0, 40.0]])
    feet = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    com = np.array([0.0, 0.0, 0.3])
    f, tau = contact_wrench(forces, feet, com)
    np.testing.assert_allclose(f, [0.0, 0.0, 80.0], atol=1e-12)
    np.testing.assert_allclose(tau, 0.0, atol=1e-12)  # 对称布置，力矩抵消


def test_contact_wrench_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="数量不一致"):
        contact_wrench(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros(3))


def test_srbd_static_equilibrium(rbd):
    """用整机模型解出的平衡接触力喂给单刚体，加速度必须严格为零。"""
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    forces = rbd.gravity_compensation_torque(q, LEGS)[0].reshape(4, 3)

    lin, ang = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
    np.testing.assert_allclose(lin, 0.0, atol=1e-9)
    np.testing.assert_allclose(ang, 0.0, atol=1e-9)


def test_equal_load_sharing_does_not_balance_the_robot(rbd):
    """把体重平均分给四条腿**不能**让机器人静止 —— 一个反直觉但很重要的事实。

    Go2 的标称站姿相对质心并不前后对称：前脚在 x=+0.178，后脚在 x=-0.209。
    等分垂直力因此留下约 2.2 N·m 的俯仰力矩，折合 4.4 rad/s^2 的角加速度。
    这就是为什么支撑力分配必须真的去解一个优化问题，而不能拍脑袋均分。
    """
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])
    forces = np.tile([0.0, 0.0, params.mass * params.gravity / 4], (4, 1))

    lin, ang = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
    np.testing.assert_allclose(lin, 0.0, atol=1e-12)  # 合力仍然平衡
    assert abs(ang[1]) > 1.0, "等分力应留下明显的俯仰角加速度"
    # 前后脚到质心的距离确实不相等，这才是根因。
    front = feet[:2, 0].mean() - com[0]
    rear = com[0] - feet[2:, 0].mean()
    assert abs(front - rear) > 0.02


def test_srbd_freefall_is_gravity(rbd):
    """没有接触力时，单刚体只受重力。"""
    params = srbd_params_from_model(rbd)
    lin, ang = srbd_acceleration(
        np.zeros(3), np.eye(3), np.array([0.3, -0.2, 0.5]), np.zeros((0, 3)), np.zeros((0, 3)), params
    )
    np.testing.assert_allclose(lin, [0.0, 0.0, -params.gravity], atol=1e-12)
    # 自由旋转时角加速度只剩陀螺项，一般不为零。
    assert np.linalg.norm(ang) > 0.0


def test_srbd_matches_full_model_com_acceleration(rbd):
    """核心交叉验证：给定同一组接触力，单刚体与整机模型的**质心线加速度**必须一致。

    线动量定理是精确的，与腿是否有质量无关，所以这一项必须严格相等。
    角加速度则不然 —— 见下一个测试。
    """
    rng = np.random.default_rng(SEED + 7)
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    for _ in range(10):
        forces = np.column_stack(
            [rng.normal(size=4) * 10, rng.normal(size=4) * 10, rng.uniform(20, 80, 4)]
        )
        # 整机：把接触力当作外力施加，求广义加速度
        J = rbd.contact_jacobian(q, LEGS)
        tau_ext = J.T @ forces.reshape(-1)
        a = rbd.forward_dynamics(q, v, tau_ext)
        Ag = rbd.centroidal_momentum_matrix(q, v)
        h_dot = Ag @ a + pin.dccrba(rbd._m, rbd._d, q, v) @ v
        com_acc_full = h_dot[:3] / rbd.total_mass

        lin, _ = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
        np.testing.assert_allclose(lin, com_acc_full, atol=1e-8)


def _locked_joint_base_angular_acceleration(rbd, q, v, forces):
    """关节锁死（``a_joints = 0``）时躯干的角加速度，来自整机模型。

    这是 SRBD 的正确对照组：它代表"腿被 WBC 牢牢伺服住"的理想情形，
    与单刚体的差别只剩腿运动带来的科氏效应。

    注意**不能**拿质心动量去比 —— ``h_dot`` 的角分量恒等于外力矩，
    再除以同一个复合惯量必然得到与 SRBD 完全相同的结果，那是循环论证。
    """
    J = rbd.contact_jacobian(q, LEGS)
    tau_ext = J.T @ np.asarray(forces).reshape(-1)
    nle = rbd.nonlinear_effects(q, v)
    M = rbd.mass_matrix(q)
    a_base = np.linalg.solve(M[:FLOATING_BASE_DOF, :FLOATING_BASE_DOF], tau_ext[:6] - nle[:6])
    return a_base[3:6]


def test_srbd_is_exact_when_the_legs_do_not_move(rbd):
    """腿静止时，单刚体与关节锁死的整机模型**完全一致**。

    这同时证明了 :func:`srbd_params_from_model` 取的复合惯量是对的 ——
    取成躯干连杆惯量的话，这个测试会以 5 到 7 倍的差距失败。
    """
    rng = np.random.default_rng(SEED + 8)
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    v = np.zeros(rbd.nv)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    for _ in range(20):
        forces = np.column_stack(
            [rng.normal(size=4) * 8, rng.normal(size=4) * 8, rng.uniform(20, 80, 4)]
        )
        ang_full = _locked_joint_base_angular_acceleration(rbd, q, v, forces)
        _, ang_srbd = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
        np.testing.assert_allclose(ang_srbd, ang_full, atol=1e-9)


def test_srbd_error_grows_with_leg_speed(rbd):
    """腿摆得越快，「腿无质量」假设越站不住 —— 量化这个代价。

    实测：关节速度 2 rad/s 时角加速度相对误差中位数约 4%，
    8 rad/s 时涨到约 43%。这就是凸 MPC 之上还必须叠一层 WBC 的原因。
    """
    rng = np.random.default_rng(SEED + 11)
    params = srbd_params_from_model(rbd)
    q = nominal_configuration(rbd)
    com = rbd.center_of_mass(q)
    feet = np.array([rbd.model.foot_position(q, leg) for leg in LEGS])

    medians = []
    for speed in (0.0, 2.0, 5.0, 8.0):
        errs = []
        for _ in range(30):
            v = np.zeros(rbd.nv)
            v[FLOATING_BASE_DOF:] = rng.normal(size=12) * speed
            forces = np.column_stack(
                [rng.normal(size=4) * 8, rng.normal(size=4) * 8, rng.uniform(20, 80, 4)]
            )
            ang_full = _locked_joint_base_angular_acceleration(rbd, q, v, forces)
            _, ang_srbd = srbd_acceleration(com, np.eye(3), np.zeros(3), forces, feet, params)
            errs.append(np.linalg.norm(ang_srbd - ang_full) / max(np.linalg.norm(ang_full), 1e-9))
        medians.append(np.median(errs))

    np.testing.assert_allclose(medians[0], 0.0, atol=1e-9)  # 腿不动时精确
    assert all(b > a for a, b in zip(medians, medians[1:])), f"误差应随腿速单调增大：{medians}"
    assert medians[-1] > 0.15, f"8 rad/s 时误差应显著，实测 {medians[-1]:.3f}"


def test_srbd_mpc_approximation_drops_gyroscopic_term(rbd):
    """MPC 版本丢掉了陀螺项，角速度越大差异越明显。"""
    params = srbd_params_from_model(rbd)
    com = np.array([0.0, 0.0, 0.3])
    feet = np.array([[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]])
    forces = np.tile([0.0, 0.0, params.mass * params.gravity / 4], (4, 1))

    diffs = []
    for w in [0.0, 1.0, 3.0, 6.0]:
        omega = np.array([w, 0.5 * w, 0.0])
        _, a_full = srbd_acceleration(com, np.eye(3), omega, forces, feet, params)
        _, a_mpc = srbd_acceleration_mpc(com, 0.0, omega, forces, feet, params)
        diffs.append(np.linalg.norm(a_full - a_mpc))

    np.testing.assert_allclose(diffs[0], 0.0, atol=1e-12)  # 静止时两者相同
    assert all(b > a for a, b in zip(diffs, diffs[1:])), "陀螺项误差应随角速度单调增大"


def test_srbd_mpc_approximation_ignores_roll_pitch(rbd):
    """MPC 版本只保留偏航，横滚俯仰越大误差越大。"""
    params = srbd_params_from_model(rbd)
    com = np.array([0.0, 0.0, 0.3])
    feet = np.array([[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]])
    # 必须用**不对称**的力，否则合力矩为零，角加速度恒为零，惯量取错也看不出来。
    forces = np.column_stack([np.zeros(4), np.zeros(4), np.array([70.0, 20.0, 50.0, 30.0])])

    diffs = []
    for tilt in [0.0, 0.1, 0.3, 0.6]:
        rpy = np.array([tilt, 0.5 * tilt, 0.4])
        R = rpy_to_matrix(rpy)
        _, a_full = srbd_acceleration(com, R, np.zeros(3), forces, feet, params)
        _, a_mpc = srbd_acceleration_mpc(com, rpy[2], np.zeros(3), forces, feet, params)
        diffs.append(np.linalg.norm(a_full - a_mpc))

    np.testing.assert_allclose(diffs[0], 0.0, atol=1e-12)
    assert all(b > a for a, b in zip(diffs, diffs[1:])), f"倾角误差应随横滚俯仰单调增大：{diffs}"


def test_srbd_is_linear_in_contact_forces(rbd):
    """凸 MPC 成立的全部前提：足端位置固定时，动力学对力是线性的。"""
    rng = np.random.default_rng(SEED + 9)
    params = srbd_params_from_model(rbd)
    com = np.array([0.0, 0.0, 0.3])
    feet = rng.normal(size=(4, 3)) * 0.2

    f1 = rng.normal(size=(4, 3)) * 20
    f2 = rng.normal(size=(4, 3)) * 20
    alpha, beta = 0.3, 1.7

    def acc(f):
        # 去掉重力与陀螺项之后，剩下的部分必须对 f 严格线性。
        lin, ang = srbd_acceleration_mpc(com, 0.0, np.zeros(3), f, feet, params)
        return np.concatenate([lin - params.gravity_vector, ang])

    np.testing.assert_allclose(acc(alpha * f1 + beta * f2), alpha * acc(f1) + beta * acc(f2), atol=1e-10)


# --------------------------------------------------------------------------
# 能量：积分器与模型的总体检
# --------------------------------------------------------------------------


def test_energy_is_conserved_in_free_flight(rbd):
    """无接触、无驱动时机械能守恒。这是最灵敏的整体一致性检查。"""
    rng = np.random.default_rng(SEED + 10)
    q = nominal_configuration(rbd)
    q[2] = 3.0
    v = np.zeros(rbd.nv)
    v[FLOATING_BASE_DOF:] = rng.normal(size=12) * 0.8
    v[:3] = rng.normal(size=3) * 0.3

    e0 = rbd.total_energy(q, v)
    dt = 5e-5
    for _ in range(400):
        # 半隐式欧拉：辛结构，能量不会系统性漂移
        a = rbd.forward_dynamics(q, v, np.zeros(rbd.nv))
        v = v + dt * a
        q = pin.integrate(rbd._m, q, dt * v)
    drift = abs(rbd.total_energy(q, v) - e0) / abs(e0)
    assert drift < 1e-3, f"能量漂移 {drift:.2e} 过大，积分器或动力学有问题"
