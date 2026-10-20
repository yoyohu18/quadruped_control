"""里程碑 11（MuJoCo 全身闭环）第一步：仿真器与控制器模型的对齐。

闭环之前必须先回答："MuJoCo 里的那只狗，和 Pinocchio 里的那只狗，是不是
同一只？"不是的话，后面任何跟踪误差都分不清是控制器的锅还是模型的锅。

* 运动学        <- 足端位置逐腿比对
* 质量矩阵 M    <- 速度约定换算后逐元素比对（扣掉 armature）
* 重力项 g      <- 同上
* 状态约定      <- 四元数顺序、线速度坐标系往返无损
* 执行器映射    <- 按名字映射，逐关节打一下看谁动
* 接触力读数    <- 静止站立时四脚竖直力之和 == 体重

运行::

    pytest tests/test_mujoco.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

mujoco = pytest.importorskip("mujoco")

from dynamics import load_go2_dynamics  # noqa: E402
from kinematics import LEGS  # noqa: E402
from kinematics.go2 import STANDING_JOINT_ANGLES, load_go2  # noqa: E402
from sim import GO2_MJCF, MujocoGo2, SimConfig  # noqa: E402

pytestmark = pytest.mark.skipif(not GO2_MJCF.is_file(), reason=f"找不到 {GO2_MJCF}")

SEED = 0
N_SAMPLES = 20


@pytest.fixture(scope="module")
def rbd():
    return load_go2_dynamics(floating_base=True)


@pytest.fixture(scope="module")
def sim_urdf():
    return MujocoGo2(SimConfig(inertia="urdf", joint_friction=False))


def random_configs(rbd, n=N_SAMPLES):
    pm = rbd.model.model
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        q = pin.neutral(pm)
        q[:3] = rng.uniform(-0.5, 0.5, 3)
        quat = rng.normal(size=4)
        q[3:7] = quat / np.linalg.norm(quat)
        q[7:] = rng.uniform(pm.lowerPositionLimit[7:], pm.upperPositionLimit[7:])
        yield q


def velocity_map(sim):
    """MuJoCo 速度 = T · Pinocchio 速度（只有基座线速度差一个旋转）。"""
    T = np.eye(18)
    T[:3, :3] = sim.base_rotation()
    return T


def standing_q(km, sim):
    q = km.neutral().copy()
    q[7:] = STANDING_JOINT_ANGLES
    q[2] = 0.0
    q[2] = -min(km.foot_position(q, leg)[2] for leg in LEGS) + sim.foot_radius
    return q


def stand(sim, q0, steps=2000, kp=40.0, kd=1.0):
    sim.reset(q0)
    for _ in range(steps):
        s = sim.state()
        sim.step(kp * (STANDING_JOINT_ANGLES - s.q[7:]) - kd * s.v[6:])
    return sim.state()


# -- 模型对齐 -------------------------------------------------------------------


def test_foot_positions_match(rbd, sim_urdf):
    km = rbd.model
    for q in random_configs(rbd):
        sim_urdf.reset(q)
        for leg in LEGS:
            b = mujoco.mj_name2id(sim_urdf.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            np.testing.assert_allclose(sim_urdf.data.xpos[b], km.foot_position(q, leg), atol=1e-12)


def test_urdf_mode_total_mass_matches(rbd, sim_urdf):
    assert sim_urdf.total_mass() == pytest.approx(pin.computeTotalMass(rbd.model.model), abs=1e-9)


def test_urdf_mode_mass_matrix_matches(rbd, sim_urdf):
    arm = np.diag(sim_urdf.model.dof_armature)
    for q in random_configs(rbd):
        sim_urdf.reset(q)
        T = velocity_map(sim_urdf)
        M_pin = T.T @ (sim_urdf.mass_matrix() - arm) @ T
        np.testing.assert_allclose(M_pin, rbd.mass_matrix(q), atol=1e-12)


def test_urdf_mode_gravity_matches(rbd, sim_urdf):
    for q in random_configs(rbd):
        sim_urdf.reset(q)
        g_pin = velocity_map(sim_urdf).T @ sim_urdf.data.qfrc_bias
        np.testing.assert_allclose(g_pin, rbd.gravity_vector(q), atol=1e-10)


def test_mjcf_mode_is_a_real_model_mismatch(rbd):
    """官方 MJCF 比 URDF 轻约 0.9 kg —— 这是鲁棒性实验要用的失配，不是 bug。"""
    sim = MujocoGo2(SimConfig(inertia="mjcf"))
    m_pin = pin.computeTotalMass(rbd.model.model)
    assert sim.total_mass() == pytest.approx(15.206, abs=1e-3)
    assert 0.04 < (m_pin - sim.total_mass()) / m_pin < 0.07


# -- 约定转换 -------------------------------------------------------------------


def test_state_round_trip(rbd, sim_urdf):
    rng = np.random.default_rng(SEED)
    for q in random_configs(rbd, 5):
        v = rng.normal(size=18)
        sim_urdf.reset(q, v)
        np.testing.assert_allclose(sim_urdf.pin_q(), q, atol=1e-12)
        np.testing.assert_allclose(sim_urdf.pin_v(), v, atol=1e-12)


def test_body_velocity_is_in_body_frame(sim_urdf):
    """绕 z 转 90° 后，机体系 +x 的速度在 MuJoCo（世界系）里应是 +y。"""
    q = np.zeros(19)
    q[2] = 1.0
    q[3:7] = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # xyzw
    q[7:] = STANDING_JOINT_ANGLES
    v = np.zeros(18)
    v[0] = 1.0
    sim_urdf.reset(q, v)
    np.testing.assert_allclose(sim_urdf.data.qvel[:3], [0, 1, 0], atol=1e-12)


def test_actuators_are_mapped_by_name(sim_urdf):
    """执行器是 FR/FL/RR/RL 顺序；给第 i 个 Pinocchio 关节力矩，只能是它加速。"""
    q = np.zeros(19)
    q[2] = 1.0
    q[6] = 1.0
    q[7:] = STANDING_JOINT_ANGLES
    sim_urdf.model.opt.gravity[:] = 0
    try:
        for i in range(12):
            sim_urdf.reset(q)
            tau = np.zeros(12)
            tau[i] = 1.0
            sim_urdf.data.ctrl[sim_urdf._act_of_joint] = tau
            mujoco.mj_forward(sim_urdf.model, sim_urdf.data)
            qacc = sim_urdf.data.qacc[sim_urdf._vadr]
            assert np.argmax(np.abs(qacc)) == i
    finally:
        sim_urdf.model.opt.gravity[:] = [0, 0, -9.81]


def test_torque_is_saturated_at_actuator_limit():
    sim = MujocoGo2()
    np.testing.assert_allclose(sim.torque_limit, [23.7, 23.7, 45.43] * 4)


# -- 接触与站立 -----------------------------------------------------------------


@pytest.mark.parametrize("inertia", ["urdf", "mjcf"])
def test_standing_contact_forces_sum_to_body_weight(inertia):
    sim = MujocoGo2(SimConfig(inertia=inertia))
    km = load_go2(floating_base=True)
    s = stand(sim, standing_q(km, sim))
    total = sum(f for f in s.contact_force.values())
    assert total[2] == pytest.approx(sim.total_mass() * 9.81, rel=1e-3)
    np.testing.assert_allclose(total[:2], 0, atol=0.5)
    assert all(s.contact_force[leg][2] > 20 for leg in LEGS)  # 四只脚都在承重


@pytest.mark.parametrize("inertia", ["urdf", "mjcf"])
@pytest.mark.parametrize("kp,kd", [(20, 0.5), (40, 1.0), (80, 2.0)])
def test_pd_standing_comes_to_rest(inertia, kp, kd):
    """回归测试：主轴升序排列时 urdf 模式会落入约 100 Hz 膝关节振荡、整机倒退。"""
    sim = MujocoGo2(SimConfig(inertia=inertia))
    km = load_go2(floating_base=True)
    x_mid = stand(sim, standing_q(km, sim), steps=1000, kp=kp, kd=kd).q[0]
    for _ in range(1000):
        s = sim.state()
        sim.step(kp * (STANDING_JOINT_ANGLES - s.q[7:]) - kd * s.v[6:])
    s = sim.state()
    assert np.linalg.norm(s.v[:6]) < 0.02
    assert abs(s.q[0] - x_mid) < 0.005  # 后一秒不再挪动（振荡时会挪 0.5 m）
