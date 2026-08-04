"""里程碑 3（状态估计）的验证。

状态估计的验证比前两个里程碑麻烦：**真机上没有真值**。所以这里先造一个
运动学完全自洽的世界（``state_estimator.simulation``），再在其中检验估计器。
这本身就是工业界的标准做法 —— 先在仿真里把估计器调对，再上真机。

交叉验证的对象：

* 数据生成器  <- FK(关节角) 必须严格等于足端位置
* IMU 模型    <- 静止时读数必须是 +g（比力，不是加速度）
* so3_exp     <- Pinocchio 的 exp3
* 腿部里程计  <- 理想条件下必须精确还原躯干速度
* ESKF        <- 与真值比对，并与纯腿部里程计对照

运行::

    pytest tests/test_state_estimation.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from dynamics import matrix_to_rpy, rpy_to_matrix, skew
from kinematics import HIP_OFFSETS, LEG_GEOMETRY, LEGS, forward_kinematics
from state_estimator import (
    ErrorStateKF,
    ESKFConfig,
    ESKFState,
    LegOdometry,
    TrajectoryConfig,
    base_velocity_from_legs,
    foot_position_in_base,
    generate_trot,
    simulate_imu,
    so3_exp,
)
from state_estimator.simulation import add_encoder_noise, inject_slip

SEED = 0
GRAVITY = 9.81


@pytest.fixture(scope="module")
def gt():
    return generate_trot(TrajectoryConfig(duration=3.0))


def make_filter(gt, cfg: ESKFConfig | None = None) -> ErrorStateKF:
    """用真值初始化滤波器 —— 位置与偏航不可观测，必须外部给起点。"""
    init = ESKFState(
        position=gt.base_position[0].copy(),
        velocity=gt.base_velocity_world[0].copy(),
        rotation=gt.base_rotation[0].copy(),
    )
    init.foot_position[:] = gt.foot_position_world[0]
    return ErrorStateKF(dt=gt.config.dt, config=cfg, initial_state=init)


def run_filter(gt, kf, joint_pos=None, accel=None, gyro=None):
    """跑完整段数据，返回逐时刻的位置、速度、姿态误差。"""
    joint_pos = gt.joint_position if joint_pos is None else joint_pos
    if accel is None or gyro is None:
        accel, gyro = simulate_imu(gt, 0.02, 0.002, seed=SEED)
    n = len(gt)
    pos_err = np.zeros((n, 3))
    vel_err = np.zeros((n, 3))
    rpy_err = np.zeros((n, 3))
    for k in range(n):
        kf.step(accel[k], gyro[k], joint_pos[k], gt.contact[k])
        pos_err[k] = kf.state.position - gt.base_position[k]
        vel_err[k] = kf.state.velocity - gt.base_velocity_world[k]
        rpy_err[k] = matrix_to_rpy(kf.state.rotation) - gt.base_rpy[k]
    return pos_err, vel_err, rpy_err


# --------------------------------------------------------------------------
# 数据生成器：估计器的验证依赖它，所以它自己必须先被验证
# --------------------------------------------------------------------------


def test_generated_data_is_kinematically_consistent(gt):
    """FK(关节角) 必须严格等于足端位置，否则整个验证都建立在沙子上。"""
    for k in range(0, len(gt), 29):
        for i, leg in enumerate(LEGS):
            p_hip = forward_kinematics(gt.joint_position[k, 3 * i : 3 * i + 3], LEG_GEOMETRY[leg])
            p_world = gt.base_position[k] + gt.base_rotation[k] @ (HIP_OFFSETS[leg] + p_hip)
            np.testing.assert_allclose(p_world, gt.foot_position_world[k, i], atol=1e-12)


def test_trot_always_has_two_stance_legs(gt):
    """trot 的定义：对角腿成对交替，任意时刻恰好两条腿支撑。"""
    assert set(np.unique(gt.contact.sum(axis=1))) == {2}
    fl, rr = LEGS.index("FL"), LEGS.index("RR")
    fr, rl = LEGS.index("FR"), LEGS.index("RL")
    np.testing.assert_array_equal(gt.contact[:, fl], gt.contact[:, rr])
    np.testing.assert_array_equal(gt.contact[:, fr], gt.contact[:, rl])
    assert np.all(gt.contact[:, fl] != gt.contact[:, fr])


def test_stance_feet_do_not_move(gt):
    """支撑脚在世界系里必须静止 —— 这是腿部里程计的全部前提。"""
    dp = np.diff(gt.foot_position_world, axis=0)
    stance = gt.contact[1:] & gt.contact[:-1]
    assert np.abs(dp[stance]).max() < 1e-12


def test_swing_feet_lift_off_the_ground(gt):
    """摆动脚必须真的抬起来，否则这段数据没有步态可言。"""
    swing_z = gt.foot_position_world[..., 2][~gt.contact]
    assert swing_z.max() > 0.9 * gt.config.swing_height
    assert gt.foot_position_world[..., 2].min() >= -1e-12


def test_base_velocity_matches_position_derivative(gt):
    """解析给定的速度必须与位置的数值导数一致。"""
    v_num = np.gradient(gt.base_position, gt.config.dt, axis=0)
    # 中心差分的截断误差是 O(dt^2)；dt=1e-3 时约 1e-5，容差按此设定。
    np.testing.assert_allclose(v_num[5:-5], gt.base_velocity_world[5:-5], atol=1e-4)


def test_base_acceleration_matches_velocity_derivative(gt):
    a_num = np.gradient(gt.base_velocity_world, gt.config.dt, axis=0)
    np.testing.assert_allclose(a_num[5:-5], gt.base_acceleration_world[5:-5], atol=1e-3)


def test_body_angular_velocity_matches_rotation_derivative(gt):
    """机体系角速度必须满足 Rdot = R * omega^。"""
    dt = gt.config.dt
    for k in range(10, len(gt) - 10, 53):
        R_dot = (gt.base_rotation[k + 1] - gt.base_rotation[k - 1]) / (2 * dt)
        np.testing.assert_allclose(R_dot, gt.base_rotation[k] @ skew(gt.omega_body[k]), atol=1e-4)


# --------------------------------------------------------------------------
# IMU 模型
# --------------------------------------------------------------------------


def test_accelerometer_measures_specific_force_not_acceleration():
    """静止水平放置时，加速度计读数是 +9.81 向上，不是零。

    这个符号搞反是 IMU 相关代码最经典的错误，会让滤波器把重力当成
    持续加速度，位置在两秒内飞出去。
    """
    gt = generate_trot(
        TrajectoryConfig(
            duration=0.2,
            forward_speed=0.0,
            lateral_amplitude=0.0,
            height_amplitude=0.0,
            yaw_rate=0.0,
            roll_amplitude=0.0,
            pitch_amplitude=0.0,
        )
    )
    accel, gyro = simulate_imu(gt, accel_noise=0.0, gyro_noise=0.0)
    expected = np.tile([0.0, 0.0, GRAVITY], (len(gt), 1))
    np.testing.assert_allclose(accel, expected, atol=1e-9)
    np.testing.assert_allclose(gyro, np.zeros_like(gyro), atol=1e-9)


def test_imu_bias_and_noise_enter_as_specified(gt):
    b_a = np.array([0.1, -0.2, 0.3])
    b_g = np.array([0.01, 0.02, -0.03])
    clean_a, clean_g = simulate_imu(gt, 0.0, 0.0)
    biased_a, biased_g = simulate_imu(gt, 0.0, 0.0, accel_bias=b_a, gyro_bias=b_g)
    np.testing.assert_allclose(biased_a - clean_a, np.tile(b_a, (len(gt), 1)), atol=1e-12)
    np.testing.assert_allclose(biased_g - clean_g, np.tile(b_g, (len(gt), 1)), atol=1e-12)


def test_gyro_measures_body_angular_velocity(gt):
    _, gyro = simulate_imu(gt, 0.0, 0.0)
    np.testing.assert_allclose(gyro, gt.omega_body, atol=1e-12)


# --------------------------------------------------------------------------
# SO(3) 工具
# --------------------------------------------------------------------------


def test_so3_exp_matches_pinocchio():
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        phi = rng.normal(size=3) * rng.uniform(0.0, 2.0)
        np.testing.assert_allclose(so3_exp(phi), pin.exp3(phi), atol=1e-12)


def test_so3_exp_is_stable_near_zero():
    """小角度必须走泰勒展开分支，不能出现 0/0。"""
    for scale in (1e-12, 1e-10, 1e-9, 1e-8):
        R = so3_exp(np.array([scale, -scale, scale]))
        assert np.all(np.isfinite(R))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_so3_exp_produces_valid_rotations():
    rng = np.random.default_rng(SEED + 1)
    for _ in range(30):
        R = so3_exp(rng.normal(size=3))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# 腿部里程计
# --------------------------------------------------------------------------


def test_leg_odometry_recovers_velocity_exactly(gt):
    """理想条件下，支撑腿反推的躯干速度必须精确等于真值。

    这一条同时验证了里程碑 1 的雅可比：如果 J 错了，这里立刻暴露。
    """
    for k in range(20, len(gt) - 20, 41):
        v_est, _ = base_velocity_from_legs(
            gt.joint_position[k],
            gt.joint_velocity[k],
            gt.base_rotation[k],
            gt.omega_body[k],
            gt.contact[k],
        )
        np.testing.assert_allclose(v_est, gt.base_velocity_world[k], atol=2e-4)


def test_all_stance_legs_agree_without_slip(gt):
    """不打滑时各条支撑腿的估计应高度一致 —— 打滑检测的基线。"""
    for k in range(20, len(gt) - 20, 61):
        _, per_leg = base_velocity_from_legs(
            gt.joint_position[k],
            gt.joint_velocity[k],
            gt.base_rotation[k],
            gt.omega_body[k],
            gt.contact[k],
        )
        assert LegOdometry.slip_indicator(per_leg) < 5e-4


def test_slip_indicator_rises_when_a_foot_slips(gt):
    """打滑时各腿估计发散，指标显著跳高。"""
    slipped = inject_slip(gt, "FL", 1.0, 1.3, np.array([0.2, 0.0, 0.0]))
    clean_max, slip_max = 0.0, 0.0
    for k in range(len(gt)):
        _, per_clean = base_velocity_from_legs(
            gt.joint_position[k], gt.joint_velocity[k], gt.base_rotation[k], gt.omega_body[k], gt.contact[k]
        )
        clean_max = max(clean_max, LegOdometry.slip_indicator(per_clean))
        if 1.05 < gt.t[k] < 1.25:
            _, per_slip = base_velocity_from_legs(
                slipped.joint_position[k],
                slipped.joint_velocity[k],
                gt.base_rotation[k],
                gt.omega_body[k],
                gt.contact[k],
            )
            slip_max = max(slip_max, LegOdometry.slip_indicator(per_slip))
    assert slip_max > 20 * clean_max, f"打滑指标区分度不足：{slip_max:.4f} vs {clean_max:.4f}"


def test_leg_odometry_raises_in_flight_phase(gt):
    with pytest.raises(ValueError, match="没有支撑腿"):
        base_velocity_from_legs(
            gt.joint_position[0], gt.joint_velocity[0], np.eye(3), np.zeros(3), np.zeros(4, dtype=bool)
        )


def test_foot_position_in_base_includes_hip_offset(gt):
    """躯干系下的足端位置必须含髋部安装偏置，漏掉会差 0.19 m。"""
    for leg in LEGS:
        p = foot_position_in_base(gt.joint_position[0], leg)
        i = LEGS.index(leg)
        p_hip_only = forward_kinematics(gt.joint_position[0, 3 * i : 3 * i + 3], LEG_GEOMETRY[leg])
        np.testing.assert_allclose(p - p_hip_only, HIP_OFFSETS[leg], atol=1e-12)


def test_leg_odometry_tracks_position_when_perfect(gt):
    odo = LegOdometry(gt.base_position[0], gt.config.dt)
    for k in range(len(gt)):
        pos, _ = odo.update(
            gt.joint_position[k],
            gt.joint_velocity[k],
            gt.base_rotation[k],
            gt.omega_body[k],
            gt.contact[k],
        )
    assert np.linalg.norm(pos - gt.base_position[-1]) < 0.01


# --------------------------------------------------------------------------
# ESKF：基本正确性
# --------------------------------------------------------------------------


def test_eskf_tracks_ground_truth(gt):
    kf = make_filter(gt)
    pos_err, vel_err, rpy_err = run_filter(gt, kf)
    assert np.linalg.norm(pos_err[-1]) < 0.01
    assert np.sqrt((vel_err**2).sum(axis=1).mean()) < 0.05
    assert np.abs(np.degrees(rpy_err[-1])).max() < 1.0


def test_eskf_covariance_stays_symmetric_positive_definite(gt):
    kf = make_filter(gt)
    accel, gyro = simulate_imu(gt, 0.02, 0.002, seed=SEED)
    for k in range(0, 600):
        kf.step(accel[k], gyro[k], gt.joint_position[k], gt.contact[k])
        if k % 137 == 0:
            np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)
            assert np.min(np.linalg.eigvalsh(kf.P)) > -1e-12


def test_eskf_rotation_stays_orthonormal(gt):
    kf = make_filter(gt)
    run_filter(gt, kf)
    R = kf.state.rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_eskf_beats_open_loop_imu_integration(gt):
    """融合的价值：与纯 IMU 积分相比误差要小几个数量级。"""
    accel, gyro = simulate_imu(gt, 0.02, 0.002, seed=SEED)

    # 纯 IMU 积分（不做任何更新）
    kf_open = make_filter(gt)
    for k in range(len(gt)):
        kf_open.predict(accel[k], gyro[k], gt.contact[k])
    open_err = np.linalg.norm(kf_open.state.position - gt.base_position[-1])

    kf = make_filter(gt)
    pos_err, _, _ = run_filter(gt, kf, accel=accel, gyro=gyro)
    fused_err = np.linalg.norm(pos_err[-1])

    assert fused_err < 0.05 * open_err, f"融合 {fused_err:.4f} m vs 开环 {open_err:.4f} m"


def test_eskf_handles_flight_phase(gt):
    """腾空时没有观测，滤波器必须靠 IMU 撑住而不是崩掉。"""
    kf = make_filter(gt)
    accel, gyro = simulate_imu(gt, 0.02, 0.002, seed=SEED)
    no_contact = np.zeros(4, dtype=bool)
    for k in range(len(gt)):
        contact = no_contact if 1.0 < gt.t[k] < 1.1 else gt.contact[k]
        kf.step(accel[k], gyro[k], gt.joint_position[k], contact)
    assert np.all(np.isfinite(kf.state.position))
    assert np.linalg.norm(kf.state.position - gt.base_position[-1]) < 0.1


def test_eskf_reanchors_feet_on_touchdown(gt):
    """刚落地的脚必须用当前位姿重新初始化，否则旧位置会污染估计。"""
    kf = make_filter(gt)
    accel, gyro = simulate_imu(gt, 0.0, 0.0)
    for k in range(len(gt)):
        kf.step(accel[k], gyro[k], gt.joint_position[k], gt.contact[k])
        if k > 10 and gt.contact[k].any():
            for i in np.flatnonzero(gt.contact[k]):
                err = np.linalg.norm(kf.state.foot_position[i] - gt.foot_position_world[k, i])
                assert err < 0.05, f"t={gt.t[k]:.3f} 腿 {LEGS[i]} 足端状态偏离 {err:.3f} m"


# --------------------------------------------------------------------------
# 可观测性：本里程碑最重要的结论
# --------------------------------------------------------------------------


def test_accelerometer_bias_is_absorbed_into_tilt(gt):
    """加速度计水平零偏与横滚俯仰**不可分辨**，滤波器把它吸收成倾角。

    物理原因：水平方向的常值零偏 b，和一个大小为 b/g 的倾角，对加速度计
    产生完全相同的读数。除非机器人做出足够丰富的机动，二者无法区分。

    实测：注入 b_y = -0.05 会让横滚估计偏 -0.29 度，而 b_y/g = -0.292 度。
    """
    b_a = np.array([0.08, -0.05, 0.10])
    accel, gyro = simulate_imu(gt, 0.02, 0.002, accel_bias=b_a, seed=SEED)
    kf = make_filter(gt)
    _, _, rpy_err = run_filter(gt, kf, accel=accel, gyro=gyro)

    # 零偏本身没被估出来
    assert np.linalg.norm(kf.state.accel_bias - b_a) > 0.5 * np.linalg.norm(b_a)
    # 但横滚误差恰好是 b_y / g
    predicted_roll = b_a[1] / GRAVITY
    np.testing.assert_allclose(rpy_err[-1][0], predicted_roll, atol=0.002)


def test_gyro_bias_is_observable(gt):
    """陀螺零偏可观测：腿部运动学不断提供姿态参考，零偏会被估出来。"""
    b_g = np.array([0.01, -0.008, 0.005])
    accel, gyro = simulate_imu(gt, 0.02, 0.002, gyro_bias=b_g, seed=SEED)
    kf = make_filter(gt)
    run_filter(gt, kf, accel=accel, gyro=gyro)
    residual = np.linalg.norm(kf.state.gyro_bias - b_g)
    assert residual < 0.3 * np.linalg.norm(b_g), f"陀螺零偏残差 {residual:.5f}"


def test_roll_pitch_converge_but_yaw_does_not(gt):
    """重力给了横滚俯仰绝对参考；偏航没有任何绝对参考。

    这是四足状态估计的根本结构：**偏航与绝对位置不可观测**，估计器
    只能保证它们的变化量正确。要消除这两个方向的漂移，必须引入外部
    传感器（视觉、激光、磁力计）。
    """
    kf = make_filter(gt)
    accel, gyro = simulate_imu(gt, 0.02, 0.002, seed=SEED)
    std_history = []
    for k in range(len(gt)):
        kf.step(accel[k], gyro[k], gt.joint_position[k], gt.contact[k])
        if k % 400 == 0:
            std_history.append(kf.attitude_std.copy())
    std_history = np.array(std_history)

    roll_ratio = std_history[-1, 0] / std_history[0, 0]
    yaw_ratio = std_history[-1, 2] / std_history[0, 2]
    assert roll_ratio < 0.6, f"横滚不确定度应显著收敛，实测比值 {roll_ratio:.3f}"
    assert yaw_ratio > 0.85, f"偏航不确定度不应收敛，实测比值 {yaw_ratio:.3f}"


# --------------------------------------------------------------------------
# 退化场景：什么会真正搞垮估计器
# --------------------------------------------------------------------------


def test_encoder_noise_degrades_accuracy_gracefully(gt):
    """编码器噪声让精度变差，但不会让估计器发散。"""
    errors = []
    for noise in (0.0, 1e-3, 5e-3):
        jp, _ = add_encoder_noise(gt, noise, seed=SEED)
        kf = make_filter(gt)
        pos_err, _, _ = run_filter(gt, kf, joint_pos=jp)
        errors.append(np.linalg.norm(pos_err[-1]))
    assert all(b > a for a, b in zip(errors, errors[1:])), f"误差应随噪声增大：{errors}"
    assert errors[-1] < 0.02, "编码器噪声不应导致发散"


def test_slip_is_far_more_damaging_than_encoder_noise(gt):
    """打滑的破坏力比编码器噪声大两个数量级 —— 本里程碑的核心结论。

    编码器噪声是零均值的，会被滤波平均掉；打滑是**有偏**的，误差直接
    积分进位置，而且没有任何观测能纠正它。
    """
    # 1 mrad 是真实编码器的量级；5 mrad 已经是坏掉的编码器了。
    jp, _ = add_encoder_noise(gt, 1e-3, seed=SEED)
    kf_noise = make_filter(gt)
    err_noise = np.linalg.norm(run_filter(gt, kf_noise, joint_pos=jp)[0][-1])

    slipped = inject_slip(gt, "FL", 1.0, 1.4, np.array([0.15, 0.0, 0.0]))
    kf_slip = make_filter(gt)
    err_slip = np.linalg.norm(run_filter(gt, kf_slip, joint_pos=slipped.joint_position)[0][-1])

    assert err_slip > 10 * err_noise, f"打滑 {err_slip:.4f} m vs 编码器噪声 {err_noise:.4f} m"


def test_slip_error_is_permanent(gt):
    """打滑结束后误差不会自己消失 —— 没有绝对参考就无法纠正。"""
    slipped = inject_slip(gt, "FL", 0.8, 1.2, np.array([0.2, 0.0, 0.0]))
    kf = make_filter(gt)
    pos_err, _, _ = run_filter(gt, kf, joint_pos=slipped.joint_position)

    during = np.linalg.norm(pos_err[gt.t <= 1.25][-1])
    after = np.linalg.norm(pos_err[-1])
    assert after > 0.7 * during, f"打滑误差应持续存在：期间 {during:.4f} m，之后 {after:.4f} m"


def test_larger_foot_process_noise_absorbs_slip_better(gt):
    """把支撑脚过程噪声调大，滤波器对打滑更宽容 —— 一个真实的调参权衡。"""
    slipped = inject_slip(gt, "FL", 1.0, 1.5, np.array([0.25, 0.0, 0.0]))
    errors = {}
    for noise in (1e-4, 1e-2):
        kf = make_filter(gt, ESKFConfig(foot_noise_stance=noise))
        errors[noise] = np.linalg.norm(run_filter(gt, kf, joint_pos=slipped.joint_position)[0][-1])
    assert errors[1e-2] < errors[1e-4], f"过程噪声调大应更抗打滑：{errors}"
