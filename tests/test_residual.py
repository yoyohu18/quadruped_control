"""里程碑 10 的测试：名义步态控制器与残差的数学。

与里程碑 9 一样**不依赖 Isaac Sim**。交叉验证的核心是：

> 批量 torch 的名义控制器，必须与里程碑 1/4/5/6 的 numpy 实现逐元素一致。

这一条测的不只是新代码 —— 它同时锁住了旧代码。任何一边被改动导致两者
分叉，都会在这里大声失败。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from footstep_planner import LIPMParams, raibert_footstep
from gait_scheduler import GAITS, GaitScheduler
from kinematics import (
    HIP_OFFSETS,
    LEG_GEOMETRY,
    LEGS,
    STANDING_JOINT_ANGLES,
    forward_kinematics,
    inverse_kinematics,
    pinocchio_to_isaac,
)
from rl.nominal_controller import (
    LEG_ORDER,
    NominalGaitConfig,
    NominalGaitController,
    batched_inverse_kinematics,
)

@pytest.fixture(autouse=True)
def double_precision():
    """本模块统一用双精度，以便与 numpy 实现逐位比对。

    **必须用带清理的 fixture，不能在模块层面直接调
    ``torch.set_default_dtype``** —— 那是个进程级的全局开关，会泄漏到同一次
    pytest 会话里的其他测试模块（里程碑 9 的测试按 float32 写判据，
    被改成 float64 之后行为和耗时都会变）。全局状态必须成对地开关。
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def controller() -> NominalGaitController:
    return NominalGaitController(NominalGaitConfig(), device="cpu")


# ====================================================== 批量 IK vs 里程碑 1


@pytest.mark.parametrize("leg", LEGS)
def test_batched_ik_matches_milestone_1(leg: str):
    """批量 torch IK 必须与 M1 的闭式 numpy IK 逐元素一致。"""
    geom = LEG_GEOMETRY[leg]
    rng = np.random.default_rng(abs(hash(leg)) % 2**32)

    q_ref = rng.uniform([-0.6, -1.5, -2.5], [0.6, 1.5, -0.5], size=(200, 3))
    targets = np.array([forward_kinematics(q, geom) for q in q_ref])

    ours = batched_inverse_kinematics(
        torch.tensor(targets), torch.tensor(geom.l0), geom.l1, geom.l2
    ).numpy()
    reference = np.array([inverse_kinematics(p, geom, clamp=True) for p in targets])

    assert np.abs(ours - reference).max() < 1e-12


def test_batched_ik_round_trips_through_forward_kinematics():
    """IK 解出的关节角经 FK 必须回到原目标 —— 与解支选择无关的判据。"""
    geom = LEG_GEOMETRY["FL"]
    rng = np.random.default_rng(0)
    targets = rng.uniform([-0.15, 0.05, -0.35], [0.15, 0.20, -0.20], size=(100, 3))

    q = batched_inverse_kinematics(torch.tensor(targets), torch.tensor(geom.l0), geom.l1, geom.l2).numpy()
    recovered = np.array([forward_kinematics(qi, geom) for qi in q])
    assert np.abs(recovered - targets).max() < 1e-12


def test_batched_ik_clamps_instead_of_raising():
    """不可达目标必须静默投影，不能抛异常 —— 4096 个环境不能被一条腿搞崩。"""
    geom = LEG_GEOMETRY["FL"]
    far = torch.tensor([[0.0, 0.0955, -5.0]])  # 远超 l1+l2
    q = batched_inverse_kinematics(far, torch.tensor(geom.l0), geom.l1, geom.l2)
    assert torch.all(torch.isfinite(q))
    # 投影到边界 = 腿完全伸直，膝角为 0
    assert q[0, 2].abs() < 1e-9


def test_batched_ik_broadcasts_over_legs():
    """一次算四条腿，每条腿的 l0 符号不同。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    p = torch.zeros(5, 4, 3)
    p[..., 2] = -0.3
    p[..., 1] = torch.tensor([0.0955, -0.0955, 0.0955, -0.0955])
    q = batched_inverse_kinematics(p, ctrl._l0[None, :], 0.213, 0.213)
    assert q.shape == (5, 4, 3)
    # 足端正好在髋俯仰轴正下方 → 侧摆角为 0
    assert q[..., 0].abs().max() < 1e-12


# ====================================================== 相位 vs 里程碑 4


@pytest.mark.parametrize("gait_name", ["trot", "pace", "bound"])
def test_phase_matches_gait_scheduler(gait_name: str):
    """名义控制器的相位必须与 M4 的调度器逐元素一致。"""
    gait = GAITS[gait_name]
    scheduler = GaitScheduler(gait)
    ctrl = NominalGaitController(
        NominalGaitConfig(
            period=gait.period,
            duty_factor=gait.duty_factor,
            phase_offsets=tuple(gait.phase_offsets[leg] for leg in LEG_ORDER),
        ),
        device="cpu",
    )

    times = np.linspace(0.0, 2.0, 41)
    ours = ctrl.phase(torch.tensor(times)).numpy()
    reference = np.array([scheduler.leg_phase(float(t)) for t in times])
    assert np.abs(ours - reference).max() < 1e-12

    assert np.array_equal(ctrl.contact(torch.tensor(times)).numpy(),
                          np.array([scheduler.contact(float(t)) for t in times]))


def test_trot_diagonal_pairs(controller: NominalGaitController):
    contact = controller.contact(torch.linspace(0.0, 1.6, 41))
    assert torch.all(contact[:, 0] == contact[:, 3])  # FL 与 RR
    assert torch.all(contact[:, 1] == contact[:, 2])  # FR 与 RL
    assert torch.all(contact[:, 0] != contact[:, 1])


# ====================================================== 落脚点 vs 里程碑 6


def test_touchdown_matches_raibert_heuristic():
    """直行时（无转向），落脚点必须等于 M6 的 Raibert 公式。

    在 :math:`t \\to T` 时，相位偏移为 0 的两条腿（FL、RR）正处于摆动末尾，
    此刻的足端水平位置就是落脚点。
    """
    cfg = NominalGaitConfig(max_stride=10.0)  # 关掉限幅，纯比公式
    ctrl = NominalGaitController(cfg, device="cpu")

    v_meas = np.array([0.6, -0.1])
    v_cmd = np.array([0.4, 0.0])
    time = torch.tensor([cfg.period * (1.0 - 1e-12)])
    feet = ctrl.foot_targets(
        time,
        torch.tensor([[v_cmd[0], v_cmd[1], 0.0]]),
        torch.tensor(v_meas)[None],
    )

    checked = 0
    for i, leg in enumerate(LEG_ORDER):
        if GAITS["trot"].phase_offsets[leg] != 0.0:
            continue  # 其余腿此刻在支撑相，比较落脚点没有意义
        nominal = np.array([HIP_OFFSETS[leg][0], HIP_OFFSETS[leg][1] + LEG_GEOMETRY[leg].l0])
        expected = raibert_footstep(
            nominal, v_meas, v_cmd, cfg.stance_duration, feedback_gain=cfg.raibert_gain
        )
        assert np.allclose(feet[0, i, :2].numpy(), expected, atol=1e-9)
        checked += 1
    assert checked == 2, "trot 里应当恰有两条腿（FL、RR）相位偏移为 0"


def test_stride_clamped_to_workspace():
    """指令速度极大时，**整条轨迹**都必须被限制在 max_stride 内。

    两个端点各自限幅，中间是它们的插值，于是整条曲线落在两点连成的线段上，
    自然也在圆内 —— 这是"限端点即限全程"能成立的原因（圆是凸的）。
    """
    cfg = NominalGaitConfig(max_stride=0.15)
    ctrl = NominalGaitController(cfg, device="cpu")
    feet = ctrl.foot_targets(
        torch.linspace(0.0, 0.4, 41),
        torch.tensor([[5.0, -3.0, 4.0]]).expand(41, 3),
        torch.tensor([[5.0, -3.0]]).expand(41, 2),
    )
    offset = (feet[..., :2] - ctrl._nominal_xy).norm(dim=-1)
    assert offset.max() <= cfg.max_stride + 1e-9


def test_turning_command_shifts_left_and_right_feet_oppositely():
    """纯转向指令下，左右腿的落脚点相对各自标称位置必须**反对称**。

    只断言反对称，不断言"哪边往前"：那取决于此刻是前馈项还是 Raibert
    反馈项占主导（起步时 :math:`v_{meas}=0`，反馈项主导，落脚点朝
    **加速**的方向偏，与稳态时相反）。断言方向会把一个正确的行为测成 bug。
    """
    cfg = NominalGaitConfig(max_stride=10.0)
    ctrl = NominalGaitController(cfg, device="cpu")
    command = torch.tensor([[0.0, 0.0, 1.0]])

    # **必须在各自的落地时刻取值。** trot 里 FL 与 FR 差半个周期：同一时刻
    # 一条在支撑起点、另一条在摆动起点，直接比会把"相位差"误读成"不对称"。
    fl = ctrl.foot_targets(torch.zeros(1), command, torch.zeros(1, 2))[0, 0, :2] - ctrl._nominal_xy[0]
    fr = ctrl.foot_targets(
        torch.tensor([cfg.stance_duration]), command, torch.zeros(1, 2)
    )[0, 1, :2] - ctrl._nominal_xy[1]

    assert fl.norm() > 1e-3, "转向指令必须让落脚点动起来"
    assert torch.allclose(fl[0], -fr[0], atol=1e-12), "左右腿的前后偏移必须反号"
    assert torch.allclose(fl[1], fr[1], atol=1e-12), "左右腿的横向偏移必须同号"


# ====================================================== 摆动轨迹 vs 里程碑 5


def test_swing_height_profile(controller: NominalGaitController):
    """摆动中点抬到 swing_height，两端贴地。"""
    cfg = controller.cfg
    # FL 的摆动相是 phase ∈ [0.5, 1) → time ∈ [0.2, 0.4)
    times = torch.linspace(0.2, 0.4, 21)
    z = controller.foot_targets(times, torch.zeros(21, 3), torch.zeros(21, 2))[:, 0, 2]

    ground = -cfg.stand_height
    assert z[0].item() == pytest.approx(ground, abs=1e-9)
    assert z.max().item() == pytest.approx(ground + cfg.swing_height, abs=1e-9)
    assert z[-1].item() == pytest.approx(ground, abs=1e-3)


def test_stance_feet_stay_on_ground(controller: NominalGaitController):
    """支撑腿的足端高度恒等于站立高度 —— 名义控制器不假设地形起伏。"""
    times = torch.linspace(0.0, 2.0, 101)
    feet = controller.foot_targets(times, torch.zeros(101, 3), torch.zeros(101, 2))
    contact = controller.contact(times)
    z = feet[..., 2]
    assert torch.allclose(z[contact], torch.full_like(z[contact], -controller.cfg.stand_height))


def test_swing_horizontal_velocity_vanishes_at_endpoints(controller: NominalGaitController):
    """smoothstep 保证离地与落地瞬间的水平速度为零 —— M5 关于落地冲击的结论。"""
    cfg = controller.cfg
    dt = 1e-5
    cmd = torch.tensor([[0.5, 0.0, 0.0]]).expand(2, 3)
    vel = torch.tensor([[0.2, 0.0]]).expand(2, 2)  # 故意让 v_meas != v_cmd
    for phase_time in (0.2, 0.4):  # FL 的离地与落地时刻
        t = torch.tensor([phase_time - dt, phase_time + dt])
        xy = controller.foot_targets(t, cmd, vel)[:, 0, :2]
        speed = ((xy[1] - xy[0]) / (2 * dt)).norm().item()
        assert speed < 0.5, f"相位切换处水平速度 {speed:.3f} m/s 过大 —— 足端目标不连续"


def test_stance_foot_moves_backward_under_forward_command(controller: NominalGaitController):
    """前进指令下，支撑足在躯干系里必须后移 —— 否则机器人不会前进。"""
    times = torch.linspace(0.0, 0.19, 20)  # FL 支撑相
    x = controller.foot_targets(
        times, torch.tensor([[0.5, 0.0, 0.0]]).expand(20, 3), torch.tensor([[0.5, 0.0]]).expand(20, 2)
    )[:, 0, 0]
    assert torch.all(x.diff() < 0)


# ====================================================== 关节输出


def test_zero_command_gives_standing_pose():
    """零指令、站立高度取 M1 标称姿态的高度时，输出必须回到 M1 的站姿。"""
    geom = LEG_GEOMETRY["FL"]
    q_stand = STANDING_JOINT_ANGLES[:3]  # FL 的 [abad, hip, knee]
    height = -forward_kinematics(q_stand, geom)[2]

    ctrl = NominalGaitController(NominalGaitConfig(stand_height=height), device="cpu")
    q = ctrl.compute(torch.zeros(1), torch.zeros(1, 3), torch.zeros(1, 2))[0].numpy()

    expected = pinocchio_to_isaac(STANDING_JOINT_ANGLES)
    assert np.abs(q[:4]).max() < 1e-12, "零指令下侧摆角必须为 0"
    assert np.abs(q - expected).max() < 0.06


def test_output_uses_isaac_joint_order():
    """输出必须是 Isaac 顺序：hip×4, thigh×4, calf×4。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    q = ctrl.compute(torch.zeros(1), torch.zeros(1, 3), torch.zeros(1, 2))[0]
    assert q.shape == (12,)
    assert torch.all(q[:4].abs() < 1e-12)  # 四个侧摆
    assert torch.all(q[4:8] > 0.3)  # 四个髋俯仰
    assert torch.all(q[8:] < -1.0)  # 四个膝


def test_custom_joint_order_is_respected():
    """按 Pinocchio 顺序请求时，输出必须相应重排。"""
    pin_order = tuple(f"{leg}_{j}_joint" for leg in LEG_ORDER for j in ("hip", "thigh", "calf"))
    ctrl = NominalGaitController(NominalGaitConfig(joint_order=pin_order), device="cpu")
    q = ctrl.compute(torch.zeros(1), torch.zeros(1, 3), torch.zeros(1, 2))[0]
    assert torch.all(q[0::3].abs() < 1e-12)  # 侧摆在 0, 3, 6, 9
    assert torch.all(q[1::3] > 0.3)  # 髋俯仰
    assert torch.all(q[2::3] < -1.0)  # 膝


def test_batch_independence():
    """不同环境互不串扰 —— 批量实现最容易写错的地方。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    t = torch.tensor([0.0, 0.13, 0.27])
    cmd = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.4, -0.5]])
    vel = torch.tensor([[0.0, 0.0], [0.6, 0.0], [0.1, 0.3]])

    together = ctrl.compute(t, cmd, vel)
    apart = torch.cat([ctrl.compute(t[i : i + 1], cmd[i : i + 1], vel[i : i + 1]) for i in range(3)])
    assert torch.allclose(together, apart, atol=1e-14)


def test_finite_for_extreme_inputs():
    """极端指令下不能出 NaN —— 域随机化与推力会造出很离谱的状态。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    cmd = torch.tensor([[10.0, -10.0, 20.0], [-8.0, 8.0, -20.0]])
    vel = torch.tensor([[15.0, -15.0], [-15.0, 15.0]])
    q = ctrl.compute(torch.tensor([0.0, 0.31]), cmd, vel)
    assert torch.all(torch.isfinite(q))


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="占空比"):
        NominalGaitConfig(duty_factor=1.0)
    with pytest.raises(ValueError, match="周期"):
        NominalGaitConfig(period=0.0)
    with pytest.raises(ValueError, match="高度"):
        NominalGaitConfig(swing_height=-0.1)


# ====================================================== 残差的数学


def test_residual_is_bounded():
    """残差 RL 的核心保证：动作与名义的距离有确定的上界。

    这是纯 RL 给不了的、可以写进安全论证的性质。
    """
    alpha = 0.1
    action_clip = 10.0  # 高斯策略的长尾在环境入口被限幅
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    q_nom = ctrl.compute(torch.zeros(16), torch.zeros(16, 3), torch.zeros(16, 2))

    actions = torch.randn(16, 12) * 5.0
    q_des = q_nom + alpha * actions.clamp(-action_clip, action_clip)
    assert (q_des - q_nom).abs().max() <= alpha * action_clip + 1e-12


def test_zero_residual_reproduces_nominal():
    """残差为零时，动作必须精确等于名义 —— α 课程从 0 起步的前提。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    t = torch.linspace(0.0, 0.4, 8)
    q_nom = ctrl.compute(t, torch.zeros(8, 3), torch.zeros(8, 2))
    q_des = q_nom + 0.1 * torch.zeros(8, 12)
    assert torch.equal(q_des, q_nom)


def test_phase_encoding_is_continuous():
    """相位的 sin/cos 编码在 0/1 交界处必须连续 —— 直接喂 φ 会有跳变。"""
    ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
    eps = 1e-6
    t = torch.tensor([ctrl.cfg.period - eps, ctrl.cfg.period + eps])
    phase = ctrl.phase(t)
    assert abs(phase[0, 0].item() - phase[1, 0].item()) > 0.9  # φ 本身跳变

    angle = 2 * math.pi * phase
    encoded = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    assert (encoded[0] - encoded[1]).abs().max() < 1e-4  # 编码后连续


# ====================================================== 课程项的降级分支


class _FakeEventManager:
    """只实现 ``active_terms`` 的假管理器。

    Isaac Lab 的 ``EventManager.active_terms`` 是 ``{模式: [名字, ...]}``，
    **不是**名字列表。这个结构差异曾经让推力课程的降级分支永远命中，
    整条课程被静默关掉而训练照跑 —— 没有任何报错。
    """

    def __init__(self, terms: dict[str, list[str]]) -> None:
        self.active_terms = terms


def test_push_curriculum_detects_term_across_modes():
    """事件项存在时，降级分支**不能**命中。这个测试就是为了撞它。"""
    manager = _FakeEventManager({"startup": ["physics_material"], "interval": ["push_robot"]})
    present = any("push_robot" in names for names in manager.active_terms.values())
    absent = any("push_robot" in names for names in _FakeEventManager({"reset": ["reset_base"]}).active_terms.values())
    assert present, "推力事件在 interval 模式下，必须被认出来"
    assert not absent, "推力事件确实不存在时，才该退化"


def test_active_terms_is_keyed_by_mode_not_name():
    """钉死这个结构假设：``name in active_terms`` 是错的写法。"""
    manager = _FakeEventManager({"interval": ["push_robot"]})
    assert "push_robot" not in manager.active_terms, "顶层键是模式名，不是事件名"
    assert "interval" in manager.active_terms
