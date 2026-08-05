"""里程碑 9 的测试：PPO 与奖励核。

**全部不依赖 Isaac Sim**，毫秒级跑完。这是把数学与仿真器解耦的直接回报。

交叉验证的三条独立路径，延续前八个里程碑的做法：

1. GAE 的**递推实现** vs **定义式的直接求和**（O(T²)，一定对但慢）；
2. GAE vs **rsl-rl 安装包里的实现**（工业界基准，逐元素比对）；
3. 解析 KL vs **torch.distributions.kl_divergence**；
4. 奖励核的 torch 版 vs **里程碑 4/6 的 numpy 实现**。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from footstep_planner import LIPMParams
from footstep_planner import capture_point as capture_point_np
from gait_scheduler import GAITS, GaitScheduler
from rl import (
    ActorCritic,
    EmpiricalNormalization,
    OnPolicyRunner,
    PPO,
    PPOConfig,
    RolloutStorage,
    RunningMeanStd,
    ToyEnvConfig,
    VelocityTrackingToyEnv,
    air_time_reward,
    capture_point,
    capture_point_error,
    compute_gae,
    exp_tracking_reward,
    foot_clearance_reward,
    foot_slip_penalty,
    gait_contact_reward,
    reference_contact,
)
from rl.reward_kernels import LEG_ORDER, TROT_OFFSETS
from rl.runner import RunnerConfig

torch.manual_seed(0)


# ====================================================================== GAE


def _gae_by_definition(rewards, values, dones, last_values, gamma, lam):
    """按定义 :math:`\\hat A_t=\\sum_l (\\gamma\\lambda)^l\\delta_{t+l}` 直接求和。

    O(T²) 且带 python 循环，慢得没法用在训练里 —— 但它是**定义本身**，
    不含任何递推技巧，因此是递推实现的独立基准。
    """
    T, N = rewards.shape
    delta = torch.zeros_like(rewards)
    for t in range(T):
        next_v = last_values if t == T - 1 else values[t + 1]
        delta[t] = rewards[t] + gamma * (1.0 - dones[t]) * next_v - values[t]

    adv = torch.zeros_like(rewards)
    for t in range(T):
        for n in range(N):
            total, coeff = 0.0, 1.0
            for l in range(t, T):  # noqa: E741
                total += coeff * delta[l, n]
                if dones[l, n] > 0.5:  # 轨迹在此断开，后续 δ 属于另一条 episode
                    break
                coeff *= gamma * lam
            adv[t, n] = total
    return adv


@pytest.fixture
def rollout():
    torch.manual_seed(7)
    T, N = 16, 6
    return {
        "rewards": torch.randn(T, N),
        "values": torch.randn(T, N),
        "dones": (torch.rand(T, N) < 0.15).float(),
        "last_values": torch.randn(N),
    }


def test_gae_recursion_matches_definition(rollout):
    """递推实现必须等于定义式的直接求和。"""
    gamma, lam = 0.97, 0.9
    adv, _ = compute_gae(**rollout, gamma=gamma, lam=lam)
    ref = _gae_by_definition(**rollout, gamma=gamma, lam=lam)
    assert torch.allclose(adv, ref, atol=1e-5)


def test_gae_matches_rsl_rl(rollout):
    """与 rsl-rl 安装包里的实现逐元素一致 —— 工业基准。"""
    tensordict = pytest.importorskip("tensordict")
    from rsl_rl.algorithms import PPO as RslPPO
    from rsl_rl.storage import RolloutStorage as RslStorage

    gamma, lam = 0.99, 0.95
    T, N = rollout["rewards"].shape
    obs = tensordict.TensorDict({"policy": torch.zeros(N, 3)}, batch_size=[N])

    storage = RslStorage("rl", N, T, obs, [2], device="cpu")
    storage.rewards.copy_(rollout["rewards"].unsqueeze(-1))
    storage.values.copy_(rollout["values"].unsqueeze(-1))
    storage.dones.copy_(rollout["dones"].unsqueeze(-1))

    alg = RslPPO.__new__(RslPPO)  # 绕过构造函数：只借它的 compute_returns
    alg.storage = storage
    alg.gamma, alg.lam = gamma, lam
    alg.normalize_advantage_per_mini_batch = True  # 关掉整批标准化，便于直接比对
    alg.critic = lambda _: rollout["last_values"].unsqueeze(-1)
    alg.compute_returns(obs)

    adv, ret = compute_gae(**rollout, gamma=gamma, lam=lam)
    assert torch.allclose(storage.returns.squeeze(-1), ret, atol=1e-6)
    assert torch.allclose(storage.advantages.squeeze(-1), adv, atol=1e-6)


def test_gae_lambda_zero_is_td_error(rollout):
    """λ=0 时优势退化为单步 TD 残差。"""
    gamma = 0.99
    adv, _ = compute_gae(**rollout, gamma=gamma, lam=0.0)
    r, v, d, lv = rollout["rewards"], rollout["values"], rollout["dones"], rollout["last_values"]
    expected = torch.zeros_like(r)
    for t in range(r.shape[0]):
        next_v = lv if t == r.shape[0] - 1 else v[t + 1]
        expected[t] = r[t] + gamma * (1 - d[t]) * next_v - v[t]
    assert torch.allclose(adv, expected, atol=1e-6)


def test_gae_lambda_one_is_monte_carlo():
    """λ=1 且无终止时，returns 等于蒙特卡洛折扣回报（含末端 bootstrap）。"""
    T, N, gamma = 10, 3, 0.9
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last_values = torch.randn(N)

    _, returns = compute_gae(rewards, values, dones, last_values, gamma=gamma, lam=1.0)

    mc = torch.zeros_like(rewards)
    running = last_values.clone()
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        mc[t] = running
    assert torch.allclose(returns, mc, atol=1e-5)


def test_timeout_bootstrap_changes_returns():
    """超时 bootstrap 必须抬高 returns —— 这是"超时≠失败"的量化体现。"""
    T, N, gamma = 6, 4, 0.99
    rewards = torch.ones(T, N)
    values = torch.full((T, N), 5.0)
    dones = torch.zeros(T, N)
    dones[-1] = 1.0  # 最后一步全部结束
    last_values = torch.zeros(N)

    _, without = compute_gae(rewards, values, dones, last_values, gamma, 0.95)
    time_out_values = torch.zeros(T, N)
    time_out_values[-1] = values[-1]  # 用 V(s_t) 近似 V(s_T)，与 rsl-rl 一致
    _, with_bootstrap = compute_gae(rewards, values, dones, last_values, gamma, 0.95, time_out_values)

    assert torch.all(with_bootstrap > without)
    # 差值恰好是最后一步多出来的 γV，再沿 GAE 链折回去
    assert with_bootstrap[-1].sub(without[-1]).allclose(torch.full((N,), gamma * 5.0), atol=1e-5)


def test_gae_shape_mismatch_raises():
    with pytest.raises(ValueError, match="形状不一致"):
        compute_gae(torch.zeros(4, 2), torch.zeros(4, 3), torch.zeros(4, 2), torch.zeros(2))


# ====================================================================== 网络


def test_log_prob_sums_over_action_dims():
    """对角高斯的 log_prob 必须**求和**而不是求均值。"""
    ac = ActorCritic(6, 4, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    obs = torch.randn(5, 6)
    actions = ac.act(obs)
    log_prob = ac.get_actions_log_prob(actions)
    manual = ac.distribution.log_prob(actions).sum(dim=-1)
    assert log_prob.shape == (5,)
    assert torch.allclose(log_prob, manual)


def test_entropy_matches_closed_form():
    """对角高斯熵 :math:`\\sum_i \\log(\\sigma_i\\sqrt{2\\pi e})`。"""
    ac = ActorCritic(3, 4, actor_hidden_dims=(8,), critic_hidden_dims=(8,), init_noise_std=0.7)
    ac.act(torch.randn(2, 3))
    expected = 4 * math.log(0.7 * math.sqrt(2 * math.pi * math.e))
    assert torch.allclose(ac.entropy, torch.full((2,), expected), atol=1e-5)


def test_kl_matches_torch_distributions():
    """解析 KL vs torch 官方实现 —— 独立路径交叉验证。"""
    torch.manual_seed(3)
    mu_old, mu_new = torch.randn(20, 5), torch.randn(20, 5)
    sigma_old, sigma_new = torch.rand(20, 5) + 0.1, torch.rand(20, 5) + 0.1

    ours = ActorCritic.kl_divergence(mu_old, sigma_old, mu_new, sigma_new)
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_old, sigma_old), torch.distributions.Normal(mu_new, sigma_new)
    ).sum(dim=-1)
    assert torch.allclose(ours, reference, atol=1e-4)


def test_kl_is_zero_for_identical_distributions():
    mu, sigma = torch.randn(8, 3), torch.rand(8, 3) + 0.5
    kl = ActorCritic.kl_divergence(mu, sigma, mu, sigma)
    assert torch.allclose(kl, torch.zeros(8), atol=1e-4)


def test_output_layer_small_gain_gives_small_initial_actions():
    """初始策略必须接近"保持默认站姿"，否则第一帧就摔。"""
    ac = ActorCritic(48, 12, actor_hidden_dims=(128, 128), critic_hidden_dims=(128, 128))
    mean = ac.act_inference(torch.randn(64, 48))
    assert mean.abs().max() < 0.2, "输出层增益过大，初始动作会把机器人抽翻"


def test_act_inference_is_distribution_mean():
    ac = ActorCritic(4, 3, actor_hidden_dims=(8,), critic_hidden_dims=(8,))
    obs = torch.randn(6, 4)
    ac.act(obs)
    assert torch.allclose(ac.act_inference(obs), ac.action_mean)


def test_critic_accepts_privileged_observations():
    """非对称 Actor-Critic：两组观测维度不同也必须能跑。"""
    ac = ActorCritic(45, 12, num_critic_obs=48, actor_hidden_dims=(32,), critic_hidden_dims=(32,))
    assert ac.evaluate(torch.randn(7, 48)).shape == (7,)
    assert ac.act(torch.randn(7, 45)).shape == (7, 12)


def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="未知激活函数"):
        ActorCritic(3, 2, activation="不存在的激活")


# ====================================================================== PPO


def _fake_storage(num_envs=32, steps=4, obs_dim=6, action_dim=3, seed=0):
    """造一个手工 rollout，让 PPO 可以脱离环境被测试。"""
    torch.manual_seed(seed)
    storage = RolloutStorage(num_envs, steps, obs_dim, obs_dim, action_dim)
    for _ in range(steps):
        storage.add(
            observations=torch.randn(num_envs, obs_dim),
            critic_observations=torch.randn(num_envs, obs_dim),
            actions=torch.randn(num_envs, action_dim),
            rewards=torch.randn(num_envs),
            dones=torch.zeros(num_envs),
            values=torch.randn(num_envs),
            log_prob=-torch.rand(num_envs) * 3,
            mu=torch.randn(num_envs, action_dim),
            sigma=torch.full((num_envs, action_dim), 1.0),
        )
    storage.compute_returns(torch.randn(num_envs), gamma=0.99, lam=0.95)
    return storage


def test_clipped_surrogate_kills_gradient_beyond_threshold():
    """优势为正、ratio 已超出 1+ε 时，策略梯度必须被截成 0。

    这是 PPO 的**核心机制**，值得单独钉一个测试：构造一个"策略已经比旧策略
    远得多"的样本，检查损失对 μ 的梯度确实为零。
    """
    clip = 0.2
    advantage = torch.tensor([1.0])
    log_ratio = torch.tensor([math.log(1.5)], requires_grad=True)  # ratio = 1.5 > 1+ε
    ratio = torch.exp(log_ratio)

    surrogate = -advantage * ratio
    surrogate_clipped = -advantage * ratio.clamp(1 - clip, 1 + clip)
    loss = torch.max(surrogate, surrogate_clipped).mean()
    loss.backward()

    assert log_ratio.grad.abs().item() == pytest.approx(0.0, abs=1e-12)


def test_clipped_surrogate_keeps_gradient_when_escaping_bad_action():
    """优势为负时不封顶 —— 允许无限制地逃离坏动作。这个不对称是刻意的。"""
    clip = 0.2
    advantage = torch.tensor([-1.0])
    log_ratio = torch.tensor([math.log(1.5)], requires_grad=True)
    ratio = torch.exp(log_ratio)

    loss = torch.max(-advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip)).mean()
    loss.backward()
    assert log_ratio.grad.abs().item() > 0.1


def test_adaptive_lr_decreases_when_kl_too_large():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    alg = PPO(ac, PPOConfig(desired_kl=0.01, learning_rate=1e-3, schedule="adaptive"))
    mu = torch.zeros(10, 3)
    # μ 挪 1.0 → KL ≈ 0.5 × 3 = 1.5，远超 2×desired_kl
    alg._adapt_learning_rate(mu, torch.ones(10, 3), mu + 1.0, torch.ones(10, 3))
    assert alg.learning_rate == pytest.approx(1e-3 / 1.5)


def test_adaptive_lr_increases_when_kl_too_small():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    alg = PPO(ac, PPOConfig(desired_kl=0.01, learning_rate=1e-3, schedule="adaptive"))
    mu = torch.zeros(10, 3)
    alg._adapt_learning_rate(mu, torch.ones(10, 3), mu + 1e-3, torch.ones(10, 3))
    assert alg.learning_rate == pytest.approx(1e-3 * 1.5)


def test_adaptive_lr_respects_bounds():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    alg = PPO(ac, PPOConfig(desired_kl=0.01, learning_rate=1e-3), learning_rate_bounds=(1e-4, 2e-3))
    mu, sigma = torch.zeros(4, 3), torch.ones(4, 3)
    for _ in range(50):
        alg._adapt_learning_rate(mu, sigma, mu + 5.0, sigma)
    assert alg.learning_rate == pytest.approx(1e-4)
    for _ in range(50):
        alg._adapt_learning_rate(mu, sigma, mu, sigma * 1.0000001)
    assert alg.learning_rate == pytest.approx(2e-3)


def test_fixed_schedule_does_not_change_lr():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    alg = PPO(ac, PPOConfig(schedule="fixed", learning_rate=7e-4))
    alg._adapt_learning_rate(torch.zeros(4, 3), torch.ones(4, 3), torch.full((4, 3), 3.0), torch.ones(4, 3))
    assert alg.learning_rate == pytest.approx(7e-4)


def test_update_returns_expected_statistics():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    alg = PPO(ac, PPOConfig(num_learning_epochs=2, num_mini_batches=2))
    stats = alg.update(_fake_storage())
    for key in ("surrogate_loss", "value_loss", "entropy", "kl", "clip_fraction", "learning_rate", "action_std"):
        assert key in stats and math.isfinite(stats[key])
    assert 0.0 <= stats["clip_fraction"] <= 1.0


def test_update_changes_parameters():
    ac = ActorCritic(6, 3, actor_hidden_dims=(16,), critic_hidden_dims=(16,))
    before = ac.actor[0].weight.clone()
    PPO(ac, PPOConfig(num_learning_epochs=1, num_mini_batches=1)).update(_fake_storage())
    assert not torch.allclose(before, ac.actor[0].weight)


def test_value_clipping_bounds_the_update():
    """clip 后的价值预测不会偏离旧值超过 ε。"""
    clip = 0.2
    old_values = torch.zeros(100)
    new_values = torch.linspace(-5.0, 5.0, 100)
    clipped = old_values + (new_values - old_values).clamp(-clip, clip)
    assert clipped.abs().max() <= clip + 1e-6


def test_invalid_ppo_config_raises():
    with pytest.raises(ValueError, match="clip_param"):
        PPOConfig(clip_param=1.5)
    with pytest.raises(ValueError, match="schedule"):
        PPOConfig(schedule="cosine")
    with pytest.raises(ValueError, match="gamma"):
        PPOConfig(gamma=1.5)


def test_minibatch_generator_covers_every_sample_once_per_epoch():
    storage = _fake_storage(num_envs=16, steps=4)
    batches = list(storage.mini_batch_generator(num_mini_batches=4, num_epochs=1))
    assert len(batches) == 4
    total = sum(b.actions.shape[0] for b in batches)
    assert total == 16 * 4


def test_minibatch_indivisible_raises():
    storage = _fake_storage(num_envs=10, steps=3)
    with pytest.raises(ValueError, match="不能被"):
        list(storage.mini_batch_generator(num_mini_batches=4, num_epochs=1))


def test_storage_overflow_raises():
    storage = RolloutStorage(4, 1, 3, 3, 2)
    args = dict(
        observations=torch.zeros(4, 3), critic_observations=torch.zeros(4, 3), actions=torch.zeros(4, 2),
        rewards=torch.zeros(4), dones=torch.zeros(4), values=torch.zeros(4),
        log_prob=torch.zeros(4), mu=torch.zeros(4, 2), sigma=torch.ones(4, 2),
    )
    storage.add(**args)
    with pytest.raises(RuntimeError, match="缓冲区已满"):
        storage.add(**args)


# ====================================================================== 归一化


def test_running_mean_std_matches_batch_statistics():
    """分 20 批增量更新的结果必须等于一次性统计。"""
    torch.manual_seed(11)
    data = torch.randn(2000, 5) * torch.tensor([1.0, 10.0, 0.1, 3.0, 100.0]) + 7.0
    rms = RunningMeanStd((5,), epsilon=1e-8)
    for chunk in data.split(100):
        rms.update(chunk)
    assert torch.allclose(rms.mean, data.mean(dim=0), atol=1e-3)
    assert torch.allclose(rms.var, data.var(dim=0, unbiased=False), rtol=1e-3)


def test_normalization_produces_zero_mean_unit_variance():
    torch.manual_seed(12)
    data = torch.randn(5000, 3) * 20.0 + 100.0
    norm = EmpiricalNormalization((3,))
    norm.train()
    out = torch.cat([norm(chunk) for chunk in data.split(500)])
    tail = out[-1000:]  # 早期统计量还没收敛，只看尾部
    assert tail.mean(dim=0).abs().max() < 0.15
    assert (tail.std(dim=0) - 1.0).abs().max() < 0.15


def test_normalization_frozen_in_eval_mode():
    """评估时统计量必须冻结，否则策略行为会漂移。"""
    norm = EmpiricalNormalization((2,))
    norm.train()
    norm(torch.randn(100, 2))
    norm.eval()
    before = norm.rms.mean.clone()
    norm(torch.randn(100, 2) + 50.0)
    assert torch.allclose(before, norm.rms.mean)


def test_normalization_until_stops_updating():
    norm = EmpiricalNormalization((2,), until=500)
    norm.train()
    norm(torch.randn(600, 2))
    assert norm.frozen
    before = norm.rms.mean.clone()
    norm(torch.randn(100, 2) + 30.0)
    assert torch.allclose(before, norm.rms.mean)


def test_normalization_inverse_round_trip():
    norm = EmpiricalNormalization((3,))
    norm.train()
    data = torch.randn(1000, 3) * 4.0 + 2.0
    norm(data)
    norm.eval()
    x = torch.randn(10, 3)
    assert torch.allclose(norm.inverse(norm(x)), x, atol=1e-4)


# ====================================================================== 奖励核


def test_capture_point_matches_milestone_6():
    """torch 版捕获点必须与 M6 的 numpy 实现逐元素一致。"""
    torch.manual_seed(5)
    pos = torch.randn(32, 2)
    vel = torch.randn(32, 2) * 0.5
    height = torch.rand(32) * 0.2 + 0.2

    ours = capture_point(pos, vel, height)
    for i in range(32):
        params = LIPMParams(height=float(height[i]))
        ref = capture_point_np(pos[i].numpy().astype(float), vel[i].numpy().astype(float), params)
        assert np.allclose(ours[i].numpy(), ref, atol=1e-5)


def test_capture_point_scalar_height_broadcasts():
    pos, vel = torch.zeros(4, 2), torch.ones(4, 2)
    xi = capture_point(pos, vel, 0.30)
    omega = math.sqrt(9.81 / 0.30)
    assert torch.allclose(xi, torch.full((4, 2), 1.0 / omega), atol=1e-5)


def test_capture_point_error_zero_when_centred():
    """静止且捕获点落在支撑形心上时，惩罚为 0。"""
    feet = torch.tensor([[[0.2, 0.1], [0.2, -0.1], [-0.2, 0.1], [-0.2, -0.1]]])
    err = capture_point_error(
        base_pos_xy=torch.zeros(1, 2),
        base_vel_xy=torch.zeros(1, 2),
        base_height=torch.tensor([0.3]),
        foot_pos_xy=feet,
        contact=torch.ones(1, 4),
    )
    assert err.item() == pytest.approx(0.0, abs=1e-9)


def test_capture_point_error_grows_with_velocity():
    feet = torch.tensor([[[0.2, 0.1], [0.2, -0.1], [-0.2, 0.1], [-0.2, -0.1]]]).repeat(3, 1, 1)
    vels = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]])
    err = capture_point_error(
        torch.zeros(3, 2), vels, torch.full((3,), 0.3), feet, torch.ones(3, 4)
    )
    assert err[0] < err[1] < err[2]


def test_capture_point_error_zero_in_flight():
    """腾空相没有支撑区域，必须返回 0 而不是发散的数值。"""
    feet = torch.randn(2, 4, 2)
    err = capture_point_error(
        torch.randn(2, 2), torch.randn(2, 2), torch.full((2,), 0.3), feet, torch.zeros(2, 4)
    )
    assert torch.all(err == 0.0)


@pytest.mark.parametrize("gait_name", ["trot", "pace", "bound", "crawl", "stand"])
def test_reference_contact_matches_gait_scheduler(gait_name):
    """torch 版参考接触必须与 M4 的调度器逐元素一致 —— 定义不许漂移。"""
    gait = GAITS[gait_name]
    scheduler = GaitScheduler(gait)
    offsets = tuple(gait.phase_offsets[leg] for leg in LEG_ORDER)

    times = np.linspace(0.0, 3.0, 61)
    ours = reference_contact(
        torch.tensor(times, dtype=torch.float64),
        period=gait.period,
        duty_factor=gait.duty_factor,
        offsets=offsets,
    ).numpy()
    reference = np.array([scheduler.contact(float(t)) for t in times])
    assert np.array_equal(ours, reference)


def test_trot_offsets_match_gait_library():
    """常量 TROT_OFFSETS 必须与 M4 的步态库一致，一旦漂移大声失败。"""
    gait = GAITS["trot"]
    assert TROT_OFFSETS == tuple(gait.phase_offsets[leg] for leg in LEG_ORDER)
    assert LEG_ORDER == ("FL", "FR", "RL", "RR")


def test_trot_reference_has_diagonal_pairs():
    """trot 的定义性质：对角腿同相，同侧腿反相。"""
    contact = reference_contact(torch.linspace(0.0, 1.6, 41))
    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
    assert torch.all(fl == rr)
    assert torch.all(fr == rl)
    assert torch.all(fl != fr)


def test_gait_contact_reward_bounds():
    ref = reference_contact(torch.tensor([0.0, 0.1, 0.25]))
    assert torch.allclose(gait_contact_reward(ref, ref), torch.ones(3))
    assert torch.allclose(gait_contact_reward(~ref, ref), torch.zeros(3))


def test_exp_tracking_reward_properties():
    """有界、单调、e^{-1} 处的宽度符合 σ 的定义。"""
    assert exp_tracking_reward(torch.zeros(1), 0.25).item() == pytest.approx(1.0)
    assert exp_tracking_reward(torch.tensor([0.25]), 0.25).item() == pytest.approx(math.exp(-1.0))
    errors = torch.linspace(0.0, 10.0, 50)
    rewards = exp_tracking_reward(errors, 0.25)
    assert torch.all(rewards.diff() < 0)
    assert torch.all((rewards > 0) & (rewards <= 1))


def test_exp_tracking_reward_rejects_bad_sigma():
    with pytest.raises(ValueError, match="sigma"):
        exp_tracking_reward(torch.zeros(1), 0.0)


def test_foot_clearance_ignores_stationary_feet():
    """速度加权：站着不动的脚不论多高都不该被罚。"""
    height = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
    still = torch.zeros(1, 4)
    assert foot_clearance_reward(height, still, 0.08).item() == pytest.approx(0.0)
    moving = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert foot_clearance_reward(height, moving, 0.08).item() > 0.1


def test_foot_clearance_minimised_at_target_height():
    vel = torch.ones(3, 4)
    heights = torch.tensor([[0.02] * 4, [0.08] * 4, [0.20] * 4])
    penalty = foot_clearance_reward(heights, vel, target_height=0.08)
    assert penalty[1] == pytest.approx(0.0, abs=1e-9)
    assert penalty[0] > 0 and penalty[2] > 0


def test_foot_slip_only_counts_stance_feet():
    vel = torch.ones(1, 4, 2)
    assert foot_slip_penalty(vel, torch.zeros(1, 4)).item() == pytest.approx(0.0)
    assert foot_slip_penalty(vel, torch.ones(1, 4)).item() == pytest.approx(8.0)


def test_air_time_only_scored_at_touchdown():
    """脚一直举着不落地拿不到分 —— 否则最优解是永远抬腿。"""
    air = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
    no_contact = torch.zeros(1, 4)
    assert air_time_reward(air, no_contact, threshold=0.2).item() == pytest.approx(0.0)
    touchdown = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert air_time_reward(air, touchdown, threshold=0.2).item() == pytest.approx(0.7)


def test_air_time_zero_for_standing_command():
    air = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
    touchdown = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reward = air_time_reward(air, touchdown, threshold=0.2, command_norm=torch.tensor([0.01]))
    assert reward.item() == pytest.approx(0.0)


# ====================================================================== 玩具环境


def test_toy_env_shapes_and_reset():
    env = VelocityTrackingToyEnv(ToyEnvConfig(num_envs=8), seed=0)
    obs = env.reset()
    assert obs.shape == (8, env.num_obs)
    obs, rewards, dones, extras = env.step(torch.zeros(8, 2))
    assert obs.shape == (8, env.num_obs)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)
    assert "time_outs" in extras and "terminated" in extras


def test_toy_env_times_out_not_terminates():
    """定速跑满 episode 必须报超时而不是终止 —— GAE 的 bootstrap 依赖它。"""
    cfg = ToyEnvConfig(num_envs=4, episode_length=10, noise_std=0.0)
    env = VelocityTrackingToyEnv(cfg, seed=0)
    env.reset()
    for _ in range(10):
        _, _, dones, extras = env.step(torch.zeros(4, 2))
    assert torch.all(dones)
    assert torch.all(extras["time_outs"])
    assert not torch.any(extras["terminated"])


def test_toy_env_reward_peaks_at_command():
    """动作正好等于指令速度时，稳态奖励接近解析上界。"""
    cfg = ToyEnvConfig(num_envs=1, noise_std=0.0, episode_length=10_000, command_resample_steps=10_000)
    env = VelocityTrackingToyEnv(cfg, seed=0)
    env.reset()
    command = env.command.clone()
    for _ in range(200):
        _, reward, _, _ = env.step(command / cfg.action_scale)
    assert reward.item() == pytest.approx(float(env.optimal_step_reward()), abs=1e-3)


def test_toy_env_terminates_on_divergence():
    cfg = ToyEnvConfig(num_envs=2, velocity_limit=1.0, noise_std=0.0, action_scale=10.0, tau=0.02)
    env = VelocityTrackingToyEnv(cfg, seed=0)
    env.reset()
    for _ in range(5):
        _, _, _, extras = env.step(torch.full((2, 2), 10.0))
    assert torch.any(extras["terminated"])


# ====================================================================== 端到端


def test_ppo_learns_toy_task():
    """端到端：PPO 必须把回报学到解析最优的 80% 以上。

    判据是**解析上界的比例**，不是"曲线在涨"。前八个里程碑都拒绝"看着对"
    这种验收方式，这里也一样。
    """
    torch.manual_seed(0)
    cfg = ToyEnvConfig(num_envs=256, episode_length=200)
    env = VelocityTrackingToyEnv(cfg, seed=0)
    runner = OnPolicyRunner(
        env,
        ActorCritic(env.num_obs, env.num_actions, actor_hidden_dims=(64, 64), critic_hidden_dims=(64, 64)),
        PPOConfig(),
        RunnerConfig(num_steps_per_env=24, seed=0),
    )
    history = runner.learn(60, verbose=False)

    optimal = float(env.optimal_step_reward().mean()) * cfg.episode_length
    final = history[-1]["mean_episode_reward"]
    assert final > 0.8 * optimal, f"回报 {final:.1f} 未达最优 {optimal:.1f} 的 80%"
    # σ 必须收缩：策略在收敛，而不是靠运气拿分
    assert history[-1]["action_std"] < 0.7 * history[0]["action_std"]


def test_deterministic_with_fixed_seed():
    """同种子两次训练必须逐位一致 —— 不可复现的实验没有调参可言。"""

    def run() -> float:
        torch.manual_seed(42)
        env = VelocityTrackingToyEnv(ToyEnvConfig(num_envs=32), seed=42)
        runner = OnPolicyRunner(
            env,
            ActorCritic(env.num_obs, env.num_actions, actor_hidden_dims=(16,), critic_hidden_dims=(16,)),
            PPOConfig(num_learning_epochs=2, num_mini_batches=2),
            RunnerConfig(num_steps_per_env=8, seed=42),
        )
        return runner.learn(5, verbose=False)[-1]["surrogate_loss"]

    assert run() == run()
