"""里程碑 9 的可视化：PPO 的每一个部件到底在干什么。

生成 ``docs/figures/m9_rl.png``，包含八组图：

1. clip 目标函数的形状 —— 优势正负时的不对称；
2. GAE 的偏差-方差权衡，λ 扫描（数值实验，不是示意图）；
3. 玩具环境上的真实学习曲线，对照解析最优；
4. 自适应学习率与 KL 的闭环 —— PPO 稳定的真正原因；
5. 探索噪声 σ 的收缩与熵；
6. 超时 bootstrap 写错会怎样；
7. 指数跟踪奖励 vs 二次惩罚，以及捕获点惩罚随速度的增长；
8. 关键数字汇总。

全部在 CPU 上跑，不需要 Isaac Sim，几十秒完成。

运行::

    python scripts/viz_rl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import (  # noqa: E402
    ActorCritic,
    OnPolicyRunner,
    PPOConfig,
    ToyEnvConfig,
    VelocityTrackingToyEnv,
    capture_point_error,
    compute_gae,
    exp_tracking_reward,
)
from rl.runner import RunnerConfig  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m9_rl.png"
CLIP = 0.2


# ---------------------------------------------------------------- 数值实验


def train_toy(iterations: int = 120, seed: int = 0):
    """在玩具环境上真训一遍，拿到真实的学习曲线。"""
    torch.manual_seed(seed)
    cfg = ToyEnvConfig(num_envs=512, episode_length=200)
    env = VelocityTrackingToyEnv(cfg, seed=seed)
    runner = OnPolicyRunner(
        env,
        ActorCritic(env.num_obs, env.num_actions, actor_hidden_dims=(64, 64), critic_hidden_dims=(64, 64)),
        PPOConfig(),
        RunnerConfig(num_steps_per_env=24, seed=seed),
    )
    history = runner.learn(iterations, verbose=False)
    optimal = float(env.optimal_step_reward().mean()) * cfg.episode_length
    return history, optimal


def gae_bias_variance(
    critic_error_std: float = 1.2,
    num_trials: int = 20_000,
    horizon: int = 80,
    gamma: float = 0.99,
    seed: int = 0,
):
    """λ 扫描：GAE 给出的**价值回归目标**的偏差平方与方差。

    被估计的量必须说清楚，否则这张图会画错。这里估计的是真实状态价值
    :math:`V^*(s_t)`，估计量是 GAE 给出的 returns
    :math:`\\hat R_t = \\hat A_t + V(s_t)` —— 也就是 critic 实际回归的那个目标。

    构造一条已知真值的链：每步奖励 :math:`r_t\\sim\\mathcal N(\\bar r_t,\\sigma_r)`，
    均值序列已知，于是 :math:`V^*(s_t)=\\sum_l \\gamma^l\\bar r_{t+l}` 可以精确求和。
    critic 被人为加上**每个状态一个固定的**估计误差 —— 这正是训练早期价值
    网络的真实状态（系统性偏，不是逐次采样的噪声）。

    两端的行为可以手推，正好是偏差-方差的两个极端：

    * :math:`\\lambda=0`：:math:`\\hat R_t = r_t + \\gamma V(s_{t+1})`，
      方差只有一步奖励的 :math:`\\sigma_r^2`，偏差是 :math:`\\gamma e(s_{t+1})`；
    * :math:`\\lambda=1`：:math:`\\hat R_t = G_t`，**无偏**，但方差累积成
      :math:`\\sigma_r^2(1-\\gamma^{2n})/(1-\\gamma^2)`。
    """
    rng = np.random.default_rng(seed)
    reward_std = 0.5
    mean_rewards = 1.0 + 0.5 * np.sin(np.arange(horizon + 1) * 0.3)

    true_values = np.array([np.sum(gamma ** np.arange(horizon + 1 - t) * mean_rewards[t:]) for t in range(horizon + 1)])
    critic = true_values + rng.normal(0.0, critic_error_std, size=horizon + 1)

    rewards = torch.tensor(
        rng.normal(mean_rewards[:horizon][None, :], reward_std, size=(num_trials, horizon)).T, dtype=torch.float32
    )
    values = torch.tensor(np.tile(critic[:horizon][:, None], (1, num_trials)), dtype=torch.float32)
    last_values = torch.full((num_trials,), float(critic[horizon]))
    dones = torch.zeros(horizon, num_trials)
    target = torch.tensor(true_values[:horizon], dtype=torch.float32)[:, None]

    lambdas = np.linspace(0.0, 1.0, 41)
    bias_sq, variance = [], []
    for lam in lambdas:
        _, returns = compute_gae(rewards, values, dones, last_values, gamma, float(lam))
        error = (returns - target)[: horizon // 2]  # 只看前半段，末端受 horizon 截断影响
        bias_sq.append(float(error.mean(dim=1).pow(2).mean()))
        variance.append(float(error.var(dim=1).mean()))
    return lambdas, np.array(bias_sq), np.array(variance)


def timeout_bootstrap_demo(horizon: int = 40, gamma: float = 0.99):
    """存活奖励恒为 1 时，超时处理正确与否给出的价值目标。"""
    rewards = torch.ones(horizon, 1)
    true_value = float(np.sum(gamma ** np.arange(400)))  # 长期存活的真实价值
    values = torch.full((horizon, 1), true_value)
    dones = torch.zeros(horizon, 1)
    dones[-1] = 1.0
    last_values = torch.zeros(1)

    _, wrong = compute_gae(rewards, values, dones, last_values, gamma, 0.95)
    time_out_values = torch.zeros(horizon, 1)
    time_out_values[-1] = values[-1]
    _, right = compute_gae(rewards, values, dones, last_values, gamma, 0.95, time_out_values)
    return wrong.squeeze(-1).numpy(), right.squeeze(-1).numpy(), true_value


# ---------------------------------------------------------------- 绘图


def main() -> None:
    print("训练玩具环境（PPO，120 次迭代）……")
    history, optimal = train_toy()
    print(f"  最终平均回报 {history[-1]['mean_episode_reward']:.1f} / 解析最优 {optimal:.1f}")

    print("GAE 的 λ 扫描（两种 critic 精度）……")
    lambdas, bias_sq, variance = gae_bias_variance(critic_error_std=1.2)
    _, bias_sq_bad, variance_bad = gae_bias_variance(critic_error_std=6.0)
    best = lambdas[np.argmin(bias_sq + variance)]
    best_bad = lambdas[np.argmin(bias_sq_bad + variance_bad)]
    print(f"  critic 较准 → 最优 λ={best:.2f}；critic 很差 → 最优 λ={best_bad:.2f}")

    print("超时 bootstrap 对照……")
    wrong, right, true_value = timeout_bootstrap_demo()

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.26)

    # -- 1. clip 目标函数 -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ratio = np.linspace(0.0, 2.0, 400)
    for adv, color, label in [(1.0, "#1f77b4", "优势 A > 0"), (-1.0, "#d62728", "优势 A < 0")]:
        obj = np.minimum(ratio * adv, np.clip(ratio, 1 - CLIP, 1 + CLIP) * adv)
        ax.plot(ratio, obj, color=color, lw=2.2, label=label)
    ax.axvspan(1 - CLIP, 1 + CLIP, color="0.9", zorder=0)
    ax.axvline(1.0, color="0.4", ls=":", lw=1)
    ax.set_xlabel(r"重要性比值 $r_t(\theta)=\pi_\theta/\pi_{old}$")
    ax.set_ylabel(r"$L^{CLIP}$")
    ax.set_title("① clip 目标：不对称是刻意的", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.text(0.03, 1.28, "A>0：越过 1+ε 就封顶（别扑得太急）", fontsize=8, color="#1f77b4", va="top")
    ax.text(1.97, -1.55, "A<0：不封顶\n（随便逃离坏动作）", fontsize=8, color="#d62728", va="top", ha="right")
    ax.grid(alpha=0.3)

    # -- 2. GAE 偏差-方差 -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(lambdas, bias_sq, "--", color="#d62728", lw=1.5, label="偏差²（critic 较准）")
    ax.plot(lambdas, variance, "--", color="#1f77b4", lw=1.5, label="方差")
    ax.plot(lambdas, bias_sq + variance, "-", color="#1f77b4", lw=2.6, label=f"总 MSE，最优 λ={best:.2f}")
    ax.plot(lambdas, bias_sq_bad + variance_bad, "-", color="#d62728", lw=2.6,
            label=f"critic 很差时，最优 λ={best_bad:.2f}")
    ax.plot([best, best_bad], [np.min(bias_sq + variance), np.min(bias_sq_bad + variance_bad)], "k*", ms=11)
    ax.axvline(0.95, color="#2ca02c", ls=":", lw=1.6)
    ax.set_ylim(0, 4.0)
    ax.text(0.955, 3.5, "业界默认 0.95", fontsize=8, color="#2ca02c", rotation=90, va="top")
    ax.set_xlabel(r"GAE 的 $\lambda$")
    ax.set_ylabel("价值目标的均方误差")
    ax.set_title("② 偏差-方差权衡：最优 λ 跟着 critic 的精度走", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # -- 3. 学习曲线 -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    iters = [h["iteration"] for h in history]
    reward = [h["mean_episode_reward"] for h in history]
    ax.plot(iters, reward, color="#1f77b4", lw=2)
    ax.axhline(optimal, color="#2ca02c", ls="--", lw=1.6, label=f"解析最优 {optimal:.0f}")
    ax.axhline(0.8 * optimal, color="0.55", ls=":", lw=1.4, label="测试判据 80%")
    ax.set_xlabel("迭代")
    ax.set_ylabel("每 episode 平均回报")
    ax.set_title("③ 玩具环境的真实学习曲线", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    # -- 4. 自适应学习率闭环 ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    kl = [h["kl"] for h in history]
    lr = [h["learning_rate"] for h in history]
    ax.semilogy(iters, kl, color="#d62728", lw=1.6, label="KL 散度")
    ax.axhline(0.01, color="#2ca02c", ls="--", lw=1.4, label="目标 KL = 0.01")
    ax.axhspan(0.005, 0.02, color="#2ca02c", alpha=0.12)
    ax.set_xlabel("迭代")
    ax.set_ylabel("KL", color="#d62728")
    ax2 = ax.twinx()
    ax2.semilogy(iters, lr, color="#1f77b4", lw=1.6)
    ax2.set_ylabel("学习率", color="#1f77b4")
    ax.set_title("④ 以 KL 为被控量的学习率控制器", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)

    # -- 5. σ 与熵 ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(iters, [h["action_std"] for h in history], color="#9467bd", lw=2, label=r"探索噪声 $\sigma$")
    ax.set_xlabel("迭代")
    ax.set_ylabel(r"$\sigma$", color="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(iters, [h["clip_fraction"] for h in history], color="#ff7f0e", lw=1.5, label="被 clip 比例")
    ax2.axhline(0.3, color="#ff7f0e", ls=":", lw=1.2)
    ax2.set_ylabel("被 clip 的样本比例", color="#ff7f0e")
    ax.set_title("⑤ 探索收缩 + clip 比例（>0.3 说明步子太大）", fontsize=11)
    ax.grid(alpha=0.3)

    # -- 6. 超时 bootstrap ------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    steps = np.arange(len(wrong))
    ax.plot(steps, right, color="#2ca02c", lw=2.2, label="正确：超时时 bootstrap")
    ax.plot(steps, wrong, color="#d62728", lw=2.2, label="错误：当成失败")
    ax.axhline(true_value, color="0.4", ls="--", lw=1.4, label="真实价值")
    ax.set_xlabel("rollout 内的步数")
    ax.set_ylabel("价值回归目标")
    ax.set_title("⑥ 超时≠失败：写错就在教策略「别活太久」", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)

    # -- 7. 奖励核形状 -----------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    err = np.linspace(0.0, 3.0, 300)
    ax.plot(err, exp_tracking_reward(torch.tensor(err**2), 0.25).numpy(), color="#1f77b4", lw=2.2,
            label=r"指数核 $e^{-e^2/\sigma}$")
    ax.plot(err, -(err**2) / 9.0, color="#d62728", lw=2.2, ls="--", label=r"二次惩罚 $-e^2$（归一化）")
    ax.set_xlabel("速度跟踪误差 [m/s]")
    ax.set_ylabel("奖励")
    ax.set_title("⑦ 为什么用指数核：有界 → 隐式课程", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # -- 8. 捕获点惩罚 -----------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    feet = torch.tensor([[[0.19, 0.11], [0.19, -0.11], [-0.19, 0.11], [-0.19, -0.11]]])
    speeds = torch.linspace(0.0, 2.0, 100)
    for height, color in [(0.24, "#d62728"), (0.30, "#1f77b4"), (0.38, "#2ca02c")]:
        vel = torch.stack([speeds, torch.zeros_like(speeds)], dim=-1)
        err_cp = capture_point_error(
            torch.zeros(len(speeds), 2), vel, torch.full((len(speeds),), height),
            feet.repeat(len(speeds), 1, 1), torch.ones(len(speeds), 4),
        )
        ax.plot(speeds.numpy(), err_cp.numpy(), color=color, lw=2, label=f"质心高 {height:.2f} m")
    ax.axhline(0.19**2, color="0.4", ls=":", lw=1.4)
    ax.text(0.05, 0.19**2 * 1.15, "捕获点跑出支撑区", fontsize=8, color="0.35")
    ax.set_xlabel("前进速度 [m/s]")
    ax.set_ylabel(r"$\|\xi - \bar p_{stance}\|^2$")
    ax.set_title("⑧ 里程碑 6 的捕获点当奖励项", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # -- 9. 汇总 -----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    final = history[-1]
    ax.text(
        0.0, 1.0,
        "\n".join([
            "里程碑 9 的关键数字",
            "",
            f"玩具环境（无仿真器）：{len(history)} 次迭代，"
            f"回报 {final['mean_episode_reward']:.0f} / 最优 {optimal:.0f}"
            f"（{100 * final['mean_episode_reward'] / optimal:.0f}%）",
            f"  σ: {history[0]['action_std']:.2f} → {final['action_std']:.2f}（策略在收敛）",
            f"  KL 中位数 {np.median(kl):.4f}，围绕目标 0.01 来回穿越 —— 控制器在工作",
            "",
            f"GAE 的最优 λ：critic 较准时 {best:.2f}，critic 很差时 {best_bad:.2f}。",
            "业界固定 0.95 是偏保守的折中 —— 训练早期 critic 很差，",
            "大 λ 才对；等 critic 学准了，小 λ 其实更省方差。",
            "",
            "交叉验证（4 条独立路径）：",
            "  GAE 递推  vs  定义式直接求和",
            "  GAE       vs  rsl-rl 安装包（逐元素一致）",
            "  解析 KL   vs  torch.distributions",
            "  奖励核    vs  M4 步态调度器 / M6 捕获点",
            "",
            "Isaac Sim 实测（RTX 5080，Go2 平地，本仓库自己的 PPO）：",
            "  1024 环境 6.0–6.7 万 step/s，观测 45(policy) / 48(critic)，",
            "  15 项奖励全部有信号，30 次迭代 explained variance 到 0.97、",
            "  episode 长度 196 → 558（机器人在学「别摔」）。",
            "1500 次迭代 ≈ 1.5 亿步 ≈ 仿真里 34 天 —— 真机不可能采到。",
        ]),
        va="top", fontsize=9.5, linespacing=1.45,
    )

    fig.suptitle("四足运动控制 —— 里程碑 9：Isaac Lab 中的 PPO", fontsize=16, y=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
