"""Rollout 缓冲区与广义优势估计（GAE）。

## 优势函数为什么存在

策略梯度的原始形式

.. math::  \\nabla_\\theta J = \\mathbb E\\bigl[\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\, G_t\\bigr]

是**无偏**的，但方差大到没法用：:math:`G_t` 是一整条轨迹的回报，
一个坏动作出现在好轨迹里照样被强化。

减去任意只依赖状态的基线 :math:`b(s)` 不改变期望（因为
:math:`\\mathbb E_a[\\nabla_\\theta\\log\\pi_\\theta] = 0`），却能大幅降方差。
取 :math:`b(s)=V(s)` 时括号里就是优势 :math:`A(s,a)=Q(s,a)-V(s)` ——
"这个动作比该状态下的平均水平好多少"。

## GAE：在偏差和方差之间连续滑动

单步 TD 残差

.. math::  \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)

方差小，但只要 :math:`V` 不准就有偏。GAE 把不同步长的估计做指数加权：

.. math::  \\hat A^{\\mathrm{GAE}(\\gamma,\\lambda)}_t
           = \\sum_{l=0}^{\\infty} (\\gamma\\lambda)^l \\delta_{t+l}

* :math:`\\lambda=0` → :math:`\\hat A_t=\\delta_t`，低方差高偏差；
* :math:`\\lambda=1` → :math:`\\hat A_t=G_t-V(s_t)`，无偏但高方差。

实现上用**倒序递推**，一次 :math:`O(T)`：

.. math::  \\hat A_t = \\delta_t + \\gamma\\lambda\\,(1-d_t)\\,\\hat A_{t+1}

这与卡尔曼平滑（RTS smoother）的倒推结构惊人地像：都是先正向跑一遍拿到
逐步的"新息"，再倒着把未来信息折算回当前时刻。

## 那个几乎人人都踩的坑：超时 ≠ 失败

四足环境里 episode 结束有两种原因：

* **终止（termination）** —— 躯干触地，真的失败了。此后没有未来回报，
  :math:`V(s_{T})` 必须按 0 处理。
* **超时（truncation / time-out）** —— 跑满 20 秒，机器人好好的。
  把它当成"未来回报为 0"会给策略灌输一个错误信念：**活到第 20 秒是件坏事**。
  正确做法是在超时那一步把 :math:`\\gamma V(s_{T})` 加回奖励里（bootstrap）。

两者都要切断 GAE 的递推链（因为下一条轨迹与本条无关），但只有前者对应
"未来价值为零"。写错的症状很典型：训练曲线在接近 episode 长度时莫名塌陷。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

__all__ = ["compute_gae", "RolloutStorage", "Batch"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    time_out_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计，返回 ``(advantages, returns)``。

    Args:
        rewards: ``(T, N)`` 每步奖励。
        values: ``(T, N)`` critic 对 :math:`s_t` 的估计。
        dones: ``(T, N)`` 0/1，episode 在该步之后结束（终止**或**超时）。
        last_values: ``(N,)`` 对 :math:`s_T` 的估计，用于最后一步 bootstrap。
        gamma: 折扣因子。
        lam: GAE 的 λ。
        time_out_values: ``(T, N)``，仅在"超时"步为 :math:`V(s_{t+1})`，
            其余为 0。传入时会按 :math:`r_t \\mathrel{+}= \\gamma V(s_{t+1})`
            修正奖励 —— 见模块文档"超时 ≠ 失败"。

    Returns:
        ``advantages``、``returns``，形状均为 ``(T, N)``。
        ``returns = advantages + values``，是价值函数的回归目标。
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(f"形状不一致：rewards {tuple(rewards.shape)}, values {tuple(values.shape)}, dones {tuple(dones.shape)}")
    num_steps = rewards.shape[0]

    rewards = rewards.clone()
    if time_out_values is not None:
        rewards += gamma * time_out_values

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(num_steps)):
        next_values = last_values if step == num_steps - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * not_done * next_values - values[step]
        running = delta + gamma * lam * not_done * running
        advantages[step] = running
    return advantages, advantages + values


@dataclass
class Batch:
    """一个 minibatch。字段名与 :class:`RolloutStorage` 的缓冲区一一对应。"""

    observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_log_prob: torch.Tensor
    old_mu: torch.Tensor
    old_sigma: torch.Tensor


class RolloutStorage:
    """按 ``(T, N, ...)`` 布局存放一次 rollout 的全部转移。

    **为什么是 (T, N) 而不是把所有环境拼成一个大 batch？** GAE 的递推必须
    沿时间倒序进行，且每个环境的时间轴互相独立。保持 ``(T, N)`` 布局让
    递推变成一次向量化的倒序循环，N 个环境天然并行；等 GAE 算完，再
    ``flatten(0, 1)`` 成 ``T*N`` 条独立样本喂给 minibatch。

    Args:
        num_envs: 并行环境数。
        num_transitions_per_env: 每个环境采集的步数 T。
        obs_dim: actor 观测维度。
        critic_obs_dim: critic 观测维度。
        action_dim: 动作维度。
        device: 缓冲区所在设备。全部留在 GPU 上 —— 4096 环境 × 24 步的
            rollout 只有几十 MB，往 CPU 搬一趟反而成为瓶颈。
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device

        shape = (num_transitions_per_env, num_envs)
        z = lambda *d: torch.zeros(*shape, *d, device=device)  # noqa: E731

        self.observations = z(obs_dim)
        self.critic_observations = z(critic_obs_dim)
        self.actions = z(action_dim)
        self.rewards = z()
        self.dones = z()
        self.time_out_values = z()
        self.values = z()
        self.log_prob = z()
        self.mu = z(action_dim)
        self.sigma = z(action_dim)

        self.advantages = z()
        self.returns = z()
        self.step = 0

    def add(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_prob: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        time_out_values: torch.Tensor | None = None,
    ) -> None:
        """记录一步转移。"""
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("rollout 缓冲区已满，忘记调用 clear() 了")
        i = self.step
        self.observations[i] = observations
        self.critic_observations[i] = critic_observations
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones.float()
        self.values[i] = values
        self.log_prob[i] = log_prob
        self.mu[i] = mu
        self.sigma[i] = sigma
        if time_out_values is not None:
            self.time_out_values[i] = time_out_values
        self.step += 1

    def clear(self) -> None:
        self.step = 0
        self.time_out_values.zero_()

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float, normalize: bool = True) -> None:
        """算 GAE，并（可选）对整批优势做标准化。

        标准化让 clip 阈值 0.2 的含义与奖励量级脱钩 —— 否则每次改奖励权重
        都得重调 ``clip_param``。代价是引入了一点偏差（batch 内的均值被
        减掉了），实践中完全值得。
        """
        advantages, returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, lam, self.time_out_values
        )
        self.returns = returns
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages = advantages

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int) -> Iterator[Batch]:
        """把 ``T*N`` 条样本随机打散成 minibatch，重复 ``num_epochs`` 轮。

        **每轮都重新打散**：同一批数据被复用多次时，固定的分组会让梯度
        噪声在各轮之间高度相关，等价于减少了有效的更新次数。
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches != 0:
            raise ValueError(f"样本数 {batch_size} 不能被 minibatch 数 {num_mini_batches} 整除")
        mini_batch_size = batch_size // num_mini_batches

        flat = {
            "observations": self.observations.flatten(0, 1),
            "critic_observations": self.critic_observations.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "old_log_prob": self.log_prob.flatten(0, 1),
            "old_mu": self.mu.flatten(0, 1),
            "old_sigma": self.sigma.flatten(0, 1),
        }

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield Batch(**{k: v[idx] for k, v in flat.items()})
