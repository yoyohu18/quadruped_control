"""PPO：从策略梯度到可以真正跑起来的更新规则。

## 一句话：PPO 是"带信任域的策略梯度"

朴素策略梯度只能走**一小步** —— 采集的数据只在当前策略下无偏，策略一变
数据就失效。于是每批数据只能更新一次，样本效率极低。

想多更新几次，就得用重要性采样把旧数据"折算"到新策略下：

.. math::  J(\\theta) = \\mathbb E_{a\\sim\\pi_{old}}
           \\Bigl[\\frac{\\pi_\\theta(a\\mid s)}{\\pi_{old}(a\\mid s)} A(s,a)\\Bigr]

问题是这个比值 :math:`r_t(\\theta)` 一旦偏离 1 太远，估计的方差会爆炸。
TRPO 的解法是显式加一个 KL 信任域约束，然后解带约束的二次规划 ——
干净、有单调改善保证，但要算 Fisher 矩阵与共轭梯度，工程上很重。

PPO 的解法是**直接把目标函数削平**：

.. math::  L^{CLIP} = \\mathbb E\\bigl[\\min\\bigl(r_t A_t,\\
           \\operatorname{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t\\bigr)\\bigr]

当更新方向对策略有利且 :math:`r_t` 已经跑出 :math:`[1-\\epsilon,1+\\epsilon]`
时，梯度被截成 0；反方向（优势为负、比值反而变大）时梯度**保留** ——
这个不对称是刻意的：允许无限制地"逃离"坏动作，但限制"扑向"好动作。

> 与无人机的类比：这是**控制增量限幅**，不是控制量限幅。NMPC 里对
> :math:`\\Delta u` 加约束是为了不让求解器给出物理上追不上的跳变；PPO 对
> :math:`\\Delta\\pi` 加约束是为了不让数据的有效性跳变。两者都是
> "信任你的模型，但只信任一小步"。

## 削平之外还要一层保险：自适应学习率

clip 只保证**单个样本**的比值不越界，不保证策略整体的 KL 不越界 ——
均值移动 + 方差收缩可以在每个 ratio 都合规的情况下把分布挪得很远。
所以 rsl-rl 这一支（legged_gym 传统）额外做一件事：每个 minibatch 算一次
解析 KL，据此调学习率

* KL > 2 × 目标 → 学习率除以 1.5（步子太大）；
* KL < 目标 / 2 → 学习率乘以 1.5（太保守）。

**这才是四足训练稳定的主因**，比 clip 本身更关键。它本质上是一个
以 KL 为被控量、学习率为控制量的 P 控制器 —— 你在无人机上熟悉的
增益调度，换了个地方出现。

## 价值函数也 clip

.. math::  L^{V} = \\max\\bigl[(V-\\hat R)^2,\\ (V^{clip}-\\hat R)^2\\bigr],\\quad
           V^{clip} = V_{old} + \\operatorname{clip}(V-V_{old}, -\\epsilon, \\epsilon)

理由相同：价值网络一步跳太远，下一批的优势估计就全歪了。

## 熵奖励

.. math::  L = L^{CLIP} + c_v L^{V} - c_e \\mathcal H[\\pi]

减去熵意味着**鼓励**策略保持随机性。没有它，σ 会在几百次迭代内塌到接近 0，
策略过早锁死在一个局部最优的步态上（典型症状：机器人学会用一条腿蹭着走，
再也跳不出来）。四足上 :math:`c_e` 常取 0.005~0.01。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorCritic
from .storage import RolloutStorage

__all__ = ["PPOConfig", "PPO"]


@dataclass
class PPOConfig:
    """PPO 超参数。缺省值取自 legged_gym / rsl-rl 在四足上的通用配置。

    Attributes:
        clip_param: :math:`\\epsilon`，同时用于策略与价值的 clip。
        value_loss_coef: 价值损失权重 :math:`c_v`。
        entropy_coef: 熵奖励权重 :math:`c_e`。
        num_learning_epochs: 同一批数据重复利用的轮数。
        num_mini_batches: 每轮切成几个 minibatch。
        learning_rate: 初始学习率；``schedule="adaptive"`` 时会被 KL 调整。
        schedule: ``"adaptive"``（按 KL 调）或 ``"fixed"``。
        gamma: 折扣因子。0.99 @ 50 Hz 对应约 2 秒的有效视野 ——
            正好覆盖两三个步态周期，这不是巧合。
        lam: GAE 的 λ。
        desired_kl: 自适应学习率的目标 KL。
        max_grad_norm: 梯度范数裁剪。
        use_clipped_value_loss: 价值损失是否 clip。
        normalize_advantage: 是否对整批优势做标准化。
    """

    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError("clip_param 应在 (0, 1) 内，典型值 0.2")
        if self.schedule not in ("adaptive", "fixed"):
            raise ValueError("schedule 只能是 'adaptive' 或 'fixed'")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma 应在 (0, 1]")
        if self.num_mini_batches < 1 or self.num_learning_epochs < 1:
            raise ValueError("minibatch 数与 epoch 数必须 ≥ 1")


class PPO:
    """PPO 更新器。只负责"给定 rollout，把网络更新一次"。

    采样、环境交互、日志都不在这里 —— 它们属于
    :class:`rl.runner.OnPolicyRunner`。这样切分之后，这个类可以脱离
    Isaac Sim 单独测试：喂一个手工构造的 :class:`RolloutStorage` 进去，
    检查损失、KL、学习率的变化是否符合公式。测试就是这么写的。

    Args:
        actor_critic: 策略与价值网络。
        config: 超参数。
        device: 计算设备。
        learning_rate_bounds: 自适应学习率的上下界。
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
        learning_rate_bounds: tuple[float, float] = (1e-5, 1e-2),
    ) -> None:
        self.cfg = config or PPOConfig()
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)
        self.learning_rate = self.cfg.learning_rate
        self.lr_min, self.lr_max = learning_rate_bounds

    # ------------------------------------------------------------------ 更新

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        """用一批 rollout 更新网络，返回本次更新的统计量。

        Returns:
            ``surrogate_loss``、``value_loss``、``entropy``、``kl``、
            ``learning_rate``、``clip_fraction``（被 clip 掉的样本比例，
            调参时最有用的一个数：长期 > 0.3 说明学习率偏大）。
        """
        totals = dict.fromkeys(
            ("surrogate_loss", "value_loss", "entropy", "kl", "clip_fraction", "explained_variance"), 0.0
        )
        num_updates = 0

        for batch in storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            self.actor_critic.update_distribution(batch.observations)
            log_prob = self.actor_critic.get_actions_log_prob(batch.actions)
            values = self.actor_critic.evaluate(batch.critic_observations)
            mu = self.actor_critic.action_mean
            sigma = self.actor_critic.action_std.expand_as(mu)
            entropy = self.actor_critic.entropy

            kl_mean = self._adapt_learning_rate(batch.old_mu, batch.old_sigma, mu, sigma)

            # ---- 策略损失：clipped surrogate ----
            ratio = torch.exp(log_prob - batch.old_log_prob)
            surrogate = -batch.advantages * ratio
            surrogate_clipped = -batch.advantages * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            # 注意是 max 不是 min —— 这里的 surrogate 已经取了负号（要最小化）
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- 价值损失 ----
            if self.cfg.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.cfg.clip_param, self.cfg.clip_param
                )
                value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (values - batch.returns).pow(2).mean()

            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                clipped = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
                residual_var = (batch.returns - values).var()
                ev = 1.0 - residual_var / (batch.returns.var() + 1e-8)

            totals["surrogate_loss"] += surrogate_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["kl"] += float(kl_mean)
            totals["clip_fraction"] += clipped.item()
            totals["explained_variance"] += ev.item()
            num_updates += 1

        stats = {k: v / max(num_updates, 1) for k, v in totals.items()}
        stats["learning_rate"] = self.learning_rate
        stats["action_std"] = float(self.actor_critic.action_std.mean())
        return stats

    # ------------------------------------------------------------------ 内部

    def _adapt_learning_rate(
        self,
        mu_old: torch.Tensor,
        sigma_old: torch.Tensor,
        mu_new: torch.Tensor,
        sigma_new: torch.Tensor,
    ) -> float:
        """按解析 KL 调整学习率，返回本 minibatch 的平均 KL。

        ``schedule="fixed"`` 时只计算 KL 用于监控，不动学习率 ——
        KL 曲线本身就是最好的诊断信号，任何情况下都值得记录。
        """
        with torch.no_grad():
            kl_mean = ActorCritic.kl_divergence(mu_old, sigma_old, mu_new, sigma_new).mean()

            if self.cfg.schedule == "adaptive":
                if kl_mean > self.cfg.desired_kl * 2.0:
                    self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.cfg.desired_kl / 2.0:
                    self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

        return float(kl_mean)

    # ------------------------------------------------------------------ 存取

    def state_dict(self) -> dict:
        return {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor_critic.load_state_dict(state["actor_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = state.get("learning_rate", self.cfg.learning_rate)
