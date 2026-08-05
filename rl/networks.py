"""Actor-Critic 网络：对角高斯策略 + 状态值函数。

## 策略为什么是"高斯"

动作是 12 个关节的目标位置增量 —— 连续量。连续控制里策略被参数化为

.. math::  \\pi_\\theta(a\\mid s) = \\mathcal N\\bigl(\\mu_\\theta(s),\\ \\operatorname{diag}(\\sigma^2)\\bigr)

网络只输出均值 :math:`\\mu_\\theta(s)`，标准差 :math:`\\sigma` 是**与状态
无关**的可学习参数。这一点常被初学者写错成"网络也输出 σ"。两种做法都能跑，
但四足这一行几乎清一色用前者，原因有三：

1. **探索与决策解耦。** σ 变成一个全局的"探索温度"，训练曲线上可以直接
   读出策略收敛到什么程度（σ 单调下降）。若 σ 依赖状态，网络会学会在难的
   状态下把 σ 调大来骗取熵奖励，探索反而失控。
2. **梯度更干净。** :math:`\\partial \\log\\pi/\\partial\\sigma` 只有 12 个
   参数承接，方差远小于经由整张网络回传。
3. **部署时直接丢掉。** 上真机用 :math:`\\mu_\\theta(s)`，σ 与网络无耦合，
   删掉不影响任何计算图。

与无人机的类比：这就像把 NMPC 的**代价权重**做成常数而不是状态相关 ——
状态相关的权重会让优化器找到"改变权重"而非"改善轨迹"的捷径。

## 为什么最后一层要缩小初始化增益

策略输出乘上 ``action_scale`` 后叠加到默认关节角上。若最后一层用标准
初始化，初始动作是幅值 ~1 rad 的随机关节角，机器人第一帧就劈叉倒地，
所有 episode 在 3 步内终止，价值函数学不到任何东西。

把最后一层增益设成 0.01，初始策略 ≈ "保持默认站姿 + 小噪声"。这是
legged locomotion 里最有效的一个"免费"技巧。

## 非对称 Actor-Critic

Critic 只在**训练时**使用，仿真里什么都能读 —— 基座真实线速度、地形高度、
摩擦系数、外力。这些量真机上没有，所以不能进 actor 的观测，但可以进
critic。价值估计越准，优势的方差越小，样本效率越高。

这就是 Isaac Lab 里 ``policy`` 与 ``critic`` 两组观测的由来。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = ["ActorCritic", "build_mlp", "ACTIVATIONS"]

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "crelu": nn.CELU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str = "elu",
    output_gain: float = 0.01,
) -> nn.Sequential:
    """构造多层感知机，隐藏层正交初始化，输出层小增益。

    正交初始化保证前向传播时各层的奇异值接近 1，深层网络不会出现
    激活值指数放大/衰减 —— 与控制里"让状态转移矩阵谱半径接近 1"是同一件事。

    Args:
        input_dim: 输入维度。
        output_dim: 输出维度。
        hidden_dims: 各隐藏层宽度。
        activation: 激活函数名，见 :data:`ACTIVATIONS`。缺省 ELU ——
            ReLU 的死区在关节角这种可正可负的输入上会丢掉半边信息。
        output_gain: 输出层权重初始化增益。策略网络取 0.01。
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"未知激活函数 {activation!r}，可选：{sorted(ACTIVATIONS)}")
    act = ACTIVATIONS[activation]

    dims = (input_dim, *hidden_dims)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers += [linear, act()]

    head = nn.Linear(dims[-1], output_dim)
    nn.init.orthogonal_(head.weight, gain=output_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """对角高斯策略 + 独立的 critic。

    两张网络**不共享**主干。共享主干能省参数，但价值损失的梯度会污染策略
    表征 —— 在 locomotion 这种价值量级远大于策略梯度的任务上尤其明显。
    参数量在这里根本不是瓶颈（策略只有十几万参数），所以分开写。

    Args:
        num_actor_obs: actor 观测维度（真机上拿得到的量）。
        num_critic_obs: critic 观测维度（可含特权信息），缺省与 actor 相同。
        num_actions: 动作维度，Go2 为 12。
        actor_hidden_dims: actor 隐藏层。
        critic_hidden_dims: critic 隐藏层。
        activation: 激活函数。
        init_noise_std: :math:`\\sigma` 初值。1.0 配合 ``action_scale=0.25``
            意味着初始探索幅度约 0.25 rad —— 足够大到能翻身，又不至于自残。
        noise_std_type: ``"scalar"`` 直接学 σ；``"log"`` 学 log σ 后取指数，
            后者保证 σ 恒正，且在 σ 很小时步长自动变小。
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        num_critic_obs: int | None = None,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
    ) -> None:
        super().__init__()
        if init_noise_std <= 0.0:
            raise ValueError("初始噪声标准差必须为正")
        if noise_std_type not in ("scalar", "log"):
            raise ValueError("noise_std_type 只能是 'scalar' 或 'log'")

        num_critic_obs = num_actor_obs if num_critic_obs is None else num_critic_obs
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type

        self.actor = build_mlp(num_actor_obs, num_actions, actor_hidden_dims, activation, output_gain=0.01)
        # critic 的输出层不需要小增益：初始价值估计是多少无所谓，几十次更新就修正了。
        self.critic = build_mlp(num_critic_obs, 1, critic_hidden_dims, activation, output_gain=1.0)

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self._distribution: Normal | None = None
        # 允许 std 在反传里被优化，但 Normal 的参数校验会拖慢速度
        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------ 分布

    @property
    def action_std(self) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std
        return torch.exp(self.log_std)

    def update_distribution(self, obs: torch.Tensor) -> Normal:
        """由观测构造动作分布并缓存。"""
        mean = self.actor(obs)
        std = self.action_std.expand_as(mean)
        self._distribution = Normal(mean, std)
        return self._distribution

    @property
    def distribution(self) -> Normal:
        if self._distribution is None:
            raise RuntimeError("必须先调用 act() 或 update_distribution()")
        return self._distribution

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def entropy(self) -> torch.Tensor:
        """每个样本的策略熵（各维求和）。"""
        return self.distribution.entropy().sum(dim=-1)

    # ------------------------------------------------------------------ 接口

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """采样一个动作（训练时用）。"""
        return self.update_distribution(obs).sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """确定性动作（评估与部署用）—— 直接取分布均值。"""
        return self.actor(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """给定动作在**当前**分布下的对数概率（各维求和）。

        各维求和而不是取均值：对角高斯的联合密度是各维密度之积，
        取对数就是求和。写成均值会让 PPO 的 ratio 变成几何平均，
        clip 阈值的含义随动作维度漂移 —— 这是一个很难查的 bug。
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """状态值 :math:`V(s)`，形状 ``(batch,)``。"""
        return self.critic(critic_obs).squeeze(-1)

    @staticmethod
    def kl_divergence(
        mu_old: torch.Tensor,
        sigma_old: torch.Tensor,
        mu_new: torch.Tensor,
        sigma_new: torch.Tensor,
    ) -> torch.Tensor:
        """两个对角高斯之间的解析 KL，逐样本返回。

        .. math::

            D_{\\mathrm{KL}} = \\sum_i \\Bigl[
                \\log\\frac{\\sigma_{new,i}}{\\sigma_{old,i}}
                + \\frac{\\sigma_{old,i}^2 + (\\mu_{old,i}-\\mu_{new,i})^2}
                        {2\\sigma_{new,i}^2}
                - \\frac12 \\Bigr]

        **用解析式而不是采样估计**是 PPO 自适应学习率能稳定工作的前提：
        采样估计的 KL 方差极大，学习率会被噪声推着乱跳。
        """
        return torch.sum(
            torch.log(sigma_new / sigma_old + 1e-5)
            + (sigma_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * sigma_new.pow(2))
            - 0.5,
            dim=-1,
        )
