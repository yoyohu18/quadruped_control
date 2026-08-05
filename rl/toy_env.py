"""玩具环境：把 Go2 换成一阶惯性环节，其余 MDP 结构原样保留。

## 它为什么必须存在

Isaac Sim 启动一次要十几秒、吃掉几 GB 显存，还依赖资产下载。如果 PPO 的
正确性只能靠"在 Go2 上跑一晚上看曲线"来验证，那这个里程碑就没有可回归的
测试，和前八个里程碑的标准不符。

所以这里造一个**纯 torch、毫秒级、可解析分析**的向量化环境，它与真实的
Go2 速度跟踪任务共享**完全相同的 MDP 骨架**：

| 结构 | 玩具环境 | Go2 |
|---|---|---|
| 动作 | 2 维加速度指令 | 12 维关节目标位置 |
| 执行器 | 一阶惯性 :math:`\\tau\\dot v = u - v` | PD + 电机模型 |
| 奖励 | :math:`\\exp(-\\lVert v-v^*\\rVert^2/\\sigma^2)` | 同一个式子 |
| 指令 | 每 N 步重采样 | 同 |
| 终止 | 速度发散 | 躯干触地 |
| 超时 | 固定步数 | 20 秒 |

**唯一被抽掉的是接触与欠驱动** —— 也就是四足真正难的那部分。所以：
它能验证 PPO 实现对不对，**不能**验证奖励设计好不好。这个界限要说清楚。

## 可解析的最优解

一阶系统在恒定指令下的最优策略是显然的：让 :math:`u = v^*`（稳态），
外加一点比例项加速收敛。所以最优回报有一个可以手算的上界，
测试里就用它当判据 —— 而不是"看着像在涨"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ToyEnvConfig", "VelocityTrackingToyEnv"]


@dataclass
class ToyEnvConfig:
    """玩具环境参数。

    Attributes:
        num_envs: 并行环境数。
        dt: 控制周期，取 0.02 s 与 Isaac Lab 的 50 Hz 策略频率一致。
        tau: 执行器一阶时间常数，秒。
        action_scale: 动作到指令速度的缩放。
        command_range: 速度指令的采样范围，m/s。
        command_resample_steps: 指令重采样间隔（步）。
        episode_length: 超时步数。
        tracking_sigma: 指数奖励的宽度。
        action_cost: 动作平方代价权重。
        velocity_limit: 超过它判定为终止（发散）。
        noise_std: 过程噪声，模拟仿真里的接触抖动。
    """

    num_envs: int = 64
    dt: float = 0.02
    tau: float = 0.15
    action_scale: float = 1.0
    command_range: tuple[float, float] = (-1.5, 1.5)
    command_resample_steps: int = 100
    episode_length: int = 200
    tracking_sigma: float = 0.25
    action_cost: float = 0.01
    velocity_limit: float = 5.0
    noise_std: float = 0.02


class VelocityTrackingToyEnv:
    """向量化的速度跟踪环境，接口与 Isaac Lab 的 RL 环境对齐。

    接口约定（与 :class:`rl.runner.OnPolicyRunner` 一致）：

    * ``reset() -> obs``
    * ``step(actions) -> (obs, rewards, dones, extras)``
    * ``extras["time_outs"]`` —— 布尔张量，标记"因超时而结束"，
      它与 ``dones`` 的区别是 GAE 里 bootstrap 与否的关键。

    观测 ``[v (2), v_cmd (2), last_action (2)]`` 共 6 维。**把上一步动作放进
    观测**是 locomotion 的标准做法：它让策略能感知自己造成的执行器滞后，
    否则 MDP 在存在执行器动态时并不马尔可夫。Go2 的 48 维观测里同样有
    ``actions`` 这一段，原因完全相同。
    """

    num_obs = 6
    num_actions = 2

    def __init__(self, config: ToyEnvConfig | None = None, device: str | torch.device = "cpu", seed: int | None = None) -> None:
        self.cfg = config or ToyEnvConfig()
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

        n = self.cfg.num_envs
        self.num_envs = n
        self.velocity = torch.zeros(n, 2, device=self.device)
        self.command = torch.zeros(n, 2, device=self.device)
        self.last_action = torch.zeros(n, 2, device=self.device)
        self.episode_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._resample_command(torch.arange(n, device=self.device))

    # ------------------------------------------------------------------ 工具

    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device) * (high - low) + low

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.command_range
        self.command[env_ids] = self._uniform((len(env_ids), 2), low, high)

    def _observations(self) -> torch.Tensor:
        return torch.cat([self.velocity, self.command, self.last_action], dim=-1)

    # ------------------------------------------------------------------ 接口

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """重置指定环境（缺省全部），返回当前观测。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) > 0:
            self.velocity[env_ids] = 0.0
            self.last_action[env_ids] = 0.0
            self.episode_step[env_ids] = 0
            self._resample_command(env_ids)
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """推进一步。

        Args:
            actions: ``(num_envs, 2)``，会被限幅到 ``[-10, 10]`` ——
                与 Isaac Lab 的 ``clip`` 动作项对应，防止高斯采样的长尾
                把仿真直接搞崩。

        Returns:
            ``obs, rewards, dones, extras``。**注意 obs 是重置之后的观测** ——
            当 episode 结束时返回的是新 episode 的第一帧，而"结束前那一帧"
            的价值通过 ``extras["time_out_values"]`` 之外的机制处理，见
            :func:`rl.storage.compute_gae` 的文档。
        """
        actions = actions.clamp(-10.0, 10.0)
        cfg = self.cfg

        # 一阶执行器：v ← v + dt/τ (u - v)
        target = cfg.action_scale * actions
        noise = torch.randn(self.velocity.shape, generator=self.generator, device=self.device) * cfg.noise_std
        self.velocity += cfg.dt / cfg.tau * (target - self.velocity) + noise * cfg.dt**0.5

        error = torch.sum((self.command - self.velocity) ** 2, dim=-1)
        rewards = torch.exp(-error / cfg.tracking_sigma) - cfg.action_cost * torch.sum(actions**2, dim=-1)

        self.last_action = actions.clone()
        self.episode_step += 1

        terminated = torch.any(self.velocity.abs() > cfg.velocity_limit, dim=-1)
        time_out = self.episode_step >= cfg.episode_length
        dones = terminated | time_out

        # 指令重采样（不重置 episode）—— 这是 locomotion 环境的标准做法：
        # 一条 episode 里要经历多个不同指令，策略才学得会"跟随"而不是"记住"。
        resample = (self.episode_step % cfg.command_resample_steps == 0) & ~dones
        if resample.any():
            self._resample_command(resample.nonzero(as_tuple=False).squeeze(-1))

        extras = {
            "time_outs": time_out,
            "terminated": terminated,
            "episode_step": self.episode_step.clone(),
        }
        if dones.any():
            self.reset(dones.nonzero(as_tuple=False).squeeze(-1))
        return self._observations(), rewards, dones, extras

    # ------------------------------------------------------------------ 基准

    def optimal_step_reward(self) -> torch.Tensor:
        """稳态最优单步奖励的解析上界。

        稳态下 :math:`v=v^*`，跟踪项为 1，动作项为
        :math:`-c\\lVert v^*/s\\rVert^2`。噪声让实际值略低，所以这是**上界**。
        测试用它来判断"学到了"而不是"看着在涨"。
        """
        return 1.0 - self.cfg.action_cost * torch.sum((self.command / self.cfg.action_scale) ** 2, dim=-1)
