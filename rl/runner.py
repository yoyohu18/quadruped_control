"""On-policy 训练循环：采样 → GAE → 多轮小批量更新 → 重复。

## 一次迭代到底在做什么

.. code-block:: text

    for it in range(max_iterations):
        with no_grad:                       # 采样阶段：网络是"冻结"的
            for t in range(num_steps):      # N 个环境同步走 T 步
                a ~ π(·|s);  s', r, d = env.step(a)
                存 (s, a, r, d, V(s), logπ(a|s), μ, σ)
        A, R = GAE(...)                     # 一次性算完 T×N 条优势
        for epoch in range(K):              # 学习阶段：同一批数据用 K 遍
            for mb in minibatches:
                更新 θ

**采样阶段必须 no_grad**：这一步只是在造数据，梯度信息毫无用处，
留着会白白占掉几个 GB 显存。忘记加 ``no_grad`` 是显存 OOM 的头号原因。

## 为什么 T 那么小（24 步）而 N 那么大（4096）

一次迭代的样本数是 :math:`T\\times N = 98304`。同样的样本量，
可以是 4096 环境 × 24 步，也可以是 96 环境 × 1024 步。**前者远远更好**：

* 24 步 @ 50 Hz 只有 0.48 秒，策略在这段时间里几乎没变，
  数据的 on-policy 性质更"新鲜"；
* 4096 条互相独立的轨迹让 batch 内的相关性极低，梯度估计方差小；
* GPU 并行度拉满 —— 这是 Isaac Sim 存在的全部理由。

代价是每条轨迹被切碎，GAE 的有效视野只有 24 步。这就是为什么
**必须有 critic** 来 bootstrap 24 步之外的价值 —— 蒙特卡洛回报在这里
根本不可用。

> 与无人机的类比：这与 MPC 的 horizon 选择是同一个权衡。有了终端代价
> （= 价值函数），短 horizon 也能得到近似最优；没有终端代价，horizon
> 必须长到覆盖整个任务。critic 就是学出来的终端代价。
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import torch

from .networks import ActorCritic
from .normalization import EmpiricalNormalization
from .ppo import PPO, PPOConfig
from .storage import RolloutStorage

__all__ = ["VecEnvProtocol", "RunnerConfig", "OnPolicyRunner"]


class VecEnvProtocol(Protocol):
    """训练循环对环境的全部要求 —— 刻意做得极窄。

    玩具环境、Isaac Lab 环境、将来的 MuJoCo 环境都只要满足这几条就能用。
    """

    num_envs: int
    num_obs: int
    num_actions: int

    def reset(self) -> torch.Tensor: ...

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...


class RunnerConfig:
    """训练循环参数。

    Args:
        num_steps_per_env: 每次迭代每个环境采集的步数 T。
        max_iterations: 迭代次数。
        save_interval: 每多少次迭代存一次 checkpoint。
        log_interval: 每多少次迭代打印一次。
        normalize_observations: 是否做观测归一化。
        experiment_name: 日志与 checkpoint 的子目录名。
        log_dir: 日志根目录，``None`` 表示不落盘。
        seed: 随机种子。
    """

    def __init__(
        self,
        num_steps_per_env: int = 24,
        max_iterations: int = 1000,
        save_interval: int = 50,
        log_interval: int = 10,
        normalize_observations: bool = True,
        experiment_name: str = "ppo",
        log_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_steps_per_env = num_steps_per_env
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.normalize_observations = normalize_observations
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.seed = seed


class OnPolicyRunner:
    """把环境、网络、PPO 粘起来的训练循环。

    Args:
        env: 满足 :class:`VecEnvProtocol` 的向量化环境。
        actor_critic: 网络；``None`` 时按环境维度自动构造。
        ppo_config: PPO 超参数。
        runner_config: 训练循环参数。
        device: 计算设备。
    """

    def __init__(
        self,
        env: VecEnvProtocol,
        actor_critic: ActorCritic | None = None,
        ppo_config: PPOConfig | None = None,
        runner_config: RunnerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.cfg = runner_config or RunnerConfig()
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        num_critic_obs = getattr(env, "num_critic_obs", env.num_obs)
        if actor_critic is None:
            actor_critic = ActorCritic(env.num_obs, env.num_actions, num_critic_obs)
        self.alg = PPO(actor_critic, ppo_config, device=self.device)

        if self.cfg.normalize_observations:
            self.obs_normalizer = EmpiricalNormalization((env.num_obs,)).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization((num_critic_obs,)).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        self.storage = RolloutStorage(
            env.num_envs,
            self.cfg.num_steps_per_env,
            env.num_obs,
            num_critic_obs,
            env.num_actions,
            device=self.device,
        )

        self.current_iteration = 0
        self.total_steps = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ 训练

    def learn(self, max_iterations: int | None = None, verbose: bool = True) -> list[dict[str, float]]:
        """跑完整个训练，返回每次迭代的统计量。"""
        max_iterations = max_iterations or self.cfg.max_iterations
        obs = self.env.reset().to(self.device)
        critic_obs = self._critic_obs(obs)

        reward_window: deque[float] = deque(maxlen=100)
        length_window: deque[float] = deque(maxlen=100)
        episode_reward = torch.zeros(self.env.num_envs, device=self.device)
        episode_length = torch.zeros(self.env.num_envs, device=self.device)

        for _ in range(max_iterations):
            t_start = time.time()
            self.storage.clear()

            # ---------------- 采样 ----------------
            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_env):
                    norm_obs = self.obs_normalizer(obs)
                    norm_critic_obs = self.critic_obs_normalizer(critic_obs)
                    actions = self.alg.actor_critic.act(norm_obs)
                    values = self.alg.actor_critic.evaluate(norm_critic_obs)
                    log_prob = self.alg.actor_critic.get_actions_log_prob(actions)
                    mu = self.alg.actor_critic.action_mean
                    sigma = self.alg.actor_critic.action_std.expand_as(mu)

                    next_obs, rewards, dones, extras = self.env.step(actions)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # 超时 bootstrap：用 V(s_t) 近似 V(s_T)。
                    # 严格做法要拿"重置前那一帧"的观测再过一次 critic；
                    # 单步之内两者差别很小，rsl-rl 与 legged_gym 都用这个近似。
                    time_outs = extras.get("time_outs")
                    time_out_values = (
                        values * time_outs.to(self.device).float() if time_outs is not None else None
                    )

                    self.storage.add(
                        norm_obs, norm_critic_obs, actions, rewards, dones,
                        values, log_prob, mu, sigma, time_out_values,
                    )

                    episode_reward += rewards
                    episode_length += 1
                    if dones.any():
                        idx = dones.nonzero(as_tuple=False).squeeze(-1)
                        reward_window.extend(episode_reward[idx].tolist())
                        length_window.extend(episode_length[idx].tolist())
                        episode_reward[idx] = 0.0
                        episode_length[idx] = 0.0

                    obs = next_obs
                    critic_obs = self._critic_obs(obs)

                last_values = self.alg.actor_critic.evaluate(self.critic_obs_normalizer(critic_obs))

            self.storage.compute_returns(
                last_values, self.alg.cfg.gamma, self.alg.cfg.lam, self.alg.cfg.normalize_advantage
            )

            # ---------------- 学习 ----------------
            stats = self.alg.update(self.storage)

            self.current_iteration += 1
            self.total_steps += self.cfg.num_steps_per_env * self.env.num_envs
            stats.update(
                iteration=self.current_iteration,
                total_steps=self.total_steps,
                fps=self.cfg.num_steps_per_env * self.env.num_envs / (time.time() - t_start),
                mean_episode_reward=statistics.fmean(reward_window) if reward_window else float("nan"),
                mean_episode_length=statistics.fmean(length_window) if length_window else float("nan"),
            )
            self.history.append(stats)

            if verbose and self.current_iteration % self.cfg.log_interval == 0:
                self._log(stats)
            if self.cfg.log_dir and self.current_iteration % self.cfg.save_interval == 0:
                self.save(self.cfg.log_dir / self.cfg.experiment_name / f"model_{self.current_iteration}.pt")

        return self.history

    # ------------------------------------------------------------------ 工具

    def _critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """取 critic 观测。环境若不提供特权观测，就复用 actor 的。"""
        getter = getattr(self.env, "get_critic_observations", None)
        return getter().to(self.device) if getter is not None else obs

    @staticmethod
    def _log(stats: dict[str, float]) -> None:
        print(
            f"[{stats['iteration']:5d}] "
            f"回报 {stats['mean_episode_reward']:8.2f} | "
            f"长度 {stats['mean_episode_length']:6.1f} | "
            f"KL {stats['kl']:.4f} | "
            f"lr {stats['learning_rate']:.2e} | "
            f"σ {stats['action_std']:.3f} | "
            f"clip {stats['clip_fraction']:.2f} | "
            f"EV {stats['explained_variance']:5.2f} | "
            f"{stats['fps']:.0f} step/s"
        )

    def get_inference_policy(self):
        """返回可直接调用的确定性策略（含观测归一化），用于评估与部署。"""
        self.alg.actor_critic.eval()
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.eval()

        def policy(obs: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return self.alg.actor_critic.act_inference(self.obs_normalizer(obs.to(self.device)))

        return policy

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **self.alg.state_dict(),
                "obs_normalizer": self.obs_normalizer.state_dict(),
                "critic_obs_normalizer": self.critic_obs_normalizer.state_dict(),
                "iteration": self.current_iteration,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.load_state_dict(state)
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.critic_obs_normalizer.load_state_dict(state["critic_obs_normalizer"])
        self.current_iteration = state.get("iteration", 0)
        self.total_steps = state.get("total_steps", 0)
