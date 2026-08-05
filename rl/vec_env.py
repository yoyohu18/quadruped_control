"""把 Isaac Lab 的 ``ManagerBasedRLEnv`` 适配成 :class:`rl.runner.VecEnvProtocol`。

**本模块不 import isaaclab** —— 它只调用被包装对象的方法。于是
``import rl`` 在没有 Isaac Sim 的机器上依然成立，而这个适配器在有
Isaac Sim 时立刻可用。这种"依赖倒置"在机器人代码里非常值钱：
控制算法不该知道自己跑在哪个仿真器里。

## 需要抹平的三处差异

1. **观测是 dict。** Isaac Lab 返回 ``{"policy": ..., "critic": ...}``，
   我们的 runner 要两个独立张量。
2. **终止分两种。** gymnasium 的 ``terminated`` / ``truncated`` 正好对应
   "真失败" / "超时"，直接映射到 ``dones`` 与 ``extras["time_outs"]``。
   Isaac Lab 在这一点上是干净的 —— 早期 legged_gym 把两者混在一个
   ``reset_buf`` 里，是无数 bug 的源头。
3. **reset 返回值。** gymnasium 的 ``reset()`` 返回 ``(obs, info)``。
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["IsaacLabVecEnv"]


class IsaacLabVecEnv:
    """Isaac Lab 环境 → 本项目 runner 的适配层。

    Args:
        env: ``ManagerBasedRLEnv`` 实例（或它的 gym wrapper 的 ``unwrapped``）。
        clip_actions: 动作限幅，``None`` 表示不限。高斯策略的长尾偶尔会给出
            幅值 10+ 的动作，不限幅的话仿真器可能直接 NaN。
        randomize_episode_starts: 首次 reset 后把各环境的 episode 计时器打散。
            **不做这件事会有一个隐蔽的后果**：4096 个环境同时开始、同时超时，
            于是每隔 1000 步就出现一次"全体重置"，价值函数看到的是一个带
            强周期性的信号，早期训练曲线会出现规律的锯齿。rsl-rl 的
            ``learn(init_at_random_ep_len=True)`` 做的就是这件事。

    Attributes:
        num_envs / num_obs / num_critic_obs / num_actions: runner 需要的维度。
    """

    def __init__(
        self,
        env: Any,
        clip_actions: float | None = 100.0,
        randomize_episode_starts: bool = True,
    ) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.randomize_episode_starts = randomize_episode_starts
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim

        obs_groups = env.observation_manager.group_obs_dim
        self.num_obs = obs_groups["policy"][0]
        self.num_critic_obs = obs_groups["critic"][0] if "critic" in obs_groups else self.num_obs
        self._critic_obs = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device)

    # ------------------------------------------------------------------ 接口

    def reset(self) -> torch.Tensor:
        obs_dict, _ = self.env.reset()
        if self.randomize_episode_starts:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        return self._split(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.float32)
        extras = dict(extras)
        extras["time_outs"] = truncated
        extras["terminated"] = terminated
        return self._split(obs_dict), rewards, dones, extras

    def get_critic_observations(self) -> torch.Tensor:
        """最近一次 step/reset 的 critic 观测。

        单独开一个方法而不是塞进 ``step`` 的返回值，是为了让**没有特权观测
        的环境**（比如玩具环境）不必实现它 —— runner 里用
        ``getattr(env, "get_critic_observations", None)`` 做能力探测。
        """
        return self._critic_obs

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------ 内部

    def _split(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        policy = obs_dict["policy"]
        self._critic_obs = obs_dict.get("critic", policy)
        return policy
