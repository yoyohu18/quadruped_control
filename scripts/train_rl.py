"""在 Isaac Lab 里训练 Go2 速度跟踪策略。

两种算法后端，同一个环境：

* ``--algo ours``   —— 本仓库 :mod:`rl` 里从零实现的 PPO；
* ``--algo rsl_rl`` —— 工业界标准实现，作为交叉验证基准。

这是本里程碑的"两遍实现"：两条独立的代码路径在同一个 MDP 上跑出统计上
一致的学习曲线，才说明我们的 PPO 是对的。

用法::

    conda activate go2_isaac_ros2

    # 平地，先跑这个（RTX 5080 上实测 2 分 45 秒）
    python scripts/train_rl.py --task Go2-Velocity-Flat-v0 --num_envs 4096 --headless

    # 用自己实现的 PPO
    python scripts/train_rl.py --task Go2-Velocity-Flat-v0 --algo ours --headless

    # 崎岖地形
    python scripts/train_rl.py --task Go2-Velocity-Rough-v0 --max_iterations 1500 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------- 命令行
parser = argparse.ArgumentParser(description="训练 Go2 运动策略")
parser.add_argument("--task", type=str, default="Go2-Velocity-Flat-v0", help="任务 id")
parser.add_argument("--algo", type=str, default="rsl_rl", choices=["rsl_rl", "ours"], help="PPO 实现")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数")
parser.add_argument("--max_iterations", type=int, default=None, help="迭代次数")
parser.add_argument("--seed", type=int, default=1, help="随机种子")
parser.add_argument("--log_dir", type=str, default="logs", help="日志根目录")
parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------------------- 启动 App
# 必须在任何 isaaclab / isaac 的 import 之前，否则 pxr 找不到。
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaac  # noqa: F401, E402  —— import 即注册任务
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _agent_cfg(task: str):
    """取出训练配置，并交给 Isaac Lab 的版本适配器过一遍。

    ``RslRlMLPModelCfg`` 里还留着几个为 rsl-rl < 5.0 保留的字段
    （``stochastic``、``init_noise_std`` 等）。装的是 5.x 时这些字段必须被
    剥掉，否则会以 ``MLPModel.__init__() got an unexpected keyword argument``
    的形式炸在很深的地方。``handle_deprecated_rsl_rl_cfg`` 就是干这个的，
    Isaac Lab 官方的 train.py 也调它。
    """
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    cfg = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]()
    return handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = _agent_cfg(args.task)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, agent_cfg.experiment_name, f"{stamp}_{args.algo}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[训练] 任务 {args.task} | 算法 {args.algo} | 环境数 {env_cfg.scene.num_envs} | 日志 {log_dir}")

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    if args.algo == "rsl_rl":
        _train_rsl_rl(env, agent_cfg, log_dir)
    else:
        _train_ours(env, agent_cfg, log_dir)

    env.close()


def _train_rsl_rl(env, agent_cfg, log_dir: str) -> None:
    """工业标准实现。Isaac Lab 提供了现成的 wrapper。"""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


def _train_ours(env, agent_cfg, log_dir: str) -> None:
    """本仓库的实现。同样的超参数，独立的代码路径。"""
    from rl import ActorCritic, PPOConfig
    from rl.runner import OnPolicyRunner, RunnerConfig
    from rl.vec_env import IsaacLabVecEnv

    vec_env = IsaacLabVecEnv(env.unwrapped, clip_actions=agent_cfg.clip_actions)
    algo_cfg = agent_cfg.algorithm
    distribution = agent_cfg.actor.distribution_cfg

    actor_critic = ActorCritic(
        num_actor_obs=vec_env.num_obs,
        num_actions=vec_env.num_actions,
        num_critic_obs=vec_env.num_critic_obs,
        actor_hidden_dims=tuple(agent_cfg.actor.hidden_dims),
        critic_hidden_dims=tuple(agent_cfg.critic.hidden_dims),
        activation=agent_cfg.actor.activation,
        init_noise_std=distribution.init_std,
        noise_std_type=distribution.std_type,
    )

    ppo_cfg = PPOConfig(
        clip_param=algo_cfg.clip_param,
        value_loss_coef=algo_cfg.value_loss_coef,
        entropy_coef=algo_cfg.entropy_coef,
        num_learning_epochs=algo_cfg.num_learning_epochs,
        num_mini_batches=algo_cfg.num_mini_batches,
        learning_rate=algo_cfg.learning_rate,
        schedule=algo_cfg.schedule,
        gamma=algo_cfg.gamma,
        lam=algo_cfg.lam,
        desired_kl=algo_cfg.desired_kl,
        max_grad_norm=algo_cfg.max_grad_norm,
        use_clipped_value_loss=algo_cfg.use_clipped_value_loss,
    )

    runner = OnPolicyRunner(
        vec_env,
        actor_critic,
        ppo_cfg,
        RunnerConfig(
            num_steps_per_env=agent_cfg.num_steps_per_env,
            max_iterations=agent_cfg.max_iterations,
            save_interval=agent_cfg.save_interval,
            log_interval=10,
            normalize_observations=agent_cfg.actor.obs_normalization,
            experiment_name=agent_cfg.experiment_name,
            log_dir=log_dir,
            seed=args.seed,
        ),
        device=vec_env.device,
    )
    if args.resume:
        runner.load(args.resume)

    history = runner.learn(agent_cfg.max_iterations)
    runner.save(os.path.join(log_dir, "model_final.pt"))

    # 学习曲线落盘，供 scripts/viz_rl.py 画图与两种实现对比
    torch.save(history, os.path.join(log_dir, "history.pt"))
    print(f"[训练] 完成，最终平均回报 {history[-1]['mean_episode_reward']:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
