"""抗推恢复评估：给机器人一脚，看它还站不站得住。

这是里程碑 10 三项内容（残差、地形、抗推）里最容易被做成"看录像"的一项。
这里把它做成**可量化、可复现、三种控制器可比**的实验：

* ``--policy nominal``  —— 纯名义步态控制器（残差恒为 0）。M1/4/5/6 的解析
  控制器单独能扛多大的推力？这是基线。
* ``--policy checkpoint --task Go2-Velocity-*``  —— 里程碑 9 的纯 RL 策略。
* ``--policy checkpoint --task Go2-Residual-*``  —— 里程碑 10 的残差策略。

实验协议（三者完全一致，否则不可比）：

1. 重置，让机器人按指令走 ``--settle`` 秒进入稳态；
2. 在**同一相位**施加一次速度突变（沿随机方向，幅值扫描）；
3. 之后 ``--recover`` 秒内只要躯干触地就算失败；
4. 每个幅值重复 ``--num_envs`` 次，报存活率。

**在同一相位施加**这一条很重要：trot 在支撑腿对角线方向上的抗扰能力比
垂直方向强得多，不控相位的话方差会把结论淹掉。

用法::

    python scripts/eval_push.py --task Go2-Residual-Flat-Play-v0 --policy nominal --headless
    python scripts/eval_push.py --task Go2-Residual-Flat-Play-v0 \\
        --checkpoint logs/go2_residual_flat/<时间戳>/model_299.pt --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="抗推恢复评估")
parser.add_argument("--task", type=str, default="Go2-Residual-Flat-Play-v0")
parser.add_argument("--policy", type=str, default="checkpoint", choices=["checkpoint", "nominal"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algo", type=str, default="rsl_rl", choices=["rsl_rl", "ours"])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=float, default=2.0, help="施推前的稳态时间，秒")
parser.add_argument("--recover", type=float, default=3.0, help="施推后的观察时间，秒")
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
parser.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="推的时候在走的速度指令")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaac  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _agent_cfg(task: str):
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    cfg = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]()
    return handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # 评估里不要任何自发的推力事件 —— 推力全部由本脚本显式控制
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    agent_cfg = _agent_cfg(args.task)
    env = gym.make(args.task, cfg=env_cfg)

    policy = _load_policy(env, agent_cfg)
    results = _sweep(env, policy)

    print("\n" + "=" * 58)
    label = "名义控制器（残差=0）" if args.policy == "nominal" else Path(args.checkpoint).name
    print(f"抗推恢复 —— {args.task}")
    print(f"策略：{label} | 每档 {args.num_envs} 次 | 指令 {args.command}")
    print("=" * 58)
    print(f"  {'推力 [m/s]':>12}  {'存活率':>8}   {'恢复后速度误差 [m/s]':>20}")
    for speed, survival, error in results:
        bar = "█" * int(round(survival * 20))
        print(f"  {speed:>12.1f}  {survival:>7.1%}   {error:>19.3f}  {bar}")
    print("=" * 58)
    print(f"  50% 存活对应的推力（越大越抗推）：{_threshold(results):.2f} m/s")
    print("=" * 58)

    env.close()


def _load_policy(env, agent_cfg):
    """返回 ``obs_dict -> action``。``nominal`` 时恒为零残差。"""
    if args.policy == "nominal":
        num_actions = env.unwrapped.action_manager.total_action_dim
        zeros = torch.zeros(env.unwrapped.num_envs, num_actions, device=env.unwrapped.device)
        return lambda _: zeros

    if args.checkpoint is None:
        raise SystemExit("--policy checkpoint 需要同时给出 --checkpoint")

    if args.algo == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        return runner.get_inference_policy(device=env.unwrapped.device)

    from rl import ActorCritic
    from rl.runner import OnPolicyRunner as OurRunner
    from rl.runner import RunnerConfig
    from rl.vec_env import IsaacLabVecEnv

    vec_env = IsaacLabVecEnv(env.unwrapped, clip_actions=agent_cfg.clip_actions)
    actor_critic = ActorCritic(
        vec_env.num_obs,
        vec_env.num_actions,
        vec_env.num_critic_obs,
        actor_hidden_dims=tuple(agent_cfg.actor.hidden_dims),
        critic_hidden_dims=tuple(agent_cfg.critic.hidden_dims),
        activation=agent_cfg.actor.activation,
    )
    runner = OurRunner(vec_env, actor_critic, runner_config=RunnerConfig(), device=vec_env.device)
    runner.load(args.checkpoint)
    inner = runner.get_inference_policy()
    return lambda obs_dict: inner(obs_dict["policy"])


@torch.inference_mode()
def _sweep(env, policy) -> list[tuple[float, float, float]]:
    """幅值扫描，返回 ``[(推力, 存活率, 恢复后速度误差)]``。"""
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    command = torch.tensor(args.command, device=device)

    settle_steps = int(args.settle / unwrapped.step_dt)
    recover_steps = int(args.recover / unwrapped.step_dt)

    results = []
    for speed in args.speeds:
        obs_dict, _ = env.reset()
        for _ in range(settle_steps):
            command_term.vel_command_b[:] = command
            obs_dict, *_ = env.step(policy(obs_dict))

        # 沿随机水平方向施加一次速度突变。方向随机、幅值固定 ——
        # 只测"能扛多大"，不测"哪个方向弱"（那需要单独的极坐标扫描）。
        angle = torch.rand(unwrapped.num_envs, device=device) * 2.0 * math.pi
        velocity = robot.data.root_vel_w.clone()
        velocity[:, 0] += speed * torch.cos(angle)
        velocity[:, 1] += speed * torch.sin(angle)
        robot.write_root_velocity_to_sim(velocity)

        fell = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=device)
        for _ in range(recover_steps):
            command_term.vel_command_b[:] = command
            obs_dict, _, terminated, _, _ = env.step(policy(obs_dict))
            fell |= terminated  # 一旦摔过就算失败，即使之后被重置

        survival = float((~fell).float().mean())
        # 只统计活下来的那些环境的稳态跟踪误差
        error = robot.data.root_lin_vel_b[:, :2] - command[None, :2]
        error = float(error.norm(dim=-1)[~fell].mean()) if (~fell).any() else float("nan")
        results.append((speed, survival, error))

    return results


def _threshold(results: list[tuple[float, float, float]]) -> float:
    """存活率跌破 50% 的推力，线性插值。"""
    for (s0, v0, _), (s1, v1, _) in zip(results, results[1:]):
        if v0 >= 0.5 > v1:
            return s0 + (s1 - s0) * (v0 - 0.5) / max(v0 - v1, 1e-9)
    return results[-1][0] if results[-1][1] >= 0.5 else results[0][0]


if __name__ == "__main__":
    main()
    simulation_app.close()
