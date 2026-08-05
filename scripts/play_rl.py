"""回放训练好的策略，并给出可量化的评估指标。

**"看着能走"不是验收标准。** 这个脚本输出的是数字：

* 线速度 / 角速度跟踪误差（RMS）；
* 存活率（未终止的比例）；
* 步态对称性 —— 对角腿接触相关系数，trot 应接近 +1；
* 能耗 —— 运输成本 COT :math:`= P/(mgv)`，与 M8 的力矩输出直接可比；
* 动作抖动 —— 动作变化率的 RMS，这是 sim-to-real 的头号预测指标。

最后三项就是"RL 学出来的东西到底比 MPC+WBC 好在哪、差在哪"的答案所在。

用法::

    python scripts/play_rl.py --task Go2-Velocity-Flat-Play-v0 \\
        --checkpoint logs/go2_flat/<时间戳>/model_300.pt

不加 ``--headless`` 会开图形界面，能直接看到机器人走。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="回放并评估 Go2 策略")
parser.add_argument("--task", type=str, default="Go2-Velocity-Flat-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
parser.add_argument("--algo", type=str, default="rsl_rl", choices=["rsl_rl", "ours"])
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1000, help="评估步数（50 Hz，1000 步 = 20 秒）")
parser.add_argument("--export_onnx", action="store_true", help="导出 ONNX，供真机部署")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaac  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _agent_cfg(task: str):
    """取出训练配置并做版本适配，与 ``train_rl.py`` 保持一致。"""
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    cfg = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]()
    return handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = _agent_cfg(args.task)
    env = gym.make(args.task, cfg=env_cfg)

    policy = _load_policy(env, agent_cfg)
    metrics = _evaluate(env, policy)

    print("\n" + "=" * 62)
    print(f"策略评估 —— {args.task} | {args.steps} 步 × {args.num_envs} 环境")
    print("=" * 62)
    for name, value, unit in metrics:
        print(f"  {name:<28} {value:>10.4f}  {unit}")
    print("=" * 62)

    if args.export_onnx:
        _export_onnx(policy, env)

    env.close()


def _load_policy(env, agent_cfg):
    """两种后端统一成同一个可调用对象：**输入是观测字典，输出是动作**。

    两边的入参约定并不一样，这里抹平：

    * rsl-rl ≥ 5.0 的推理策略直接吃整个观测字典（它内部按
      ``obs_groups`` 自己挑用哪几组）；
    * 本仓库的策略吃 ``policy`` 那一组的张量。

    统一在这一层，评估代码里就只剩一种调用方式。
    """
    if args.algo == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        return runner.get_inference_policy(device=env.unwrapped.device)

    from rl import ActorCritic
    from rl.runner import OnPolicyRunner, RunnerConfig
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
    runner = OnPolicyRunner(vec_env, actor_critic, runner_config=RunnerConfig(), device=vec_env.device)
    runner.load(args.checkpoint)
    inner = runner.get_inference_policy()
    return lambda obs_dict: inner(obs_dict["policy"])


@torch.inference_mode()
def _evaluate(env, policy) -> list[tuple[str, float, str]]:
    """跑 ``--steps`` 步，累计各项指标。"""
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    foot_ids, _ = contact_sensor.find_bodies(".*_foot")
    mass = float(robot.data.default_mass.sum(dim=1)[0])

    obs_dict, _ = env.reset()

    lin_err_sq = ang_err_sq = 0.0
    power_sum = speed_sum = 0.0
    action_rate_sq = 0.0
    terminations = 0
    contact_log = []
    last_action = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=device)

    for _ in range(args.steps):
        action = policy(obs_dict)
        obs_dict, _, terminated, truncated, _ = env.step(action)

        command = unwrapped.command_manager.get_command("base_velocity")
        lin_err_sq += float(torch.mean(torch.sum((command[:, :2] - robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1)))
        ang_err_sq += float(torch.mean((command[:, 2] - robot.data.root_ang_vel_b[:, 2]) ** 2))

        power_sum += float(torch.mean(torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)))
        speed_sum += float(torch.mean(torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)))
        action_rate_sq += float(torch.mean(torch.sum((action - last_action) ** 2, dim=1)))
        last_action = action.clone()

        terminations += int(terminated.sum())
        forces = contact_sensor.data.net_forces_w_history[:, :, foot_ids, :]
        contact_log.append((forces.norm(dim=-1).max(dim=1)[0] > 1.0).float().mean(dim=0).cpu())

    n = args.steps
    contacts = torch.stack(contact_log)  # (steps, 4) 各腿的平均接触率时间序列
    # 对角腿相关：FL(0) 与 RR(3) 应同相，FR(1) 与 RL(2) 应同相
    diag_corr = 0.5 * (_corr(contacts[:, 0], contacts[:, 3]) + _corr(contacts[:, 1], contacts[:, 2]))
    cot = (power_sum / n) / (mass * 9.81 * max(speed_sum / n, 1e-3))

    return [
        ("线速度跟踪 RMS", (lin_err_sq / n) ** 0.5, "m/s"),
        ("角速度跟踪 RMS", (ang_err_sq / n) ** 0.5, "rad/s"),
        ("终止次数", float(terminations), f"/ {args.num_envs * n} 步"),
        ("对角腿接触相关", diag_corr, "（trot 应接近 +1）"),
        ("平均机械功率", power_sum / n, "W"),
        ("平均速度", speed_sum / n, "m/s"),
        ("运输成本 COT", cot, "无量纲"),
        ("动作变化率 RMS", (action_rate_sq / n) ** 0.5, "（越小越好上真机）"),
    ]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if denom > 1e-9 else 0.0


def _export_onnx(policy, env) -> None:
    """导出 ONNX。真机部署跑的是它，不是 PyTorch。

    Go2 的板载算力有限，ONNX Runtime / TensorRT 上一次前向约几十微秒，
    远小于 20 ms 的控制周期 —— **推理从来不是四足 RL 的瓶颈**。
    """
    path = os.path.join(os.path.dirname(args.checkpoint), "policy.onnx")
    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    dummy = torch.zeros(1, num_obs, device=env.unwrapped.device)
    model = policy.__self__ if hasattr(policy, "__self__") else policy
    torch.onnx.export(model, dummy, path, input_names=["observations"], output_names=["actions"], opset_version=17)
    print(f"[导出] ONNX → {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
