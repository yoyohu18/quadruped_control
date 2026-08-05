# quadruped_control

在宇树 Go2 上从第一性原理搭建的一整套四足运动控制栈 —— 运动学、动力学、状态估计、步态与落脚点规划、凸 MPC、全身控制（WBC），以及 Isaac Lab 中的强化学习。

每个模块都能独立运行、独立测试，并配有完整推导文档。

---

## 进度

| # | 里程碑 | 模块 | 状态 |
|---|---|---|---|
| 1 | 单腿运动学 —— FK / IK / 雅可比 | `kinematics/` | ✅ **已完成** —— [文档](docs/01_kinematics.md) —— 24 项测试 |
| 2 | 刚体动力学 —— RNEA、CRBA、质心动量、SRBD | `dynamics/` | ✅ **已完成** —— [文档](docs/02_dynamics.md) —— 45 项测试 |
| 3 | 状态估计 —— 腿部里程计 + IMU 融合（ESKF） | `state_estimator/` | ✅ **已完成** —— [文档](docs/03_state_estimation.md) —— 32 项测试 |
| 4 | 步态调度器 —— 相位、占空比、接触时序 | `gait_scheduler/` | ✅ **已完成** —— [文档](docs/04_gait_scheduler.md) —— 34 项测试 |
| 5 | 摆动腿规划器 —— 足端轨迹与落地冲击 | `swing_planner/` | ✅ **已完成** —— [文档](docs/05_swing_planner.md) —— 32 项测试 |
| 6 | 落脚点规划器 —— LIPM、捕获点、Raibert | `footstep_planner/` | ✅ **已完成** —— [文档](docs/06_footstep_planner.md) —— 44 项测试 |
| 7 | 凸 MPC —— 单刚体模型 + 摩擦锥 QP | `mpc/` | ✅ **已完成** —— [文档](docs/07_convex_mpc.md) —— 26 项测试 |
| 8 | 全身控制 —— 完整动力学加权 QP | `whole_body_controller/` | ✅ **已完成** —— [文档](docs/08_whole_body_control.md) —— 28 项测试 |
| 9 | 强化学习运动 —— Isaac Lab 中的 PPO | `rl/`, `isaac/` | ✅ **已完成** —— [文档](docs/09_rl_ppo.md) —— 58 项测试 |
| 10 | 残差强化学习、地形自适应、抗推恢复 | `rl/` | 计划中 |

---

## 快速开始

```bash
conda activate go2_isaac_ros2

# 运行目前已完成部分的全部测试
pytest tests/ -v

# 重新生成各里程碑的插图
python scripts/viz_kinematics.py
python scripts/viz_dynamics.py
python scripts/viz_state_estimation.py
python scripts/viz_gait.py
python scripts/viz_swing.py
python scripts/viz_footstep.py
python scripts/viz_mpc.py
python scripts/viz_wbc.py
python scripts/viz_rl.py

# 测一测 1 kHz 控制周期里到底塞得下什么
python scripts/benchmark_kinematics.py

# 在 Isaac Lab 里训练 RL 策略（平地 300 次迭代，RTX 5080 上实测 2 分 45 秒）
python scripts/train_rl.py --task Go2-Velocity-Flat-v0 --num_envs 4096 --headless

# 回放并给出量化指标
python scripts/play_rl.py --task Go2-Velocity-Flat-Play-v0 --checkpoint logs/go2_flat/<时间戳>/model_300.pt
```

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.11（conda 环境 `go2_isaac_ros2`） |
| Pinocchio | 2.7.0 |
| Isaac Sim / Isaac Lab | 5.1.0 / 0.54.3 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| rsl-rl | 5.0.1 |
| QP 求解器 | daqp、quadprog（经 qpsolvers 调用；条件化后是稠密 QP） |
| GPU | RTX 5080, 16 GB |

`casadi` 与 `acados` 装在 base 环境里，从里程碑 7 开始会用到。

---

## 目录结构

```
quadruped_control/
├── robot_description/go2/   Go2 URDF + 网格模型（自包含）
├── kinematics/              ✅ FK、IK、雅可比（闭式解 + Pinocchio）
├── dynamics/                ✅ RNEA、CRBA、质心动量、单刚体模型 SRBD
├── state_estimator/         ✅ 支撑腿里程计、ESKF 融合、真值数据生成
├── gait_scheduler/          ✅ 步态库、相位调度、支撑多边形与稳定裕度
├── footstep_planner/        ✅ 线性倒立摆、捕获点、Raibert 与精确极限环系数
├── swing_planner/           ✅ 四种足端轨迹、落地冲击分析、IK 进控制回路
├── mpc/                     ✅ 凸 MPC：条件化 QP、摩擦金字塔、实时求解
├── whole_body_controller/   ✅ 完整 18 自由度动力学 QP，1 kHz
├── rl/                      ✅ 从零实现的 PPO、GAE、奖励核、玩具环境（不依赖仿真器）
├── isaac/                   ✅ Isaac Lab 的 Go2 速度跟踪环境（自己写的 MDP 配置）
├── configs/                 机器人与控制器参数
├── tests/                   交叉验证测试
├── scripts/                 可视化与性能基准
└── docs/                    推导文档，每个里程碑一份
```

`configs/` 是尚未开始的里程碑的占位。

**经典技术栈（M1–M8）已完整**：状态估计 → 步态 → 落脚点 → 摆动轨迹 → 凸 MPC → WBC → 关节力矩。

**两条路线都通了。** M9 换成无模型强化学习：同一个 Go2，不建模，直接从数据里学。
M1–M8 在这里以三种身份继续参与 —— 性能基线、奖励设计的依据（M4 的相位、M5 的摆动
高度、M6 的捕获点、M7 的摩擦锥都变成了奖励项）、以及 M10 残差 RL 的基础控制器。

---

## 设计原则

**每样东西都写两遍，互相交叉验证。** 每个算法都实现两次：一次手推闭式解，一次走通用库（Pinocchio）。测试断言二者一致到 `1e-12`。**单一实现是无法被测试的** —— 推导里的错误和你预期里的同一个错误永远会互相印证。

**实时路径不带重依赖。** 那些将来要跑在真机 1 kHz 线程里的模块，只 import NumPy，不 import 别的，这样移植到 C++ 是机械劳动而不是重写。

**每个里程碑都能单独跑起来。** 明确目标、可量化输出、可视化、测试、调试清单 —— 全部齐了才进入下一个。

**约定先写清楚，再用测试钉死。** 坐标系、关节顺序、雅可比参考系是四足绝大多数 bug 的来源。每一项都文档化一次，并配一个一旦漂移就大声失败的测试。里程碑 1 推出来的 `ISAAC_JOINT_ORDER` 在里程碑 9 被运行中的 Isaac Sim 逐项验证 —— 约定确实没漂。

**学习类模块也不许"看着对"。** M9 的 PPO 有一个纯 torch 的玩具环境，它的最优回报可以手算，端到端测试的判据是"达到解析最优的 80% 以上"，而不是"曲线在涨"；GAE 与 rsl-rl 的实现逐元素比对。**整套 RL 测试不需要启动仿真器。**

---

## 参考文献

Featherstone, *Rigid Body Dynamics Algorithms* · Lynch & Park, *Modern Robotics* · Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 via Convex MPC* (IROS 2018) · Carpentier et al., *The Pinocchio C++ library* (SII 2019)

机器人模型来自宇树 [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros)。
