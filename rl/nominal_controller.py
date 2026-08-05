"""名义步态控制器：把里程碑 1/4/5/6 整体搬上 GPU。

## 为什么必须重写一遍

残差强化学习要算的是

.. math::  a = a_{\\text{nom}}(s) + \\alpha\\,a_{\\text{res}}(s)

其中 :math:`a_{\\text{nom}}` **每一个仿真步、每一个环境都要算一次**。
4096 个环境 × 50 Hz = 每秒 20 万次调用。里程碑 1/4/5/6 的实现是纯 NumPy、
逐样本、带 python 循环的，单次几十微秒 —— 直接用的话名义控制器一家就要吃掉
每秒 10 秒的 CPU 时间，比整个仿真还慢两个数量级。

所以这里把它们重写成**批量 torch**：同样的公式，同样的约定，一次算 4096 个。
测试 `tests/test_residual.py` 断言两条路径在相同输入下逐元素一致 ——
这是本仓库第十次使用"每样东西写两遍"这条规矩，也是最有价值的一次：
**它同时证明了新实现是对的、旧实现没有被悄悄改坏。**

## 为什么不是 MPC + WBC

里程碑 7 的凸 MPC 和里程碑 8 的 WBC 才是"完整"的名义控制器，但它们**进不来**：
两者都要解 QP，而 QP 求解器是串行的、CPU 上的、迭代次数依数据而变的。
没有任何办法把它批量化到 GPU 上跑 4096 份。

能进来的，恰好是闭式的那一半：

| 里程碑 | 用到的部分 | 为什么能批量化 |
|---|---|---|
| M4 步态调度器 | 相位、占空比、摆动进度 | 纯取模运算 |
| M6 落脚点规划 | Raibert 启发式 + 转向项 | 闭式代数 |
| M5 摆动轨迹 | 正弦抬腿 + smoothstep 水平插值 | 闭式，且解析可导 |
| M1 单腿逆运动学 | 闭式 IK（余弦定理） | 只有 atan2/acos/sqrt |

**这条分界线本身就是本里程碑最重要的知识点**：优化型控制器和解析型控制器
在大规模并行仿真里的地位完全不同。工业界的解法是把 MPC+WBC **蒸馏**成一个
网络（拿它在 CPU 上离线跑出的数据做监督学习），再把网络当名义策略 ——
见 `docs/10_residual_rl.md` 第 3 节。

## 名义控制器在做什么

一个**无状态**的（stateless）位置型步态生成器：

1. 由时间和步态定义算出每条腿的相位（M4）；
2. 支撑腿：足端在躯干系里以 :math:`-v_{\\text{cmd}}` 匀速后移 —— 等价于
   躯干以 :math:`+v_{\\text{cmd}}` 前进；
3. 摆动腿：从对称的离地点插值到 Raibert 落脚点（M6），竖直方向叠加正弦
   抬腿（M5）；
4. 足端位置从躯干系转到髋系，闭式 IK 解出关节角（M1）。

**无状态**是刻意的：不维护"上一次离地点在哪"这类内部状态，全部由当前时刻
解析地给出。代价是离地点用了对称步幅假设（不完全等于真实离地位置），
收益是可以被向量化、可以被随机重置、可以被并行 4096 份 —— 而残差策略
本来就是用来补偿这类近似的。**这正是残差 RL 的分工：名义控制器负责
"大致对"，策略负责"补上差的那部分"。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["NominalGaitConfig", "NominalGaitController", "batched_inverse_kinematics"]

#: 腿序，与 :data:`kinematics.LEGS`、:data:`rl.reward_kernels.LEG_ORDER` 一致。
LEG_ORDER = ("FL", "FR", "RL", "RR")


def batched_inverse_kinematics(
    p: torch.Tensor,
    l0: torch.Tensor,
    l1: float,
    l2: float,
    knee_backward: bool = True,
) -> torch.Tensor:
    """闭式逆运动学的批量 torch 版，与 :func:`kinematics.inverse_kinematics` 等价。

    三步与里程碑 1 完全相同，只是把标量换成张量、把异常换成 clamp
    （控制回路里不能因为一个不可达目标就抛异常）：

    1. :math:`y^2+z^2 = l_0^2 + \\ell^2` 定出面内伸展量 :math:`\\ell`，
       再由 :math:`q_0 = \\operatorname{atan2}(z,y) + \\operatorname{atan2}(\\ell,l_0)`
       得侧摆角；
    2. 矢状面内对 2R 链用余弦定理得膝角 :math:`q_2`；
    3. 把两连杆折叠成等效连杆 :math:`(a,b)`，从足端方向里减去它的相位得 :math:`q_1`。

    Args:
        p: ``(..., 3)`` 髋系下的期望足端位置。
        l0: ``(...,)`` 或可广播，带符号的侧摆偏置（左腿正、右腿负）。
        l1: 大腿长度。
        l2: 小腿长度。
        knee_backward: 解支选择，Go2 限位内只有 ``True`` 可行。

    Returns:
        ``(..., 3)`` 关节角 ``[q_abad, q_hip, q_knee]``。

    Note:
        **永远 clamp，不抛异常。** 目标不可达时静默投影到工作空间边界 ——
        一个过于激进的落脚点不应该把 4096 个环境一起搞崩。里程碑 1 的
        ``clamp=True`` 分支就是为这一刻准备的。
    """
    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    # --- 第一步：侧摆角 ---
    radial_sq = (y * y + z * z - l0 * l0).clamp(min=0.0)
    leg_len = torch.sqrt(radial_sq)
    q0 = torch.atan2(z, y) + torch.atan2(leg_len, l0)

    # --- 第二步：膝关节角 ---
    d_sq = x * x + leg_len * leg_len
    cos_knee = ((d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0)
    q2 = -torch.acos(cos_knee) if knee_backward else torch.acos(cos_knee)

    # --- 第三步：髋俯仰角 ---
    a = l1 + l2 * torch.cos(q2)
    b = l2 * torch.sin(q2)
    q1 = torch.atan2(-x, leg_len) - torch.atan2(b, a)

    return torch.stack([q0, q1, q2], dim=-1)


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
    """三次 smoothstep :math:`3s^2-2s^3`，两端一阶导为零。

    摆动腿的水平位移用它而不是线性插值：线性插值在离地和落地瞬间有非零
    水平速度，落地时会横向刮擦地面。里程碑 5 量化过这件事对落地冲击的影响。
    """
    return s * s * (3.0 - 2.0 * s)


@dataclass
class NominalGaitConfig:
    """名义控制器参数。缺省值取自里程碑 4/5/6 的 Go2 配置。

    Attributes:
        period: 步态周期，秒。trot 缺省 0.4。
        duty_factor: 占空比。trot 缺省 0.5。
        phase_offsets: 四条腿的相位偏移，按 :data:`LEG_ORDER`。
        stand_height: 标称躯干离足端高度，米。
        swing_height: 摆动最高点相对地面的抬起高度，米。
        raibert_gain: Raibert 速度反馈增益 :math:`k`，秒。
        hip_offsets: ``(4, 3)`` 各侧摆轴在躯干系下的位置。
        abad_offset: 侧摆轴到髋俯仰轴沿 y 的偏置（正值，符号按腿自动定）。
        thigh_length / calf_length: 连杆长度。
        max_stride: 落脚点相对标称位置的最大水平位移，米。**必须有** ——
            指令速度很大时 Raibert 会给出腿根本够不到的落脚点。
        lateral_offset_scale: 标称步宽相对"侧摆轴正下方 + 侧摆偏置"的缩放。
            与 :class:`footstep_planner.FootstepPlannerConfig` 同名参数一致：
            小于 1 收窄步宽（更省力、更容易自碰），大于 1 加宽（更稳）。
        height_gain: 比例高度伺服的增益 :math:`k`，取值 [0, 1]。
            足端深度取 :math:`h_{meas} + k(h^* - h_{meas})`。
            0 表示完全跟随实测高度（没有高度回复力），1 表示完全按期望站高
            （摆动腿会提前触地）。**两端都是失效模式**，见
            :meth:`NominalGaitController.foot_targets` 的详细说明。
    """

    period: float = 0.4
    duty_factor: float = 0.5
    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    stand_height: float = 0.30
    swing_height: float = 0.08
    raibert_gain: float = 0.03
    hip_offsets: tuple[tuple[float, float, float], ...] = (
        (0.1934, 0.0465, 0.0),
        (0.1934, -0.0465, 0.0),
        (-0.1934, 0.0465, 0.0),
        (-0.1934, -0.0465, 0.0),
    )
    abad_offset: float = 0.0955
    thigh_length: float = 0.213
    calf_length: float = 0.213
    max_stride: float = 0.20
    lateral_offset_scale: float = 1.0
    height_gain: float = 0.35
    joint_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf") for leg in LEG_ORDER)
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError("占空比必须在 (0, 1) 内 —— 1.0 表示永远站立，没有摆动相")
        if self.period <= 0.0:
            raise ValueError("步态周期必须为正")
        if self.stand_height <= 0.0 or self.swing_height <= 0.0:
            raise ValueError("站立高度与抬腿高度必须为正")
        if not 0.0 <= self.height_gain <= 1.0:
            raise ValueError("高度伺服增益必须在 [0, 1] 内")

    @property
    def stance_duration(self) -> float:
        return self.period * self.duty_factor

    @property
    def swing_duration(self) -> float:
        return self.period * (1.0 - self.duty_factor)


class NominalGaitController:
    """批量的解析步态控制器，输出 Isaac 关节顺序的 12 维关节角。

    Args:
        config: 参数。
        device: 张量设备。

    Example:
        >>> ctrl = NominalGaitController(NominalGaitConfig(), device="cpu")
        >>> q = ctrl.compute(
        ...     time=torch.zeros(8),
        ...     velocity_command=torch.zeros(8, 3),
        ...     base_lin_vel_b=torch.zeros(8, 2),
        ... )
        >>> q.shape
        torch.Size([8, 12])
    """

    def __init__(self, config: NominalGaitConfig | None = None, device: str | torch.device = "cpu") -> None:
        self.cfg = config or NominalGaitConfig()
        self.device = torch.device(device)

        self._offsets = torch.tensor(self.cfg.phase_offsets, device=self.device)
        self._hips = torch.tensor(self.cfg.hip_offsets, device=self.device)
        # l0 的符号把右侧腿镜像过来 —— 与 kinematics.LEG_GEOMETRY 的约定一致
        signs = torch.tensor([1.0 if leg.endswith("L") else -1.0 for leg in LEG_ORDER], device=self.device)
        self._l0 = self.cfg.abad_offset * signs

        # 标称足端的水平位置：**髋偏置 + 侧摆偏置**，与
        # :meth:`footstep_planner.FootstepPlanner.nominal_hip_projection` 同一约定。
        # 只用髋偏置是错的 —— 那会把足端放在侧摆轴正下方，逼出一个非零的
        # 侧摆角，整条腿一直斜着站（实测 ±0.32 rad，膝角也随之偏 0.18 rad）。
        self._nominal_xy = self._hips[:, :2].clone()
        self._nominal_xy[:, 1] += self._l0 * self.cfg.lateral_offset_scale

        # 从 (腿, 关节) 排布重排到 Isaac 的 (关节, 腿) 排布的索引
        pin_names = [f"{leg}_{j}_joint" for leg in LEG_ORDER for j in ("hip", "thigh", "calf")]
        self._to_isaac = torch.tensor(
            [pin_names.index(name) for name in self.cfg.joint_order], device=self.device, dtype=torch.long
        )

    # ------------------------------------------------------------------ 相位

    def phase(self, time: torch.Tensor) -> torch.Tensor:
        """各腿相位，``(N, 4)``，取值 [0, 1)，0 表示刚落地。与 M4 同定义。"""
        return ((time.unsqueeze(-1) / self.cfg.period) - self._offsets) % 1.0

    def contact(self, time: torch.Tensor) -> torch.Tensor:
        """参考接触状态 ``(N, 4)``，与 :func:`rl.reward_kernels.reference_contact` 同定义。"""
        return self.phase(time) < self.cfg.duty_factor

    # ------------------------------------------------------------------ 足端

    def foot_targets(
        self,
        time: torch.Tensor,
        velocity_command: torch.Tensor,
        base_lin_vel_b: torch.Tensor,
        base_height: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """躯干系下的名义足端位置，``(N, 4, 3)``。

        Args:
            time: ``(N,)`` 每个环境自己的 episode 时间，秒。
            velocity_command: ``(N, 3)`` ``[vx, vy, wz]``，躯干系。
            base_lin_vel_b: ``(N, 2)`` 实测水平速度，躯干系。真机上来自
                里程碑 3 的 ESKF —— **名义控制器同样需要状态估计**。
            base_height: ``(N,)`` **实测**的躯干离地高度。``None`` 时退回
                用 ``cfg.stand_height``，但那会带来一个很隐蔽的失效模式，
                见下。

        Returns:
            ``(N, 4, 3)`` 足端位置，躯干系（z 向下为负）。

        Note:
            **竖直基准是一个比例高度伺服，不是常数。** 这是本模块调出来的
            第二个 bug，也是试了三版才对的一个。三版实测（零速度指令）：

            ==================================  ==========  ==========
            足端深度取法                        实际速度    躯干高度
            ==================================  ==========  ==========
            期望站高（常数 0.30）               −0.35 m/s   0.26 m
            实测高度                            +0.01 m/s   **0.11 m**
            支撑用期望、摆动用实测              −0.49 m/s   0.22 m
            实测 + 增益 ×（期望 − 实测）        见下        见下
            ==================================  ==========  ==========

            两端各自的失效机理是对称的：

            * **取常数**：纯位置控制下 Go2 的 PD 刚度扛不住自重，躯干比目标
              下沉约 5 cm。足端目标于是落在真实地面**以下**，摆动腿在轨迹
              还没走完时就撞地 —— 而它那一刻正在向前运动，摩擦力把机身
              往后推，**零指令下匀速倒退 0.35 m/s**。
            * **取实测**：足端目标永远等于足端现在的位置，高度误差恒为零，
              **高度控制权整个消失**。身体下沉 → 目标跟着下沉 → 下沉更多，
              正反馈，躯干塌到 0.11 m。

            所以正确的写法是二者之间的一个**比例控制器**：

            .. math::  d = h_{\\text{meas}} + k\\,(h^* - h_{\\text{meas}})

            :math:`k=0` 退化成"取实测"，:math:`k=1` 退化成"取常数"。
            取中间值就同时得到了落地时机的正确性和有限的高度回复力。
            **这本来就该是一个反馈问题，写成常数才是那个 bug。**

            这也让"名义控制器离不开状态估计"又多了一条：它不只要里程碑 3 估的
            **速度**，还要**高度**。
        """
        cfg = self.cfg
        if base_height is None:
            base_height = torch.full_like(time, cfg.stand_height)
        base_height = base_height.clamp(min=0.05)
        # 比例高度伺服：k=0 取实测（无高度控制权），k=1 取期望（过度下压）
        depth = base_height + cfg.height_gain * (cfg.stand_height - base_height)
        ground_z = -depth.unsqueeze(-1)  # (N, 1)
        phase = self.phase(time)  # (N, 4)
        in_stance = phase < cfg.duty_factor

        v_cmd = velocity_command[:, :2]  # (N, 2)
        w_cmd = velocity_command[:, 2]  # (N,)
        nominal_xy = self._nominal_xy  # (4, 2) 髋偏置 + 侧摆偏置

        # -- 落脚点（M6 Raibert + 转向前馈）----------------------------------
        # 转向项：绕 z 转 w 时，标称点自身有速度 w × p，落脚点要跟着走
        turn = torch.stack([-nominal_xy[:, 1], nominal_xy[:, 0]], dim=-1)  # (4, 2) = ẑ × p
        hip_velocity = w_cmd[:, None, None] * turn[None]  # (N, 4, 2)
        v_cmd_leg = v_cmd[:, None, :] + hip_velocity  # (N, 4, 2) 各腿处的期望地面速度

        v_meas = base_lin_vel_b[:, None, :].expand_as(v_cmd_leg)
        touchdown = (
            nominal_xy[None]
            + 0.5 * cfg.stance_duration * v_meas
            + cfg.raibert_gain * (v_meas - v_cmd_leg)
        )
        # 离地点 = 支撑相扫完之后的位置。**必须由落脚点减去支撑扫程得到，
        # 而不是独立地由标称位置对称构造** —— 否则落地瞬间足端目标会跳变。
        #
        # 这是本模块调出来的真 bug：原先支撑相起点写成
        # ``nominal + 0.5·T·v_cmd``，而摆动终点是 Raibert 落脚点
        # ``nominal + 0.5·T·v_meas + k·(v_meas − v_cmd)``。两者只在
        # ``v_meas == v_cmd``（完美跟踪）时相等，**而那恰恰是永远不成立的
        # 那个假设** —— 起步、转向、被推的瞬间都不成立。实测跳变 0.065 m，
        # 足端在触地那一帧被瞬移，机器人被自己的腿拽着倒退。
        liftoff = touchdown - cfg.stance_duration * v_cmd_leg

        # 工作空间限幅 —— 指令过大时 Raibert 会给出腿够不到的点。
        # 两个端点都限，中间靠插值，于是整条轨迹都在工作空间内。
        touchdown = self._clamp_stride(touchdown, nominal_xy)
        liftoff = self._clamp_stride(liftoff, nominal_xy)

        # -- 摆动相（M5 正弦抬腿 + smoothstep 水平插值）----------------------
        swing_s = ((phase - cfg.duty_factor) / (1.0 - cfg.duty_factor)).clamp(0.0, 1.0)  # (N, 4)
        swing_xy = liftoff + (touchdown - liftoff) * _smoothstep(swing_s).unsqueeze(-1)
        swing_z = ground_z + cfg.swing_height * torch.sin(torch.pi * swing_s)

        # -- 支撑相：从落脚点匀速扫到离地点 ----------------------------------
        # s=0 时正好是 touchdown（与摆动末尾接上），s=1 时正好是 liftoff
        # （与下一次摆动起点接上）。**两个边界都是构造上连续的。**
        stance_s = (phase / cfg.duty_factor).clamp(0.0, 1.0)  # (N, 4)
        stance_xy = touchdown + (liftoff - touchdown) * stance_s.unsqueeze(-1)
        stance_z = ground_z.expand_as(swing_z)

        mask = in_stance.unsqueeze(-1)
        xy = torch.where(mask, stance_xy, swing_xy)
        z = torch.where(in_stance, stance_z, swing_z)
        return torch.cat([xy, z.unsqueeze(-1)], dim=-1)

    def _clamp_stride(self, target: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
        """把落脚点限制在标称位置周围 ``max_stride`` 的圆内。"""
        delta = target - centre[None]
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.cfg.max_stride / norm).clamp(max=1.0)
        return centre[None] + delta * scale

    # ------------------------------------------------------------------ 关节

    def compute(
        self,
        time: torch.Tensor,
        velocity_command: torch.Tensor,
        base_lin_vel_b: torch.Tensor,
        base_height: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """名义关节角，``(N, 12)``，**Isaac 关节顺序**。

        这是残差 RL 的 :math:`a_{\\text{nom}}`。整条链路上没有任何迭代求解、
        没有任何 python 循环，4096 个环境一次算完。
        """
        feet_body = self.foot_targets(time, velocity_command, base_lin_vel_b, base_height)  # (N, 4, 3)
        feet_hip = feet_body - self._hips[None]  # 躯干系 → 髋系（只差一个平移）
        q = batched_inverse_kinematics(feet_hip, self._l0[None, :], self.cfg.thigh_length, self.cfg.calf_length)
        return q.flatten(start_dim=1)[:, self._to_isaac]
