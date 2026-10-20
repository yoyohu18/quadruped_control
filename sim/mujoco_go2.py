"""MuJoCo 里的 Go2：把宇树官方 MJCF 包成一个"说 Pinocchio 约定"的仿真器。

前八个里程碑的所有代码都用 Pinocchio 约定。MuJoCo 在三处跟它不一样，
每一处搞错都是"看上去几乎对、然后摔倒"那一类 bug：

=================  ==================================  ==================================
量                 MuJoCo                              Pinocchio（本仓库）
=================  ==================================  ==================================
四元数             ``qpos[3:7] = [w, x, y, z]``        ``q[3:7] = [x, y, z, w]``
基座线速度         **世界系**                           **机体系**
基座角速度         机体系                               机体系
关节量顺序         MJCF 声明顺序 FL/FR/RL/RR            URDF 顺序 FL/FR/RL/RR（相同）
执行器顺序         SDK 顺序 **FR/FL/RR/RL**             —（我们直接给 12 维力矩）
=================  ==================================  ==================================

所以边界上只做两件事：状态读出时转成 Pinocchio 的 ``(q, v)``，力矩写入时
**按关节名**映射到执行器 —— 绝不按位置对。

模型文件不拷进仓库（网格 28 MB），默认指向 ``unitree_mujoco``，可用环境变量
``GO2_MJCF`` 覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from kinematics.robot_model import LEGS, PINOCCHIO_JOINT_ORDER

__all__ = [
    "GO2_MJCF",
    "MujocoGo2",
    "SimConfig",
    "SimState",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
]

GO2_MJCF = Path(
    os.environ.get(
        "GO2_MJCF",
        "/home/clear/workspace/rluni_hjh/unitree_mujoco/unitree_robots/go2/go2.xml",
    )
)

# 平地场景。官方 scene.xml 在 x = 1.2 m 处摆了台阶，不适合做平地基准。
_FLOOR = """
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.5 0.55 0.6 1"
          friction="{mu} 0.02 0.01"/>
  </worldbody>
"""


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


@dataclass
class SimConfig:
    """仿真参数。

    Attributes:
        dt: 物理步长。1 kHz 与 WBC 同频。
        floor_friction: 地面摩擦系数。MuJoCo 取两个 geom 摩擦的**最大值**，
            足端是 0.4，所以这里决定了实际 μ（≥ 0.4）。
        joint_friction: 是否保留 MJCF 里的关节阻尼 / 库仑摩擦。
            关掉后与 Pinocchio 模型只差 armature，适合做对齐检验。
        inertia: ``"mjcf"`` 用宇树官方 MJCF 的质量惯量（15.21 kg）；
            ``"urdf"`` 改写成本仓库 URDF 的值（16.09 kg），与控制器模型
            逐连杆一致。前者是真实的模型失配，后者用来先把控制器调通。
    """

    dt: float = 0.001
    floor_friction: float = 0.6
    joint_friction: bool = True
    inertia: str = "mjcf"


@dataclass
class SimState:
    """一帧仿真状态，已转为 Pinocchio 约定。"""

    t: float
    q: np.ndarray  # (19,) [p, quat_xyzw, 12 关节]
    v: np.ndarray  # (18,) [v_body, ω_body, 12 关节]
    contact_force: dict[str, np.ndarray]  # 每只脚的地面反力，世界系，(3,)


class MujocoGo2:
    """Go2 的 MuJoCo 仿真，接口全部是 Pinocchio 约定。"""

    def __init__(self, cfg: SimConfig | None = None, mjcf: str | Path = GO2_MJCF) -> None:
        self.cfg = cfg or SimConfig()
        mjcf = Path(mjcf)
        if not mjcf.is_file():
            raise FileNotFoundError(f"找不到 Go2 的 MJCF：{mjcf}（可用环境变量 GO2_MJCF 指定）")

        xml = mjcf.read_text()
        xml = xml.replace('meshdir="assets"', f'meshdir="{mjcf.parent / "assets"}"')
        xml = xml.replace("</mujoco>", _FLOOR.format(mu=self.cfg.floor_friction) + "</mujoco>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.cfg.dt
        if not self.cfg.joint_friction:
            self.model.dof_damping[6:] = 0.0
            self.model.dof_frictionloss[6:] = 0.0
        self.data = mujoco.MjData(self.model)
        if self.cfg.inertia == "urdf":
            self._load_urdf_inertia()
        elif self.cfg.inertia != "mjcf":
            raise ValueError(f"未知的 inertia 选项：{self.cfg.inertia}")

        # 关节 -> qpos / qvel 下标，按名字查，不假设声明顺序。
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PINOCCHIO_JOINT_ORDER]
        self._qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self._vadr = np.array([self.model.jnt_dofadr[j] for j in jid])
        # 执行器 -> 关节：ctrl[k] 驱动的是哪个 Pinocchio 关节。
        act_joint = [self.model.actuator_trnid[k, 0] for k in range(self.model.nu)]
        self._act_of_joint = np.array([act_joint.index(j) for j in jid])
        self.torque_limit = self.model.actuator_ctrlrange[self._act_of_joint, 1].copy()

        self._foot_geom = {leg: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS}
        self._floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_radius = float(self.model.geom_size[self._foot_geom["FL"], 0])

    def _load_urdf_inertia(self) -> None:
        """把每个刚体的质量、质心、惯量改写成 Pinocchio（URDF）的值。

        两边的连杆坐标系是重合的（足端位置对齐到 1e-16），所以 Pinocchio
        每个关节上挂的合成惯量可以直接搬进同名 MuJoCo 刚体。MuJoCo 要求
        惯量写成主轴形式，所以先对角化，再把主轴旋转存进 ``body_iquat``。
        """
        from kinematics.go2 import load_go2

        pm = load_go2(floating_base=True).model
        pairs = [("base_link", 1)] + [
            (n.replace("_joint", ""), pm.getJointId(n)) for n in PINOCCHIO_JOINT_ORDER
        ]
        for body, jid in pairs:
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            I = pm.inertias[jid]
            # 降序排列主轴（最小惯量放最后），与宇树 MJCF 的写法一致。数学上
            # 与升序等价，但实测升序时 PD 站立会落入约 100 Hz 的膝关节振荡、
            # 整机以 0.5 m/s 倒退 —— MuJoCo 对主轴顺序数值敏感，见文档第 3 节。
            eigval, eigvec = np.linalg.eigh(I.inertia)
            eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
            if np.linalg.det(eigvec) < 0:
                eigvec[:, 0] *= -1
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, eigvec.flatten())
            self.model.body_mass[b] = I.mass
            self.model.body_ipos[b] = I.lever
            self.model.body_inertia[b] = eigval
            self.model.body_iquat[b] = quat
        mujoco.mj_setConst(self.model, self.data)

    # -- 状态 -----------------------------------------------------------------

    def reset(self, q: np.ndarray, v: np.ndarray | None = None) -> SimState:
        """用 Pinocchio 约定的 ``(q, v)`` 设定初始状态。"""
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(q, np.zeros(18) if v is None else v)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        d = self.data
        d.qpos[0:3] = q[0:3]
        d.qpos[3:7] = quat_xyzw_to_wxyz(q[3:7])
        d.qpos[self._qadr] = q[7:]
        R = self.base_rotation()
        d.qvel[0:3] = R @ v[0:3]  # 机体系 -> 世界系
        d.qvel[3:6] = v[3:6]
        d.qvel[self._vadr] = v[6:]

    def base_rotation(self) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    def pin_q(self) -> np.ndarray:
        d = self.data
        return np.concatenate([d.qpos[0:3], quat_wxyz_to_xyzw(d.qpos[3:7]), d.qpos[self._qadr]])

    def pin_v(self) -> np.ndarray:
        d = self.data
        R = self.base_rotation()
        return np.concatenate([R.T @ d.qvel[0:3], d.qvel[3:6], d.qvel[self._vadr]])

    def contact_forces(self) -> dict[str, np.ndarray]:
        """每只脚受到的地面反力（世界系）—— 足端力估计器的**真值**。"""
        out = {leg: np.zeros(3) for leg in LEGS}
        leg_of_geom = {g: leg for leg, g in self._foot_geom.items()}
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._floor not in pair:
                continue
            other = (pair - {self._floor}).pop()
            if other not in leg_of_geom:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            frame = c.frame.reshape(3, 3)  # 行向量是接触系三个轴（世界系）
            f_world = frame.T @ f6[:3]
            # mj_contactForce 给的是 geom1 对 geom2 的作用力；要的是地面推脚。
            if c.geom1 == self._floor:
                out[leg_of_geom[other]] += f_world
            else:
                out[leg_of_geom[other]] -= f_world
        return out

    def state(self) -> SimState:
        return SimState(self.data.time, self.pin_q(), self.pin_v(), self.contact_forces())

    # -- 推进 -----------------------------------------------------------------

    def step(self, tau: np.ndarray, n_substeps: int = 1) -> SimState:
        """施加 12 维关节力矩（Pinocchio 顺序）并推进。力矩按执行器上限饱和。"""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limit, self.torque_limit)
        self.data.ctrl[self._act_of_joint] = tau
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def apply_push(self, force_world: np.ndarray) -> None:
        """对躯干施加外力（世界系，作用于质心）。持续到下一次调用；传零即撤掉。"""
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.data.xfrc_applied[body, :3] = force_world

    # -- 与 Pinocchio 对齐用 ---------------------------------------------------

    def mass_matrix(self) -> np.ndarray:
        """MuJoCo 的广义质量矩阵（MuJoCo 速度约定，含 armature）。"""
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def total_mass(self) -> float:
        return float(self.model.body_subtreemass[1])
