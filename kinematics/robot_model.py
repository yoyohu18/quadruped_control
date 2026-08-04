"""基于 Pinocchio 的整机模型。

``leg_kinematics`` 是快速、手工推导、与具体机器人绑定的那条路径。
本模块是通用路径：直接从 URDF 构建刚体模型，对固定基座或浮动基座、
对任意坐标系，都能给出位姿和雅可比。

后续里程碑（动力学、WBC、MPC）都建立在这个对象之上，所以约定值得
在这里一次性定对。

有两套**坐标系**很容易混：

* **躯干系（base frame）** —— 固连在躯干上。固定基座模型完全活在这个系里。
* **世界系（world frame）** —— 惯性系。只有在模型带浮动基座之后才存在。

还有两套**关节顺序**，混起来更要命：

* **Pinocchio 顺序** —— URDF 声明顺序，即 FL(hip,thigh,calf)、FR(...)、
  RL(...)、RR(...)。
* **Isaac Lab 顺序** —— 对运动树做广度优先，即先四条腿的 hip，
  再四条腿的 thigh，再四条腿的 calf。

搞混的结果是机器人*看上去几乎是对的*，然后摔倒。
请在每个边界上使用 :func:`isaac_to_pinocchio` / :func:`pinocchio_to_isaac`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

__all__ = [
    "LEGS",
    "ISAAC_JOINT_ORDER",
    "PINOCCHIO_JOINT_ORDER",
    "isaac_to_pinocchio",
    "pinocchio_to_isaac",
    "QuadrupedModel",
]

LEGS = ("FL", "FR", "RL", "RR")

PINOCCHIO_JOINT_ORDER = tuple(
    f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")
)
ISAAC_JOINT_ORDER = tuple(
    f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in LEGS
)

_ISAAC_TO_PIN = np.array([ISAAC_JOINT_ORDER.index(n) for n in PINOCCHIO_JOINT_ORDER])
_PIN_TO_ISAAC = np.array([PINOCCHIO_JOINT_ORDER.index(n) for n in ISAAC_JOINT_ORDER])


def isaac_to_pinocchio(q_isaac: np.ndarray) -> np.ndarray:
    """把 12 维关节量从 Isaac Lab 顺序重排为 Pinocchio 顺序。"""
    return np.asarray(q_isaac)[..., _ISAAC_TO_PIN]


def pinocchio_to_isaac(q_pin: np.ndarray) -> np.ndarray:
    """把 12 维关节量从 Pinocchio 顺序重排为 Isaac Lab 顺序。"""
    return np.asarray(q_pin)[..., _PIN_TO_ISAAC]


class QuadrupedModel:
    """四足机器人的刚体模型，由 Pinocchio 从 URDF 载入。

    Args:
        urdf_path: 机器人 URDF 的路径。
        floating_base: 若为 ``True``，在最前面加一个自由飞行关节，
            使躯干可以在世界系中运动。此时 ``nq`` 变为 19（3 位置 +
            4 四元数 + 12 关节），``nv`` 变为 18。
        foot_frames: 每条腿对应的足端坐标系名，默认 ``{leg}_foot``。
        mesh_dir: URDF 中 ``filename=`` 相对解析的根目录，仅可视化需要。
    """

    def __init__(
        self,
        urdf_path: str | Path,
        floating_base: bool = False,
        foot_frames: dict[str, str] | None = None,
        mesh_dir: str | Path | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"找不到 URDF：{self.urdf_path}")
        self.floating_base = floating_base
        self.mesh_dir = Path(mesh_dir) if mesh_dir else self.urdf_path.parent

        if floating_base:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path), pin.JointModelFreeFlyer())
        else:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        self.foot_frames = foot_frames or {leg: f"{leg}_foot" for leg in LEGS}
        self.foot_frame_ids = {}
        for leg, frame in self.foot_frames.items():
            if not self.model.existFrame(frame):
                raise ValueError(f"URDF 中不存在腿 '{leg}' 的坐标系 '{frame}'。")
            self.foot_frame_ids[leg] = self.model.getFrameId(frame)

        # 每个驱动关节在位置/速度向量中的下标。
        self.joint_names = list(PINOCCHIO_JOINT_ORDER)
        self.q_index = {n: self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names}
        self.v_index = {n: self.model.joints[self.model.getJointId(n)].idx_v for n in self.joint_names}
        # 完整雅可比中属于每条腿的列下标，按 (侧摆, 髋俯仰, 膝) 排列。
        self.leg_v_index = {
            leg: np.array([self.v_index[f"{leg}_{j}_joint"] for j in ("hip", "thigh", "calf")])
            for leg in LEGS
        }

    # -- 构型相关的辅助函数 ---------------------------------------------------

    @property
    def nq(self) -> int:
        """位置向量的维数。"""
        return self.model.nq

    @property
    def nv(self) -> int:
        """速度向量的维数（即自由度数）。"""
        return self.model.nv

    def neutral(self) -> np.ndarray:
        """零构型（躯干位姿为单位元，所有关节角为零）。"""
        return pin.neutral(self.model)

    def make_configuration(
        self,
        q_joints: np.ndarray,
        base_position: np.ndarray | None = None,
        base_quaternion_xyzw: np.ndarray | None = None,
    ) -> np.ndarray:
        """由各部分拼装出完整的位置向量。

        Args:
            q_joints: 12 个驱动关节角，按 Pinocchio 顺序。
            base_position: 躯干在世界系中的位置，仅浮动基座可用。
            base_quaternion_xyzw: 躯干姿态四元数 ``[x, y, z, w]``，
                仅浮动基座可用。

        Returns:
            长度为 :attr:`nq` 的位置向量。
        """
        q_joints = np.asarray(q_joints, dtype=float).reshape(12)
        if not self.floating_base:
            if base_position is not None or base_quaternion_xyzw is not None:
                raise ValueError("固定基座模型没有可设置的躯干位姿。")
            return q_joints
        pos = np.zeros(3) if base_position is None else np.asarray(base_position, dtype=float)
        quat = (
            np.array([0.0, 0.0, 0.0, 1.0])
            if base_quaternion_xyzw is None
            else np.asarray(base_quaternion_xyzw, dtype=float)
        )
        return np.concatenate([pos.reshape(3), quat.reshape(4) / np.linalg.norm(quat), q_joints])

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """从完整位置向量中取出 12 个驱动关节角。"""
        return np.asarray(q, dtype=float)[-12:]

    def random_joint_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """在 URDF 位置限位内均匀采样 12 个关节角。"""
        lo = self.model.lowerPositionLimit[-12:]
        hi = self.model.upperPositionLimit[-12:]
        return rng.uniform(lo, hi)

    # -- 运动学 ---------------------------------------------------------------

    def update(self, q: np.ndarray) -> None:
        """按构型 ``q`` 刷新所有坐标系的位姿。"""
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model, self.data)

    def frame_placement(self, q: np.ndarray, frame: str) -> pin.SE3:
        """``frame`` 相对模型根坐标系的位姿（浮动基座时即世界系）。"""
        self.update(q)
        return self.data.oMf[self.model.getFrameId(frame)].copy()

    def foot_position(self, q: np.ndarray, leg: str) -> np.ndarray:
        """某条腿的足端相对模型根坐标系的位置。"""
        self.update(q)
        return self.data.oMf[self.foot_frame_ids[leg]].translation.copy()

    def foot_positions(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """四条腿的足端相对模型根坐标系的位置。"""
        self.update(q)
        return {leg: self.data.oMf[fid].translation.copy() for leg, fid in self.foot_frame_ids.items()}

    def foot_position_in_base(self, q: np.ndarray, leg: str) -> np.ndarray:
        """某条腿的足端位置，表达在躯干系下。

        对固定基座模型，这与 :meth:`foot_position` 完全相同。
        """
        self.update(q)
        p_world = self.data.oMf[self.foot_frame_ids[leg]].translation
        if not self.floating_base:
            return p_world.copy()
        base = self.data.oMi[1]  # 自由飞行关节即躯干
        return base.actInv(p_world)

    def foot_jacobian(self, q: np.ndarray, leg: str, local_frame: bool = False) -> np.ndarray:
        """某条腿足端相对该腿三个关节的 3x3 平动雅可比。

        Args:
            q: 位置向量。
            leg: ``"FL"``、``"FR"``、``"RL"``、``"RR"`` 之一。
            local_frame: 若为 ``True``，速度表达在足端自身坐标系
                （``LOCAL``）；否则表达在与根坐标系平行的轴上
                （``LOCAL_WORLD_ALIGNED``），后者才是控制器想要的。

        Returns:
            形状 (3, 3) 的雅可比。
        """
        q = np.asarray(q, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        reference = pin.ReferenceFrame.LOCAL if local_frame else pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        J = pin.getFrameJacobian(self.model, self.data, self.foot_frame_ids[leg], reference)
        return J[:3, self.leg_v_index[leg]]

    def full_foot_jacobian(self, q: np.ndarray, leg: str) -> np.ndarray:
        """某条腿足端的 3 x nv 平动雅可比，保留全部列。

        全身控制需要它：基座那几列承载着接触约束。
        :meth:`foot_jacobian` 就是它的腿部切片。
        """
        q = np.asarray(q, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.foot_frame_ids[leg], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J[:3, :].copy()

    def inverse_kinematics_numeric(
        self,
        target: np.ndarray,
        leg: str,
        q_init: np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-10,
        damping: float = 1e-6,
        step: float = 1.0,
    ) -> tuple[np.ndarray, bool]:
        """单腿的阻尼最小二乘（DLS）逆运动学，作用于固定基座模型。

        它存在的意义是**交叉校验闭式解**，以及服务于那些腿部没有解析解的
        机器人。它**不是**控制回路的路径：每次迭代都要做一次雅可比分解。

        Args:
            target: 根坐标系下期望的足端位置，形状 (3,)。
            leg: 腿标识。
            q_init: 12 个关节角的初值，默认取零构型。
            max_iter: 最大迭代次数。
            tol: 位置误差范数的收敛阈值，单位米。
            damping: Levenberg-Marquardt 阻尼；在奇异构型附近把步长
                限制住，而不是让它爆掉。
            step: 步长缩放系数，取值 (0, 1]。

        Returns:
            ``(q_joints, converged)`` —— 完整的 12 维关节角向量，
            以及误差是否降到了 ``tol`` 以下。
        """
        if self.floating_base:
            raise NotImplementedError("数值腿部 IK 需要固定基座模型。")
        target = np.asarray(target, dtype=float).reshape(3)
        q = self.neutral() if q_init is None else np.asarray(q_init, dtype=float).copy()
        cols = self.leg_v_index[leg]
        eye = np.eye(3)

        for _ in range(max_iter):
            err = target - self.foot_position(q, leg)
            if np.linalg.norm(err) < tol:
                return q, True
            J = self.foot_jacobian(q, leg)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * eye, err)
            q[cols] += step * dq

        return q, np.linalg.norm(target - self.foot_position(q, leg)) < tol
