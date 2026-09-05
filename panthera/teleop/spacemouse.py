"""SpaceMouse 遥操作：6 维笛卡尔速度 → 关节位置指令。

⭐ **为什么要自己写**

官方 `Panthera-HT_lerobot` 的遥操是 **主从双臂**（`panthera_leader` +
`panthera_follower`，同构关节映射）——**需要两台机器**。
我们只有一台，所以 SpaceMouse 不是"锦上添花"，而是**采示教数据的唯一途径**。

而示教数据正是 **BC（行为克隆）** 的输入。官方全组织搜索 `spacemouse`
结果为 **0**，这块必须我们自己做。

链路
----
::

    SpaceMouse (6 轴模拟量)
      → 死区 + 缩放            ⚠️ 不做死区，手不碰它也会漂
      → 笛卡尔速度 v_des
      → 微分 IK: q̇ = J⁺_λ v    ⚠️ 6 轴方阵，奇异点必须用阻尼伪逆
      → 积分 q_des += q̇·dt     ⚠️⚠️ 积分漂移，见 §anti-windup
      → 夹到关节限位            ⚠️ 不夹的话 SDK 会**静默丢弃整条指令**
      → 关节位置指令（= LeRobot 的 action，也 = RL 的 action）

⭐ 最后一行很重要：输出的动作空间和 **LeRobot 的 action_features
（`{joint1.pos, ...}`）完全一致**，也和 RL 的标准动作空间一致。
所以这一个模块同时服务遥操、BC 数据采集、和 RL 部署。

⚠️⚠️ 积分漂移是最大的坑
-----------------------
$q_{des}$ 靠积分累积，而真机 $q$ 有跟踪误差（摩擦、限幅、负载）。
两者会**越差越远**：

* 手已经停了，但 $q_{des}$ 还在手臂到不了的地方
* 松手后手臂继续朝那个位置冲 —— **看起来像失控**

⭐ 对策：把 $q_{des}$ 与实测 $q$ 的偏差**钳制**在一个上限内
（:attr:`TeleopConfig.max_lag`）。这与 PID 的 anti-windup 同理：
积分器不许跑到执行器跟不上的地方。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeleopConfig:
    """遥操作参数。⭐ 默认值偏保守——遥操是**人在回路**，宁慢勿惊。"""

    #: 平移速度标度 (m/s)，对应摇杆推到底
    trans_scale: float = 0.08
    #: 旋转速度标度 (rad/s)
    rot_scale: float = 0.5
    #: ⚠️ 死区。SpaceMouse 静止时输出**不是精确的 0**，
    #: 不设死区手不碰它手臂也会缓慢漂移。
    deadzone: float = 0.08
    #: 输入低通系数（0~1，越小越平滑）。⚠️ 太小会有明显滞后感。
    smoothing: float = 0.3
    #: DLS 阻尼系数。⭐ 越大越稳但精度越低；奇异点附近靠它兜底。
    damping: float = 0.05
    #: 关节速度上限 (rad/s)。⚠️ 官方 ``velocity_limits`` 就是 1.0
    qd_max: float = 1.0
    #: ⚠️⚠️ 期望位置允许领先实测位置的最大量 (rad)。见模块文档 anti-windup。
    max_lag: float = 0.15
    #: 距关节限位多近开始减速 (rad)
    limit_margin: float = 0.1


class SpaceMouseInput:
    """把 SpaceMouse 原始读数变成干净的笛卡尔速度。

    ⭐ 设计成**可注入**：``read_fn`` 返回 6 个 [-1,1] 的量。
    真机用 ``pyspacemouse``，测试用假函数——
    这样这段逻辑在没有硬件时就能被完整验证。

    Args:
        read_fn: ``() -> (6,)`` 数组，顺序 [x, y, z, roll, pitch, yaw]，
            范围 [-1, 1]。
    """

    def __init__(self, read_fn, cfg: TeleopConfig | None = None):
        self.read_fn = read_fn
        self.cfg = cfg or TeleopConfig()
        self._filtered = np.zeros(6)

    @staticmethod
    def _deadzone(x: np.ndarray, dz: float) -> np.ndarray:
        """死区 + **重新拉伸**。

        ⭐ 不能简单地"小于死区就置零"——那样输出在死区边缘会**跳变**。
        正确做法是把 [dz, 1] 线性映射回 [0, 1]，保证连续。
        """
        mag = np.abs(x)
        out = np.where(mag > dz, np.sign(x) * (mag - dz) / (1.0 - dz), 0.0)
        return np.clip(out, -1.0, 1.0)

    def read_velocity(self) -> np.ndarray:
        """返回 6 维笛卡尔速度 [vx,vy,vz,wx,wy,wz]，单位 m/s 与 rad/s。"""
        raw = np.asarray(self.read_fn(), dtype=float).ravel()
        if raw.size != 6:
            raise ValueError(f"SpaceMouse 应返回 6 个量，收到 {raw.size}")
        clean = self._deadzone(np.clip(raw, -1.0, 1.0), self.cfg.deadzone)
        a = self.cfg.smoothing
        self._filtered = a * clean + (1.0 - a) * self._filtered
        return self._filtered * np.concatenate(
            [np.full(3, self.cfg.trans_scale), np.full(3, self.cfg.rot_scale)])

    def reset(self) -> None:
        self._filtered[:] = 0.0


class CartesianTeleop:
    """笛卡尔速度 → 关节位置指令。

    Args:
        robot: :class:`~panthera.core.robot.ArmModel`
        q_lower / q_upper: 关节限位。⚠️ 用**官方 Follower.yaml** 的值。

    用法::

        teleop = CartesianTeleop(robot, lower, upper)
        teleop.reset(q_now)
        while True:
            v = mouse.read_velocity()
            q_des = teleop.step(v, q_measured, dt)
            backend.send_mit(q_des, 0, 0, kp, kd)
    """

    def __init__(self, robot, q_lower, q_upper,
                 cfg: TeleopConfig | None = None):
        self.robot = robot
        self.cfg = cfg or TeleopConfig()
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.n = len(self.q_lower)
        self.q_des = np.zeros(self.n)
        #: 诊断读数——⭐ 直接暴露物理量，而不是只看"能不能动"
        self.last = {}

    def reset(self, q) -> None:
        """⚠️ 每次开始遥操**必须**调用，把期望位置对齐到当前实测位置。

        不 reset 就直接 step，$q_{des}$ 会从上一次的残留值开始，
        手臂会瞬间往那里冲。
        """
        self.q_des = np.clip(np.asarray(q, dtype=float),
                             self.q_lower, self.q_upper)

    def _dls_pinv(self, J: np.ndarray) -> np.ndarray:
        """阻尼最小二乘伪逆 $J^{\\mathsf T}(JJ^{\\mathsf T}+\\lambda^2 I)^{-1}$。

        ⚠️ Panthera 是 6 轴，$J$ 是 **6×6 方阵，没有零空间**。
        奇异构型附近 $J^{-1}$ 会爆炸——$\\lambda$ 牺牲精度换有界性。
        """
        m = J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + self.cfg.damping ** 2 * np.eye(m))

    def _limit_scale(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """接近限位时，**只**衰减朝向限位的分量。

        ⭐ 和 :class:`~panthera.driver.safety.SafetyLayer` 同一个道理：
        反方向必须保留，否则一旦贴住限位就再也回不来。
        """
        m = self.cfg.limit_margin
        f_lo = np.clip((q - self.q_lower) / m, 0.0, 1.0)
        f_hi = np.clip((self.q_upper - q) / m, 0.0, 1.0)
        scale = np.ones_like(qd)
        scale = np.where(qd < 0, f_lo, scale)
        scale = np.where(qd > 0, f_hi, scale)
        return scale

    def step(self, v_cart, q_measured, dt: float) -> np.ndarray:
        """推进一步，返回关节位置指令。

        Args:
            v_cart: 6 维笛卡尔速度（来自 :meth:`SpaceMouseInput.read_velocity`）
            q_measured: **实测**关节位置。⚠️ 必须是实测值，不能用上一次的 q_des，
                否则 anti-windup 完全失效。
        """
        q_measured = np.asarray(q_measured, dtype=float)
        v = np.asarray(v_cart, dtype=float)

        # ⭐ 雅可比在**实测**构型上算，不是在 q_des 上——
        #    q_des 可能领先，用它算出的映射与真实运动学不符。
        J = self.robot.jacobian(q_measured)
        qd = self._dls_pinv(J) @ v

        qd = qd * self._limit_scale(q_measured, qd)

        # 速度限幅：⚠️ 按**最大分量**整体缩放，保持方向不变。
        #    逐关节 clip 会改变末端运动方向，手感会很怪。
        peak = np.abs(qd).max()
        if peak > self.cfg.qd_max:
            qd = qd * (self.cfg.qd_max / peak)

        self.q_des = self.q_des + qd * dt

        # ⚠️⚠️ anti-windup：不许期望位置跑到实测跟不上的地方
        lag = self.q_des - q_measured
        over = np.abs(lag) > self.cfg.max_lag
        if np.any(over):
            self.q_des = np.where(
                over,
                q_measured + np.sign(lag) * self.cfg.max_lag,
                self.q_des)

        # ⚠️ 必须夹紧：SDK 在 pos 超限时会 return False **静默丢弃整条指令**
        self.q_des = np.clip(self.q_des, self.q_lower, self.q_upper)

        self.last = {
            "qd_peak": float(np.abs(qd).max()),
            "lag_max": float(np.abs(self.q_des - q_measured).max()),
            "windup_clamped": bool(np.any(over)),
            "manipulability": float(self.robot.manipulability(q_measured)),
        }
        return self.q_des.copy()


def make_reader(device_path: str | None = None):
    """真机用的 SpaceMouse 读取函数。

    ⚠️ 需要 ``pip install pyspacemouse``（依赖 hidapi）。
    Linux 上还要给 hidraw 设备权限::

        sudo tee /etc/udev/rules.d/99-spacemouse.rules <<'EOF'
        KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"
        KERNEL=="hidraw*", ATTRS{idVendor}=="046d", MODE="0666"
        EOF
        sudo udevadm control --reload-rules && sudo udevadm trigger

    ⚠️ 3Dconnexion 的 vendor id 有两个：老款 046d（Logitech 时期）、
    新款 256f。两个都要加。
    """
    import pyspacemouse                      # pragma: no cover - 需要硬件

    if not pyspacemouse.open():
        raise RuntimeError(
            "打不开 SpaceMouse。检查：① 是否插好 ② hidraw 权限（见 docstring）")

    def read():
        s = pyspacemouse.read()
        # ⚠️ pyspacemouse 的 y/z 与机器人坐标系不同，且 roll/pitch 符号相反。
        #    这里的映射**必须在真机上逐轴确认**——推一个方向，看手臂往哪走。
        return np.array([s.y, -s.x, s.z, s.roll, -s.pitch, -s.yaw])

    return read
