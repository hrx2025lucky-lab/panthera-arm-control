"""MuJoCo 仿真后端。

实现 :class:`~panthera.driver.backend.ArmBackend`，让控制代码在仿真里跑。
与真机后端的差别只在两处，都刻意做成"仿真更严格"而不是"仿真更宽松"：

1. **力矩限幅照样生效**。仿真里放开限幅很方便，但那会让你在仿真里调出一组
   真机根本发不出来的增益——等搬到真机才发现，白调。
2. **可选注入延迟**。真机的 CAN 往返 + 零阶保持带来半拍以上延迟（见讲义
   15 篇 §五）。默认关闭，做 sim2real 对照时打开，让两边的相位账一致。
"""

from __future__ import annotations

import collections
import time

import numpy as np
import mujoco

from panthera.core.robot import make_panthera
from panthera.driver.backend import ArmBackend, ArmState


class MujocoBackend(ArmBackend):
    """把 MuJoCo 仿真包装成统一后端。

    Args:
        xml_path: MJCF 路径，默认用仓库自带模型
        latency_steps: 人为注入的指令延迟（控制周期数）。
            ⭐ 真机至少有半拍零阶保持延迟，再加 CAN 往返；
            做 sim2real 对照时把它调到实测值，否则仿真会"偏乐观"。
    """

    def __init__(self, xml_path: str | None = None,
                 latency_steps: int = 0) -> None:
        self.robot = make_panthera(xml_path)
        self.model = self.robot.model
        self.data = self.robot.data
        self.n = self.robot.n
        self.tau_limit = self.robot.tau_limit.copy()
        self.dt = float(self.model.opt.timestep)
        self._queue: collections.deque[np.ndarray] = collections.deque(
            [np.zeros(self.n)] * max(latency_steps, 0), maxlen=None)
        self._latency = max(latency_steps, 0)
        self._t = 0.0

    def read(self) -> ArmState:
        return ArmState(
            q=self.data.qpos[self.robot.qpos_idx].copy(),
            qd=self.data.qvel[self.robot.qvel_idx].copy(),
            # 仿真里这是求解器算出的真值；真机上是电流估算，可信度不同
            tau=self.data.actuator_force[self.robot.arm_actuator_ids].copy(),
            stamp=self._t,
        )

    def send_torque(self, tau: np.ndarray) -> None:
        tau = self.saturate(np.asarray(tau, dtype=float))
        if self._latency:
            self._queue.append(tau)
            tau = self._queue.popleft()
        self.data.ctrl[self.robot.arm_actuator_ids] = tau

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return self.robot.gravity(q)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self._t += self.dt

    def reset(self, q: np.ndarray) -> None:
        """把仿真状态设到给定构型。**真机后端没有这个方法**——真机不能瞬移。"""
        self.robot.set_state(q)
        self._t = 0.0
        self._queue = collections.deque([np.zeros(self.n)] * self._latency)

    def close(self) -> None:
        self.data.ctrl[self.robot.arm_actuator_ids] = 0.0


class RealBackend(ArmBackend):
    """高擎 Panthera-HT 真机后端（**骨架，待接入 SDK**）。

    接入时要包的是官方预编译包 ``hightorque_robot``（见 ``Panthera-HT_SDK``
    的 ``panthera_python/motor_whl/``），对应关系::

        read()        → get_current_pos / get_current_vel / get_current_torque
        send_torque() → pos_vel_tqe_kp_kd(0, 0, tau, kp=0, kd=0)   # 纯力矩
        gravity()     → get_Gravity()

    ⚠️⚠️ 接入前必须先做的四件事（顺序不能反）：

    1. **实测控制频率**。官方示例是 ``sleep(0.002)``＝500 Hz，但那是**期望值**
       不是实测值。先测出真实节拍，再把 ``dt`` 设成它——
       否则所有基于 Δt 的推导（15 篇 §五）都是错的。
    2. **单关节先行**。从一个关节、``tau=0`` 开始逐步加，手放急停旁边。
    3. **看门狗**。控制循环卡住时必须自动置零力矩。
    4. **确认限幅生效**：故意下发一个超限力矩，确认被截断而不是被执行。

    ⛔ 在这四件事做完之前，不要跑整臂力矩控制。
    """

    def __init__(self) -> None:  # pragma: no cover - 需要真实硬件
        raise NotImplementedError(
            "真机后端尚未接入。接入步骤见本类 docstring，"
            "以及 docs/参数辨识与sim2real.md 的『真机上电流程』。"
        )
