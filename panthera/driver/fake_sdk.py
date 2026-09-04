"""假 SDK：把官方 ``hightorque_robot`` 的接口转发到 MuJoCo。

⭐ **为什么值得单独写一个假的**

真机到货前，我们没法验证 :class:`~panthera.driver.real_backend.RealBackend`
里的任何一行。而那些代码恰恰是**最不能出错**的部分——
它直接决定给电机发什么。

这个假 SDK 让整条链路

    控制器 → 安全层 → RealBackend → SDK → 电机

在**没有硬件**的情况下端到端跑通。真机到货时只需要把
``Panthera`` 从这里换成官方包，其余一行不改。

⚠️ 它复刻了官方 SDK 的**行为**，不只是接口签名。特别是那些"坑"：

* ``pos_vel_tqe_kp_kd()`` 在 ``pos`` 超限时 **``return False`` 静默丢弃整条指令**
* ``motor_count`` 是 ``len(Motors) - 1``（最后一个是夹爪）
* 读状态要先 ``send_get_motor_state_cmd()`` + ``motor_send_cmd()``
* MIT 模式：$\\tau = k_p(q_{des}-q) + k_d(\\dot q_{des}-\\dot q) + \\tau_{ff}$

⚠️⚠️ **它不能替代真机验证**。它验证的是"我们的代码调用 SDK 的方式对不对"，
**不是**"真机会不会这样反应"。延迟、力矩精度、摩擦这些只有真机知道。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panthera.core.robot import Q_HOME

#: 官方 ``Follower.yaml`` 的关节限位
JOINT_LOWER = np.array([-2.4, -0.1, -0.1, -1.6, -1.7, -2.5])
JOINT_UPPER = np.array([2.4, 3.2, 4.0, 1.6, 1.7, 2.5])
GRIPPER_LIMITS = (0.0, 2.0)
#: ⚠️ 这是**堵转**扭矩，不是可持续输出。见 docs/给客服的问题清单.md Q9
MAX_TORQUE = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])


@dataclass
class MotorState:
    """对应官方 ``get_current_motor_state()`` 的返回。"""

    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0


class FakeMotor:
    def __init__(self):
        self.state = MotorState()
        self.cmd = np.zeros(5)      # pos, vel, tqe, kp, kd

    def get_current_motor_state(self) -> MotorState:
        return self.state

    def pos_vel_tqe_kp_kd(self, pos, vel, tqe, kp, kd) -> None:
        self.cmd = np.array([pos, vel, tqe, kp, kd], dtype=float)


class FakePanthera:
    """官方 ``Panthera`` 类的仿真替身。

    Args:
        backend: 一个 :class:`~panthera.driver.mujoco_backend.MujocoBackend`。
        strict_pos_check: 是否复刻"pos 超限就丢弃整条指令"的行为。
            ⭐ 默认 True——**这个坑必须能被测到**。

    Attributes:
        dropped_commands: 被静默丢弃的指令数。
            ⭐ 真机 SDK 不会告诉你这件事（只 print 一行然后 return False），
            这里显式计数，让守护测试能断言它。
    """

    def __init__(self, backend, strict_pos_check: bool = True):
        self.backend = backend
        self.strict_pos_check = strict_pos_check
        self.Motors = [FakeMotor() for _ in range(7)]   # 6 关节 + 1 夹爪
        self.motor_count = len(self.Motors) - 1
        self.gripper_id = 7
        self.joint_limits = {"lower": JOINT_LOWER.copy(),
                             "upper": JOINT_UPPER.copy()}
        self.gripper_limits = {"lower": GRIPPER_LIMITS[0],
                               "upper": GRIPPER_LIMITS[1]}
        self.max_torque = MAX_TORQUE.copy()
        self.dropped_commands = 0
        self._pending = np.zeros((self.motor_count, 5))
        self._stopped = False
        self.backend.reset(np.array(Q_HOME))
        self._refresh()

    # ------------------------------------------------------------ 状态

    def _refresh(self) -> None:
        st = self.backend.read()
        for i in range(self.motor_count):
            m = self.Motors[i].state
            m.position, m.velocity, m.torque = st.q[i], st.qd[i], st.tau[i]

    def send_get_motor_state_cmd(self) -> None:
        """⚠️ 官方注释："利用运控模式发送控制帧以读取电机反馈状态"。"""
        self._refresh()

    def motor_send_cmd(self) -> None:
        """把缓存的指令按 MIT 模式算成力矩，推进一步仿真。"""
        if self._stopped:
            return
        st = self.backend.read()
        pos, vel, tqe, kp, kd = (self._pending[:, i] for i in range(5))
        tau = kp * (pos - st.q) + kd * (vel - st.qd) + tqe
        self.backend.send_torque(tau)
        self.backend.step()
        self._refresh()

    def get_current_pos(self) -> np.ndarray:
        return np.array([m.state.position for m in self.Motors[:self.motor_count]])

    def get_current_vel(self) -> np.ndarray:
        return np.array([m.state.velocity for m in self.Motors[:self.motor_count]])

    def get_current_torque(self) -> np.ndarray:
        return np.array([m.state.torque for m in self.Motors[:self.motor_count]])

    # ------------------------------------------------------------ 控制

    def pos_vel_tqe_kp_kd(self, pos, vel, tqe, kp, kd) -> bool:
        """关节五参数 MIT 控制模式。

        ⚠️⚠️ **复刻了官方最危险的行为**：``pos`` 超限时打印警告并
        ``return False``——**整条指令被丢弃，包括力矩项**。
        纯力矩模式下这意味着手臂瞬间失去所有力矩。
        """
        params = [pos, vel, tqe, kp, kd]
        if not all(len(p) == self.motor_count for p in params):
            raise ValueError(f"关节参数长度必须为{self.motor_count}")

        pos = np.asarray(pos, dtype=float)
        if self.strict_pos_check:
            out = np.logical_or(pos < self.joint_limits["lower"],
                                pos > self.joint_limits["upper"])
            if np.any(out):
                self.dropped_commands += 1
                return False              # ⚠️ 静默丢弃，力矩项也没了

        self._pending = np.column_stack(
            [pos, np.asarray(vel, float), np.asarray(tqe, float),
             np.asarray(kp, float), np.asarray(kd, float)])
        self.motor_send_cmd()
        return True

    def get_Gravity(self) -> np.ndarray:
        return self.backend.robot.gravity(self.get_current_pos())

    def get_friction_compensation(self, vel, Fc, Fv, vel_threshold) -> np.ndarray:
        """⚠️ 复刻官方的**速度阈值**写法：低于阈值时不加库伦项。

        注意这与我们 ``momentum_observer`` 里的 ``tanh`` 平滑写法不同——
        阈值法在阈值处**不连续**。做对照实验时要注意这个差异。
        """
        vel = np.asarray(vel, dtype=float)
        coulomb = np.where(np.abs(vel) > vel_threshold,
                           np.asarray(Fc) * np.sign(vel), 0.0)
        return coulomb + np.asarray(Fv) * vel

    def set_stop(self) -> None:
        """⚠️ 官方 ``Panthera.py`` 里**没有**这个方法的定义
        （在预编译包里），而且力矩控制示例中它全部被注释掉了。
        这里按"力矩置零"实现——**真机行为待客服确认**（问题清单 Q3）。
        """
        self._stopped = True
        self.backend.send_torque(np.zeros(self.motor_count))

    # ------------------------------------------------------------ 夹爪

    def gripper_control_MIT(self, pos, vel, tqe, kp, kd) -> bool:
        lo, hi = self.gripper_limits["lower"], self.gripper_limits["upper"]
        if pos < lo or pos > hi:
            self.dropped_commands += 1
            return False
        self.Motors[6].pos_vel_tqe_kp_kd(pos, vel, tqe, kp, kd)
        return True

    def get_current_pos_gripper(self) -> float:
        return self.Motors[6].state.position
