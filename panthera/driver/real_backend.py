"""真机后端：把 :class:`~panthera.driver.backend.ArmBackend` 接到官方 SDK。

⭐ **设计目标：真机到货时改动量最小。**

上层（控制器、安全层、rollout、测试）全部走 ``ArmBackend`` 接口，
所以从仿真切到真机只需要换一个对象::

    # 仿真
    backend = MujocoBackend()
    # 真机
    backend = RealBackend()

而 ``RealBackend`` 本身也能在**没有硬件**时跑——
传一个 :class:`~panthera.driver.fake_sdk.FakePanthera` 进去即可。
这样它的每一行代码在真机到货前就被测过了。

⚠️⚠️ 上电前必须做完的四件事（顺序不能反）
------------------------------------------

1. **实测控制频率**。官方示例是 ``sleep(0.002)``＝500 Hz，但那是**期望值**
   不是实测值；另一些示例用的是 200 Hz 甚至 100 Hz。
   先测出真实节拍再把 ``dt`` 设成它——否则所有基于 Δt 的推导都是错的。
2. **单关节先行**。从一个关节、``tau=0`` 开始逐步加，手放急停旁边。
3. **看门狗生效**。:class:`~panthera.driver.safety.SafetyLayer` 必须在链路里。
4. **确认限幅生效**：故意下发一个超限力矩，确认被截断而不是被执行。

⛔ 这四件事做完之前，不要跑整臂力矩控制。
:func:`~panthera.scripts.commission` 会带你依次做完。
"""

from __future__ import annotations

import time

import numpy as np

from .backend import ArmBackend, ArmState


class RealBackend(ArmBackend):
    """高擎 Panthera-HT 真机后端。

    Args:
        sdk: 官方 ``Panthera`` 实例。传 ``None`` 时尝试导入官方包；
            传 :class:`FakePanthera` 可在无硬件时跑通整条链路。
        dt: 标称控制周期。⚠️ **必须设成实测值**，见模块文档第 1 条。
        tau_limit: 力矩限幅。默认取官方示例里**最保守**的一套。
            ⚠️ 官方示例存在三套矛盾值，见 docs/给客服的问题清单.md Q9。
        model: 用于运动学/动力学计算的 MuJoCo 模型（做重力补偿等）。
            ⭐ 真机没有"模型"，但控制律需要——所以这里仍然挂一个。

    ⚠️ **没有 reset()**。真机不能瞬移。上层若调用会抛异常，
    这是**有意的**：让"仿真专用"的代码路径在真机上立刻暴露，
    而不是悄悄做了别的事。
    """

    #: 官方示例里最保守的一套限幅
    DEFAULT_TAU_LIMIT = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])

    def __init__(self, sdk=None, dt: float = 0.005,
                 tau_limit=None, model=None):
        if sdk is None:                       # pragma: no cover - 需要硬件
            from Panthera_lib import Panthera
            sdk = Panthera()
        self.sdk = sdk
        self.n = int(sdk.motor_count)
        self.dt = float(dt)
        self.tau_limit = (self.DEFAULT_TAU_LIMIT.copy() if tau_limit is None
                          else np.asarray(tau_limit, dtype=float))
        self.robot = model
        self.model = getattr(model, "model", None)

        self._zero = np.zeros(self.n)
        self._t0 = time.monotonic()
        #: ⭐ 累计被 SDK 丢弃的指令数。真机 SDK 只 print 一行就 return False，
        #: 不显式跟踪的话，"手臂突然软了"会查不出原因。
        self.dropped = 0
        #: 实测周期统计，用于验证第 1 件事
        self.periods: list[float] = []
        self._last = None

    # ------------------------------------------------------------ 读

    def read(self) -> ArmState:
        """⚠️ 必须先 ``send_get_motor_state_cmd()`` 刷新。

        官方**力矩控制示例里都没有调用它**，而遥操作示例全都调用了——
        这可能意味着力矩环用的是上一周期的旧状态。
        待客服确认（问题清单 Q7）。在确认之前，我们**显式刷新**，宁慢勿错。
        """
        self.sdk.send_get_motor_state_cmd()
        now = time.monotonic()
        if self._last is not None:
            self.periods.append(now - self._last)
        self._last = now
        return ArmState(
            q=np.asarray(self.sdk.get_current_pos(), dtype=float),
            qd=np.asarray(self.sdk.get_current_vel(), dtype=float),
            # ⚠️ 真机这是**电流估算**，不是力矩传感器。可信度与仿真不同。
            tau=np.asarray(self.sdk.get_current_torque(), dtype=float),
            stamp=now - self._t0)

    # ------------------------------------------------------------ 写

    def send_torque(self, tau: np.ndarray) -> None:
        """纯力矩模式（MIT kp=kd=0）下发。

        ⚠️⚠️ ``pos`` 参数传的是**当前实测位置**而不是 0。

        官方 SDK 在 ``pos`` 超限时会 ``return False`` **静默丢弃整条指令**，
        纯力矩模式下这意味着手臂瞬间失去所有力矩。
        而传 ``pos=0`` 的余量极薄：Follower 配置下 J2/J3 只有 0.1 rad，
        Leader 配置下恰好为 0。
        传当前位置在数学上不起作用（kp=kd=0），但永远落在限位内。
        """
        tau = self.saturate(np.asarray(tau, dtype=float))
        q_now = np.asarray(self.sdk.get_current_pos(), dtype=float)
        lower = self.sdk.joint_limits["lower"]
        upper = self.sdk.joint_limits["upper"]
        pos_arg = np.clip(q_now, lower, upper)

        ok = self.sdk.pos_vel_tqe_kp_kd(pos_arg, self._zero, tau,
                                        self._zero, self._zero)
        if not ok:
            # ⚠️ 不抛异常——控制循环崩掉时电机会保持上一条指令继续跑，
            #    那比丢一帧更危险。计数并让上层去查。
            self.dropped += 1

    def send_mit(self, q_des, qd_des, tau_ff, kp, kd) -> bool:
        """完整 MIT 模式。⭐ **RL 策略走这条路**（输出位置目标，电机侧 PD 执行）。

        与 :meth:`send_torque` 的区别：这里 ``q_des`` 是**真的期望位置**，
        必须自己保证它在限位内，否则整条指令被丢弃。
        """
        q_des = np.clip(np.asarray(q_des, dtype=float),
                        self.sdk.joint_limits["lower"],
                        self.sdk.joint_limits["upper"])
        ok = self.sdk.pos_vel_tqe_kp_kd(
            q_des, np.asarray(qd_des, dtype=float),
            self.saturate(np.asarray(tau_ff, dtype=float)),
            np.asarray(kp, dtype=float), np.asarray(kd, dtype=float))
        if not ok:
            self.dropped += 1
        return ok

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """⭐ 优先用 SDK 自己的重力模型——它和真机的标定是一致的。"""
        if hasattr(self.sdk, "get_Gravity"):
            return np.asarray(self.sdk.get_Gravity(), dtype=float)
        return self.robot.gravity(q)

    def step(self) -> None:
        """真机没有"步进"，靠调用节拍推进。

        ⚠️ 这里**不 sleep**。节拍由上层的调度器控制——
        官方 ``7_gamepad`` 用的是**绝对截止时间调度**而不是固定 sleep，
        因为"运算耗时 + 固定 sleep"会造成周期漂移。
        """

    def reset(self, q: np.ndarray) -> None:
        """⛔ 真机不能瞬移。"""
        raise NotImplementedError(
            "真机不能 reset。要回零位请用 moveJ 慢速移动，"
            "并确认路径上无障碍。")

    def close(self) -> None:
        """⚠️ 退出时置零力矩。

        官方力矩控制示例里 ``set_stop()`` **全部被注释掉了**（13 处），
        而配置里 ``exit_motor_brake_flag: true`` 写着"退出时释放刹车"——
        所以退出行为待客服确认（问题清单 Q3）。
        在确认之前，我们**显式置零并调用 set_stop**。
        """
        try:
            self.send_torque(np.zeros(self.n))
        finally:
            if hasattr(self.sdk, "set_stop"):
                self.sdk.set_stop()

    # ------------------------------------------------------------ 诊断

    def period_stats(self) -> dict:
        """实测控制周期统计。⭐ 上电第一件事就是看这个。"""
        if not self.periods:
            return {"n": 0}
        p = np.array(self.periods)
        return {"n": len(p), "mean_ms": float(p.mean() * 1e3),
                "std_ms": float(p.std() * 1e3),
                "min_ms": float(p.min() * 1e3),
                "max_ms": float(p.max() * 1e3),
                "p99_ms": float(np.percentile(p, 99) * 1e3),
                "hz": float(1.0 / p.mean())}
