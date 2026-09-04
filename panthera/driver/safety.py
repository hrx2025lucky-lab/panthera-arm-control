"""安全层：给真机发力矩之前，最后一道闸。

⛔ **在这个模块投入使用之前，不要给真机发第一条力矩指令。**

为什么需要它
------------
力矩控制没有位置控制的"自然安全性"。位置模式下发一个错误目标，
手臂会慢慢走过去；力矩模式下发一个错误力矩，手臂会**立刻加速**。

而下面每一种情况都会产生错误力矩，且都不会抛异常：

* 控制循环卡住（GC、页错误、Python 抢不到 CPU）→ 电机保持上一条指令**继续加速**
* 控制律算出 NaN（矩阵奇异、除零）→ 下发 NaN
* 增益配错 → 输出瞬间顶到限幅
* 关节接近限位 → 力矩还在往限位方向推

⭐ 这些在 MuJoCo 里**全部可以测**，所以安全层必须在真机到货前写完并测透。

一个真实的陷阱
--------------
⚠️ 官方 SDK 的 ``pos_vel_tqe_kp_kd()`` 在位置超限时**直接 return False
不发指令**——纯力矩模式下这意味着**手臂瞬间失去所有力矩**。

而我们如果传 ``pos=0``，余量薄得可怕：

==================  ===========  ==========
配置                 J2/J3 下限    pos=0 余量
==================  ===========  ==========
Follower（我们的）    −0.1         0.1 rad
Leader              0.0          **0**
==================  ===========  ==========

⚠️ 换配置、改 URDF、或者零位漂移都可能吃掉这 0.1 rad。

所以 :class:`SafetyLayer` 把 ``pos`` 参数取成**当前实测位置**而不是 0：
纯力矩模式下（kp=kd=0）它在数学上不起作用，但保证永远不触发那个拒绝分支。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class SafetyViolation(RuntimeError):
    """安全层拒绝了一条指令。"""


@dataclass
class SafetyConfig:
    """安全限值。⭐ 默认值全部来自官方 ``Follower.yaml``，不要手填。"""

    q_lower: np.ndarray
    q_upper: np.ndarray
    qd_max: np.ndarray
    tau_max: np.ndarray

    #: ⚠️ 控制周期超过这个值就判定"循环卡住"，强制置零力矩。
    #: 取 3 倍标称周期：偶发抖动不误触发，真卡住立刻抓到。
    watchdog_timeout: float = 0.02

    #: 上电力矩斜坡时长。⭐ 不要一上来就给满力矩。
    ramp_time: float = 1.0

    #: 距离限位多远开始刹车（rad）。在这个区间内，
    #: **指向限位方向**的力矩被线性衰减到零。
    limit_margin: float = 0.15

    #: 超速时的力矩衰减区间（rad/s）
    speed_margin: float = 0.2


def config_from_backend(backend, **overrides) -> SafetyConfig:
    """从后端读限值构造配置。⭐ 单一事实来源，避免两处数字不一致。"""
    m = backend.model
    cfg = SafetyConfig(
        q_lower=m.jnt_range[:6, 0].copy(),
        q_upper=m.jnt_range[:6, 1].copy(),
        qd_max=np.full(6, 1.0),          # 官方 velocity_limits
        tau_max=backend.tau_limit.copy())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@dataclass
class SafetyStatus:
    """一次 :meth:`SafetyLayer.filter` 的结果说明。⭐ 每一次干预都要留痕。"""

    ok: bool = True
    ramp_scale: float = 1.0
    limit_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    speed_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    saturated: bool = False
    watchdog_tripped: bool = False
    nan_blocked: bool = False
    reason: str = ""


class SafetyLayer:
    """把控制器输出过一遍安全检查，返回可以真正下发的力矩。

    用法::

        safety = SafetyLayer(config_from_backend(backend))
        safety.arm()                       # 开始上电斜坡
        while True:
            st = backend.read()
            tau = controller(st.q, st.qd)
            tau, status = safety.filter(tau, st.q, st.qd)
            backend.send_torque(tau)

    ⚠️ **顺序不能变**：NaN → 看门狗 → 限位刹车 → 超速刹车 → 斜坡 → 限幅。
    限幅必须在最后，否则前面的缩放可能把已经限过的值放大回去。
    """

    def __init__(self, cfg: SafetyConfig, clock=time.monotonic):
        self.cfg = cfg
        self._clock = clock
        self._armed = False
        self._arm_time = 0.0
        self._last_call = None
        self._latched_fault = ""

    # ------------------------------------------------------------ 生命周期

    def arm(self) -> None:
        """开始上电斜坡。⭐ 每次从静止启动都要重新调用。"""
        self._armed = True
        self._arm_time = self._clock()
        self._last_call = None
        self._latched_fault = ""

    def disarm(self, reason: str = "手动停止") -> None:
        self._armed = False
        self._latched_fault = reason

    @property
    def faulted(self) -> bool:
        return bool(self._latched_fault)

    # ------------------------------------------------------------ 核心

    def filter(self, tau, q, qd) -> tuple[np.ndarray, SafetyStatus]:
        """返回 ``(安全力矩, 状态)``。

        ⚠️ **永远不抛异常**。抛异常会让控制循环崩掉，而崩掉时电机
        保持上一条指令继续跑——这正是最危险的情况。
        出问题就返回零力矩并在 status 里说明。
        """
        cfg = self.cfg
        status = SafetyStatus()
        tau = np.asarray(tau, dtype=float).copy()
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)

        if self._latched_fault:
            status.ok = False
            status.reason = f"已锁定故障：{self._latched_fault}"
            return np.zeros_like(tau), status

        if not self._armed:
            status.ok = False
            status.reason = "未 arm，拒绝输出力矩"
            return np.zeros_like(tau), status

        # ① NaN / inf —— 必须最先查，否则后面所有比较都失效
        if not np.isfinite(tau).all():
            self.disarm("控制器输出 NaN/inf")
            status.ok = False
            status.nan_blocked = True
            status.reason = "控制器输出 NaN/inf，已锁定"
            return np.zeros_like(tau), status

        # ② 看门狗：本次调用距上次太久 = 控制循环卡过
        now = self._clock()
        if self._last_call is not None:
            gap = now - self._last_call
            if gap > cfg.watchdog_timeout:
                self._last_call = now
                self.disarm(f"控制周期 {gap * 1000:.1f} ms 超过看门狗 "
                            f"{cfg.watchdog_timeout * 1000:.0f} ms")
                status.ok = False
                status.watchdog_tripped = True
                status.reason = self._latched_fault
                return np.zeros_like(tau), status
        self._last_call = now

        # ③ 限位刹车：只衰减**指向限位方向**的力矩。
        #    ⭐ 反方向（把手臂拉回安全区）的力矩必须保留，否则一旦越界就再也回不来。
        d_lo = q - cfg.q_lower
        d_hi = cfg.q_upper - q
        scale = np.ones_like(tau)
        near_lo = d_lo < cfg.limit_margin
        near_hi = d_hi < cfg.limit_margin
        f_lo = np.clip(d_lo / cfg.limit_margin, 0.0, 1.0)
        f_hi = np.clip(d_hi / cfg.limit_margin, 0.0, 1.0)
        scale = np.where(near_lo & (tau < 0), f_lo, scale)
        scale = np.where(near_hi & (tau > 0), f_hi, scale)
        status.limit_scale = scale
        tau = tau * scale

        # ④ 超速刹车：同样只衰减"让它更快"的方向
        over = np.abs(qd) - cfg.qd_max
        s_scale = np.ones_like(tau)
        speeding = over > -cfg.speed_margin
        f_v = np.clip(-over / cfg.speed_margin, 0.0, 1.0)
        same_dir = np.sign(tau) == np.sign(qd)
        s_scale = np.where(speeding & same_dir, f_v, s_scale)
        status.speed_scale = s_scale
        tau = tau * s_scale

        # ⑤ 上电斜坡
        elapsed = now - self._arm_time
        ramp = 1.0 if cfg.ramp_time <= 0 else min(elapsed / cfg.ramp_time, 1.0)
        status.ramp_scale = float(ramp)
        tau = tau * ramp

        # ⑥ 限幅（⚠️ 必须最后做）
        #
        # ⚠️⚠️ 限幅上限**本身**也要乘斜坡，否则上电斜坡在大指令下完全失效：
        # 指令 100×限幅、斜坡 50% ⇒ 缩放后仍是 50×限幅 ⇒ 截断后还是**满限幅**。
        # `实测` 这个 bug 被 test_ramp_and_clamp_compose_correctly 抓到——
        # 斜坡走到一半却输出满力矩，正是"上电别一下给满"要防的那件事。
        #
        # ⭐ 教训：一个"缩放 + 饱和"的链路里，**只缩信号不缩上限等于没缩**。
        limit = cfg.tau_max * ramp
        clipped = np.clip(tau, -limit, limit)
        status.saturated = bool(np.any(np.abs(clipped - tau) > 1e-12))
        return clipped, status

    def safe_position_argument(self, q) -> np.ndarray:
        """给 ``pos_vel_tqe_kp_kd`` 的 ``pos`` 参数用。

        ⚠️⚠️ **不要传 0**。官方 SDK 在 ``pos`` 超限时会 ``return False``
        静默丢弃整条指令——纯力矩模式下这意味着**手臂瞬间失去所有力矩**。
        而 ``pos=0`` 的余量极薄：Follower 配置下 J2/J3 只有 0.1 rad，
        Leader 配置下**恰好为 0**。

        ⭐ 传当前实测位置：kp=kd=0 时它在数学上不起作用，
        但保证永远落在限位内。这里再夹一次是为了防传感器野值。
        """
        return np.clip(np.asarray(q, dtype=float),
                       self.cfg.q_lower, self.cfg.q_upper)
