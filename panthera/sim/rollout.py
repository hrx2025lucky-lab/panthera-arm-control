"""闭环仿真跑道：让控制器在**真的积分了动力学**的回路里跑。

⭐ 为什么必须有这个模块
----------------------
只验证"控制律的公式算得对"是不够的。公式对、闭环发散的情况太常见了：

* 增益配得太高 → 采样率下离散化失稳（连续域看着好好的）
* 力矩饱和 → 实际力矩 ≠ 指令力矩，抵消项全错
* 观测器带宽和控制带宽打架 → 互相激励
* 通信延迟 → 相位裕度被吃掉

⚠️ 这些**只有真的把回路闭起来跑**才会暴露。
公式级的单元测试对它们完全无感。

⭐ 这个 runner 走 :class:`~panthera.driver.backend.ArmBackend` 接口，
所以**同一段测试代码将来可以直接对着真机跑**——
把 ``MujocoBackend`` 换成 ``RealBackend`` 即可。
这就是"统一后端"的兑现点。

两个必须守住的规矩
------------------
⚠️ **① 力矩饱和永远打开。** armctrl 的 ``_convert_to_torque_actuators()``
会把限幅设成 ±1e6 让仿真好看，本项目刻意不这么做：
仿真里"用得起"的力矩，真机上必须也用得起，否则整套结论作废。

⚠️ **② 发散不抛异常。** "这组增益会发散"本身就是一条需要被断言的结论，
把它变成异常就没法写成测试了。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutLog:
    """一次闭环 rollout 的全部记录。"""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    q_des: np.ndarray
    dt: float
    #: 逐步的额外读数，由 ``extra_fn`` 塞进来（如 ‖J̇q̇‖、观测器残差 r）
    extra: dict = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        return self.q_des - self.q

    def rms(self, skip: float = 0.0) -> float:
        """跟踪误差 RMS。

        Args:
            skip: ⚠️ **跳过前多少秒**。参考轨迹在 $t=0$ 往往速度非零，
                而手臂静止，这个启动瞬态会淹没稳态差异。
                armctrl 迁移时踩过这个坑——不跳瞬态，PD 和 CTC 看起来一样好。
        """
        e = self.error
        if skip > 0:
            e = e[int(skip / self.dt):]
        return float(np.sqrt(np.mean(e ** 2)))

    def max_abs_torque(self) -> float:
        return float(np.abs(self.tau).max())

    def saturation_pct(self, tau_limit, skip: float = 0.0) -> float:
        """力矩触到限幅的时间比例。

        Args:
            skip: 跳过前多少秒。⚠️ 与 :meth:`rms` 用同一个 ``skip``——
                启动瞬态几乎必然会顶一下限幅（参考轨迹 $t=0$ 速度非零而手臂静止），
                `实测` PD kp=20 全程饱和 0.0167%（12000 步里的 2 步），
                跳掉 1 s 之后是 0。**比较稳态性能时必须跳，否则"是否饱和"
                这个前提判据本身就不干净。**

        ⚠️ 大于 0 就意味着**实际力矩 ≠ 指令力矩**，
        这段数据不能用于任何依赖 $\\tau$ 的推断（辨识、能耗、模型验证）。
        """
        tau = self.tau[int(skip / self.dt):] if skip > 0 else self.tau
        hit = np.abs(tau) >= np.asarray(tau_limit)[None, :] - 1e-9
        return float(hit.mean() * 100.0)

    def overshoot(self, target, index: int = 0) -> float:
        """阶跃响应超调量（相对阶跃幅度的百分比）。"""
        q0 = self.q[0, index]
        target = float(np.atleast_1d(target)[index])
        span = target - q0
        if abs(span) < 1e-12:
            return 0.0
        peak = (self.q[:, index].max() if span > 0 else self.q[:, index].min())
        return float((peak - target) / span * 100.0)

    def settling_time(self, target, tol: float = 0.02, index: int = 0) -> float:
        """进入并保持在 ±tol×阶跃幅度内所需的时间。没进去返回 inf。"""
        q0 = self.q[0, index]
        target = float(np.atleast_1d(target)[index])
        band = abs(target - q0) * tol
        if band < 1e-12:
            return 0.0
        outside = np.abs(self.q[:, index] - target) > band
        if not outside.any():
            return 0.0
        last = int(np.nonzero(outside)[0][-1])
        if last >= len(self.t) - 1:
            return float("inf")
        return float(self.t[last + 1])

    def diverged(self, q_bound: float = 1e3) -> bool:
        """闭环是否发散。NaN 也算。"""
        return bool(not np.isfinite(self.q).all()
                    or np.abs(self.q).max() > q_bound)


def rollout(backend, controller_fn, reference_fn, duration: float,
            q0, extra_fn=None) -> RolloutLog:
    """跑一段闭环。

    Args:
        backend: 实现了 ``read`` / ``send_torque`` / ``step`` / ``saturate``
            的后端。
        controller_fn: ``f(t, q, qd, ref) -> tau``。
        reference_fn: ``f(t) -> (q_des, v_des, a_des)``。
        extra_fn: 可选，``f(t, q, qd, tau) -> dict``，把额外读数记进日志。
            ⭐ 用它来**直接暴露物理量**（‖J̇q̇‖、观测器残差 r），
            而不是只看一个代理量——代理量会把 5% 的差异吃掉。
            这是 armctrl 元教训 #10 的直接对策。
    """
    backend.reset(np.asarray(q0, dtype=float))
    dt = backend.dt
    n_steps = int(round(duration / dt))
    ts, qs, qds, taus, q_dess = [], [], [], [], []
    extras: dict[str, list] = {}

    for k in range(n_steps):
        t = k * dt
        state = backend.read()
        q, qd = state.q.copy(), state.qd.copy()
        ref = reference_fn(t)

        tau = np.asarray(controller_fn(t, q, qd, ref), dtype=float)
        if not np.isfinite(tau).all():
            tau = np.zeros_like(tau)      # 已经发散，别再往里灌 NaN
        tau = backend.saturate(tau)       # ⚠️ 限幅永远打开

        if extra_fn is not None:
            for key, val in extra_fn(t, q, qd, tau).items():
                extras.setdefault(key, []).append(val)

        backend.send_torque(tau)
        backend.step()

        ts.append(t)
        qs.append(q)
        qds.append(qd)
        taus.append(tau)
        q_dess.append(np.asarray(ref[0], dtype=float))

    return RolloutLog(
        t=np.array(ts), q=np.array(qs), qd=np.array(qds),
        tau=np.array(taus), q_des=np.array(q_dess), dt=dt,
        extra={k: np.array(v) for k, v in extras.items()})


# ---------------------------------------------------------------- 参考轨迹

def hold_reference(q_hold):
    """恒定参考。用于重力补偿、观测器等"不该动"的场景。"""
    q_hold = np.asarray(q_hold, dtype=float)
    zero = np.zeros_like(q_hold)

    def ref(t):
        return (q_hold, zero, zero)
    return ref


def step_reference(q_start, q_end, t_step: float = 0.5):
    """阶跃参考。⭐ 用来看超调和调节时间，是最直接的性能读数。"""
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    zero = np.zeros_like(q_start)

    def ref(t):
        return ((q_end if t >= t_step else q_start), zero, zero)
    return ref


def sine_reference(q0, amp, freq: float):
    """正弦参考，解析给出速度和加速度。

    ⚠️ $t=0$ 时 $\\dot q_d = A\\omega \\ne 0$ 而手臂静止——
    这就是"启动瞬态"的来源。用 :meth:`RolloutLog.rms` 的 ``skip`` 跳掉。
    """
    q0 = np.asarray(q0, dtype=float)
    amp = np.asarray(amp, dtype=float)
    w = 2.0 * np.pi * freq

    def ref(t):
        return (q0 + amp * np.sin(w * t),
                amp * w * np.cos(w * t),
                -amp * w * w * np.sin(w * t))
    return ref


def pd_controller(kp, kd, gravity_fn=None):
    """基线 PD（可选重力补偿）。CTC 的对照组。

    ⚠️ **没有重力补偿的 PD 会有稳态误差**——重力是常值扰动，
    纯 PD 抵抗不了，误差稳定在 $e_{ss}=g(q)/K_p$。
    做对照时必须说清楚比的是哪一种，否则 CTC 的"优势"里
    有一大半只是重力补偿的功劳。
    """
    kp = np.atleast_1d(np.asarray(kp, dtype=float))
    kd = np.atleast_1d(np.asarray(kd, dtype=float))

    def ctrl(t, q, qd, ref):
        q_des, v_des, _ = ref
        tau = kp * (q_des - q) + kd * (v_des - qd)
        if gravity_fn is not None:
            tau = tau + gravity_fn(q)
        return tau
    return ctrl
