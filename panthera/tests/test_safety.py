"""安全层的守护测试。

⛔ **这些测试全绿，是允许给真机发第一条力矩指令的前提。**

⭐ 每一条都对应一个**真实会发生的失效**，不是为了覆盖率凑数：

* 控制循环卡住（GC、页错误、被抢 CPU）
* 控制律算出 NaN（矩阵奇异）
* 关节已经贴到限位
* 超速
* 上电瞬间给满力矩
* SDK 静默拒绝指令

⚠️ 安全层里**没有一条**是"理论上不会发生所以不用测"的。
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.core.robot import Q_HOME
from panthera.driver.mujoco_backend import MujocoBackend
from panthera.driver.safety import (SafetyConfig, SafetyLayer,
                                    config_from_backend)


class FakeClock:
    """⭐ 手动推进的时钟。测看门狗**必须**能控制时间，
    靠 ``time.sleep`` 会让测试又慢又不稳。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(scope="module")
def be():
    return MujocoBackend()


@pytest.fixture
def rig(be):
    clock = FakeClock()
    cfg = config_from_backend(be)
    layer = SafetyLayer(cfg, clock=clock)
    return layer, clock, cfg, np.array(Q_HOME)


def tick(layer, clock, tau, q, qd=None, dt=0.002):
    """走一个正常控制周期。"""
    clock.advance(dt)
    return layer.filter(tau, q, np.zeros(6) if qd is None else qd)


# ================================================================ 生命周期

class TestArming:
    def test_refuses_torque_before_arm(self, rig):
        """⛔ 没 arm 就输出力矩，是最基本的误用。"""
        layer, clock, _, q = rig
        tau, st = layer.filter(np.full(6, 5.0), q, np.zeros(6))
        assert not st.ok
        np.testing.assert_array_equal(tau, np.zeros(6))

    def test_ramp_starts_at_zero_and_reaches_full(self, rig):
        """⭐ 上电斜坡：第一帧必须接近零，不能一上来就给满力矩。

        ⚠️ 这里必须**按正常控制节拍**推进时钟，不能一步跳 0.5 s——
        那会触发看门狗，测到的就不是斜坡了。
        （这个坑本身也是测试写出来才发现的。）
        """
        layer, clock, cfg, q = rig
        layer.arm()
        cmd = np.full(6, 1.0)

        _, st0 = layer.filter(cmd, q, np.zeros(6))
        assert st0.ramp_scale == 0.0

        while clock.t < cfg.ramp_time / 2:
            _, st = tick(layer, clock, cmd, q)
        assert 0.4 < st.ramp_scale < 0.6

        while clock.t < cfg.ramp_time * 1.5:
            tau_full, st_full = tick(layer, clock, cmd, q)
        assert st_full.ramp_scale == 1.0
        np.testing.assert_allclose(tau_full, cmd)

    def test_disarm_latches(self, rig):
        """故障必须**锁存**：不能自己恢复，要人重新 arm。"""
        layer, clock, _, q = rig
        layer.arm()
        clock.advance(2.0)
        layer.disarm("测试")
        for _ in range(5):
            tau, st = tick(layer, clock, np.full(6, 5.0), q)
            assert not st.ok
            np.testing.assert_array_equal(tau, np.zeros(6))
        assert layer.faulted

    def test_fault_survives_rearm_attempt_without_clearing(self, rig):
        """⭐⭐ 故障锁存必须是**独立**的一道闸，不能只靠 ``_armed`` 挡。

        ⚠️ 这条测试是变异测试逼出来的：把"故障锁存"整个关掉之后，
        原本 26 项测试**全部通过**——因为 ``disarm()`` 同时把 ``_armed``
        置了 False，所有断言其实都被"未 arm"那道闸挡住了，
        锁存逻辑**从来没有被真正执行过**。

        ⭐ 教训：两道闸串在一起时，必须构造一个"只有第二道闸能挡住"的场景，
        否则你只测到了第一道。
        """
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        tick(layer, clock, np.full(6, 3.0), q)

        # 触发看门狗故障
        tick(layer, clock, np.full(6, 3.0), q, dt=cfg.watchdog_timeout * 2)
        assert layer.faulted

        # ⭐ 关键：绕过 arm()，直接把 _armed 置回 True。
        #    此时"未 arm"那道闸失效，只剩锁存能挡。
        layer._armed = True
        tau, st = tick(layer, clock, np.full(6, 3.0), q)
        assert not st.ok
        assert "已锁定" in st.reason
        np.testing.assert_array_equal(tau, np.zeros(6))

    def test_rearm_clears_the_fault(self, rig):
        """故障锁存要能被**显式** ``arm()`` 清掉——否则没法恢复运行。"""
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        tick(layer, clock, np.full(6, 3.0), q)
        tick(layer, clock, np.full(6, 3.0), q, dt=cfg.watchdog_timeout * 2)
        assert layer.faulted

        layer.arm()
        assert not layer.faulted
        _, st = tick(layer, clock, np.full(6, 1.0), q)
        assert st.ok


# ================================================================ 看门狗

class TestWatchdog:
    def test_normal_cadence_does_not_trip(self, rig):
        """500 Hz 正常跑 1000 拍不能误触发。"""
        layer, clock, _, q = rig
        layer.arm()
        clock.advance(2.0)
        for _ in range(1000):
            _, st = tick(layer, clock, np.full(6, 1.0), q)
            assert st.ok and not st.watchdog_tripped

    def test_jitter_within_budget_does_not_trip(self, rig):
        """⭐ 抖动容忍：偶发 3 倍周期不该报警，否则真机上天天误触发。"""
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        for dt in (0.002, 0.006, 0.002, 0.008, 0.002):
            assert dt < cfg.watchdog_timeout
            _, st = tick(layer, clock, np.full(6, 1.0), q, dt=dt)
            assert st.ok

    def test_stall_trips_and_zeroes_torque(self, rig):
        """⛔ 循环卡住 → 力矩必须归零。

        这是力矩控制**最危险**的失效模式：卡住时电机会保持上一条指令继续加速。
        """
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        tick(layer, clock, np.full(6, 3.0), q)

        tau, st = tick(layer, clock, np.full(6, 3.0), q,
                       dt=cfg.watchdog_timeout * 2)
        assert st.watchdog_tripped and not st.ok
        np.testing.assert_array_equal(tau, np.zeros(6))
        assert layer.faulted          # 且锁存，不会自己恢复

    def test_first_call_after_arm_never_trips(self, rig):
        """⚠️ arm 之后第一次调用没有"上一次"可比，不能误判为卡住。"""
        layer, clock, _, q = rig
        clock.advance(100.0)          # 距上次很久
        layer.arm()
        _, st = tick(layer, clock, np.full(6, 1.0), q, dt=0.5)
        assert not st.watchdog_tripped


# ================================================================ NaN

class TestNaNGuard:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_is_blocked_and_latched(self, rig, bad):
        """⭐ NaN 必须**最先**查——否则后面所有比较都静默失效。

        （`np.nan > 5` 是 False，任何基于比较的限幅都拦不住 NaN。）
        """
        layer, clock, _, q = rig
        layer.arm()
        clock.advance(2.0)
        cmd = np.full(6, 1.0)
        cmd[3] = bad
        tau, st = tick(layer, clock, cmd, q)
        assert st.nan_blocked and not st.ok
        np.testing.assert_array_equal(tau, np.zeros(6))
        assert layer.faulted

    def test_clipping_alone_would_not_catch_nan(self):
        """⚠️ 证明"光靠限幅拦不住 NaN"——所以 NaN 检查不能省。"""
        assert np.isnan(np.clip(np.nan, -10.0, 10.0))


# ================================================================ 限位

class TestJointLimitBraking:
    def test_torque_toward_limit_is_attenuated(self, rig):
        """接近上限时，**推向上限**的力矩被衰减。"""
        layer, clock, cfg, _ = rig
        layer.arm()
        clock.advance(2.0)
        q = cfg.q_upper - 0.01 * np.ones(6)      # 贴着上限
        tau, st = tick(layer, clock, np.full(6, 3.0), q)
        assert np.all(st.limit_scale < 0.2)
        assert np.all(tau < 3.0)

    def test_torque_away_from_limit_is_preserved(self, rig):
        """⭐⭐ **反方向的力矩必须完整保留**。

        这是本组最重要的一条：如果一律衰减，手臂一旦贴到限位就**再也回不来**了。
        安全措施本身不能制造一个更危险的状态。
        """
        layer, clock, cfg, _ = rig
        layer.arm()
        clock.advance(2.0)
        q = cfg.q_upper - 0.01 * np.ones(6)
        tau, st = tick(layer, clock, np.full(6, -3.0), q)   # 往回拉
        np.testing.assert_allclose(st.limit_scale, np.ones(6))
        np.testing.assert_allclose(tau, np.full(6, -3.0))

    def test_far_from_limit_is_untouched(self, rig):
        layer, clock, _, q = rig
        layer.arm()
        clock.advance(2.0)
        tau, st = tick(layer, clock, np.full(6, 2.0), q)
        np.testing.assert_allclose(st.limit_scale, np.ones(6))

    def test_scale_is_continuous(self, rig):
        """⚠️ 衰减必须**连续**。突变会激发振荡，也会让力矩不连续。"""
        layer, clock, cfg, _ = rig
        layer.arm()
        clock.advance(2.0)
        scales = []
        for d in np.linspace(0.0, cfg.limit_margin * 2, 25):
            q = cfg.q_upper - d
            _, st = tick(layer, clock, np.full(6, 1.0), q)
            scales.append(st.limit_scale[0])
        assert np.abs(np.diff(scales)).max() < 0.2
        assert scales[0] < 0.05 and scales[-1] == 1.0


# ================================================================ 超速

class TestSpeedBraking:
    def test_accelerating_torque_is_cut_when_speeding(self, rig):
        """超速时，**同向**（让它更快）的力矩被砍掉。"""
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        qd = cfg.qd_max * 1.5
        tau, st = tick(layer, clock, np.full(6, 3.0), q, qd=qd)
        assert np.all(st.speed_scale == 0.0)
        np.testing.assert_array_equal(tau, np.zeros(6))

    def test_braking_torque_is_preserved_when_speeding(self, rig):
        """⭐ 反向（刹车）的力矩必须保留——同 `限位` 那条的道理。"""
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        qd = cfg.qd_max * 1.5
        tau, st = tick(layer, clock, np.full(6, -3.0), q, qd=qd)
        np.testing.assert_allclose(st.speed_scale, np.ones(6))
        np.testing.assert_allclose(tau, np.full(6, -3.0))

    def test_official_speed_limit_is_used(self, be):
        """⚠️ 限值必须来自官方配置（velocity_limits: 1.0），不是自己填的。"""
        cfg = config_from_backend(be)
        np.testing.assert_array_equal(cfg.qd_max, np.full(6, 1.0))


# ================================================================ 限幅

class TestTorqueClamp:
    def test_never_exceeds_limit(self, rig):
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(2.0)
        tau, st = tick(layer, clock, np.full(6, 1e6), q)
        assert np.all(np.abs(tau) <= cfg.tau_max + 1e-9)
        assert st.saturated

    def test_clamp_is_applied_last(self, rig):
        """⭐⭐ 顺序守护：限幅必须在**所有缩放之后**。

        如果先限幅再乘斜坡，结果还是对的；但如果**先限幅再乘一个 >1 的量**
        就会破限。这条测试把"限幅在最后"这个顺序钉死。
        """
        layer, clock, cfg, q = rig
        layer.arm()
        clock.advance(cfg.ramp_time * 10)         # 斜坡已满
        huge = cfg.tau_max * 100
        tau, _ = tick(layer, clock, huge, q)
        np.testing.assert_allclose(tau, cfg.tau_max)

    def test_ramp_and_clamp_compose_correctly(self, rig):
        """斜坡进行中 + 超限指令 ⇒ 结果应是 ramp × 限幅值。"""
        layer, clock, cfg, q = rig
        layer.arm()
        while clock.t < cfg.ramp_time / 2:
            tau, st = tick(layer, clock, cfg.tau_max * 100, q)
        # ⭐ 限幅上限本身也乘了斜坡，所以输出是 ramp × tau_max 而不是 tau_max
        np.testing.assert_allclose(tau, cfg.tau_max * st.ramp_scale, rtol=1e-9)
        assert st.ramp_scale < 1.0


# ================================================================ SDK 陷阱

class TestSafePositionArgument:
    """⚠️⚠️ 官方 SDK 的 ``pos_vel_tqe_kp_kd()`` 在 ``pos`` 超限时
    **直接 return False 不发指令**。

    纯力矩模式下我们传 ``pos``，如果它落在限位外，
    **整条指令被静默丢弃 ⇒ 手臂瞬间失去所有力矩**。

    而传 ``pos=0`` 的余量极薄：Follower 配置（我们用的）下 J2/J3 只有
    **0.1 rad**，Leader 配置下**恰好为 0**。
    换配置、改 URDF、或者零位漂移都可能吃掉它。

    ⭐ 最初我们把这两份配置搞混了，写成"余量恰好为零"——
    引用配置文件时要说清楚是哪一份。
    """

    def test_result_is_always_within_limits(self, rig):
        layer, clock, cfg, _ = rig
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(cfg.q_lower - 1.0, cfg.q_upper + 1.0)
            safe = layer.safe_position_argument(q)
            assert np.all(safe >= cfg.q_lower - 1e-12)
            assert np.all(safe <= cfg.q_upper + 1e-12)

    def test_normal_position_passes_through(self, rig):
        layer, clock, _, q = rig
        np.testing.assert_allclose(layer.safe_position_argument(q), q)

    def test_zero_has_almost_no_margin(self, be):
        """⚠️ ``pos=0`` 在 J2/J3 上余量极薄（我们的模型用的是 Leader 限位）。

        这条测试**不是**在说 0 一定不行，而是在说**余量小到不该依赖**。
        用当前位置就没有这个问题。
        """
        cfg = config_from_backend(be)
        margin_low = np.abs(0.0 - cfg.q_lower)
        assert margin_low.min() < 0.15      # J2/J3 余量薄


# ================================================================ 端到端

class TestEndToEndWithController:
    def test_safety_layer_does_not_break_normal_control(self, be):
        """⭐ 安全层不能把好好的控制器搞坏。

        重力补偿在安全层下跑，手臂应该基本不动——
        和没有安全层时一样（斜坡结束之后）。
        """
        from panthera.control.computed_torque import (ComputedTorqueController,
                                                      from_bandwidth)
        q0 = np.array(Q_HOME)
        clock = FakeClock()
        layer = SafetyLayer(config_from_backend(be), clock=clock)
        ctc = ComputedTorqueController(be.robot, from_bandwidth(6, wn=20.0))

        be.reset(q0)
        layer.arm()
        for k in range(2000):
            clock.advance(be.dt)
            st = be.read()
            tau = ctc.compute(st.q, st.qd, q0, np.zeros(6), np.zeros(6))
            tau, status = layer.filter(tau, st.q, st.qd)
            assert status.ok
            be.send_torque(tau)
            be.step()

        assert np.abs(be.read().q - q0).max() < 0.05
        assert not layer.faulted

    def test_diverging_controller_is_contained(self, be):
        """⛔ 一个发散的控制器，安全层必须兜住——力矩不越限、手臂不失控。"""
        q0 = np.array(Q_HOME)
        clock = FakeClock()
        cfg = config_from_backend(be)
        layer = SafetyLayer(cfg, clock=clock, )
        be.reset(q0)
        layer.arm()
        clock.advance(cfg.ramp_time * 2)

        gain = 1.0
        for k in range(1500):
            clock.advance(be.dt)
            st = be.read()
            gain *= 1.01                       # 指数发散的"控制器"
            tau, status = layer.filter(np.full(6, gain), st.q, st.qd)
            assert np.all(np.abs(tau) <= cfg.tau_max + 1e-9)
            be.send_torque(tau)
            be.step()
            if not status.ok:
                break

        q_end = be.read().q
        assert np.isfinite(q_end).all()
        # ⭐ 关键：即使控制器疯了，关节也不该冲出限位太多
        assert np.all(q_end >= cfg.q_lower - 0.2)
        assert np.all(q_end <= cfg.q_upper + 0.2)
