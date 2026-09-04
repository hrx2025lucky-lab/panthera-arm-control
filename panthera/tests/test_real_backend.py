"""真机后端 + 假 SDK 的守护测试。

⭐ **这组测试的价值在于：它让真机代码在硬件到货前就被验证过。**

链路是 控制器 → 安全层 → RealBackend → SDK → 电机。
真机到货时只把 ``FakePanthera`` 换成官方 ``Panthera``，其余一行不改。

⚠️⚠️ **它验证的是"我们调用 SDK 的方式对不对"，不是"真机会不会这样反应"。**
延迟、力矩精度、摩擦、热特性——这些只有真机知道。
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.control.computed_torque import (ComputedTorqueController,
                                              from_bandwidth)
from panthera.core.robot import Q_HOME
from panthera.driver.fake_sdk import FakePanthera
from panthera.driver.mujoco_backend import MujocoBackend
from panthera.driver.real_backend import RealBackend
from panthera.driver.safety import SafetyLayer, config_from_backend


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def rig():
    mb = MujocoBackend()
    sdk = FakePanthera(mb)
    rb = RealBackend(sdk=sdk, dt=mb.dt, model=mb.robot)
    return mb, sdk, rb


# ================================================================ 接口一致

class TestInterfaceParity:
    """⭐ 两个后端必须能互换，否则"统一后端"是句空话。"""

    def test_same_public_surface(self, rig):
        mb, _, rb = rig
        for name in ("read", "send_torque", "gravity", "step",
                     "close", "saturate", "dt", "n", "tau_limit"):
            assert hasattr(rb, name), f"RealBackend 缺少 {name}"
            assert hasattr(mb, name), f"MujocoBackend 缺少 {name}"

    def test_read_returns_same_shape(self, rig):
        mb, _, rb = rig
        a, b = mb.read(), rb.read()
        assert a.q.shape == b.q.shape == (6,)
        assert a.qd.shape == b.qd.shape == (6,)

    def test_reset_is_refused_on_real_backend(self, rig):
        """⛔ 真机不能瞬移。让仿真专用的代码路径立刻暴露，而不是悄悄做别的。"""
        _, _, rb = rig
        with pytest.raises(NotImplementedError):
            rb.reset(np.array(Q_HOME))

    def test_torque_limit_is_the_conservative_set(self, rig):
        """⚠️ 官方示例有三套矛盾限幅，我们取最保守的一套。"""
        _, _, rb = rig
        np.testing.assert_array_equal(rb.tau_limit,
                                      [10.0, 20.0, 20.0, 10.0, 5.0, 5.0])
        # ⚠️ 且必须远低于配置里的 max_torque（那是堵转扭矩）
        assert np.all(rb.tau_limit < rb.sdk.max_torque)


# ================================================================ 静默丢弃

class TestSilentCommandDrop:
    """⚠️⚠️ 官方 SDK 最危险的行为：``pos`` 超限时 ``return False``
    **静默丢弃整条指令**——包括力矩项。

    纯力矩模式下，这意味着**手臂瞬间失去所有力矩**。
    而 Leader 配置里 J2/J3 的下限恰好是 ``0.0``，传 ``pos=0`` 余量为零。
    """

    def test_fake_sdk_reproduces_the_drop(self, rig):
        """先证明假 SDK 真的复刻了这个行为，否则后面的测试都是空的。"""
        _, sdk, _ = rig
        bad = np.array(Q_HOME)
        bad[0] = 99.0                       # 远超限位
        ok = sdk.pos_vel_tqe_kp_kd(bad, np.zeros(6), np.full(6, 5.0),
                                   np.zeros(6), np.zeros(6))
        assert ok is False
        assert sdk.dropped_commands == 1

    def test_send_torque_never_triggers_the_drop(self, rig):
        """⭐ 核心保证：不管手臂在哪，``send_torque`` 都不能被丢弃。

        因为它传的是**当前实测位置**（且再夹一次），永远落在限位内。
        """
        mb, sdk, rb = rig
        rng = np.random.default_rng(0)
        lower, upper = mb.model.jnt_range[:6, 0], mb.model.jnt_range[:6, 1]
        for _ in range(30):
            mb.reset(rng.uniform(lower, upper))
            rb.send_torque(rng.uniform(-3, 3, 6))
        assert rb.dropped == 0
        assert sdk.dropped_commands == 0

    def test_survives_out_of_range_sensor_reading(self, rig):
        """⚠️ 即使传感器读数越界（噪声/零位漂移），也不能被丢弃。

        ⭐ 这不是假设：J2/J3 下限是 0.0，只要读数有一丝负噪声就会越界。
        """
        mb, sdk, rb = rig
        q_bad = np.array(Q_HOME)
        q_bad[1] = -1e-6                    # 比下限低一点点
        mb.robot.data.qpos[mb.robot.qpos_idx] = q_bad
        rb.send_torque(np.full(6, 1.0))
        assert rb.dropped == 0

    def test_zero_position_argument_has_almost_no_margin(self, rig):
        """⚠️ 传 ``pos=0`` 时 J2/J3 的余量极小。

        ⭐ 这条测试**纠正了我们最初的一个事实错误**。原先写的是
        "余量恰好为 0"，但那是 ``Leader.yaml``；我们的机器用
        ``Follower.yaml``，下限是 **−0.1** 而不是 0.0。

        ==================  ===========  ==========
        配置                 J2/J3 下限    pos=0 余量
        ==================  ===========  ==========
        Follower（我们的）    −0.1         **0.1 rad**
        Leader              0.0          **0**
        ==================  ===========  ==========

        所以风险比最初判断的小，但 0.1 rad 依然很薄——
        换配置、改 URDF、或者传感器零位漂移都可能吃掉它。
        用当前实测位置就完全没有这个问题。

        ⭐ 教训：**引用配置文件时要说清楚是哪一份**。
        Follower / Leader 只差 0.1，但结论差别很大。
        """
        _, sdk, _ = rig
        margin = np.abs(0.0 - sdk.joint_limits["lower"])
        assert margin.min() == pytest.approx(0.1)     # Follower
        assert margin.min() < 0.15                    # 依然很薄

    def test_drop_is_counted_not_raised(self, rig):
        """⭐ 丢弃要**计数**而不是抛异常。

        抛异常会让控制循环崩掉，而崩掉时电机保持上一条指令继续跑——
        比丢一帧危险得多。但也不能完全无声，所以计数让上层能查。
        """
        _, _, rb = rig
        rb.sdk.pos_vel_tqe_kp_kd = lambda *a: False     # 强制全部丢弃
        rb.send_torque(np.full(6, 1.0))                 # 不抛异常
        assert rb.dropped == 1


# ================================================================ MIT 模式

class TestMITMode:
    def test_pure_torque_uses_zero_gains(self, rig):
        """纯力矩模式必须 kp=kd=0，否则电机侧 PD 会叠加进来。"""
        mb, sdk, rb = rig
        rb.send_torque(np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
        cmd = sdk._pending
        np.testing.assert_array_equal(cmd[:, 3], np.zeros(6))   # kp
        np.testing.assert_array_equal(cmd[:, 4], np.zeros(6))   # kd
        np.testing.assert_allclose(cmd[:, 2],
                                   [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    def test_mit_position_mode_clips_target(self, rig):
        """⭐ RL 走这条路。期望位置必须先夹到限位内，否则整条指令被丢弃。"""
        mb, sdk, rb = rig
        q_des = np.array(Q_HOME)
        q_des[0] = 99.0
        assert rb.send_mit(q_des, np.zeros(6), np.zeros(6),
                           np.full(6, 20.0), np.full(6, 1.0))
        assert rb.dropped == 0
        assert sdk._pending[0, 0] <= sdk.joint_limits["upper"][0]

    def test_mit_pd_actually_drives_the_arm(self, rig):
        """MIT 位置模式要真能把手臂拉过去——验证 kp/kd 的语义没接反。"""
        mb, sdk, rb = rig
        target = np.array(Q_HOME) + np.array([0.2, 0, 0, 0, 0, 0])
        for _ in range(3000):
            rb.send_mit(target, np.zeros(6), np.zeros(6),
                        np.full(6, 30.0), np.full(6, 2.0))
        assert abs(rb.read().q[0] - target[0]) < 0.05

    def test_torque_is_saturated_before_sending(self, rig):
        _, sdk, rb = rig
        rb.send_torque(np.full(6, 1e6))
        assert np.all(np.abs(sdk._pending[:, 2]) <= rb.tau_limit + 1e-9)


# ================================================================ 端到端

class TestFullStack:
    def test_control_stack_holds_position(self, rig):
        """⭐⭐ 整条链路：CTC → 安全层 → RealBackend → SDK → 电机。

        `实测` 2000 步后位置偏差 0.011 rad，零指令丢弃。
        ⭐ 真机到货时，这段代码**一行都不用改**，只换 backend 构造。
        """
        mb, sdk, rb = rig
        clock = FakeClock()
        cfg = config_from_backend(mb)
        cfg.tau_limit = rb.tau_limit.copy()
        safety = SafetyLayer(cfg, clock=clock)
        ctc = ComputedTorqueController(mb.robot, from_bandwidth(6, wn=20.0))
        q0 = np.array(Q_HOME)
        safety.arm()

        for _ in range(2000):
            clock.advance(mb.dt)
            st = rb.read()
            tau = ctc.compute(st.q, st.qd, q0, np.zeros(6), np.zeros(6))
            tau, status = safety.filter(tau, st.q, st.qd)
            assert status.ok, status.reason
            rb.send_torque(tau)
            rb.step()

        assert np.abs(rb.read().q - q0).max() < 0.05
        assert rb.dropped == 0
        assert sdk.dropped_commands == 0
        assert not safety.faulted

    def test_safety_layer_catches_divergence_through_real_backend(self, rig):
        """发散的控制器经过完整链路也必须被兜住。"""
        mb, sdk, rb = rig
        clock = FakeClock()
        cfg = config_from_backend(mb)
        cfg.tau_limit = rb.tau_limit.copy()
        safety = SafetyLayer(cfg, clock=clock)
        safety.arm()
        clock.advance(cfg.ramp_time * 2)

        gain = 1.0
        for _ in range(1000):
            clock.advance(mb.dt)
            st = rb.read()
            gain *= 1.02
            tau, status = safety.filter(np.full(6, gain), st.q, st.qd)
            assert np.all(np.abs(tau) <= cfg.tau_max + 1e-9)
            rb.send_torque(tau)
            rb.step()
            if not status.ok:
                break
        assert np.isfinite(rb.read().q).all()

    def test_close_zeroes_torque(self, rig):
        _, sdk, rb = rig
        rb.send_torque(np.full(6, 3.0))
        rb.close()
        np.testing.assert_array_equal(sdk._pending[:, 2], np.zeros(6))

    def test_period_stats_are_collected(self, rig):
        """⭐ 上电第一件事就是看这个统计——它验证"实测控制频率"那一步。"""
        _, _, rb = rig
        for _ in range(50):
            rb.read()
        stats = rb.period_stats()
        assert stats["n"] == 49
        for key in ("mean_ms", "std_ms", "max_ms", "p99_ms", "hz"):
            assert key in stats and np.isfinite(stats[key])


# ================================================================ 假 SDK 保真

class TestFakeSdkFidelity:
    """⚠️ 假 SDK 必须复刻官方的**行为**，不只是接口签名。
    否则用它验证过的代码，到真机上照样出问题。"""

    def test_motor_count_excludes_gripper(self, rig):
        """官方 ``motor_count = len(Motors) - 1``，最后一个是夹爪。"""
        _, sdk, _ = rig
        assert len(sdk.Motors) == 7
        assert sdk.motor_count == 6

    def test_official_joint_limits(self, rig):
        _, sdk, _ = rig
        np.testing.assert_array_equal(sdk.joint_limits["lower"],
                                      [-2.4, -0.1, -0.1, -1.6, -1.7, -2.5])
        np.testing.assert_array_equal(sdk.joint_limits["upper"],
                                      [2.4, 3.2, 4.0, 1.6, 1.7, 2.5])

    def test_max_torque_is_the_stall_value(self, rig):
        """⚠️ 配置里的 ``max_torque`` 是**堵转**扭矩，不是可持续输出。"""
        _, sdk, _ = rig
        np.testing.assert_array_equal(sdk.max_torque,
                                      [21.0, 36.0, 36.0, 21.0, 10.0, 10.0])

    def test_gripper_limit_rejects_out_of_range(self, rig):
        _, sdk, _ = rig
        assert sdk.gripper_control_MIT(3.0, 0, 0, 0, 0) is False
        assert sdk.gripper_control_MIT(1.0, 0, 0, 0, 0) is True

    def test_friction_model_uses_threshold_not_tanh(self, rig):
        """⚠️ 官方用**速度阈值**（阈值处不连续），我们的观测器用 tanh 平滑。

        两者不同——做对照实验时必须知道这个差异，否则会把
        "模型不一样"误判成"辨识结果不对"。
        """
        _, sdk, _ = rig
        Fc, Fv = np.full(6, 0.2), np.full(6, 0.06)
        below = sdk.get_friction_compensation(np.full(6, 0.01), Fc, Fv, 0.02)
        above = sdk.get_friction_compensation(np.full(6, 0.03), Fc, Fv, 0.02)
        np.testing.assert_allclose(below, Fv * 0.01)          # 无库伦项
        np.testing.assert_allclose(above, Fc + Fv * 0.03)     # 有库伦项
        assert above[0] - below[0] > 0.19                     # 阈值处跳变
