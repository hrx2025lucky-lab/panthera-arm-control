"""SpaceMouse 遥操作的守护测试。

⚠️ 遥操是**人在回路**且**直接驱动真机**的——它出 bug 的后果是手臂突然乱动。

⭐ 这组测试守的三件事：

1. 手不碰设备时，手臂**绝对不能动**
2. 推一个方向，末端**真的往那个方向走**（而不是"能动就行"）
3. ⚠️⚠️ 松手之后手臂**不能继续冲**（积分漂移）
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.core.robot import Q_HOME, make_panthera
from panthera.teleop.spacemouse import (CartesianTeleop, SpaceMouseInput,
                                        TeleopConfig)


@pytest.fixture(scope="module")
def robot():
    return make_panthera()


@pytest.fixture(scope="module")
def limits(robot):
    m = robot.model
    return m.jnt_range[:6, 0].copy(), m.jnt_range[:6, 1].copy()


@pytest.fixture
def teleop(robot, limits):
    tp = CartesianTeleop(robot, *limits)
    tp.reset(np.array(Q_HOME))
    return tp


def track(tp, v, q_start, steps=200, dt=0.005, follow=0.5):
    """跑一段遥操，``follow`` 模拟真机的跟踪滞后（1.0 = 完美跟随）。"""
    q = np.asarray(q_start, dtype=float).copy()
    for _ in range(steps):
        q_des = tp.step(v, q, dt)
        q = q + (q_des - q) * follow
    return q, q_des


# ================================================================ 输入处理

class TestSpaceMouseInput:
    def test_idle_produces_exactly_zero(self):
        """⭐ 最基本的安全属性：手不碰它，输出必须是**精确的零**。

        ⚠️ SpaceMouse 静止时原始读数**不是** 0（有零漂）。
        没有死区的话，手臂会自己缓慢移动——这在真机上非常吓人。
        """
        drift = np.array([0.03, -0.02, 0.05, -0.01, 0.04, 0.02])
        mouse = SpaceMouseInput(lambda: drift)
        for _ in range(50):
            v = mouse.read_velocity()
        assert np.abs(v).max() == 0.0

    def test_deadzone_is_continuous(self):
        """⚠️ 死区边缘不能跳变。

        简单的"小于死区就置零"会在边缘产生阶跃——
        手轻轻推过临界点，手臂会突然一顿。正确做法是把
        [dz, 1] 重新拉伸回 [0, 1]。
        """
        cfg = TeleopConfig()
        xs = np.linspace(0.0, 0.4, 200)
        ys = [SpaceMouseInput._deadzone(np.array([x]), cfg.deadzone)[0]
              for x in xs]
        assert np.abs(np.diff(ys)).max() < 0.02
        assert ys[0] == 0.0

    def test_full_deflection_reaches_scale(self):
        """推到底应该拿到设定的速度标度。"""
        cfg = TeleopConfig(smoothing=1.0)      # 关掉滤波看稳态
        mouse = SpaceMouseInput(lambda: np.ones(6), cfg)
        v = mouse.read_velocity()
        np.testing.assert_allclose(v[:3], cfg.trans_scale, rtol=1e-9)
        np.testing.assert_allclose(v[3:], cfg.rot_scale, rtol=1e-9)

    def test_smoothing_reduces_jitter(self):
        """低通必须真的在滤波——否则抖动会直接进关节指令。"""
        rng = np.random.default_rng(0)
        noisy = SpaceMouseInput(lambda: rng.uniform(-1, 1, 6),
                                TeleopConfig(deadzone=0.0, smoothing=0.1))
        raw_std = np.std([rng.uniform(-1, 1) for _ in range(300)])
        out = np.array([noisy.read_velocity()[0] for _ in range(300)])
        assert out[50:].std() < raw_std * TeleopConfig().trans_scale

    def test_rejects_wrong_size(self):
        with pytest.raises(ValueError):
            SpaceMouseInput(lambda: np.zeros(3)).read_velocity()


# ================================================================ 方向正确

class TestDirectionMapping:
    """⭐ "能动"不等于"动对了"。这组测的是**末端真的往指定方向走**。"""

    @pytest.mark.parametrize("axis,name", [(0, "X"), (1, "Y"), (2, "Z")])
    def test_translation_axis_is_pure(self, robot, teleop, axis, name):
        """推单一平移轴，末端位移应几乎全在那个轴上。"""
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        v = np.zeros(6)
        v[axis] = 0.08
        q_end, _ = track(teleop, v, q0)
        d = robot.tcp_position(q_end) - x0

        assert np.linalg.norm(d) > 0.02, f"{name} 方向几乎没动"
        purity = abs(d[axis]) / np.linalg.norm(d)
        assert purity > 0.9, f"{name} 方向纯度只有 {purity:.1%}"
        assert np.sign(d[axis]) > 0                  # 方向不能反

    def test_zero_input_zero_motion(self, robot, teleop):
        """⭐ 零输入必须**零位移**，不能有任何漂移。"""
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        q_end, _ = track(teleop, np.zeros(6), q0, steps=1000)
        assert np.linalg.norm(robot.tcp_position(q_end) - x0) < 1e-9


# ================================================================ anti-windup

class TestAntiWindup:
    """⚠️⚠️ 本文件最重要的一组。

    $q_{des}$ 靠积分累积，真机 $q$ 有跟踪误差。两者会越差越远——
    **手已经停了，手臂还在朝一个很远的目标冲**。

    `实测`（手臂完全卡住，持续推 3 秒）：

    ====================  ==================
    anti-windup           最终领先量
    ====================  ==================
    开启（max_lag=0.15）   0.15 rad（8.6°）
    **关闭**              **1.01 rad（57.7°）**
    ====================  ==================

    ⭐ 领先量就是**松手瞬间手臂还要走的距离**。57.7° 是会吓到人的。
    """

    def test_lag_is_bounded_when_arm_is_stuck(self, robot, limits):
        """手臂完全不动时，期望位置不能无限领先。"""
        cfg = TeleopConfig(max_lag=0.15)
        tp = CartesianTeleop(robot, *limits, cfg)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        for _ in range(600):                       # 3 秒
            q_des = tp.step(np.array([0.08, 0, 0, 0, 0, 0]), q0, 0.005)
        assert np.abs(q_des - q0).max() <= cfg.max_lag + 1e-9

    def test_without_antiwindup_it_runs_away(self, robot, limits):
        """⚠️ 反向守护：关掉 anti-windup 后领先量**必须**大很多。

        这条测试证明该机制真的在起作用，而不是"恰好没触发"。
        哪天它变绿了，说明有人把限幅或积分改坏了。
        """
        q0 = np.array(Q_HOME)
        v = np.array([0.08, 0, 0, 0, 0, 0])

        def run(max_lag):
            tp = CartesianTeleop(robot, *limits, TeleopConfig(max_lag=max_lag))
            tp.reset(q0)
            for _ in range(600):
                q_des = tp.step(v, q0, 0.005)
            return np.abs(q_des - q0).max()

        assert run(1e9) > 5 * run(0.15)            # `实测` 1.01 vs 0.15

    def test_windup_flag_is_reported(self, robot, limits):
        """⭐ 触发钳制时要**留痕**，否则遥操手感变差时查不出原因。"""
        tp = CartesianTeleop(robot, *limits)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        for _ in range(400):
            tp.step(np.array([0.08, 0, 0, 0, 0, 0]), q0, 0.005)
        assert tp.last["windup_clamped"] is True

    def test_recovers_after_arm_catches_up(self, robot, limits):
        """手臂跟上之后，钳制应自动解除。"""
        tp = CartesianTeleop(robot, *limits)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        for _ in range(400):
            tp.step(np.array([0.08, 0, 0, 0, 0, 0]), q0, 0.005)
        q_now = tp.q_des.copy()
        tp.step(np.zeros(6), q_now, 0.005)        # 手臂跟上了，且松手
        assert tp.last["windup_clamped"] is False


# ================================================================ 安全边界

class TestSafetyBounds:
    def test_never_exceeds_joint_limits(self, robot, limits):
        """⚠️ 必须夹紧——SDK 在 pos 超限时会**静默丢弃整条指令**，
        纯力矩模式下等于瞬间失去所有力矩。"""
        lower, upper = limits
        tp = CartesianTeleop(robot, *limits)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        q = q0.copy()
        for _ in range(3000):                      # 15 秒持续下压
            q_des = tp.step(np.array([0, 0, -0.08, 0, 0, 0]), q, 0.005)
            assert np.all(q_des >= lower - 1e-9)
            assert np.all(q_des <= upper + 1e-9)
            q = q + (q_des - q) * 0.5

    def test_joint_speed_is_capped(self, robot, limits):
        """速度上限必须生效——官方 ``velocity_limits`` 就是 1.0 rad/s。"""
        cfg = TeleopConfig(qd_max=1.0, trans_scale=5.0)   # 故意给大输入
        tp = CartesianTeleop(robot, *limits, cfg)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        for _ in range(100):
            tp.step(np.array([5.0, 0, 0, 0, 0, 0]), q0, 0.005)
            assert tp.last["qd_peak"] <= cfg.qd_max + 1e-9

    def test_speed_cap_preserves_direction(self, robot, limits):
        """⭐ 限速要按最大分量**整体缩放**，不能逐关节 clip。

        逐关节 clip 会改变末端运动方向——手往前推，手臂却斜着走。
        """
        q0 = np.array(Q_HOME)
        v = np.array([0.08, 0.04, 0, 0, 0, 0])

        slow = CartesianTeleop(robot, *limits, TeleopConfig(qd_max=10.0))
        fast = CartesianTeleop(robot, *limits, TeleopConfig(qd_max=0.05))
        slow.reset(q0)
        fast.reset(q0)
        d_slow = slow.step(v, q0, 0.005) - q0
        d_fast = fast.step(v, q0, 0.005) - q0
        # 方向（单位向量）应一致，只是长度不同
        u1 = d_slow / np.linalg.norm(d_slow)
        u2 = d_fast / np.linalg.norm(d_fast)
        np.testing.assert_allclose(u1, u2, atol=1e-6)

    def test_reset_aligns_to_measured(self, robot, limits):
        """⚠️ 不 reset 就 step，手臂会朝上次的残留目标冲。"""
        tp = CartesianTeleop(robot, *limits)
        q_odd = np.array(Q_HOME) + 0.3
        tp.reset(q_odd)
        np.testing.assert_allclose(tp.q_des, np.clip(q_odd, *limits))

    def test_singularity_does_not_explode(self, robot, limits):
        """⭐ **真正**奇异的构型上，DLS 必须兜住。

        ⚠️ 最初这条测试用 ``q = zeros(6)``，以为"伸直=接近奇异"——
        `实测` 那个构型的最小奇异值是 **4e-2**，一点都不奇异，
        所以把 DLS 换成普通求逆测试照样全绿。

        下面这个构型是随机搜出来的，最小奇异值 **1.1e-05**（小 4000 倍）。

        ⭐ 教训：**"接近奇异"要用数字确认，不能靠直觉。**
        """
        q_sing = np.array([0.389, 1.969, 3.941, -0.743, -1.328, -0.605])
        sv = np.linalg.svd(robot.jacobian(q_sing), compute_uv=False)
        assert sv[-1] < 1e-4, "这个构型不够奇异，测试没有分辨力"

        tp = CartesianTeleop(robot, *limits)
        tp.reset(q_sing)
        for _ in range(50):
            tp.step(np.array([0.08, 0, 0, 0, 0, 0]), q_sing, 0.005)
            assert np.isfinite(tp.q_des).all()
            assert tp.last["qd_peak"] <= TeleopConfig().qd_max + 1e-9

    def test_damping_actually_attenuates_at_singularity(self, robot, limits):
        """⭐ 直接测 DLS 本身，而不是绕一圈测系统行为。

        ⚠️ 最初这条想通过"输出速度不爆"来间接验证 DLS——但**测不到**：
        速度限幅早就把幅值兜住了。也就是说那个判据测的是限幅，不是 DLS。

        ⭐ DLS 的真正作用是**在奇异方向上主动衰减**，而不是"防止溢出"。
        所以直接比较：同一个奇异构型下，有阻尼 vs 无阻尼的输出范数。

        `实测` 该构型最小奇异值 1.1e-05 ⇒ 无阻尼时增益约 1e5 倍。
        """
        q_sing = np.array([0.389, 1.969, 3.941, -0.743, -1.328, -0.605])
        J = robot.jacobian(q_sing)
        v = np.array([0.08, 0, 0, 0, 0, 0])

        damped = CartesianTeleop(robot, *limits, TeleopConfig(damping=0.05))
        raw = np.linalg.inv(J) @ v
        out = damped._dls_pinv(J) @ v
        assert np.linalg.norm(out) < 0.1 * np.linalg.norm(raw)


# ================================================================ 接口对齐

class TestLeRobotCompatibility:
    """⭐⭐ 输出的动作空间必须和 LeRobot / RL 一致，
    否则采的数据训不了、训的策略部署不了。"""

    def test_output_is_joint_positions(self, robot, teleop):
        """LeRobot 的 ``action_features`` 是 ``{joint1.pos, ...}``，
        RL 的动作也是关节位置目标 —— 我们输出同一个东西。"""
        q_des = teleop.step(np.zeros(6), np.array(Q_HOME), 0.005)
        assert q_des.shape == (6,)
        assert np.isfinite(q_des).all()

    def test_can_be_serialized_to_lerobot_dict(self, robot, teleop):
        """能直接转成 LeRobot 的 action dict。"""
        q_des = teleop.step(np.zeros(6), np.array(Q_HOME), 0.005)
        action = {f"joint{i + 1}.pos": float(v) for i, v in enumerate(q_des)}
        assert len(action) == 6
        assert all(isinstance(v, float) for v in action.values())


class TestSecondLineOfDefence:
    """⭐⭐ 变异测试逼出来的一组：**两道闸串在一起时，只测到第一道**。

    最初 8 个变异里有 4 个存活，全是这个原因：

    ================================  ==========================
    变异                               为什么原来的测试抓不到
    ================================  ==========================
    去掉关节限位 clip                  ``_limit_scale`` 已把速度降到 0
    雅可比用 q_des 而非实测            正常工况下两者几乎一样
    限位减速不看方向                    没测过"从限位往回走"
    ================================  ==========================

    ⚠️ 这和 :mod:`panthera.tests.test_safety` 里故障锁存那条是同一个病。
    必须**主动构造只有第二道闸能挡住的场景**。
    """

    def test_clip_catches_what_speed_scaling_misses(self, robot, limits):
        """⭐ 把限位减速关掉（margin=0），此时只剩 clip 能挡。"""
        lower, upper = limits
        cfg = TeleopConfig(limit_margin=1e-9)      # 几乎关掉减速
        tp = CartesianTeleop(robot, *limits, cfg)
        q0 = np.array(Q_HOME)
        tp.reset(q0)
        q = q0.copy()
        for _ in range(2000):
            q_des = tp.step(np.array([0, 0, -0.08, 0, 0, 0]), q, 0.005)
            assert np.all(q_des >= lower - 1e-9)
            assert np.all(q_des <= upper + 1e-9)
            q = q + (q_des - q) * 0.5

    def test_reset_clamps_an_out_of_range_start(self, robot, limits):
        """⚠️ 起始位置就越界时（传感器野值/零位漂移），也不能输出越界指令。"""
        lower, upper = limits
        tp = CartesianTeleop(robot, *limits)
        tp.reset(upper + 1.0)                      # 全部超上限
        assert np.all(tp.q_des <= upper + 1e-9)
        q_des = tp.step(np.zeros(6), upper + 1.0, 0.005)
        assert np.all(q_des <= upper + 1e-9)

    def test_jacobian_uses_measured_not_commanded(self, robot, limits):
        """⭐ 雅可比必须在**实测**构型上算。

        $q_{des}$ 可能领先实测很多（anti-windup 允许 0.15 rad）。
        用领先的构型算映射，等于用一个手臂**还没到达**的姿态
        去解算当下该怎么动——方向会偏。

        构造：让 q_des 明显领先，然后比较两种算法给出的关节速度。
        """
        q0 = np.array(Q_HOME)
        v = np.array([0.08, 0, 0, 0, 0, 0])
        tp = CartesianTeleop(robot, *limits)
        tp.reset(q0)
        # 先推到 anti-windup 上限，制造明显领先
        for _ in range(400):
            tp.step(v, q0, 0.005)
        assert np.abs(tp.q_des - q0).max() > 0.1

        # ⭐ 先确认两个构型给出的映射**确实不同**，
        #    否则这条测试本身没有分辨力。
        qd_meas = tp._dls_pinv(robot.jacobian(q0)) @ v
        qd_cmd = tp._dls_pinv(robot.jacobian(tp.q_des)) @ v
        assert np.abs(qd_meas - qd_cmd).max() > 1e-3

        # ⭐ 直接判据：这一步 q_des 的增量方向，必须与**实测构型**
        #    算出的关节速度一致（而不是与 q_des 构型算出的一致）。
        #    这里换一个不触发 anti-windup 钳制的小输入。
        tp2 = CartesianTeleop(robot, *limits)
        tp2.reset(q0)
        q_lead = q0 + 0.12                     # 人为制造领先
        tp2.q_des = q_lead.copy()
        before = tp2.q_des.copy()
        tp2.step(v, q0, 0.005)
        delta = tp2.q_des - before

        u_meas = qd_meas / np.linalg.norm(qd_meas)
        u_cmd = (tp._dls_pinv(robot.jacobian(q_lead)) @ v)
        u_cmd = u_cmd / np.linalg.norm(u_cmd)
        u_actual = delta / max(np.linalg.norm(delta), 1e-12)
        # 与"实测构型"的夹角必须比与"指令构型"的更小
        assert float(u_actual @ u_meas) > float(u_actual @ u_cmd)

    def test_can_always_move_away_from_a_limit(self, robot, limits):
        """⭐⭐ 贴住限位后，**往回走的指令必须畅通**。

        ⚠️ 如果减速项不看方向（``f_lo * f_hi``），贴住限位时两个方向
        都被衰减到零 —— 手臂**再也回不来**。
        安全措施本身不能制造一个更危险的状态。
        """
        lower, upper = limits
        tp = CartesianTeleop(robot, *limits)
        q_at_limit = upper.copy() - 1e-4           # 紧贴上限
        tp.reset(q_at_limit)

        # 朝限位方向：应被压住
        tp.step(np.zeros(6), q_at_limit, 0.005)
        J = robot.jacobian(q_at_limit)
        qd_out = np.ones(6) * 0.5                  # 全部朝上限
        s_out = tp._limit_scale(q_at_limit, qd_out)
        assert np.all(s_out < 0.05)

        # ⭐ 背离限位方向：必须完整保留
        s_back = tp._limit_scale(q_at_limit, -qd_out)
        np.testing.assert_allclose(s_back, np.ones(6))
