"""⭐ 闭环守护测试：控制器必须在**真的积分了动力学**的回路里被验证。

为什么单独开这一组
------------------
在这个文件之前，本仓库有 51 项测试**全部通过**，而模型里躺着一个
致命 bug：底座与 link1 永久穿模，接触摩擦把 J1 完全锁死
（施加 5 N·m 一秒钟只转了 0.0001 rad）。

⚠️ 51 项测试没有一项发现它。因为它们验的是运动学、动力学、回归矩阵、
条件数——**没有一项触碰接触求解器，没有一项把回路闭起来**。

`实测` 修复后同一组增益的跟踪 RMS 从 0.087 降到 0.006（**14 倍**）。
在此之前所有"PD vs CTC"的对比数字全部无效。

⭐ 本文件的每一条测试，都是只有闭环才测得出来的东西。
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.control.adaptive import SlotineLiAdaptiveController, scaled_gains
from panthera.control.computed_torque import (ComputedTorqueController,
                                              TaskSpaceCTC, from_bandwidth,
                                              jacobian_dot_qd)
from panthera.control.momentum_observer import MomentumObserver
from panthera.core.robot import Q_HOME
from panthera.driver.mujoco_backend import MujocoBackend
from panthera.identification.regressor import DynamicsRegressor
from panthera.sim.rollout import (hold_reference, pd_controller, rollout,
                                  sine_reference, step_reference)

XML = "models/panthera/panthera.xml"
AMP = np.array([0.3, 0.2, 0.3, 0.2, 0.3, 0.3])


@pytest.fixture(scope="module")
def be():
    return MujocoBackend()


@pytest.fixture(scope="module")
def q0():
    return np.array(Q_HOME)


@pytest.fixture(scope="module")
def reg():
    return DynamicsRegressor(XML, n_arm=6)


def ctc_fn(controller):
    return lambda t, q, qd, ref: controller.compute(q, qd, *ref)


# ============================================================ 跑道本身

class TestHarness:
    """⚠️ 一个结果依赖执行顺序的测试跑道，是没有判据可言的。"""

    def test_rollout_is_order_independent(self, be, q0):
        """同一组增益放在不同位置跑，必须**逐位一致**。

        `实测` bug：``MujocoBackend.reset`` 原先只写 qpos，
        ``ArmModel.set_state(q)`` 在 qd=None 时**不清零速度**，
        上一次 rollout 的末速度被带进下一次——同一组增益（CTC wn=45）
        读出 4.86 或 15.88 N·m 两个峰值力矩，取决于它排在第几个跑。

        ⭐ 单次运行永远看不出来。必须"同一件事做两遍，中间插点别的"。
        """
        ref = sine_reference(q0, AMP, 0.5)

        def run(wn):
            c = ComputedTorqueController(be.robot, from_bandwidth(6, wn=wn))
            return rollout(be, ctc_fn(c), ref, 2.0, q0)

        first = run(45.0)
        run(200.0)                      # 中间插一个完全不同的工况
        again = run(45.0)
        np.testing.assert_array_equal(first.q, again.q)
        np.testing.assert_array_equal(first.tau, again.tau)

    def test_torque_saturation_is_always_on(self, be, q0):
        """⚠️ 限幅是保护真机的最后一道闸，任何时候都不许放开。"""
        ref = step_reference(q0, q0 + 1.0, t_step=0.0)
        log = rollout(be, pd_controller(np.full(6, 5e3), np.full(6, 50.0)),
                      ref, 0.5, q0)
        assert np.all(np.abs(log.tau) <= be.tau_limit + 1e-9)
        assert log.saturation_pct(be.tau_limit) > 0     # 确实撞到了闸


# ============================================================ 模型 bug 回归

class TestBaseContactRegression:
    """⭐⭐ 底座 ↔ link1 碰撞排除。这个 bug 修好了就不许再回来。"""

    def test_no_resting_contacts_at_home(self, be, q0):
        import mujoco
        be.reset(q0)
        mujoco.mj_forward(be.model, be.data)
        assert be.data.ncon == 0, (
            f"零位存在 {be.data.ncon} 个接触。MuJoCo 不会自动过滤 "
            "world↔子体 的碰撞，MJCF 里必须显式 <exclude>。")

    def test_joint1_actually_rotates(self, be, q0):
        """⭐ 直接判据：给 J1 恒定力矩，它**必须转起来**。

        ⚠️ 这条测试的价值在于它测的是"物理上会不会动"，
        而不是"矩阵算得对不对"。前者才是接触 bug 能被抓住的地方。
        """
        def push(t, q, qd, ref):
            tau = np.zeros(6)
            tau[0] = 5.0
            return tau

        log = rollout(be, push, hold_reference(q0), 1.0, q0)
        assert abs(log.q[-1, 0] - q0[0]) > 0.5


# ============================================================ CTC

class TestComputedTorque:
    def test_ctc_beats_pd_at_matched_torque_budget(self, be, q0):
        """⭐ 公平对比的前提：**力矩预算相同，且都不饱和**。

        `实测`（正弦 0.5 Hz，跳 1 s 瞬态）：

        ==================  ========  ==========  ========
        控制器               RMS       |τ|max      饱和
        ==================  ========  ==========  ========
        PD + 重力补偿 kp=20   0.00614   11.80       0%
        CTC wn=30            0.00492   11.99       0%
        ==================  ========  ==========  ========

        ⚠️ 不控制力矩预算的对比毫无意义：PD 加大增益会饱和，
        饱和后实际力矩 ≠ 指令力矩，比的已经不是控制律了。
        """
        ref = sine_reference(q0, AMP, 0.5)
        pd_log = rollout(be, pd_controller(np.full(6, 20.0), np.full(6, 8.0),
                                           be.robot.gravity), ref, 4.0, q0)
        ctc = ComputedTorqueController(be.robot, from_bandwidth(6, wn=30.0))
        ctc_log = rollout(be, ctc_fn(ctc), ref, 4.0, q0)

        # ⚠️ 跳掉启动瞬态再判饱和：t=0 处参考速度非零而手臂静止，
        #    必然顶一下限幅（`实测` 12000 步里的 2 步）。
        assert pd_log.saturation_pct(be.tau_limit, skip=1.0) == 0.0
        assert ctc_log.saturation_pct(be.tau_limit, skip=1.0) == 0.0
        # 力矩预算相当（20% 以内）
        assert abs(ctc_log.max_abs_torque() / pd_log.max_abs_torque() - 1) < 0.2
        assert ctc_log.rms(skip=1.0) < pd_log.rms(skip=1.0)

    def test_pd_gets_worse_when_it_saturates(self, be, q0):
        """⚠️ 反直觉但重要：PD 增益加大**反而变差**，因为撞了限幅。

        这条测试防的是"增益越大越好"的想当然。
        """
        ref = sine_reference(q0, AMP, 0.5)
        low = rollout(be, pd_controller(np.full(6, 20.0), np.full(6, 8.0),
                                        be.robot.gravity), ref, 4.0, q0)
        high = rollout(be, pd_controller(np.full(6, 100.0), np.full(6, 18.0),
                                         be.robot.gravity), ref, 4.0, q0)
        assert low.saturation_pct(be.tau_limit, skip=1.0) == 0.0
        assert high.saturation_pct(be.tau_limit, skip=1.0) > 10.0
        assert high.rms(skip=1.0) > low.rms(skip=1.0)

    def test_ctc_monotonically_improves_with_bandwidth(self, be, q0):
        """CTC 把系统线性化之后，带宽是真的能换精度的。"""
        ref = sine_reference(q0, AMP, 0.5)
        rms = []
        for wn in (8.0, 20.0, 45.0):
            c = ComputedTorqueController(be.robot, from_bandwidth(6, wn=wn))
            rms.append(rollout(be, ctc_fn(c), ref, 4.0, q0).rms(skip=1.0))
        assert rms[0] > rms[1] > rms[2]
        assert rms[0] / rms[2] > 10        # `实测` 0.0554 / 0.00223 ≈ 25×

    def test_coriolis_compensation_matters_at_speed(self, be, q0):
        """关掉科氏力补偿，高速下必须变差。低速下差别很小是正常的。"""
        ref = sine_reference(q0, AMP, 1.5)
        gains = from_bandwidth(6, wn=30.0)
        on = rollout(be, ctc_fn(ComputedTorqueController(be.robot, gains, True)),
                     ref, 3.0, q0).rms(skip=1.0)
        off = rollout(be, ctc_fn(ComputedTorqueController(be.robot, gains, False)),
                      ref, 3.0, q0).rms(skip=1.0)
        assert off > on


class TestJacobianDerivativeTerm:
    """⭐⭐ 任务空间 CTC 漏掉 $\\dot J\\dot q$ 的后果——**随速度平方增长**。

    `实测`（笛卡尔圆弧，wn=25）：

    ========  ==========  ============
    f (Hz)    ‖J̇q̇‖ 均值   漏掉后劣化
    ========  ==========  ============
    0.5       0.12        1.02×  ← **看不见**
    1.0       0.56        1.14×
    1.5       1.32        1.49×
    2.0       2.43        **2.20×**
    ========  ==========  ============

    ⚠️⚠️ **低速下这个 bug 是隐形的。** armctrl 当年就是这么漏过去的：
    测试跑在低速工况，含与不含差 2%，被容差吃掉。
    所以这里的测试**必须跑在 2 Hz**，并且**直接暴露 ‖J̇q̇‖ 这个物理量**，
    而不是只看一个 RMS 比值。（元教训 #10）
    """

    @staticmethod
    def _run(be, q0, freq, include_jdot):
        R = be.robot
        x0 = R.tcp_position(q0)
        amp = np.array([0.06, 0.06, 0.05])
        w = 2.0 * np.pi * freq
        ctrl = TaskSpaceCTC(R, from_bandwidth(6, wn=25.0),
                            include_jdot=include_jdot)
        be.reset(q0)
        errs, jn = [], []
        for k in range(1500):
            t = k * be.dt
            st = be.read()
            ref = (x0 + amp * np.sin(w * t), amp * w * np.cos(w * t),
                   -amp * w * w * np.sin(w * t))
            tau = be.saturate(ctrl.compute(st.q, st.qd, *ref))
            errs.append(np.linalg.norm(R.tcp_position(st.q) - ref[0]))
            jn.append(np.linalg.norm(jacobian_dot_qd(R, st.q, st.qd)[:3]))
            be.send_torque(tau)
            be.step()
        e = np.array(errs)[400:]
        return float(np.sqrt((e ** 2).mean())), float(np.mean(jn))

    def test_jdot_term_matters_at_high_speed(self, be, q0):
        with_j, norm = self._run(be, q0, 2.0, True)
        without, _ = self._run(be, q0, 2.0, False)
        assert norm > 1.0                      # 确认这个工况真的激发了该项
        assert without / with_j > 1.5          # `实测` 2.20×

    def test_jdot_term_is_invisible_at_low_speed(self, be, q0):
        """⚠️ 反向守护：低速下差别 < 5%。

        这条测试在**记录一个陷阱**，不是在庆祝。它证明了
        "在低速工况测这个 bug 一定测不出来"。
        """
        with_j, norm = self._run(be, q0, 0.5, True)
        without, _ = self._run(be, q0, 0.5, False)
        assert norm < 0.3
        assert without / with_j < 1.05

    def test_jdot_norm_scales_quadratically_with_speed(self, be, q0):
        """$\\dot J\\dot q$ 是速度的二次型，频率翻倍应约 4 倍。"""
        _, n1 = self._run(be, q0, 1.0, True)
        _, n2 = self._run(be, q0, 2.0, True)
        assert 3.0 < n2 / n1 < 5.5             # `实测` 4.32×


# ============================================================ 自适应

class TestAdaptiveControl:
    """⚠️ 本组测试经历过一次"漂亮但假"的结果，处理过程本身是主要产出。

    最初用**标量** $\\Gamma$ + 教科书自适应律，跑出 "3.44× 改善"。
    把同一个脚本**重跑一遍**，结果变成 1.01×——因为那时 $\\hat\\beta$
    已经冲到投影上界 200 并卡死，系统是发散的，轨迹是混沌的。

    ⭐ **一个会因 1e-16 扰动而翻转的结论不是结论。**
    发现它只需要一件事：**同一个实验做两遍**。

    修法是改用归一化自适应律（见 :class:`SlotineLiAdaptiveController` ⑥）。
    """

    @staticmethod
    def _run(be, q0, reg, kd_ref, gamma, duration=20.0, normalized=True,
             pi_init=None):
        lam, kd = scaled_gains(be.tau_limit, kd_ref=kd_ref, lam_ref=5.0)
        ad = SlotineLiAdaptiveController(reg, lam=lam, kd=kd, gamma=gamma,
                                         normalized=normalized, pi_init=pi_init)
        ref = sine_reference(q0, np.array([0.2, 0.15, 0.2, 0.15, 0.2, 0.2]), 0.3)
        log = rollout(be, lambda t, q, qd, r: ad.compute(q, qd, *r, dt=be.dt)[0],
                      ref, duration, q0)
        e = np.abs(log.error).mean(axis=1)
        n = len(e) // 4
        return ad, log, float(e[:n].mean()), float(e[-n:].mean())

    def test_scaled_gains_track_torque_capacity(self, be):
        """腕部只有 5 N·m，肩肘有 20 N·m——增益必须跟着容量走。"""
        _, kd = scaled_gains(be.tau_limit, kd_ref=10.0)
        assert kd[1] > kd[0] > kd[4]
        np.testing.assert_allclose(kd / kd.max(),
                                   be.tau_limit / be.tau_limit.max())

    def test_scalar_gain_saturates_the_wrist(self, be, q0, reg):
        """⚠️⚠️ 标量 $K_D$ 会把腕部关节按在限幅上。

        ⭐ 这里把 ``gamma=0``、``pi_init`` 设成真值，**把自适应完全关掉**，
        剩下的只有 $K_D s$ 这一项——这样测的才纯粹是"标量增益"这一件事，
        不会被自适应的动态混进来。（判据要能归因到唯一一个原因。）

        `实测` 每关节饱和比例：

        ==================  ===  ===  ===  ======  ======  ======
        增益                 J1   J2   J3   J4      J5      J6
        ==================  ===  ===  ===  ======  ======  ======
        标量 kd=20           0%   0%   0%   49.8%   79.3%   88.5%
        缩放 kd_ref=10       0%   0%   0%   0%      0%      0%
        ==================  ===  ===  ===  ======  ======  ======
        """
        ref = sine_reference(q0, np.array([0.2, 0.15, 0.2, 0.15, 0.2, 0.2]), 0.3)

        def sat_profile(lam, kd):
            ad = SlotineLiAdaptiveController(reg, lam=lam, kd=kd, gamma=0.0,
                                             pi_init=reg.true_parameters())
            log = rollout(be, lambda t, q, qd, r: ad.compute(q, qd, *r, dt=be.dt)[0],
                          ref, 8.0, q0)
            return (np.abs(log.tau) >= be.tau_limit[None, :] - 1e-9).mean(axis=0)

        flat = sat_profile(np.full(6, 5.0), np.full(6, 20.0))
        assert flat[4] > 0.5 and flat[5] > 0.5      # 腕部严重饱和
        assert flat[0] < 0.05                       # 肩部完全不饱和

        lam, kd = scaled_gains(be.tau_limit, kd_ref=10.0, lam_ref=5.0)
        assert sat_profile(lam, kd).max() == 0.0    # 缩放之后全部不饱和

    def test_zero_initial_parameters_means_no_gravity_compensation(self, be, q0, reg):
        r"""⭐⭐ ``pi_init=0`` 意味着**开机第一帧完全没有重力补偿**。

        $\tau = Y_b\hat\beta + K_D s$，$\hat\beta=0$ 时只剩 $K_D s$。
        在零位静止、参考也静止时 $s=0$，于是 $\tau=0$——
        而 $g(q_0)$ 在 J3 上有 **4.22 N·m**。手臂直接掉下去。

        ⚠️ 真机上**永远不要**从零参数起估。用 CAD 名义模型做初值：
        它不准，但至少重力方向是对的。

        ⭐ 这条测试只看**第一帧**，是完全确定的一步计算。
        刻意不去断言"长期会发散"——零初值 + 非归一化律的长期行为是
        **混沌的**（实测过一次琐碎重构就把结果从 ‖β‖=200 翻成 ‖β‖=0.18），
        对混沌行为写断言就是写 flaky 测试。
        """
        lam, kd = scaled_gains(be.tau_limit, kd_ref=6.0, lam_ref=5.0)
        zero = SlotineLiAdaptiveController(reg, lam=lam, kd=kd, gamma=0.0,
                                           pi_init=None)
        warm = SlotineLiAdaptiveController(reg, lam=lam, kd=kd, gamma=0.0,
                                           pi_init=reg.true_parameters())
        z = np.zeros(6)
        tau_zero, _ = zero.compute(q0, z, q0, z, z, dt=be.dt)
        tau_warm, _ = warm.compute(q0, z, q0, z, z, dt=be.dt)
        g = be.robot.gravity(q0)

        assert np.abs(tau_zero).max() < 1e-9                 # 零初值：什么都不出
        np.testing.assert_allclose(tau_warm, g, atol=1e-9)   # 名义初值：正好托住重力
        assert np.abs(g).max() > 4.0                         # 而重力确实不小

    def test_normalized_law_is_insensitive_to_gamma(self, be, q0, reg):
        """⭐ 归一化律在 **40 倍** 的 $\\gamma$ 范围内表现几乎不变。

        `实测`（初值 0.7×真值）：

        ========  ========  ========
        $\\gamma$  改善      ‖β‖ 末
        ========  ========  ========
        50        3.23×     0.61
        200       3.16×     0.63
        1000      3.78×     0.65
        2000      3.76×     0.65
        ========  ========  ========

        ⭐ 而非归一化律在同样的相对范围里从 1.18× 变到 3.36×——
        **需要调参才能用**。归一化的价值就在这里：少一个要调的旋钮。
        """
        pi0 = 0.7 * reg.true_parameters()
        results = []
        for gamma in (50.0, 2000.0):
            ad, log, first, last = self._run(be, q0, reg, kd_ref=6.0,
                                             gamma=gamma, pi_init=pi0)
            assert log.saturation_pct(be.tau_limit, skip=1.0) == 0.0
            assert not log.diverged()
            # ⭐ ‖β‖ 必须远低于投影上界，否则"误差下降"可能只是
            #    安全阀按住了一个发散系统造成的假象。
            assert np.linalg.norm(ad.beta) < 0.1 * ad.proj_bound
            results.append(first / last)

        assert min(results) > 2.5                      # 两端都好
        assert max(results) / min(results) < 1.5       # 且彼此接近

    def test_unnormalized_law_needs_gamma_tuning(self, be, q0, reg):
        """⚠️ 对照组：非归一化律在小 $\\gamma$ 下几乎学不动。

        `实测` γ=1e-3 只有 1.18×，而归一化律在其推荐范围内恒 >3×。
        """
        pi0 = 0.7 * reg.true_parameters()
        _, _, first, last = self._run(be, q0, reg, kd_ref=6.0, gamma=1e-3,
                                      normalized=False, pi_init=pi0)
        assert first / last < 1.5

    def test_convergence_is_reproducible(self, be, q0, reg):
        """⭐⭐ 同一个实验做两遍，结果必须**逐位一致**。

        这条测试是一次翻车的直接产物：早期版本用标量 Γ + 零初值，
        跑出过 "3.44× 改善"，重跑变成 1.01×——因为那时系统在发散，
        轨迹是混沌的，位级扰动就翻盘。

        ⭐ **"可复现"本身就是"真的收敛了"的一个独立判据。**
        """
        pi0 = 0.7 * reg.true_parameters()
        _, log_a, a1, a2 = self._run(be, q0, reg, 6.0, 1000.0,
                                     duration=8.0, pi_init=pi0)
        self._run(be, q0, reg, 6.0, 50.0, duration=8.0)     # 中间插别的工况
        _, log_b, b1, b2 = self._run(be, q0, reg, 6.0, 1000.0,
                                     duration=8.0, pi_init=pi0)
        assert (a1, a2) == (b1, b2)
        np.testing.assert_array_equal(log_a.tau, log_b.tau)

    def test_regressor_identity_holds_exactly(self, reg):
        """⭐⭐ 自适应的**全部前提**：$Y_{SL}\\pi$ 必须精确等于它声称的那个量。

        .. math::
            Y_{SL}(q,\\dot q,\\dot q_r,\\ddot q_r)\\,\\pi
            = M\\ddot q_r + C(q,\\dot q)\\dot q_r + g
              + \\text{armature}\\odot\\ddot q_r
              + F_v\\odot\\dot q_r + F_c\\odot\\tanh(200\\dot q)

        ⚠️ 这个恒等式一旦不成立，Lyapunov 论证整个崩掉，
        自适应就变成"一个带积分的随机数发生器"。

        ⚠️ 检验式**必须把摩擦项写全**。我第一次写这条检验时漏了摩擦，
        得到 8% 的偏差，差点去"修"一个根本没坏的回归矩阵。
        `实测` 写全之后残差 6×10⁻¹⁶。
        """
        pi = reg.true_parameters()
        rng = np.random.default_rng(0)
        for _ in range(3):
            q, v = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
            vr, ar = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
            lhs = reg.slotine_li_regressor(q, v, vr, ar) @ pi
            rhs = (reg.mass_matrix(q) @ ar + reg.coriolis_times(q, v, vr)
                   + reg.gravity(q) + reg.fv_true * vr
                   + reg.fc_true * np.tanh(200.0 * v))
            assert np.linalg.norm(lhs - rhs) < 1e-12

    def test_christoffel_coriolis_is_symmetric(self, reg):
        """$C(q,v)w=C(q,w)v$——极化恒等式能用的前提。

        Christoffel 符号 $\\Gamma_{ijk}$ 关于 $j\\leftrightarrow k$ 对称，
        所以 ``coriolis_times`` 用极化恒等式算是**精确**的，不是近似。
        """
        rng = np.random.default_rng(1)
        for _ in range(2):
            q, v, w = (rng.uniform(-1, 1, 6) for _ in range(3))
            assert np.linalg.norm(reg.coriolis_times(q, v, w)
                                  - reg.coriolis_times(q, w, v)) < 1e-12

    def test_tracking_convergence_is_not_parameter_convergence(self, be, q0, reg):
        """⭐⭐ 跟踪收敛 ≠ 参数收敛。

        自适应只保证 $s\\to0$，**不保证** $\\hat\\pi\\to\\pi$——
        后者还需要持续激励（PE）。

        ⚠️ 拿自适应跑出来的 $\\hat\\pi$ 当辨识结果用，是常见错误。
        要准的参数就老老实实做辨识（``identification`` 模块）。
        """
        pi0 = 0.7 * reg.true_parameters()
        ad, _, first, last = self._run(be, q0, reg, 6.0, 1000.0, pi_init=pi0)
        assert first / last > 2.0                    # 跟踪确实收敛了

        beta_true = ad.P.T @ reg.true_parameters()
        err = np.linalg.norm(ad.beta - beta_true) / np.linalg.norm(beta_true)
        assert err > 0.1                             # 参数却没收敛到真值

    def test_projection_bound_keeps_parameters_finite(self, be, q0, reg):
        """PE 不足时参数会漂移，投影是安全阀。"""
        lam, kd = scaled_gains(be.tau_limit, kd_ref=6.0)
        ad = SlotineLiAdaptiveController(reg, lam=lam, kd=kd, gamma=1e-2,
                                         normalized=False, proj_bound=10.0)
        ref = sine_reference(q0, np.full(6, 0.1), 0.3)
        rollout(be, lambda t, q, qd, r: ad.compute(q, qd, *r, dt=be.dt)[0],
                ref, 6.0, q0)
        assert np.linalg.norm(ad.beta) <= 10.0 + 1e-6


# ============================================================ 动量观测器

class TestMomentumObserver:
    @staticmethod
    def _run(be, q0, tau_ext, t_on=2.0, duration=10.0, k_i=25.0,
             feed_unsaturated=False, wn=20.0, q_target=None):
        R = be.robot
        target = q0 if q_target is None else q_target
        ctc = ComputedTorqueController(R, from_bandwidth(6, wn=wn))
        obs = MomentumObserver(R, k_i=k_i, dt=be.dt)
        be.reset(q0)
        obs.reset(q0, np.zeros(6))
        rs = []
        for k in range(int(duration / be.dt)):
            t = k * be.dt
            st = be.read()
            raw = ctc.compute(st.q, st.qd, target, np.zeros(6), np.zeros(6))
            tau = be.saturate(raw)
            be.data.qfrc_applied[R.qvel_idx] = tau_ext if t >= t_on else 0.0
            obs.update(st.q, st.qd, raw if feed_unsaturated else tau)
            rs.append(obs.r.copy())
            be.send_torque(tau)
            be.step()
        be.data.qfrc_applied[R.qvel_idx] = 0.0
        return np.array(rs)

    def test_estimates_known_external_torque(self, be, q0):
        """⭐ 直接判据：外力矩是我们**自己加的**，真值已知。

        `实测` 真值 −1.5 N·m，10 s 后估到 −1.462（97.5%）。
        """
        tau_ext = np.array([0.0, 0.0, -1.5, 0.0, 0.0, 0.0])
        rs = self._run(be, q0, tau_ext)
        assert abs(rs[-100:, 2].mean() - (-1.5)) < 0.15
        assert np.abs(rs[-100:, [0, 4, 5]]).max() < 0.1     # 串扰要小

    def test_residual_is_quiet_without_external_force(self, be, q0):
        rs = self._run(be, q0, np.zeros(6))
        assert np.abs(rs).max() < 0.05

    def test_bandwidth_matches_k_i(self, be, q0):
        """一阶低通：上升到 63% 的时间应约等于 $1/K_I$。

        `实测` $K_I=25$ ⇒ 46 ms（理论 40 ms）。
        """
        tau_ext = np.array([0.0, 0.0, -1.5, 0.0, 0.0, 0.0])
        n_on = int(2.0 / be.dt)
        for k_i, expect in ((25.0, 1.0 / 25.0), (50.0, 1.0 / 50.0)):
            rs = self._run(be, q0, tau_ext, k_i=k_i, duration=6.0)
            idx = np.argmax(np.abs(rs[n_on:, 2]) > 0.63 * 1.5)
            rise = idx * be.dt
            assert 0.5 * expect < rise < 2.0 * expect

    def test_threshold_calibration_separates_the_two_cases(self, be, q0):
        """⭐ 阈值必须由**空跑数据**定，不能拍脑袋。"""
        tau_ext = np.array([0.0, 0.0, -1.5, 0.0, 0.0, 0.0])
        rs = self._run(be, q0, tau_ext)
        n_on = int(2.0 / be.dt)
        thr = MomentumObserver.calibrate_threshold(rs[:n_on])

        obs = MomentumObserver(be.robot, dt=be.dt)
        obs.r = rs[n_on - 1]
        assert not obs.detect(thr)          # 施力前：不报警
        obs.r = rs[-1]
        assert obs.detect(thr)              # 施力后：报警

    def test_feeding_commanded_torque_creates_a_phantom_collision(self, be, q0):
        """⭐⭐ 喂**饱和前**的指令力矩，会凭空造出一个巨大的假外力。

        观测器估的是"模型没算到的一切"。力矩饱和时实际力矩 ≠ 指令力矩，
        被限幅削掉的那部分会被整个误判成外力。

        `实测`（高增益 wn=60 阶跃，指令超限 39 步）：

        ================  ============
        喂进去的力矩       |r|max
        ================  ============
        实际（饱和后）      1.13 N·m
        指令（饱和前）      **135 N·m**
        ================  ============

        ⚠️ 120 倍的假信号。在真机上这会表现为"手臂一动就报碰撞"。
        """
        # ⚠️ 必须让手臂真的动起来（阶跃到 +0.5 rad）才会撞限幅。
        #    停在原地不动是不会饱和的，那样这条测试测不到任何东西。
        step = q0 + np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        rs_ok = self._run(be, q0, np.zeros(6), duration=3.0, wn=60.0,
                          q_target=step, feed_unsaturated=False)
        rs_bad = self._run(be, q0, np.zeros(6), duration=3.0, wn=60.0,
                           q_target=step, feed_unsaturated=True)
        assert np.abs(rs_bad).max() > 20 * np.abs(rs_ok).max()


class TestObserverUnderFastMotion:
    """⭐⭐ 速度二次项的通病：**低速下隐形**。

    删掉动量观测器的 $C^{\\mathsf T}\\dot q$ 项，上面 22 项测试**全部通过**——
    因为它们都跑在近似静止的工况下。这和任务空间漏 $\\dot J\\dot q$
    是同一个病（见 :class:`TestJacobianDerivativeTerm`）。

    `实测` 正弦跟踪，无外力，残差应恒为噪声级：

    ========  ==========  ============  ==============  ======
    f (Hz)    |q̇|max      |r|max 正确    |r|max 漏 Cᵀq̇   倍数
    ========  ==========  ============  ==============  ======
    0.3       0.62        0.2336        0.2336          1.0×
    1.0       1.98        0.3001        0.3630          1.2×
    2.0       3.73        0.3290        **0.6457**      2.0×
    ========  ==========  ============  ==============  ======

    ⭐ 教训：**凡是速度二次项，守护测试必须跑在高速工况。**
    """

    @staticmethod
    def _run(be, q0, freq, drop_coriolis=False):
        R = be.robot
        ctc = ComputedTorqueController(R, from_bandwidth(6, wn=30.0))
        obs = MomentumObserver(R, k_i=25.0, dt=be.dt)
        if drop_coriolis:
            obs._coriolis_transpose_qd = lambda q, qd: np.zeros(6)
        be.reset(q0)
        obs.reset(q0, np.zeros(6))
        w = 2.0 * np.pi * freq
        rs, vmax = [], 0.0
        for k in range(2000):
            t = k * be.dt
            st = be.read()
            ref = (q0 + AMP * np.sin(w * t), AMP * w * np.cos(w * t),
                   -AMP * w * w * np.sin(w * t))
            tau = be.saturate(ctc.compute(st.q, st.qd, *ref))
            obs.update(st.q, st.qd, tau)
            rs.append(obs.r.copy())
            vmax = max(vmax, float(np.abs(st.qd).max()))
            be.send_torque(tau)
            be.step()
        return float(np.abs(np.array(rs)[500:]).max()), vmax

    def test_coriolis_term_matters_under_fast_motion(self, be, q0):
        """⭐ 这条测试专门杀"漏掉 $C^{\\mathsf T}\\dot q$"这个变异。"""
        ok, vmax = self._run(be, q0, 2.0, drop_coriolis=False)
        bad, _ = self._run(be, q0, 2.0, drop_coriolis=True)
        assert vmax > 3.0                 # 确认这个工况真的跑到高速了
        assert bad / ok > 1.5             # `实测` 2.0×

    def test_coriolis_term_is_invisible_at_low_speed(self, be, q0):
        """⚠️ 反向守护：低速下删掉该项**没有任何影响**。

        记录陷阱用，不是庆祝。它证明"在低速工况测这个 bug 一定测不出来"。
        """
        ok, vmax = self._run(be, q0, 0.3, drop_coriolis=False)
        bad, _ = self._run(be, q0, 0.3, drop_coriolis=True)
        assert vmax < 1.0
        assert abs(bad / ok - 1.0) < 0.05

    def test_residual_floor_grows_with_speed(self, be, q0):
        """⚠️ 无外力时残差**并不为零**——摩擦没在观测器里建模。

        `实测` |r|max：0.3 Hz 时 0.234，2 Hz 时 0.329。

        ⭐ 后果：**碰撞检测阈值必须随工况标定，不能定一个全局常数。**
        静止时标出来的阈值拿到高速下用，会天天误报。
        这也是为什么摩擦辨识是碰撞检测的前置工作。
        """
        slow, _ = self._run(be, q0, 0.3)
        fast, _ = self._run(be, q0, 2.0)
        assert slow > 0.05                # 静态模型下也不是零
        assert fast > slow
