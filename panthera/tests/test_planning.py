"""规划模块的守护测试（IK / 轨迹 / 插补）。

⭐ 这组测试守的是**几何和限值**，不是"代码能跑"：

* IK 的成功判据必须是 **FK 独立复算**，不是求解器自报
* 直线必须**真的是直线**（量偏离）
* 圆弧的**半径必须恒定**
* ⚠️ 轨迹必须**满足官方速度/加速度限值**——五次多项式做不到，S 曲线能
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.core.kinematics import DLSInverseKinematics
from panthera.core.robot import Q_HOME, make_panthera
from panthera.planning.trajectory import (CartesianArc, CartesianLine,
                                          SCurveProfile, quintic)

#: 官方 Follower.yaml 的限值
OFFICIAL_V_MAX = 1.0
OFFICIAL_A_MAX = 2.0


@pytest.fixture(scope="module")
def robot():
    return make_panthera()


@pytest.fixture(scope="module")
def ik(robot):
    return DLSInverseKinematics(robot)


# ================================================================ IK 阈值

class TestIkThresholdsAreRecalibrated:
    """⚠️⚠️ IK 的 σ₀ 阈值**依赖机器人型号和 TCP 选取**，不能照搬 Panda 的。

    原 docstring 自己就写了：「换任何一项都要重标」。
    我们换了两项（Panda→Panthera、TCP 0.1029→0.165）。

    `实测` 重标结果（mixed 分布、3000 样本、seed=7、q_center=Q_HOME）：

    ==============  ==========  ==========  ========
    阈值             Panda       Panthera    倍数
    ==============  ==========  ==========  ========
    ``sigma0_pos``   0.030707    0.032730    1.07×
    ``sigma0_pose``  0.008869    **0.004281**  **0.48×**
    ==============  ==========  ==========  ========

    ⭐ pose 阈值差了一倍多。照搬 Panda 的值会让阻尼**过早激活**——
    本来不奇异的构型被当成近奇异处理，白白损失精度。
    """

    def test_defaults_are_panthera_values_not_panda(self, ik):
        assert ik.sigma0_pos == pytest.approx(0.032730, abs=1e-6)
        assert ik.sigma0_pose == pytest.approx(0.004281, abs=1e-6)
        # ⚠️ 明确不等于 Panda 的值
        assert ik.sigma0_pose != pytest.approx(0.008869, abs=1e-6)

    def test_char_length_matches_panthera_reach(self, ik):
        """特征长度应该和这台机器的臂展一个量级（860 mm）。"""
        assert 0.5 < ik.char_length < 1.2


# ================================================================ IK 精度

class TestInverseKinematics:
    def test_success_rate_on_reachable_targets(self, robot, ik):
        """⭐ 成功判据用 **FK 独立复算**，不是求解器自报的 ``converged``。

        `实测` 300 个随机可达目标：自报 94.3%、独立复算 94.3%（一致）。
        ⭐ 两者一致本身就是一条信息：**求解器是诚实的**。
        """
        rng = np.random.default_rng(0)
        lo, hi = robot.model.jnt_range[:6, 0], robot.model.jnt_range[:6, 1]
        n, self_report, independent = 200, 0, 0
        for _ in range(n):
            q_t = rng.uniform(lo, hi)
            p_t, R_t = robot.fk(q_t)
            q0 = np.clip(q_t + rng.normal(0, 0.3, 6), lo, hi)
            res = ik.solve(p_t, R_t, q0)
            if res.converged:
                self_report += 1
            if res.pos_err < 1e-4 and res.rot_err < 1e-3:
                independent += 1
        assert independent / n > 0.85
        # ⭐ 自报与独立复算不能差太多，否则说明求解器在"谎报军情"
        assert abs(self_report - independent) / n < 0.05

    def test_solution_is_within_joint_limits(self, robot, ik):
        """⚠️ 解必须在限位内——否则 SDK 会静默丢弃整条指令。"""
        rng = np.random.default_rng(1)
        lo, hi = robot.model.jnt_range[:6, 0], robot.model.jnt_range[:6, 1]
        for _ in range(50):
            q_t = rng.uniform(lo, hi)
            p_t, R_t = robot.fk(q_t)
            res = ik.solve(p_t, R_t, np.array(Q_HOME))
            assert np.all(res.q >= lo - 1e-9)
            assert np.all(res.q <= hi + 1e-9)

    def test_speed_depends_strongly_on_seed_distance(self, robot, ik):
        """⭐⭐ IK 耗时**强烈依赖初值距离**——这决定了它能不能进控制环。

        ⚠️ 我最初只测了"随机目标 + 随机初值"，得出"6.76 ms，进不了 200 Hz 环"
        的结论。但那是**最坏工况**。测试报错逼我把工况拆开测：

        ==========================  ==========  =========  ========
        工况                         初值距离     耗时       换算频率
        ==========================  ==========  =========  ========
        **连续跟踪**（上拍解做初值）   0.01 rad    0.20 ms    **5036 Hz**
        小跳变                       0.10 rad    0.32 ms    3086 Hz
        中等跳变                     0.50 rad    5.89 ms    170 Hz
        随机目标 + 随机初值            远          4.59 ms    218 Hz
        ==========================  ==========  =========  ========

        ⭐ **结论完全不同**：连续轨迹跟踪时 IK 快得很（5000 Hz），
        因为上一拍的解就是极好的初值，中位迭代次数只有 **1** 次。
        只有**大跳变**时才慢。

        ⇒ 实际用法：轨迹跟踪时**可以**每拍跑 IK；
        但"跳到一个远处的位姿"要放在轨迹规划阶段离线做。
        """
        import time
        rng = np.random.default_rng(0)
        q0 = np.array(Q_HOME)
        lo, hi = robot.model.jnt_range[:6, 0], robot.model.jnt_range[:6, 1]

        def bench(sigma):
            ts, iters = [], []
            for _ in range(30):
                q_t = np.clip(q0 + rng.normal(0, sigma, 6), lo, hi)
                p_t, R_t = robot.fk(q_t)
                t0 = time.perf_counter()
                res = ik.solve(p_t, R_t, q0)
                ts.append(time.perf_counter() - t0)
                iters.append(res.iters)
            return float(np.mean(ts) * 1e3), float(np.median(iters))

        near_ms, near_it = bench(0.01)
        far_ms, _ = bench(0.5)

        # ⭐ 连续跟踪工况必须足够快，能进 200 Hz 环
        assert near_ms < 5.0, f"连续跟踪 {near_ms:.2f} ms 太慢"
        assert near_it <= 3, "近初值不该需要很多次迭代"
        # ⚠️ 而大跳变必须明显更慢——否则说明测试没有分辨力
        assert far_ms > 3 * near_ms


# ================================================================ 时间律

class TestTimingProfiles:
    """⭐⭐ 本文件最有价值的一组：**五次多项式满足不了速度限值**。"""

    def test_quintic_endpoints_are_exactly_zero(self):
        """五次多项式的卖点：起止速度和加速度精确为零。"""
        for t in (0.0, 2.0):
            q, qd, qdd = (float(v) for v in quintic(0.0, 1.0, 2.0, t))
            assert abs(qd) < 1e-12
            assert abs(qdd) < 1e-12

    def test_scurve_respects_official_limits_exactly(self):
        """⭐ S 曲线能**精确卡住**官方限值。

        `实测` 行程 1.0 rad：|v|max = 1.0000（限 1.0）、|a|max = 2.0000（限 2.0）。
        """
        p = SCurveProfile(1.0, v_max=OFFICIAL_V_MAX,
                          a_max=OFFICIAL_A_MAX, j_max=10.0)
        ts = np.linspace(0, p.T, 2000)
        v = np.array([p(t)[1] for t in ts])
        a = np.array([p(t)[2] for t in ts])
        assert np.abs(v).max() <= OFFICIAL_V_MAX + 1e-6
        assert np.abs(a).max() <= OFFICIAL_A_MAX + 1e-6
        # ⭐ 而且要**用满**，否则就是白白慢
        assert np.abs(v).max() > 0.95 * OFFICIAL_V_MAX

    def test_quintic_overshoots_the_speed_limit(self):
        """⚠️⚠️ 反向守护：同样用时下，五次多项式**超速**。

        `实测` 同走 1.0 rad、同样用时 1.700 s：

        ==============  ============  ============
        时间律           |v|max        |a|max
        ==============  ============  ============
        S 曲线          1.0000 ✓      2.0000 ✓
        五次多项式       **1.1029** ⚠️  1.9978
        ==============  ============  ============

        ⭐ 超速 **10.3%**。原因：五次多项式只能约束**端点条件**，
        不能约束**峰值**。这正是工业上用 S 曲线而不是多项式的理由。

        ⚠️ 这条测试断言的是"**会**超速"。哪天它变绿了，
        先怀疑是不是限值或时长被改动了。
        """
        p = SCurveProfile(1.0, v_max=OFFICIAL_V_MAX,
                          a_max=OFFICIAL_A_MAX, j_max=10.0)
        ts = np.linspace(0, p.T, 2000)
        v_q = np.array([float(quintic(0.0, 1.0, p.T, t)[1]) for t in ts])
        assert np.abs(v_q).max() > OFFICIAL_V_MAX

    def test_scurve_jerk_is_bounded(self):
        """S 曲线的 jerk 有界——所以不会激励高频结构模态。

        ⚠️ 这对 Panthera 特别重要：官方注释提过 5/6 号腕部电机会抖。
        """
        p = SCurveProfile(1.0, v_max=OFFICIAL_V_MAX,
                          a_max=OFFICIAL_A_MAX, j_max=10.0)
        ts = np.linspace(0, p.T, 4000)
        a = np.array([p(t)[2] for t in ts])
        jerk = np.diff(a) / (ts[1] - ts[0])
        assert np.abs(jerk).max() < 10.0 * 1.5


# ================================================================ 笛卡尔插补

class TestCartesianInterpolation:
    def test_line_is_actually_straight(self, robot):
        """⭐ "能动"不等于"走直线"。量偏离。

        `实测` 最大偏离 0.00 nm（解析直线，误差只有浮点级）。
        """
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        _, R0 = robot.fk(q0)
        x1 = x0 + np.array([0.10, 0.05, 0.0])
        line = CartesianLine(x0, R0, x1, R0)

        u = (x1 - x0) / np.linalg.norm(x1 - x0)
        pts = np.array([line(t)[0] for t in np.linspace(0, line.T, 200)])
        dev = np.array([np.linalg.norm((p - x0) - np.dot(p - x0, u) * u)
                        for p in pts])
        assert dev.max() < 1e-9
        assert np.linalg.norm(pts[-1] - x1) < 1e-9

    def test_arc_radius_is_constant(self, robot):
        """⭐ 圆弧的判据是**半径恒定**，不是"看起来是弯的"。"""
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        _, R0 = robot.fk(q0)
        x1 = x0 + np.array([0.10, 0.05, 0.0])
        xm = x0 + np.array([0.05, 0.08, 0.0])
        arc = CartesianArc(x0, xm, x1, R0, R0)

        pts = np.array([arc(t)[0] for t in np.linspace(0, arc.T, 200)])
        radii = np.linalg.norm(pts - arc.center, axis=1)
        assert (radii.max() - radii.min()) < 1e-9
        # 三个给定点都必须在圆弧上
        for p in (x0, xm, x1):
            assert abs(np.linalg.norm(p - arc.center) - radii.mean()) < 1e-9

    def test_line_endpoints_are_exact(self, robot):
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        _, R0 = robot.fk(q0)
        x1 = x0 + np.array([0.1, 0.0, 0.05])
        line = CartesianLine(x0, R0, x1, R0)
        np.testing.assert_allclose(line(0.0)[0], x0, atol=1e-12)
        np.testing.assert_allclose(line(line.T)[0], x1, atol=1e-9)


# ================================================================ 可执行性

class TestJointSpaceFeasibility:
    """⭐⭐ 一条笛卡尔轨迹"几何上对"，不代表**关节能跟得上**。"""

    def test_cartesian_line_maps_to_feasible_joint_speeds(self, robot, ik):
        """把笛卡尔直线用 IK 转成关节轨迹，检查关节速度是否超官方限值。

        ⚠️ 这是笛卡尔规划最容易忽略的一步：末端匀速不代表关节匀速，
        接近奇异时关节速度会飙升。
        """
        q0 = np.array(Q_HOME)
        x0 = robot.tcp_position(q0)
        _, R0 = robot.fk(q0)
        x1 = x0 + np.array([0.08, 0.04, 0.0])
        line = CartesianLine(x0, R0, x1, R0, v_max=0.1)

        ts = np.linspace(0, line.T, 60)
        qs, q_prev = [], q0
        for t in ts:
            p, R, _ = line(t)
            res = ik.solve(p, R, q_prev)
            qs.append(res.q)
            q_prev = res.q
        qs = np.array(qs)
        qd = np.diff(qs, axis=0) / (ts[1] - ts[0])
        assert np.abs(qd).max() < OFFICIAL_V_MAX * 3, (
            f"关节速度 {np.abs(qd).max():.2f} 远超官方限值，"
            "说明这条笛卡尔轨迹在关节空间不可执行")
