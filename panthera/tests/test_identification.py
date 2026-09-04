"""激励轨迹与辨识流水线的守护测试。

⭐ 这些测试守的是**读数是对的**，不是"代码能跑"。
最重要的是 :class:`TestResidualIsNotACriterion`——它把一条
反直觉的事实钉死：**训练集力矩残差越小，参数可能越错**。
"""

import numpy as np
import pytest

from panthera.core.robot import make_panthera, Q_HOME
from panthera.identification import excitation as ex
from panthera.identification import pipeline as pl
from panthera.identification.regressor import DynamicsRegressor

XML = "models/panthera/panthera.xml"
TAU_MAX = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])


@pytest.fixture(scope="module")
def rig():
    robot = make_panthera()
    reg = DynamicsRegressor(XML, n_arm=6)
    P, _ = reg.base_parameter_projection(samples=400)
    m = robot.model
    limits = ex.official_limits(m)
    return reg, P, limits


# ---------------------------------------------------------------- 轨迹本身

class TestFourierTrajectory:
    def test_derivatives_match_numerical_differentiation(self):
        """解析导数必须和数值差分对上。

        ⭐ 这条测的是"回归矩阵吃进去的 qdd 是对的"。如果解析式写错，
        辨识会得到一个**自洽但错误**的结果——因为 RNEA 和回归矩阵
        用的是同一个错 qdd，残差照样很小。
        """
        traj = ex.random_trajectory(np.zeros(6), n_harmonics=4, seed=1)
        t, h = 1.234, 1e-6
        _, qd, qdd = traj(t)
        q_p, qd_p, _ = traj(t + h)
        q_m, qd_m, _ = traj(t - h)
        np.testing.assert_allclose(qd, (q_p - q_m) / (2 * h), atol=1e-6)
        np.testing.assert_allclose(qdd, (qd_p - qd_m) / (2 * h), atol=1e-5)

    def test_zero_boundary_makes_start_velocity_zero(self):
        """真机安全条款：起始速度必须是零，不是"接近零"。"""
        traj = ex.random_trajectory(np.zeros(6), n_harmonics=5, seed=2)
        _, qd0, _ = traj.zero_boundary()(0.0)
        assert np.abs(qd0).max() < 1e-12

    def test_period_matches_base_frequency(self):
        traj = ex.random_trajectory(np.zeros(6), w_f=0.6, seed=3)
        q_a, _, _ = traj(0.0)
        q_b, _, _ = traj(traj.period)
        np.testing.assert_allclose(q_a, q_b, atol=1e-9)


# ------------------------------------------------------- 条件数怎么算才对

class TestConditionNumber:
    def test_unprojected_condition_number_is_structurally_singular(self, rig):
        """⚠️ 不投影 → 条件数顶穿双精度，且**与激励质量无关**。

        这条测试存在的意义：把"为什么必须投影"变成一个可执行的事实，
        而不是一句注释。
        """
        reg, P, _ = rig
        traj = ex.random_trajectory(np.array(Q_HOME), n_harmonics=5, seed=3)
        t = np.linspace(0, traj.period, 60, endpoint=False)
        q, qd, qdd = traj(t)
        W = np.vstack([reg.regressor(q[i], qd[i], qdd[i]) for i in range(len(t))])
        sv = np.linalg.svd(W, compute_uv=False)
        assert np.log10(sv[0] / max(sv[-1], 1e-300)) > 100

        projected = np.log10(ex.condition_number(traj, reg.regressor, P, 60))
        assert projected < 5

    def test_projection_rank_is_52(self, rig):
        """Panthera 的基参数维数。变了就说明模型被改过。"""
        _, P, _ = rig
        assert P.shape == (78, 52)

    def test_condition_number_distinguishes_good_from_bad(self, rig):
        reg, P, _ = rig
        bad = ex.FourierTrajectory(np.array(Q_HOME), np.zeros((6, 5)),
                                   np.zeros((6, 5)), 0.6)
        bad.a[1, 0] = 0.15
        good = ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.35, seed=3)
        k_bad = np.log10(ex.condition_number(bad.zero_boundary(),
                                             reg.regressor, P, 60))
        k_good = np.log10(ex.condition_number(good, reg.regressor, P, 60))
        assert k_bad - k_good > 10


# ------------------------------------------------------------------ 优化器

class TestOptimizer:
    @pytest.fixture(scope="class")
    def optimized(self, rig):
        reg, P, limits = rig
        start = ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.35, seed=3)
        best, hist = ex.optimize(start, reg.regressor, P, limits,
                                 iterations=250, n_samples=60, seed=1)
        return start, best, hist

    def test_optimizer_reduces_condition_number(self, rig, optimized):
        reg, P, _ = rig
        start, best, _ = optimized
        k0 = np.log10(ex.condition_number(start, reg.regressor, P, 60))
        k1 = np.log10(ex.condition_number(best, reg.regressor, P, 60))
        assert k1 < k0

    def test_optimized_trajectory_satisfies_all_constraints(self, rig, optimized):
        """⭐ 不可执行的最优解没有意义。约束是硬的。"""
        reg, P, limits = rig
        _, best, _ = optimized
        assert ex.violation(best, limits) == 0.0
        rep = ex.evaluate(best, reg.regressor, P, limits, rnea_fn=reg.rnea)
        assert rep["tau_saturation_pct"] == 0.0
        assert rep["qd_start"] < 1e-9

    def test_infeasible_start_is_repaired(self, rig, optimized):
        """起点违反限位时，优化器必须把它拉回可行域。"""
        reg, P, limits = rig
        start, best, _ = optimized
        assert ex.violation(start, limits) > 0
        assert ex.violation(best, limits) == 0.0


# --------------------------------------------------------- 辨识结果对不对

class TestIdentification:
    def test_noise_free_identification_is_near_exact(self, rig):
        """无噪声时，基参数必须被精确还原。做不到就是流水线本身有 bug。"""
        reg, P, limits = rig
        traj = ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.3, seed=7)
        Y, tau = pl.collect(traj, reg.regressor, reg.rnea, dt=0.004)
        res = pl.identify(Y, tau, P, pi_true=reg.true_parameters())
        assert res["rank"] == 52
        assert res["beta_error"] < 1e-8

    def test_full_parameter_error_stays_large_even_when_correct(self, rig):
        """⚠️ 反向守护：``pi_error`` 大是**正常的**，不是失败。

        不可辨识方向上的分量无法确定。谁要是把 ``pi_error`` 当判据，
        会得出"辨识永远失败"的错误结论。
        """
        reg, P, _ = rig
        traj = ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.3, seed=7)
        Y, tau = pl.collect(traj, reg.regressor, reg.rnea, dt=0.004)
        res = pl.identify(Y, tau, P, pi_true=reg.true_parameters())
        assert res["beta_error"] < 1e-8       # 基参数：准
        assert res["pi_error"] > 1e-3          # 完整参数：差得多，且正常


class TestResidualIsNotACriterion:
    """⭐⭐ 本文件最重要的一组：**训练集残差不能当判据**。

    实测（噪声 0.02 N·m）：

    ======================  ========  ==========  ==============
    轨迹                     log10κ    β 误差       训练力矩残差
    ======================  ========  ==========  ==============
    A 单关节慢摆              19.11     **49.9%**   **0.046%** ←最小
    C 优化后傅里叶             2.10     0.696%      0.065%  ←最大
    ======================  ========  ==========  ==============

    参数错了 50% 的那条，训练残差**反而最小**——因为它几乎不动，
    力矩又小又单调，当然好拟合。这是元教训 #10 的现场重演。
    """

    @pytest.fixture(scope="class")
    def two_runs(self, rig):
        reg, P, limits = rig
        q0 = np.array(Q_HOME)
        bad = ex.FourierTrajectory(q0, np.zeros((6, 5)), np.zeros((6, 5)), 0.6)
        bad.a[1, 0] = 0.15
        bad = bad.zero_boundary()
        good, _ = ex.optimize(ex.random_trajectory(q0, 5, 0.6, 0.35, seed=3),
                              reg.regressor, P, limits,
                              iterations=250, n_samples=60, seed=1)
        pi_true = reg.true_parameters()
        out = []
        for traj in (bad, good):
            Y, tau = pl.collect(traj, reg.regressor, reg.rnea,
                                dt=0.004, noise_std=0.02, seed=5)
            out.append((traj, pl.identify(Y, tau, P, pi_true=pi_true)))
        return out

    def test_bad_excitation_gives_wrong_parameters(self, two_runs):
        (_, bad_res), (_, good_res) = two_runs
        assert bad_res["beta_error"] > 0.1        # 错得离谱
        assert good_res["beta_error"] < 0.02      # 准

    def test_training_residual_fails_to_detect_the_bad_run(self, two_runs):
        """⭐ 核心断言：训练残差**没有**把坏轨迹标出来，反而更小。

        这条测试如果哪天变红，说明有人"修好"了残差指标——
        那反而要警惕：更可能是数据或判据被改动了。
        """
        (_, bad_res), (_, good_res) = two_runs
        assert bad_res["torque_error"] <= good_res["torque_error"]

    def test_holdout_residual_does_detect_the_bad_run(self, rig, two_runs):
        """⭐ 留出集是有效判据：它把坏轨迹拉开了两个数量级。"""
        reg, _, _ = rig
        holdout = ex.random_trajectory(np.array(Q_HOME), 5, 0.55, 0.3, seed=11)
        (_, bad_res), (_, good_res) = two_runs
        h_bad = pl.holdout_check(bad_res, holdout, reg.regressor, reg.rnea, dt=0.01)
        h_good = pl.holdout_check(good_res, holdout, reg.regressor, reg.rnea, dt=0.01)
        assert h_bad > 20 * h_good

    def test_condition_number_also_detects_the_bad_run(self, rig, two_runs):
        """条件数是**事前**判据：不用等辨识跑完就能否掉坏轨迹。"""
        reg, P, _ = rig
        (bad_traj, _), (good_traj, _) = two_runs
        k_bad = np.log10(ex.condition_number(bad_traj, reg.regressor, P, 60))
        k_good = np.log10(ex.condition_number(good_traj, reg.regressor, P, 60))
        assert k_bad > 10 > k_good


class TestNoiseAmplification:
    def test_error_scales_with_noise(self, rig):
        """噪声翻倍，参数误差大致同比例放大——这是条件数的物理含义。"""
        reg, P, limits = rig
        traj, _ = ex.optimize(
            ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.35, seed=3),
            reg.regressor, P, limits, iterations=250, n_samples=60, seed=1)
        pi_true = reg.true_parameters()
        errs = []
        for std in (0.01, 0.04):
            Y, tau = pl.collect(traj, reg.regressor, reg.rnea,
                                dt=0.004, noise_std=std, seed=5)
            errs.append(pl.identify(Y, tau, P, pi_true=pi_true)["beta_error"])
        assert 2.0 < errs[1] / errs[0] < 8.0


class TestOfficialLimits:
    """⚠️⚠️ 限值必须来自官方配置，不能自己填。

    本项目最初把 ``qd_max`` 拍成 2.0、**完全没有加速度约束**，
    优化出的"合规"轨迹 `实测` 超官方速度 **68%**、超加速度 **57%**——
    而 ``violation()`` 报 0，因为它照着错误的限值检查。

    ⭐ **一个照着错误标准检查的检查器，比没有检查器更危险。**

    权威出处 ``Panthera-HT_SDK/panthera_python/robot_param/Follower.yaml``::

        velocity_limits:     [1.0]*6
        acceleration_limits: [2.0]*6
    """

    def test_official_values_match_the_sdk_config(self):
        np.testing.assert_array_equal(ex.OFFICIAL_QD_MAX, np.full(6, 1.0))
        np.testing.assert_array_equal(ex.OFFICIAL_QDD_MAX, np.full(6, 2.0))
        np.testing.assert_array_equal(ex.SDK_TAU_MAX,
                                      [10.0, 20.0, 20.0, 10.0, 5.0, 5.0])

    def test_acceleration_is_actually_constrained(self, rig):
        """⭐ 不带 qdd_max 的约束对超加速度**完全无感**。

        这条测试证明加速度约束真的在起作用，而不只是个没人读的字段。
        """
        reg, P, limits = rig
        fast = ex.random_trajectory(np.array(Q_HOME), 5, 2.0, 0.6, seed=5)
        _, _, qdd = fast(np.linspace(0, fast.period, 200))
        assert np.abs(qdd).max() > 2.0                     # 确实超了

        no_acc = ex.TrajectoryLimits(
            limits.q_lower, limits.q_upper, limits.qd_max, limits.tau_max)
        assert ex.violation(fast, limits) > ex.violation(fast, no_acc)

    def test_optimized_trajectory_respects_official_limits(self, rig):
        """⭐ 端到端：优化结果必须同时满足官方速度**和**加速度限值。"""
        reg, P, limits = rig
        best, _ = ex.optimize(
            ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.35, seed=3),
            reg.regressor, P, limits, iterations=250, n_samples=60, seed=1)
        rep = ex.evaluate(best, reg.regressor, P, limits, rnea_fn=reg.rnea)

        assert rep["violation"] == 0.0
        assert rep["qd_max_abs"] <= 1.0                    # 官方速度限值
        assert rep["qdd_max_abs"] <= 2.0                   # 官方加速度限值
        assert rep["tau_saturation_pct"] == 0.0
        assert rep["cond_log"] < 3.0                       # 合规的代价可接受

    def test_compliance_costs_little_conditioning(self, rig):
        """⚠️ 记录代价：收紧到官方限值后 log10κ 从 2.10 涨到约 2.35。

        条件数从 126 涨到 224——`理论` 噪声放大约 1.8 倍，仍然完全可用。
        **合规的代价很小，没有理由为了好看的条件数去超限。**
        """
        reg, P, limits = rig
        loose = ex.TrajectoryLimits(
            limits.q_lower, limits.q_upper, np.full(6, 2.0), limits.tau_max)
        start = ex.random_trajectory(np.array(Q_HOME), 5, 0.6, 0.35, seed=3)
        k_official = ex.optimize(start, reg.regressor, P, limits,
                                 iterations=250, n_samples=60, seed=1)[1][-1][1]
        k_loose = ex.optimize(start, reg.regressor, P, loose,
                              iterations=250, n_samples=60, seed=1)[1][-1][1]
        assert k_official > k_loose                        # 确实有代价
        assert k_official - k_loose < 0.5                  # 但很小
