"""调参台四个场景的守护测试。

⭐ 这一组测试守的不是「代码能跑」，而是「**读数是对的**」。

迁移调参台时踩到的三个坑，全部是**指标测错了对象**——代码不报错、
界面看着也正常，但那个数字反映的不是你以为的东西。这类错误只能靠
「拿已知答案去考它」抓出来，所以每条测试都配一个独立可验的判据。

三个坑分别是：

1. **CTC 漏了 J̇q̇ 项** ⇒ CTC 相对 PD 的优势被系统性削弱
2. **RMS 把启动瞬态算进去** ⇒ 稳态差异被淹没
3. ⭐⭐ **条件数没投影到基参数子空间** ⇒ 恒为 10¹⁸，**调什么都不变**

第 3 条最典型：一个「永远不变」的指标比没有指标更危险，
因为它会让你以为自己在监控。（armctrl 元教训 #28）
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from panthera.core.robot import Q_HOME
from panthera.tuner.scenes import (SCENES, SCENE_BY_NAME, GravityScene,
                                   ImpedanceScene, IdentifyScene, TrackingScene)


def run_scene(scene_cls, steps: int, **params):
    """跑一个场景若干步，返回最后一次遥测。"""
    scene = scene_cls()
    scene.build()
    scene.reset()
    for key, value in params.items():
        scene.set(key, value)
    telemetry = {}
    for _ in range(steps):
        q = scene.data.qpos[scene.robot.qpos_idx].copy()
        v = scene.data.qvel[scene.robot.qvel_idx].copy()
        scene.data.ctrl[scene.robot.arm_actuator_ids] = scene.control(
            scene.data.time, q, v)
        mujoco.mj_step(scene.model, scene.data)
        telemetry = scene.telemetry(scene.data.time, q, v)
    return telemetry


class TestAllScenesRun(unittest.TestCase):
    """四个场景都能跑起来，且声明的读数都真的被产出。"""

    def test_scene_registry(self):
        self.assertEqual(len(SCENES), 4)
        self.assertEqual(set(SCENE_BY_NAME), {"gravity", "impedance",
                                              "tracking", "identify"})

    def test_every_declared_readout_is_produced(self):
        """⭐ 声明了读数却不产出，前端会显示空白而不报错——必须测。"""
        for scene_cls in SCENES:
            with self.subTest(scene=scene_cls.name):
                telemetry = run_scene(scene_cls, 300)
                declared = {r.key for r in scene_cls.readouts}
                missing = declared - set(telemetry)
                self.assertEqual(missing, set(),
                                 f"{scene_cls.name} 声明了却没产出：{missing}")

    def test_torque_never_exceeds_limit(self):
        """⚠️ 任何场景都不得下发超限力矩——这条守的是真机安全。"""
        for scene_cls in SCENES:
            with self.subTest(scene=scene_cls.name):
                scene = scene_cls()
                scene.build()
                scene.reset()
                for _ in range(500):
                    q = scene.data.qpos[scene.robot.qpos_idx].copy()
                    v = scene.data.qvel[scene.robot.qvel_idx].copy()
                    tau = scene.control(scene.data.time, q, v)
                    self.assertTrue(
                        np.all(np.abs(tau) <= scene.robot.tau_limit + 1e-9),
                        f"{scene_cls.name} 下发了超限力矩")
                    scene.data.ctrl[scene.robot.arm_actuator_ids] = tau
                    mujoco.mj_step(scene.model, scene.data)


class TestGravityScene(unittest.TestCase):

    def test_full_compensation_holds_still(self):
        """η=1 时应当停住不动。"""
        telemetry = run_scene(GravityScene, 1500, comp_ratio=1.0, damping=0.5)
        self.assertLess(telemetry["drift"], 1.0, "完全补偿下不该漂移")

    def test_under_compensation_sinks(self):
        """⭐ 独立判据：η=0.5 必须明显下沉。

        只测「η=1 不动」是不够的——如果 gravity() 恒返回一个能托住手臂的
        常数，那条测试照样通过。必须再验一个**已知会失败**的工况。
        """
        telemetry = run_scene(GravityScene, 1500, comp_ratio=0.5, damping=0.5)
        self.assertGreater(telemetry["drift"], 10.0,
                           "只补一半重力却没下沉，检查 gravity() 是否真的在用")


class TestImpedanceScene(unittest.TestCase):
    """⭐ 核心判据：稳态偏移精确等于 F/K，且判据独立于实现。"""

    def test_steady_state_matches_f_over_k(self):
        telemetry = run_scene(ImpedanceScene, 8000, k_trans=500.0, fz=-10.0)
        self.assertLess(telemetry["err_pct"], 1.0,
                        f"实测 {telemetry['dz']:.3f} mm vs "
                        f"理论 {telemetry['dz_theory']:.3f} mm")

    def test_stiffness_scaling(self):
        """⭐ K 翻倍，偏移减半——比单点吻合更强的判据。"""
        soft = run_scene(ImpedanceScene, 8000, k_trans=250.0, fz=-10.0)
        stiff = run_scene(ImpedanceScene, 8000, k_trans=1000.0, fz=-10.0)
        ratio = abs(soft["dz"]) / abs(stiff["dz"])
        self.assertAlmostEqual(ratio, 4.0, delta=0.2,
                               msg=f"K 变 4 倍，偏移比应为 4，实测 {ratio:.2f}")


class TestTrackingScene(unittest.TestCase):

    def test_ctc_beats_pd(self):
        """⭐ CTC 应优于 PD+重力补偿。

        ⚠️ 这条曾经几乎测不出差异，根因是 CTC 漏了 J̇q̇ 项
        （ẍ = J q̈ + J̇ q̇，映射回关节空间时必须减掉它）。
        漏掉它会**系统性削弱 CTC**，而代码不会报任何错。
        """
        period = 1.5
        steps = int(period * 4 / 0.002)
        pd = run_scene(TrackingScene, steps, law="PD+重力", period=period)
        ctc = run_scene(TrackingScene, steps, law="计算力矩", period=period)
        self.assertLess(ctc["e_rms"], pd["e_rms"] * 0.95,
                        f"CTC({ctc['e_rms']:.2f}) 未明显优于 "
                        f"PD({pd['e_rms']:.2f})，检查 J̇q̇ 项")

    def test_jacobian_derivative_term_is_computed(self):
        """⭐⭐ 直接测 J̇q̇ 项，而不是通过 RMS 间接推断。

        ⚠️ 教训：最初这条是靠「CTC 的 RMS 比 PD 好多少」来判的，
        结果把 J̇q̇ 整项删掉后测试**照样全绿**——因为该项只贡献约 5% 的 RMS，
        被判据的容差吃掉了。

        ⭐ 元教训 #10：**判据必须直接测你要的那个物理量，不能用代理量。**
        RMS 是代理量（它同时受 Kp/Kd、惯量项、摩擦影响）；
        ‖J̇q̇‖ 才是要测的那个量本身。
        """
        telemetry = run_scene(TrackingScene, 1500, period=1.5, radius=0.10)
        self.assertGreater(telemetry["jdot_v"], 1e-3,
                           "‖J̇q̇‖ 恒为 0，说明这一项没有被计算")

    def test_jacobian_derivative_scales_with_speed(self):
        """J̇q̇ 含两个速度因子，应随速度平方增长（周期减半 ⇒ 约 4 倍）。"""
        slow = run_scene(TrackingScene, 1500, period=3.0, radius=0.10)
        fast = run_scene(TrackingScene, 1500, period=1.5, radius=0.10)
        ratio = fast["jdot_v"] / max(slow["jdot_v"], 1e-12)
        self.assertGreater(ratio, 2.0,
                           f"‖J̇q̇‖ 随速度增长不足（{ratio:.2f}×），"
                           f"检查它是否真的由速度算出")

    def test_rms_skips_the_startup_transient(self):
        """⭐ 直接测「瞬态被跳过」，而不是指望它在对照里显现出来。

        ⚠️ 又一次同样的教训：最初没有这条，把「跳过瞬态」这行删掉后
        所有对照测试**照样全绿**——因为 PD 和 CTC 吃到的是同一段瞬态，
        比值不受影响。

        但 e_rms 是**四象限实验的主指标**：比的是 CAD 模型 vs 辨识模型，
        瞬态混进去会稀释掉真正要看的稳态差异。
        ⭐ 所以必须直接盯住「有没有跳」，而不是盯住某个比值。
        """
        period = 2.0
        # 跑不满一个周期：RMS 不该计入任何样本
        early = run_scene(TrackingScene, int(period * 0.8 / 0.002), period=period)
        self.assertEqual(early["n_rms"], 0.0,
                         "第一个周期内就开始统计 RMS，瞬态会污染主指标")

        # 跑满两个周期：应当已经计入约一个周期的样本
        later = run_scene(TrackingScene, int(period * 2 / 0.002), period=period)
        self.assertGreater(later["n_rms"], period / 0.002 * 0.8,
                           "过了瞬态期却没开始统计")

    def test_faster_motion_widens_the_gap(self):
        """⭐⭐ 越快，CTC 的优势应越大——因为惯量项才是它补的东西。

        这条比上一条强：它验的是**因果方向**，不只是「谁大谁小」。
        如果 CTC 的优势与速度无关，说明它补的不是惯量。
        """
        def gap(period):
            steps = int(period * 4 / 0.002)
            pd = run_scene(TrackingScene, steps, law="PD+重力", period=period)
            ctc = run_scene(TrackingScene, steps, law="计算力矩", period=period)
            return (pd["e_rms"] - ctc["e_rms"]) / pd["e_rms"]

        self.assertGreater(gap(1.5), gap(2.5),
                           "运动变快时 CTC 的优势没有扩大，检查惯量项")

    def test_reference_acceleration_scales_with_omega_squared(self):
        """参考加速度应为 R·ω²（`理论` 档，独立解析算出）。"""
        slow = run_scene(TrackingScene, 600, period=3.0, radius=0.10)
        fast = run_scene(TrackingScene, 600, period=1.5, radius=0.10)
        self.assertAlmostEqual(fast["a_tcp"] / slow["a_tcp"], 4.0, delta=0.1)


class TestIdentifyScene(unittest.TestCase):
    """⭐⭐ 这一组守的是「条件数这个读数真的有分辨力」。"""

    def test_base_rank_is_structural(self):
        """基参数秩只跟模型有关，调轨迹不该改变它。"""
        a = run_scene(IdentifyScene, 400, n_harm=1)
        b = run_scene(IdentifyScene, 400, n_harm=5)
        self.assertEqual(a["base_rank"], b["base_rank"])
        self.assertGreater(a["base_rank"], 0, "基参数投影没算出来")

    def test_condition_number_responds_to_excitation(self):
        """⭐⭐ 最关键的一条：条件数必须随激励质量变化。

        ⚠️ 迁移时踩过的坑：对**完整**参数集算条件数，得到的恒是 10¹⁸——
        那反映的是**结构性不可辨识**（78 维里有 26 维任何轨迹都辨不出来），
        与「这条轨迹好不好」无关，而且**不管怎么调都不变**。

        ⭐ 一个永远不变的指标，比没有指标更危险：它会让你以为自己在监控。
        （armctrl 元教训 #28）

        修法是先投影到基参数子空间再算。这条测试就是防止有人把投影去掉。
        """
        weak = run_scene(IdentifyScene, 2500, n_harm=3, amp=0.1)
        strong = run_scene(IdentifyScene, 2500, n_harm=3, amp=0.5)
        self.assertGreater(
            weak["cond_log"] - strong["cond_log"], 0.5,
            f"幅值 0.1→0.5 条件数几乎没变"
            f"（{weak['cond_log']:.2f} → {strong['cond_log']:.2f}）——"
            f"检查是否漏了基参数投影")

    def test_condition_number_is_finite_and_meaningful(self):
        """条件数应落在有意义的范围，而不是顶到数值上限。"""
        telemetry = run_scene(IdentifyScene, 2500, n_harm=3, amp=0.3)
        self.assertGreater(telemetry["cond_log"], 0.0)
        self.assertLess(telemetry["cond_log"], 16.0,
                        "条件数顶到双精度上限，说明投影没生效")


if __name__ == "__main__":
    unittest.main()
