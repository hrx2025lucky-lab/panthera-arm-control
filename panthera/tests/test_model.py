"""模型与控制栈的冒烟测试。

这一组测试守的是**迁移过程中最容易悄悄坏掉**的东西：
模型能不能加载、维度对不对、力矩限幅有没有被放开、阻抗控制还成不成立。

⚠️ 与 armctrl 的一个重要差别：那边是纯仿真，力矩限幅放开无所谓；
这边的代码要原样下发到真机，**限幅被放开就是安全事故**，所以专门有一条测试锁它。
"""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from panthera.core.robot import make_panthera, Q_HOME, PANTHERA_TCP_OFFSET


class TestModelLoads(unittest.TestCase):
    """模型本身的结构性质。"""

    @classmethod
    def setUpClass(cls):
        cls.robot = make_panthera()

    def test_six_dof_arm(self):
        self.assertEqual(self.robot.n, 6, "Panthera 是 6 轴")
        self.assertEqual(self.robot.model.nq, 6)
        self.assertEqual(self.robot.model.nv, 6)

    def test_actuators_exist(self):
        """⚠️ 官方 URDF 里 nu=0，力矩控制无从谈起。转换脚本必须补上执行器。"""
        self.assertEqual(self.robot.model.nu, 6,
                         "没有执行器 ⇒ 力矩控制不可用，转换脚本失效了")

    def test_torque_limits_are_not_disabled(self):
        """⭐ 力矩限幅是保护真机的最后一道闸，任何时候都不许被放开。

        armctrl 的 `_convert_to_torque_actuators` 会把 ctrlrange 设成 ±1e6；
        本项目刻意去掉了那两行。这条测试就是防止它被"顺手改回去"。
        """
        expect = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])
        np.testing.assert_allclose(self.robot.tau_limit, expect,
                                   err_msg="力矩上限被改动，真机会有风险")
        for i in self.robot.arm_actuator_ids:
            self.assertTrue(self.robot.model.actuator_ctrllimited[i],
                            "ctrllimited 被关掉了")
            hi = self.robot.model.actuator_ctrlrange[i, 1]
            self.assertLess(hi, 100.0, f"执行器 {i} 的 ctrlrange 被放开到 {hi}")

    def test_rotor_inertia_is_present(self):
        """转子反射惯量必须存在，且与连杆惯量同量级。

        ⚠️ 这一项 CAD 导不出来（不是连杆的几何属性），是转换脚本补的。
        减速比 36 ⇒ 反射放大 1296 倍，忽略它会让仿真的"轻快程度"完全失真。
        """
        armature = self.robot.model.dof_armature
        self.assertTrue(np.all(armature > 0), "armature 全为 0 ⇒ 转子惯量被忽略")
        # link2 的 izz ≈ 0.0227；反射惯量应当可比，而不是小两个数量级
        self.assertGreater(armature[1], 0.005,
                           "反射惯量太小，检查 ROTOR_INERTIA_GUESS 与减速比")

    def test_friction_is_present(self):
        """关节摩擦必须存在。⚠️ 目前是占位值，辨识后要回填。"""
        self.assertTrue(np.all(self.robot.model.dof_frictionloss > 0),
                        "库仑摩擦为 0 ⇒ 仿真比真机滑得多，sim2real 必然失败")
        self.assertTrue(np.all(self.robot.model.dof_damping > 0),
                        "粘滞摩擦为 0")


class TestKinematics(unittest.TestCase):
    """运动学的基本自洽性。"""

    @classmethod
    def setUpClass(cls):
        cls.robot = make_panthera()

    def test_jacobian_shape_and_rank(self):
        jac = self.robot.jacobian(Q_HOME)
        self.assertEqual(jac.shape, (6, 6))
        self.assertEqual(np.linalg.matrix_rank(jac), 6,
                         "Q_HOME 处不该是奇异位形")

    def test_no_redundancy(self):
        """⭐ 6 轴对 6 维任务空间没有冗余，零空间维度必须是 0。

        这不是缺陷，是 6 轴工业臂的常态。写成测试是为了防止有人
        照搬 armctrl（7 轴 Panda）的零空间代码却以为它在起作用——
        那边 N 的秩是 1，这边恒等于 0，投影结果永远是零向量。
        """
        jac = self.robot.jacobian(Q_HOME)
        null_proj = np.eye(self.robot.n) - jac.T @ np.linalg.pinv(jac).T
        self.assertEqual(np.linalg.matrix_rank(null_proj, tol=1e-9), 0)
        self.assertLess(np.linalg.norm(null_proj), 1e-9)

    def test_tcp_is_offset_from_link6(self):
        """TCP 必须在法兰外侧，不能等于连杆原点。"""
        p_tcp, _ = self.robot.fk(Q_HOME)
        flange = make_panthera(tcp="link6")
        p_flange, _ = flange.fk(Q_HOME)
        dist = np.linalg.norm(p_tcp - p_flange)
        self.assertAlmostEqual(dist, np.linalg.norm(PANTHERA_TCP_OFFSET),
                               places=6)


class TestGravityCompensation(unittest.TestCase):
    """重力补偿——真机上的第一个实验，也是最基本的正确性判据。"""

    def test_gravity_holds_the_arm_still(self):
        robot = make_panthera()
        robot.set_state(Q_HOME)
        for _ in range(1500):
            q = robot.data.qpos[robot.qpos_idx].copy()
            robot.data.ctrl[robot.arm_actuator_ids] = robot.saturate_torque(
                robot.gravity(q))
            mujoco.mj_step(robot.model, robot.data)
        drift = np.abs(robot.data.qpos[robot.qpos_idx] - Q_HOME).max()
        self.assertLess(drift, 1e-3, f"纯重力补偿下漂移 {drift:.4f} rad")

    def test_gravity_torque_is_within_rated_limits(self):
        """重力力矩不能超过额定扭矩，否则这个姿态在真机上保持不住。"""
        robot = make_panthera()
        rated = np.array([6.0, 10.0, 10.0, 6.0, 6.0, 6.0])  # 参数手册额定值
        self.assertTrue(np.all(np.abs(robot.gravity(Q_HOME)) < rated))


class TestImpedanceControl(unittest.TestCase):
    """⭐ 阻抗控制的核心判据：稳态偏移精确等于 F/K。

    判据独立于被测实现——理论值由解析式 F/K 算出，不读控制器的任何中间量。
    """

    def test_steady_state_deflection_equals_force_over_stiffness(self):
        robot = make_panthera()
        robot.set_state(Q_HOME)
        p0, _ = robot.fk(Q_HOME)

        stiffness = 500.0
        damping = 40.0
        f_ext = np.array([0.0, 0.0, -10.0])

        for _ in range(8000):
            q = robot.data.qpos[robot.qpos_idx].copy()
            qd = robot.data.qvel[robot.qvel_idx].copy()
            p, _ = robot.fk(q)
            jac = robot.jacobian(q)[:3]
            force = stiffness * (p0 - p) - damping * (jac @ qd)
            tau = jac.T @ (force + f_ext) + robot.gravity(q)
            robot.data.ctrl[robot.arm_actuator_ids] = robot.saturate_torque(tau)
            mujoco.mj_step(robot.model, robot.data)

        measured = (robot.fk(robot.data.qpos[robot.qpos_idx])[0] - p0)[2]
        theory = f_ext[2] / stiffness
        self.assertAlmostEqual(measured, theory, places=4,
                               msg=f"实测 {measured * 1000:.3f} mm "
                                   f"vs 理论 {theory * 1000:.3f} mm")


if __name__ == "__main__":
    unittest.main()
