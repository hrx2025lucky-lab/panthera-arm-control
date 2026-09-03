"""模型一致性守护：让 sim2sim 的差异**只可能来自求解器**。

为什么需要这一组测试
--------------------
标准的 RL sim2real 流程是三段：

    IsaacLab（训练，几千并行） → MuJoCo（sim2sim 验证） → 真机

⭐ 中间那一步的价值在于**两个物理引擎是独立实现**（PhysX vs MuJoCo 求解器）：
策略在 A 里能跑、到 B 里就废 ⇒ 它过拟合了 A 的数值特性，而不是学到了物理。
这是「判据必须独立」在仿真环节的直接应用。

⚠️ **但这个判据有个前提：两边的模型必须是同一个模型。**

USD 和 MJCF 是**两条独立的转换链**，其中 ``armature``（转子反射惯量）最容易
在某一条里悄悄丢掉——它不是 URDF 的标准字段，各家导入器处理方式不一。

一旦丢了，后果非常隐蔽：

    sim2sim 对不上 → 你以为是"引擎差异"→ 实际是"模型没对齐"

两种原因混在一起，判据就脏了（armctrl 元教训 #10：
「我这个判据，除了我关心的那个原因，还有别的原因能让它变化吗？」）。

所以这一组测试逐项断言 MJCF 里的物理参数，并提供一个可对 USD 复用的导出，
让「模型一致」变成**每次提交都被检查**的事，而不是"我记得对过一次"。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from panthera.assets import panthera_xml
from panthera.core.robot import make_panthera

#: 参数手册 §2.2.2 给出的减速比
GEAR_RATIO = np.array([36, 36, 36, 36, 30, 30])

#: 力矩限幅（官方 SDK 示例 tau_limit）
TORQUE_LIMIT = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])


def physical_fingerprint() -> dict:
    """导出模型的物理指纹，用于跨引擎比对。

    ⭐ 只放**物理量**，不放几何/视觉——后者两个引擎本来就允许不同表示。
    把这份指纹在 Isaac 侧也导一遍，逐项对比，就能把"模型不一致"排除掉。
    """
    robot = make_panthera()
    model = robot.model
    return {
        "n_joints": int(robot.n),
        "link_mass": [round(float(v), 6) for v in model.body_mass[1:]],
        "armature": [round(float(v), 6) for v in model.dof_armature],
        "damping": [round(float(v), 6) for v in model.dof_damping],
        "frictionloss": [round(float(v), 6) for v in model.dof_frictionloss],
        "torque_limit": [round(float(v), 6) for v in robot.tau_limit],
        "joint_range": [[round(float(a), 6), round(float(b), 6)]
                        for a, b in model.jnt_range],
        "gear_ratio": GEAR_RATIO.tolist(),
    }


class TestPhysicalParametersArePresent(unittest.TestCase):
    """MJCF 里那些「CAD 给不出、必须补」的量，一个都不能缺。"""

    @classmethod
    def setUpClass(cls):
        cls.robot = make_panthera()
        cls.model = cls.robot.model

    def test_armature_matches_gear_ratio_scaling(self):
        """转子反射惯量必须随 N² 缩放。

        ⚠️ J1–J4 减速比 36、J5–J6 减速比 30，所以前四个关节的 armature
        应当明显大于后两个。若全部相等，说明有人填了个常数而没按 N² 算——
        那等于没有建模这一项。
        """
        armature = self.model.dof_armature
        self.assertTrue(np.all(armature > 0), "armature 为 0 ⇒ 转子惯量被忽略")
        ratio = armature[0] / armature[4]
        expect = (36 / 30) ** 2
        self.assertAlmostEqual(
            ratio, expect, places=3,
            msg=f"armature 的比值 {ratio:.4f} 与 (36/30)²={expect:.4f} 不符，"
                "检查是否按 N² 缩放")

    def test_armature_is_comparable_to_link_inertia(self):
        """⭐ 反射惯量必须与连杆惯量同量级，否则它形同虚设。

        减速比 36 ⇒ 放大 1296 倍。link2 的 izz≈0.0227，
        典型转子 1e-5 反射后是 0.013——**57%**，不能当成小量忽略。
        """
        armature_j2 = self.model.dof_armature[1]
        self.assertGreater(armature_j2, 0.005,
                           "反射惯量太小，仿真里的臂会比真机轻快得多")

    def test_friction_is_nonzero(self):
        """摩擦不能为 0。⚠️ 当前是占位值，辨识后必须回填。"""
        self.assertTrue(np.all(self.model.dof_frictionloss > 0))
        self.assertTrue(np.all(self.model.dof_damping > 0))

    def test_torque_limits_match_sdk(self):
        """力矩限幅必须与官方 SDK 示例一致，且未被放开。"""
        np.testing.assert_allclose(self.robot.tau_limit, TORQUE_LIMIT)


class TestFingerprintIsStable(unittest.TestCase):
    """物理指纹本身要能稳定导出——它是跨引擎比对的依据。"""

    def test_fingerprint_has_all_fields(self):
        fingerprint = physical_fingerprint()
        for key in ("link_mass", "armature", "damping", "frictionloss",
                    "torque_limit", "joint_range", "gear_ratio"):
            self.assertIn(key, fingerprint)
            self.assertEqual(len(fingerprint[key]), 6,
                             f"{key} 应当有 6 项（6 个关节/连杆）")

    def test_fingerprint_is_json_serialisable(self):
        """⭐ 必须能落盘成 JSON，才能拿到 Isaac 那边逐项 diff。"""
        text = json.dumps(physical_fingerprint(), ensure_ascii=False)
        self.assertGreater(len(text), 100)


class TestModelFileIsSelfContained(unittest.TestCase):
    """模型随仓库分发，clone 下来就该能用。

    ⭐ armctrl 元教训 #66：「在我这台机器上是好的」和「别人 clone 下来是好的」
    是两件事，而且只有真的 clone 一次才知道差在哪。
    """

    def test_mjcf_exists_and_uses_relative_mesh_paths(self):
        path = Path(panthera_xml())
        self.assertTrue(path.is_file(), "模型文件缺失")
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("/home/", text, "MJCF 里残留了本机绝对路径")
        self.assertNotIn("/tmp/", text, "MJCF 里残留了临时目录路径")
        self.assertIn('meshdir="meshes"', text)

    def test_all_referenced_meshes_exist(self):
        path = Path(panthera_xml())
        text = path.read_text(encoding="utf-8")
        import re
        missing = [name for name in re.findall(r'file="([^"]+)"', text)
                   if not (path.parent / "meshes" / name).is_file()]
        self.assertEqual(missing, [], f"MJCF 引用的网格不存在：{missing}")


if __name__ == "__main__":
    print(json.dumps(physical_fingerprint(), ensure_ascii=False, indent=2))
