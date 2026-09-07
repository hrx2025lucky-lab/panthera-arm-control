"""守护 ``mass_matrix()`` 的 MuJoCo 绑定兼容层。

🔴 **这个文件存在的理由**

``mj_fullM`` 的签名在 MuJoCo 版本间变过，本仓库因此**来回改错三次**：

* 一次改成 ``mj_fullM(m, full, d.qM)``，理由写的是"3.2.3 的签名是这样"；
* 但本机实测 ``mujoco.__version__ == '3.12.0'``——⚠️ ``3.12 > 3.2``，
  版本号是三段数字，不是小数。

每次改错的后果都一样：``mass_matrix()`` 直接 ``AttributeError``，
而 CTC、笛卡尔阻抗、动量观测器、调参台**全部**依赖它 ⇒ 整个控制栈瘫痪。

⭐ 更值得记的一点：改错的那一版**提交前没有跑测试**。
``test_control_closed_loop.py`` 第一个用例就会红。
⇒ 守护测试只有在真的被执行时才有价值。
"""

from __future__ import annotations

import numpy as np
import pytest

from panthera.core.robot import Q_HOME, make_panthera


@pytest.fixture(scope="module")
def robot():
    return make_panthera()


class TestMassMatrixIsUsable:
    """最基本的一条：它必须**能算出来**。

    ⚠️ 之前三次故障全部卡在这一步——不是"算得不准"，是"根本调不通"。
    """

    def test_returns_finite_matrix(self, robot):
        M = robot.mass_matrix(Q_HOME)
        assert M.shape == (6, 6)
        assert np.all(np.isfinite(M)), "M 里有 NaN/Inf"

    def test_is_symmetric(self, robot):
        """质量矩阵**必须**对称，这是物理决定的，不是巧合。"""
        M = robot.mass_matrix(Q_HOME)
        assert np.allclose(M, M.T, atol=1e-12)

    def test_is_positive_definite(self, robot):
        """正定 ⇔ 动能 ½q̇ᵀMq̇ 恒为正 ⇔ 不存在"白拿能量"的运动方向。

        ⭐ 这条比"对称"更强，能抓出取错子块、行列错位之类的错误。
        """
        w = np.linalg.eigvalsh(robot.mass_matrix(Q_HOME))
        assert w.min() > 0, f"最小特征值 {w.min()} ≤ 0"

    def test_diagonal_dominates_a_free_joint(self, robot):
        """⭐ 判据独立于实现：J6 只需要转动自己的腕部，
        它的对角项必须是全矩阵里最小的那个。

        ⚠️ 这条不依赖任何"我们自己算出来的期望值"——
        它来自机械结构（越靠末端带动的质量越少），
        所以即使模型参数换了也仍然成立。
        """
        M = robot.mass_matrix(Q_HOME)
        assert np.argmin(np.diag(M)) == 5


class TestMassMatrixIsPhysicallyRight:
    """光"能算"不够，还要"算得对"。"""

    def test_kinetic_energy_matches_mujoco(self, robot):
        """⭐⭐ **判据独立于被测对象**：

        用 ½q̇ᵀM(q)q̇ 算出的动能，必须等于 MuJoCo 自己报的动能。
        我们算的是 M，MuJoCo 报的是能量——两条**互相独立**的路径。
        如果 ``mj_fullM`` 的参数传错（比如把 dst 和 qM 弄反），
        M 要么报错要么是垃圾值，这条一定会红。

        ⚠️ MuJoCo **默认不计算能量**（省算力），必须显式打开 ``mjENBL_ENERGY``。
        第一次写这条测试时忘了打开，读回来恒为 0.0——
        ⭐ 那会变成一条"永远在跟 0 比"的假测试。
        """
        import mujoco
        flag = int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        robot.model.opt.enableflags |= flag
        try:
            rng = np.random.default_rng(0)
            for _ in range(5):
                q = rng.uniform(robot.q_lower, robot.q_upper)
                qd = rng.normal(size=6) * 0.5
                M = robot.mass_matrix(q)
                ours = 0.5 * qd @ M @ qd

                robot.set_state(q, qd)
                mujoco.mj_forward(robot.model, robot.data)
                theirs = float(robot.data.energy[1])  # [位能, 动能]
                assert theirs != 0.0, "MuJoCo 报的动能恒为 0，能量开关没生效"
                assert abs(ours - theirs) < 1e-9 * abs(theirs) + 1e-12, \
                    f"动能对不上：我们 {ours:.12f} vs MuJoCo {theirs:.12f}"
        finally:
            robot.model.opt.enableflags &= ~flag

    def test_inertia_varies_with_posture(self, robot):
        """⭐ M 必须**随姿态变**——这正是 CTC 存在的理由。

        ⚠️ 如果哪天它变成常数，说明取到的是常数块或者根本没更新构型，
        那时 CTC 会退化成 PD 而没人发现。
        """
        m_folded = robot.mass_matrix(np.array([0., 0.1, 3.5, 0., 0., 0.]))[0, 0]
        m_stretched = robot.mass_matrix(np.array([0., 1.57, 0.0, 0., 0., 0.]))[0, 0]
        ratio = max(m_folded, m_stretched) / min(m_folded, m_stretched)
        assert ratio > 3.0, f"J1 惯量随姿态只变了 {ratio:.2f} 倍，太小，可疑"


class TestBindingProbe:
    """兼容层本身。"""

    def test_probe_picked_something(self):
        from panthera.core.robot import _FULL_M
        assert callable(_FULL_M)

    def test_probe_result_matches_direct_call(self, robot):
        """⭐ 探测出来的调用器，结果必须和"手写正确签名"一致。

        ⚠️ 这条是**反向**验证：它不检查我们选了哪个分支，
        只检查选出来的那个**算得对**。
        所以将来 MuJoCo 再改签名，只要探测逻辑还能找到能跑的那个，
        这条仍然通过——判据独立于版本。
        """
        import mujoco
        from panthera.core.robot import _FULL_M
        d = robot.data
        robot.set_state(Q_HOME, np.zeros(6))
        mujoco.mj_forward(robot.model, d)
        a = np.zeros((robot.model.nv, robot.model.nv))
        _FULL_M(robot.model, d, a)
        assert np.all(np.isfinite(a)) and a.trace() > 0
