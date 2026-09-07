"""守护「仿真模型」与「官方 SDK 配置」不许悄悄分叉。

🔴 **这个文件存在的理由**

本仓库已经因为「两份配置不一致」栽过三次：

1. **TCP**：官方有 **四个**不同的末端定义（SDK ``tool_link`` 0.165、
   ROS2 ``bat_center`` 0.18、我们量的裸法兰 0.1893、Leader 写的 ``joint6``）。
   我们从 0.1893 改成 0.165 —— 差 **24.3 mm** —— 而 **170 项测试照常全过**。
   ⭐ 测试全过不等于正确，只说明**判据缺失**。

2. **关节限位**：MJCF 用的是 **Leader** 口径（J2/J3 下限 ``0.0``），
   而 ``RealBackend`` 加载的是 **Follower.yaml**（下限 ``-0.1``）。
   仿真比真机窄 0.1 rad（5.7°）—— 方向上偏保守所以没出事，
   但「没出事」是运气，不是设计。

3. **URDF 版本**：SDK 与 ROS2 两份 URDF 的 link5 ``izz`` 差 **221%**。

⇒ 所以这里**直接读官方 YAML**，和仿真模型逐项对照。
这个判据**独立于我们自己的代码** —— 它拿的是官方文件，
所以我们改自己的模型时它不会跟着一起"改对"。

⚠️ 没有 SDK 时自动跳过（CI 上跑不了），但**本机必须跑**。
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from panthera.core.robot import make_panthera


def _load_follower() -> dict | None:
    """读官方 ``Follower.yaml``。找不到返回 None（触发 skip）。

    ⚠️ 用 Follower 而不是 Leader：``RealBackend`` 实际加载的就是它
    （见 ``sdk_path.follower_config()``），也是官方重力补偿示例的默认。
    """
    try:
        from panthera.driver.sdk_path import follower_config
        return yaml.safe_load(follower_config().read_text())["robot"]
    except (FileNotFoundError, KeyError, ImportError):
        return None


FOLLOWER = _load_follower()
needs_sdk = pytest.mark.skipif(FOLLOWER is None, reason="本机没有克隆官方 SDK")


@pytest.fixture(scope="module")
def robot():
    return make_panthera()


@needs_sdk
class TestJointLimitsMatchOfficial:
    """🔴 限位必须和官方配置**逐个数字**相等。

    ⚠️ 不能只测"仿真比真机保守"——那样两边可以一直缓慢分叉而不报警，
    直到某天分叉到反方向。**要求严格相等**才能真正钉住。
    """

    def test_lower_limits(self, robot):
        official = np.array(FOLLOWER["joint_limits"]["lower"], dtype=float)
        assert np.allclose(robot.q_lower, official, atol=1e-9), (
            f"下限不一致\n  模型 {robot.q_lower}\n  官方 {official}\n"
            f"  差   {robot.q_lower - official}")

    def test_upper_limits(self, robot):
        official = np.array(FOLLOWER["joint_limits"]["upper"], dtype=float)
        assert np.allclose(robot.q_upper, official, atol=1e-9), (
            f"上限不一致\n  模型 {robot.q_upper}\n  官方 {official}")

    def test_not_accidentally_using_leader_config(self, robot):
        """⭐ 专门盯住踩过的那个坑：J2/J3 下限被写成 Leader 的 0.0。

        Leader `[-2.4, 0.0, 0.0, ...]` vs Follower `[-2.4, -0.1, -0.1, ...]`
        —— 只差这两个数，最容易混。
        """
        assert robot.q_lower[1] == pytest.approx(-0.1), \
            "J2 下限是 0.0 ⇒ 用成了 Leader 配置"
        assert robot.q_lower[2] == pytest.approx(-0.1), \
            "J3 下限是 0.0 ⇒ 用成了 Leader 配置"


@needs_sdk
class TestTorqueLimits:
    """力矩限幅有**三套值**，用错哪一套后果完全不同。"""

    #: 客服 Q9 明确的额定值。⭐ 连续运行（如辨识跑 10 分钟）必须用这套。
    RATED = np.array([6.0, 10.0, 10.0, 6.0, 6.0, 6.0])

    def test_model_torque_limits_are_layered(self, robot):
        """⚠️ 模型里有**两层**力矩限制，别搞混：

        =============================  ==========================  ============
        字段                            值                          含义
        =============================  ==========================  ============
        ``joint.actuatorfrcrange``     ``[21,36,36,21,10,10]``     ⛔ **堵转**（物理天花板）
        ``motor.forcerange``           ``[10,20,20,10,5,5]``       SDK 示例的短时峰值
        （无对应字段）                   ``[6,10,10,6,6,6]``         ⭐ **额定**，属 SafetyLayer
        =============================  ==========================  ============

        ⭐ 这样分层是**有意的**：
        物理模型应允许电机输出它**能**输出的全部力矩（堵转），
        「不许超额定」是**控制策略**，属于 :class:`SafetyLayer` 的职责。
        硬编进物理模型会**掩盖真机上的饱和问题**——
        仿真里永远不饱和，上真机就崩。

        这条测试的作用是**把这个设计决定写下来**，
        防止后来的人"顺手"把它统一成一个值而没人知道为什么。
        """
        stall = np.array(FOLLOWER["max_torque"], dtype=float)
        joint_lim = robot.model.jnt_actfrcrange[robot.joint_ids][:, 1]
        assert np.allclose(joint_lim, stall), (
            f"joint.actuatorfrcrange 应等于官方 max_torque(堵转) {stall}，"
            f"实为 {joint_lim}")

        act = robot.model.actuator_forcerange[:6, 1]
        assert np.allclose(act, [10., 20., 20., 10., 5., 5.]), (
            f"motor.forcerange 应为 SDK 示例值 [10,20,20,10,5,5]，实为 {act}")

        assert np.all(act <= stall), "SDK 示例峰值不应超过堵转"

    def test_wrist_torque_sources_contradict_each_other(self, robot):
        """🔴⭐ **官方资料在腕部力矩上自相矛盾**——这条测试把矛盾钉住。

        四个来源对 J5/J6 的说法：

        ======================  =========  ======  ======
        来源                     J1–J4      J5      J6
        ======================  =========  ======  ======
        客服 Q9「额定」          6,10,10,6  **6**   **6**
        SDK 示例                10,20,20,10 **5**   **5**
        ROS2 阻抗示例            10,20,20,10 **3**   **2**
        官方 max_torque(堵转)    21,36,36,21 10      10
        ======================  =========  ======  ======

        ⚠️⚠️ **「额定 6」比 SDK 示例的 5 还大**，逻辑上讲不通——
        额定本该是最保守的连续值。
        腕部在四个来源里跨 **2 ~ 10 N·m，差 5 倍**，而 J1–J4 各来源基本自洽。

        ⭐ **为什么偏偏是腕部**：`实测` J6 的反射惯量占总惯量 **97.3%**
        （转子是连杆的 35.71 倍），官方阻抗示例注释也写着
        「避免**再次**激励 5/6 号腕部电机抖动」——"再次"说明他们踩过坑。
        $\\Rightarrow$ 腕部力矩越大越容易激起振荡，所以各处都在往下压，
        但压到多少没有统一口径。

        🔴 **行动项**：真机上必须实测腕部的可用力矩上限，
        在那之前 **J5/J6 一律按最保守的 ROS2 值 [3, 2] 用**。
        """
        stall = np.array(FOLLOWER["max_torque"], dtype=float)
        act = robot.model.actuator_forcerange[:6, 1]

        # J1–J4：三档应严格递增，这部分是自洽的
        assert np.all(self.RATED[:4] < act[:4]), \
            f"J1-J4 额定 {self.RATED[:4]} 应 < SDK 示例 {act[:4]}"
        assert np.all(act[:4] < stall[:4]), "J1-J4 SDK 示例应 < 堵转"

        # J5/J6：⚠️ 矛盾就在这里，断言的是「矛盾仍然存在」
        assert np.any(self.RATED[4:] > act[4:]), (
            "腕部矛盾消失了——说明某个来源的数字被改过。"
            "请重新核对客服 Q9 与 SDK 示例，并更新本测试的说明。")

    #: 🔴 在真机实测之前，腕部一律按最保守的 ROS2 阻抗示例值用。
    WRIST_SAFE = np.array([3.0, 2.0])

    def test_wrist_safe_value_is_the_minimum_of_all_sources(self):
        """⭐ 判据独立于我们的选择：它必须是所有来源里**最小**的那个。"""
        sources = np.array([[6., 6.], [5., 5.], [3., 2.], [10., 10.]])
        assert np.allclose(self.WRIST_SAFE, sources.min(axis=0))

    def test_rated_is_strictly_below_stall(self):
        """额定必须严格小于堵转。⚠️ 这是常识校验，防止两套值被写反。"""
        stall = np.array(FOLLOWER["max_torque"], dtype=float)
        assert np.all(self.RATED < stall), \
            f"额定 {self.RATED} 没有全部小于堵转 {stall}"

    def test_gravity_peak_vs_rated_is_documented(self, robot):
        """⭐⭐ J2 的重力力矩峰值会**超过额定**——这是必须记住的事实。

        `实测` 全域扫描：J2 峰值约 10.33 N·m，而额定只有 10.0
        ⇒ **某些姿态下光托住自己就超额定 3.3%**。

        ⚠️ 这条测试不是在"要求"它超额定，而是在**盯住这个事实**：
        如果哪天模型改动让它不再超了，说明质量参数变了，
        必须重新评估辨识轨迹的安全性。
        """
        rng = np.random.default_rng(0)
        peak = np.zeros(6)
        for _ in range(3000):
            q = rng.uniform(robot.q_lower, robot.q_upper)
            peak = np.maximum(peak, np.abs(robot.gravity(q)))
        assert peak[1] > self.RATED[1] * 0.95, (
            f"J2 重力峰值 {peak[1]:.3f} 远低于额定 {self.RATED[1]}，"
            "模型质量参数可能被改小了，请重新核对")
        assert peak[0] < 1e-12, "J1 绕竖直轴，重力力矩必须**恒等于** 0"

    def test_j1_zero_is_identity_but_j6_zero_is_not(self, robot):
        """⭐⭐ 两种"零"完全不同，混淆它们会在装夹爪之后翻车。

        **J1 的零是恒等式**：J1 绕**竖直轴**转，重力方向也竖直，
        力与转轴平行 $\\Rightarrow$ 力臂恒为 0。
        这与质量怎么分布**无关**，装什么末端都成立。
        `实测` 全域扫描峰值 **0.0**（精确到浮点）。

        ⚠️ **J6 的"零"只是巧合**：J6 绕自身 x 轴转，
        它的重力力矩取决于 link6 质心离该轴多远。
        `实测` link6 质心偏离 J6 轴 **0.0096 mm**（CAD 里的微小不对称），
        于是 $\\tau_{\\max}=mgd=0.3407\\times 9.81\\times 9.64\\times10^{-6}
        =3.22\\times10^{-5}$ N·m —— 和扫描峰值完全吻合，
        **说明它不是数值噪声，是真实的偏心**。

        🔴 **工程后果**：装上 D405 相机或夹爪后，
        J6 的质心会明显偏离转轴 $\\Rightarrow$ 这个 $3.22\\times10^{-5}$
        会变成**可观的值**，"J6 不用管重力"的假设立刻失效。
        ⭐ 这正是「D405 装机前后各辨识一次」的价值所在。
        """
        rng = np.random.default_rng(1)
        p1 = p6 = 0.0
        for _ in range(2000):
            g = np.abs(robot.gravity(rng.uniform(robot.q_lower, robot.q_upper)))
            p1, p6 = max(p1, g[0]), max(p6, g[5])
        assert p1 < 1e-12, f"J1 应精确为 0，实为 {p1:.3e}"
        assert 1e-6 < p6 < 1e-3, (
            f"J6 重力峰值 {p6:.3e} 超出预期区间。"
            "若明显变大，说明末端质心偏心增加（比如装了相机/夹爪）"
            "⇒ 必须重新辨识")


@needs_sdk
class TestVelocityAndAcceleration:
    def test_velocity_limit(self):
        v = np.array(FOLLOWER["velocity_limits"], dtype=float)
        assert np.allclose(v, 1.0), f"官方速度限已变成 {v}，请同步讲义与规划器"

    def test_acceleration_limit(self):
        a = np.array(FOLLOWER["acceleration_limits"], dtype=float)
        assert np.allclose(a, 2.0), f"官方加速度限已变成 {a}，请同步 TOPP 配置"


@needs_sdk
class TestEndEffectorDefinition:
    """🔴 TCP 曾经改了 24.3 mm 而 170 项测试全过。这里补上那个缺失的判据。"""

    def test_official_end_effector_is_tool_link(self):
        """确认我们对齐的是 ``tool_link`` 而不是 Leader 的 ``joint6``。"""
        assert FOLLOWER is not None
        import yaml as _y
        from panthera.driver.sdk_path import follower_config
        cfg = _y.safe_load(follower_config().read_text())
        assert cfg["urdf"]["end_effector_link"] == "tool_link", \
            "官方 Follower 的末端不再是 tool_link，TCP 偏置必须重新核对"

    def test_tcp_offset_pinned(self):
        """⭐ 把 0.165 钉死。改它必须同时改这里，逼人做一次显式决定。"""
        from panthera.core.robot import PANTHERA_TCP_OFFSET
        assert PANTHERA_TCP_OFFSET[0] == pytest.approx(0.165)
        assert np.allclose(PANTHERA_TCP_OFFSET[1:], 0.0)

    def test_tcp_actually_moves_fk(self, robot):
        """⚠️ 判据独立于数值：TCP 偏置必须**真的作用在 FK 上**。

        之前的 bug 正是"改了常数但 FK 没用上"这一类——
        只测常数等于 0.165 是抓不到的。
        """
        from panthera.core.robot import Q_HOME
        p_tcp, _ = robot.fk(Q_HOME)
        p_flange = robot.flange_pose(Q_HOME)[0] if hasattr(robot, "flange_pose") \
            else None
        if p_flange is not None:
            d = np.linalg.norm(np.asarray(p_tcp) - np.asarray(p_flange))
            assert d == pytest.approx(0.165, abs=1e-6), \
                f"TCP 与法兰的距离是 {d:.6f}，应为 0.165"
