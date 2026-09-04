"""Panthera 调参台的四个场景。

和 armctrl 那 12 个场景的区别
------------------------------
armctrl 的场景是**教学**用的：每个都带大段 Lesson，目的是让零基础的人看懂原理。
这里的四个是**真机对照**用的：目的是让「仿真读数」和「真机读数」并排放，
一眼看出 sim2real gap 在哪一项上。

所以这边刻意做减法——不重复讲原理（那在 armctrl 讲义里），
把版面留给「理论值 / 仿真实测 / 真机实测」这三列。

⚠️ 一条贯穿全篇的规矩
--------------------
每个读数都要标清楚它属于哪一档（见 armctrl 讲义 §九「数字的可信度分四类」）：

* ``理论``  —— 由解析式**独立**算出，不读被测实现的任何中间量。⭐ 只有它能验证实现对不对
* ``实测``  —— 从仿真或真机读出来的物理量
* ``自报``  —— 实现自己报的量（例如真机的 ``get_current_torque()``，它是
  电流×固定系数估算的），**不能用来自证**

⭐ 真机上力矩读数属于 ``自报``：官方 SDK 就是拿电流乘一个固定系数换算的，
不含摩擦、不含减速器效率。做动量观测器时按惯例用**指令力矩**而不是这个读数。
"""

from __future__ import annotations

import numpy as np
import mujoco

from panthera.assets import panthera_xml
from panthera.core.robot import make_panthera, Q_HOME
from panthera.tuner.base import (Arrow, Bars, ControlLaw, LawTerm, Param,
                                 Readout, Scene, Sphere, Trace, hex_rgba)

#: 轨迹跟踪与辨识场景共用的圆心（Q_HOME 处的 TCP）
CIRCLE_CENTER = np.array([0.3304, 0.0, 0.308])


class _PantheraScene(Scene):
    """四个场景的公共部分：加载模型、力矩下发、饱和度统计。"""

    dt = 0.002
    camera = dict(azimuth=135.0, elevation=-18.0, distance=1.6,
                  lookat=(0.30, 0.0, 0.35))
    gripper_width = None          # Panthera 暂无夹爪模型

    def build(self) -> mujoco.MjModel:
        self.robot = make_panthera()
        self.model = self.robot.model
        self.data = self.robot.data
        return self.model

    def reset(self) -> None:
        self.robot.set_state(Q_HOME)
        self.tau_last = np.zeros(self.robot.n)

    def _send(self, tau: np.ndarray) -> np.ndarray:
        """限幅并记录。⚠️ 限幅不是可选项——真机上它是保护硬件的最后一道闸。"""
        tau = self.robot.saturate_torque(tau)
        self.tau_last = tau
        return tau

    def _saturation_pct(self) -> float:
        """最大关节力矩饱和度（%）。⭐ 力矩饱和是真机翻车的头号原因，必须能一眼看见。"""
        return float(np.max(np.abs(self.tau_last) / self.robot.tau_limit) * 100.0)


# ====================================================================== 重力补偿

class GravityScene(_PantheraScene):
    """重力补偿：真机上电后的第一个实验。

    ⭐ 判据：纯重力补偿下手臂应当**停在原地不动**。
    掉下去 ⇒ 重力项算错或力矩不够；往上飘 ⇒ 补过头。
    """

    name = "gravity"
    title = "重力补偿"
    intro = (
        "只给重力补偿力矩 g(q)，不加任何位置反馈。\n\n"
        "手臂应当停在原地——这是真机上电后第一个该做的实验，"
        "也是后面所有力矩级控制的地基：g(q) 不对，上面全白搭。"
    )

    params = [
        Param("comp_ratio", "补偿比例 η", 0.0, 1.2, 1.0,
              effect="η=1 完全补偿；η<1 手臂会缓慢下沉；"
                     "η>1 会往上飘。⭐ 真机上先从 0.8 试起，别一上来给满。"),
        Param("damping", "关节阻尼", 0.0, 5.0, 0.5, unit="N·m·s/rad",
              effect="纯重力补偿下手臂是中性平衡，一点扰动就会漂。"
                     "加一点阻尼让它稳住——⚠️ 但这已经不是纯重力补偿了，"
                     "报结果时要说明。"),
    ]

    readouts = [
        Readout("drift", "位置漂移", "mrad", 2,
                help="相对初始构型的最大关节偏差。⭐ 判据：应当趋近 0。"),
        Readout("g_norm", "重力力矩 ‖g(q)‖", "N·m", 3, theory=True,
                help="由 MuJoCo 的 qfrc_bias 在零速下算出，属 `理论` 档。"),
        Readout("g_j3", "J3 重力力矩", "N·m", 3, theory=True,
                help="⭐ J3 通常最吃力。额定 10 N·m，看它占多少。"),
        Readout("tau_sat", "最大力矩饱和度", "%", 1,
                help="⚠️ 接近 100% 说明这个姿态真机上保持不住。"),
    ]

    traces = [Trace("drift", "位置漂移 (mrad)"), Trace("g_j3", "J3 重力 (N·m)")]
    bars = [Bars("tau", "关节力矩 τ", "N·m")]

    law = ControlLaw(
        formula=r"\tau = \eta\,g(q) - D\,\dot q",
        terms=[LawTerm("g(q)", "重力力矩", key="g_norm", unit="N·m", digits=3),
               LawTerm("η", "补偿比例", key="comp_ratio", digits=2)])

    def reset(self) -> None:
        super().reset()
        self.q0 = Q_HOME.copy()

    def control(self, t, q, v):
        tau = self.get("comp_ratio") * self.robot.gravity(q) - self.get("damping") * v
        return self._send(tau)

    def telemetry(self, t, q, v):
        g = self.robot.gravity(q)
        return dict(drift=float(np.abs(q - self.q0).max() * 1e3),
                    g_norm=float(np.linalg.norm(g)), g_j3=float(g[2]),
                    tau_sat=self._saturation_pct(), tau=self.tau_last.tolist())

    def overlays(self):
        p, _ = self.robot.fk(self.data.qpos[self.robot.qpos_idx])
        return [Sphere(pos=p, size=0.012, rgba=hex_rgba("#22c55e", 0.6))]


# ====================================================================== 阻抗控制

class ImpedanceScene(_PantheraScene):
    """笛卡尔阻抗控制。

    ⭐ 核心判据：**稳态偏移精确等于 F/K**。
    这个判据的价值在于它**独立于实现**——F/K 由解析式算出，
    不读控制器的任何中间量。两者对不上，就说明有一方错了。
    """

    name = "impedance"
    title = "笛卡尔阻抗控制"
    intro = (
        "把末端塑造成一个可调的弹簧-阻尼系统：推得越用力偏得越多，松手弹回。\n\n"
        "⭐ 拖「刚度 K」，看「实测偏移」和「理论 F/K」是否始终对得上——"
        "这是全项目最有价值的一个对照。"
    )

    params = [
        Param("k_trans", "平移刚度 K", 20.0, 600.0, 140.0, unit="N/m",
              effect="⭐ 稳态偏移应精确等于 F/K。K 翻倍，偏移减半。<br>"
                     "⚠️ 默认 140 是<b>官方 ROS2 阻抗示例的实测值</b>"
                     "（armctrl 从 Panda 沿用的 500 在这台机器上偏高）。"),
        Param("zeta", "阻尼比 ζ", 0.1, 2.0, 0.8,
              effect="⚠️ 恒力下读数一动不动——阻尼只在「变化过程」中出力。"
                     "想看它的作用，把外力模式切成方波。"),
        Param("fz", "外力 Fz", -30.0, 0.0, -10.0, unit="N",
              effect="末端竖直方向的外力。"),
        Param("force_mode", "外力模式", choices=["恒力", "方波"], default="恒力",
              effect="方波模式下才能看出阻尼比的作用。"),
    ]

    readouts = [
        Readout("dz", "实测偏移 Δz", "mm", 3,
                compare_with="dz_theory",
                help="末端相对目标点的竖直偏移。"),
        Readout("dz_theory", "理论 F/K", "mm", 3, theory=True,
                help="⭐ 由 F/K 解析算出，**不读控制器任何中间量**。"
                     "这是独立判据——和实测对不上就说明有一方错了。"),
        Readout("err_pct", "相对误差", "%", 2,
                help="⭐ 稳态下应趋近 0。"),
        Readout("f_now", "当前外力", "N", 2),
        Readout("tau_sat", "最大力矩饱和度", "%", 1,
                help="⚠️ K 调很大时会饱和，此时 F/K 判据不再成立。"),
    ]

    traces = [Trace("dz", "实测偏移 (mm)"), Trace("dz_theory", "理论 F/K (mm)")]
    bars = [Bars("tau", "关节力矩 τ", "N·m")]

    law = ControlLaw(
        formula=r"\tau = J^{\mathsf T}\big[K(x_d-x) - D\dot x + F_{ext}\big] + g(q)",
        terms=[LawTerm("K", "平移刚度", key="k_trans", unit="N/m", digits=0),
               LawTerm("Δz", "实测偏移", key="dz", unit="mm", digits=3),
               LawTerm("F/K", "理论偏移", key="dz_theory", unit="mm", digits=3)])

    def reset(self) -> None:
        super().reset()
        self.p0, _ = self.robot.fk(Q_HOME)

    def _force(self, t: float) -> np.ndarray:
        fz = self.get("fz")
        if self.choice("force_mode") == "方波":
            fz = fz if (t % 2.0) < 1.0 else 0.0
        self._fz_now = fz
        return np.array([0.0, 0.0, fz])

    def control(self, t, q, v):
        k = self.get("k_trans")
        # 临界阻尼口径：D = 2ζ√(K·m_eq)，这里用任务空间惯量的迹做等效质量
        d = 2.0 * self.get("zeta") * np.sqrt(k * 1.0)
        p, _ = self.robot.fk(q)
        jac = self.robot.jacobian(q)[:3]
        force = k * (self.p0 - p) - d * (jac @ v)
        tau = jac.T @ (force + self._force(t)) + self.robot.gravity(q)
        return self._send(tau)

    def telemetry(self, t, q, v):
        p, _ = self.robot.fk(q)
        dz = float((p - self.p0)[2] * 1e3)
        dz_th = float(self._fz_now / self.get("k_trans") * 1e3)
        err = abs(dz - dz_th) / abs(dz_th) * 100.0 if abs(dz_th) > 1e-9 else 0.0
        return dict(dz=dz, dz_theory=dz_th, err_pct=err, f_now=self._fz_now,
                    tau_sat=self._saturation_pct(), tau=self.tau_last.tolist())

    def overlays(self):
        p, _ = self.robot.fk(self.data.qpos[self.robot.qpos_idx])
        return [Sphere(pos=self.p0, size=0.012, rgba=hex_rgba("#22c55e", 0.5)),
                Sphere(pos=p, size=0.010, rgba=hex_rgba("#e5e7eb", 0.9)),
                Arrow(frm=self.p0, to=p, size=0.004,
                      rgba=hex_rgba("#f97316", 0.9))]


# ====================================================================== 轨迹跟踪

class TrackingScene(_PantheraScene):
    """PD+重力补偿 vs 计算力矩（CTC）。

    ⭐ 这个场景是四象限实验里 A/B 两格的载体：
    同一条轨迹，分别用 CAD 模型和辨识后模型跑 CTC，比 TCP 轨迹 RMS 误差。
    """

    name = "tracking"
    title = "轨迹跟踪：PD vs 计算力矩"
    intro = (
        "末端画一个圆。对比两种控制律：\n"
        "• PD+重力补偿——只补重力，不管惯量与耦合\n"
        "• 计算力矩 CTC——显式用 M(q)、C(q,q̇) 做前馈\n\n"
        "⭐ 把周期调短（运动变快），两者差距会拉开——因为惯量项开始起作用。"
    )

    params = [
        Param("law", "控制律", choices=["PD+重力", "计算力矩"],
              default="计算力矩",
              effect="⭐ 慢速时两者差不多；快起来 PD 就跟不上了。"),
        Param("period", "圆周期 T", 1.2, 5.0, 2.5, unit="s",
              effect="⭐ 任务的速度是实验设计的一部分。"
                     "T 太大 ⇒ 动力学不起作用 ⇒ 四象限实验会失效。"),
        Param("radius", "圆半径", 0.04, 0.15, 0.10, unit="m"),
        Param("kp", "位置增益 Kp", 20.0, 400.0, 150.0),
        Param("kd", "速度增益 Kd", 2.0, 60.0, 24.0),
    ]

    readouts = [
        Readout("e_now", "当前跟踪误差", "mm", 3),
        Readout("e_rms", "TCP 轨迹 RMS 误差", "mm", 3,
                help="⭐ **四象限对照的主指标**。外部可观测，不依赖控制器内部量。"
                     "⚠️ 跳过第一个周期（启动瞬态），只统计稳态。"),
        Readout("n_rms", "RMS 已计入样本", "", 0,
                help="⚠️ 第一个周期内应恒为 0——参考轨迹 t=0 处速度非零而手臂静止，"
                     "那段瞬态若计入会淹没稳态差异，而稳态差异正是要比的东西。"),
        Readout("v_tcp", "TCP 速度", "m/s", 3),
        Readout("a_tcp", "TCP 加速度", "m/s²", 2, theory=True,
                help="由参考轨迹解析算出：a = R·ω²。⭐ 它决定惯量项有多重要。"),
        Readout("jdot_v", "‖J̇q̇‖", "m/s²", 3,
                help="⭐ 雅可比时变带来的加速度项。ẍ = J q̈ + **J̇q̇**，"
                     "把任务加速度映射回关节空间时必须减掉它。"
                     "⚠️ 漏掉它不会报错，只会让 CTC 的优势被系统性削弱——"
                     "所以把它显式暴露出来，好让测试能直接盯住。"),
        Readout("tau_sat", "最大力矩饱和度", "%", 1),
    ]

    traces = [Trace("e_now", "跟踪误差 (mm)")]
    bars = [Bars("tau", "关节力矩 τ", "N·m")]

    law = ControlLaw(
        formula=r"\tau = M(q)\big[\ddot q_d + K_d\dot e + K_p e\big] + C(q,\dot q)\dot q + g(q)",
        terms=[LawTerm("e_rms", "RMS 误差", key="e_rms", unit="mm", digits=3),
               LawTerm("a", "TCP 加速度", key="a_tcp", unit="m/s²", digits=2)])

    def reset(self) -> None:
        super().reset()
        self._err_sq = 0.0
        self._n = 0
        self._jdot_v = 0.0

    def _ref(self, t: float):
        """参考轨迹：任务空间画圆。⚠️ 必须二阶连续可导——CTC 要用 p̈_d 做前馈。"""
        w = 2.0 * np.pi / self.get("period")
        r = self.get("radius")
        p = CIRCLE_CENTER + np.array([0.0, r * np.sin(w * t), r * (1 - np.cos(w * t))])
        pd = np.array([0.0, r * w * np.cos(w * t), r * w * np.sin(w * t)])
        pdd = np.array([0.0, -r * w * w * np.sin(w * t), r * w * w * np.cos(w * t)])
        return p, pd, pdd

    def control(self, t, q, v):
        p_d, pd_d, pdd_d = self._ref(t)
        p, _ = self.robot.fk(q)
        jac = self.robot.jacobian(q)[:3]
        jac_pinv = np.linalg.pinv(jac)

        e = p_d - p
        ed = pd_d - jac @ v
        self._e_now = float(np.linalg.norm(e) * 1e3)
        self._v_tcp = float(np.linalg.norm(jac @ v))
        self._a_ref = float(np.linalg.norm(pdd_d))

        # ⚠️ 任务空间加速度 → 关节加速度：ẍ = J q̈ + J̇ q̇，所以要减掉 J̇q̇。
        # 漏掉这一项会**系统性削弱 CTC 的优势**——它正是 CTC 相对 PD 的关键增量之一。
        # 这里用数值差分估 J̇q̇（解析形式要三阶张量，代价不值）。
        eps = 1e-6
        jac_next = self.robot.jacobian(q + v * eps)[:3]
        jdot_v = ((jac_next - jac) / eps) @ v

        self._jdot_v = float(np.linalg.norm(jdot_v))

        acc_task = pdd_d + self.get("kd") * ed + self.get("kp") * e
        qdd_des = jac_pinv @ (acc_task - jdot_v)

        if self.choice("law") == "计算力矩":
            # ⭐ 显式用 M(q) 和偏置项——这一层正是 RL 路线所没有的
            tau = self.robot.mass_matrix(q) @ qdd_des + self.robot.bias(q, v)
        else:
            # PD 只补重力，惯量与耦合都不管
            tau = jac.T @ (self.get("kp") * e + self.get("kd") * ed) \
                  + self.robot.gravity(q)
        return self._send(tau)

    def telemetry(self, t, q, v):
        # ⚠️ 参考轨迹在 t=0 处速度非零（ṗ=[0, Rω, 0]）而手臂静止，
        # 必然有一段启动瞬态。把它算进 RMS 会淹没稳态跟踪能力的差异——
        # ⭐ 而稳态差异正是 PD 与 CTC 的分野所在。故跳过第一个周期。
        if t >= self.get("period"):
            self._err_sq += (self._e_now ** 2)
            self._n += 1
        return dict(e_now=self._e_now,
                    e_rms=float(np.sqrt(self._err_sq / max(self._n, 1))),
                    n_rms=float(self._n),
                    v_tcp=self._v_tcp, a_tcp=self._a_ref,
                    jdot_v=self._jdot_v,
                    tau_sat=self._saturation_pct(), tau=self.tau_last.tolist())

    def overlays(self):
        t = self.data.time
        p_d, _, _ = self._ref(t)
        p, _ = self.robot.fk(self.data.qpos[self.robot.qpos_idx])
        return [Sphere(pos=p_d, size=0.010, rgba=hex_rgba("#22c55e", 0.6)),
                Sphere(pos=p, size=0.008, rgba=hex_rgba("#e5e7eb", 0.9)),
                Arrow(frm=p, to=p_d, size=0.003, rgba=hex_rgba("#eab308", 0.9))]


# ====================================================================== 参数辨识

class IdentifyScene(_PantheraScene):
    """激励轨迹与可辨识性 —— ⭐ armctrl 没有这个场景，是本项目新增的。

    参数辨识准不准，取决于激励轨迹够不够「花哨」（持续激励 PE 条件）。
    这个场景把**条件数**实时显示出来，让你能一边调轨迹一边看它变好还是变坏。

    ⚠️ 条件数这个读数本身有个坑（armctrl 元教训 #28）：
    如果按某个阈值截断秩再算 σ₀/σ_{r-1}，它**按定义永远不会太大**——
    一个「永远报不出坏消息」的指标比没有指标更危险。
    所以这里用**全谱** σ_max/σ_min，烂轨迹会立刻顶到很大的值。
    """

    name = "identify"
    title = "激励轨迹与可辨识性"
    intro = (
        "参数辨识的成败取决于激励轨迹。轨迹越「花哨」，回归矩阵条件数越小，"
        "辨识出的参数越可信。\n\n"
        "⭐ 拖动谐波数和幅值，看条件数怎么变——目标是让它尽可能小。"
    )

    params = [
        Param("n_harm", "谐波数 K", 1, 5, 3, step=1,
              effect="⭐ 谐波越多轨迹越花哨，激励越充分。但也越难跑、越容易超限。"),
        Param("amp", "幅值", 0.05, 0.6, 0.3, unit="rad",
              effect="幅值太小 ⇒ 激励不足；太大 ⇒ 撞限位。"),
        Param("w_f", "基频 ωf", 0.2, 1.5, 0.6, unit="rad/s",
              effect="⚠️ 频率高了力矩会饱和，饱和的数据不能用来辨识。"),
        Param("kp", "跟踪增益 Kp", 20.0, 400.0, 200.0),
    ]

    readouts = [
        Readout("cond", "回归矩阵条件数 κ", "", 2,
                help="⭐ 越小越好。⚠️ 这里用**全谱** σmax/σmin，"
                     "不是按秩截断的版本——后者永远报不出坏消息。"),
        Readout("cond_log", "log₁₀ κ", "", 2,
                help="数量级更直观。10² 很好，10⁸ 以上基本没法辨识。"),
        Readout("base_rank", "基参数秩", "", 0,
                help="⭐ 完整参数集里真正可辨识的维数。Panthera 为 52（完整 78）。"
                     "⚠️ 这是**结构**性质，只跟模型有关，调轨迹不会改变它。"),
        Readout("n_samples", "已采样本数", "", 0,
                help="辨识需要足够多的样本，且要覆盖一个完整周期。"),
        Readout("tau_sat", "最大力矩饱和度", "%", 1,
                help="⚠️⚠️ **饱和期间的数据必须丢弃**——"
                     "那时实际力矩不等于指令力矩，回归方程不成立。"),
        Readout("q_margin", "距关节限位最近", "rad", 3,
                help="⚠️ 接近 0 说明轨迹要撞限位了。"),
    ]

    traces = [Trace("cond_log", "log₁₀ 条件数"), Trace("tau_sat", "饱和度 (%)")]
    bars = [Bars("tau", "关节力矩 τ", "N·m")]

    law = ControlLaw(
        formula=r"q_i(t)=q_{i,0}+\sum_{k=1}^{K}\frac{a_{ik}}{k\omega_f}\sin(k\omega_f t)"
                r"-\frac{b_{ik}}{k\omega_f}\cos(k\omega_f t)",
        terms=[LawTerm("κ", "条件数", key="cond", digits=2),
               LawTerm("K", "谐波数", key="n_harm", digits=0)])

    def reset(self) -> None:
        super().reset()
        rng = np.random.default_rng(0)      # 固定种子，保证可复现
        self._a = rng.uniform(-1, 1, (self.robot.n, 5))
        self._b = rng.uniform(-1, 1, (self.robot.n, 5))
        self._rows: list[np.ndarray] = []
        self._cond = 1.0
        # Pinocchio 需要额外依赖；缺失时降级为「不显示条件数」而不是崩掉，
        # 让调参台在没装 pinocchio 的机器上仍然能用其余三个场景。
        try:
            from panthera.identification.regressor import DynamicsRegressor
            self._reg = DynamicsRegressor(panthera_xml(), n_arm=self.robot.n)
            # 基参数投影只跟模型有关、与轨迹无关，算一次缓存起来
            self._proj, self._rank = self._reg.base_parameter_projection(samples=300)
        except Exception:
            self._reg = None
            self._proj, self._rank = None, 0

    def _ref(self, t: float):
        """傅里叶级数激励轨迹。⭐ 起止速度为零，便于真机上安全启停。"""
        k_max = int(self.get("n_harm"))
        amp = self.get("amp")
        w_f = self.get("w_f")
        q = Q_HOME.copy()
        qd = np.zeros(self.robot.n)
        qdd = np.zeros(self.robot.n)
        for k in range(1, k_max + 1):
            wk = k * w_f
            a, b = self._a[:, k - 1] * amp, self._b[:, k - 1] * amp
            q += a / wk * np.sin(wk * t) - b / wk * np.cos(wk * t)
            qd += a * np.cos(wk * t) + b * np.sin(wk * t)
            qdd += -a * wk * np.sin(wk * t) + b * wk * np.cos(wk * t)
        return q, qd, qdd

    def control(self, t, q, v):
        q_d, qd_d, qdd_d = self._ref(t)
        kp = self.get("kp")
        kd = 2.0 * np.sqrt(kp)
        tau = self.robot.mass_matrix(q) @ (
            qdd_d + kd * (qd_d - v) + kp * (q_d - q)) + self.robot.bias(q, v)
        self._q_margin = float(np.min(np.minimum(q - self.robot.q_lower,
                                                 self.robot.q_upper - q)))
        return self._send(tau)

    def telemetry(self, t, q, v):
        # ⭐ 用**真实回归矩阵** Y(q,q̇,q̈) 累积，而不是 [q,q̇,sign(q̇)] 那种代理。
        #
        # ⚠️ 为什么不能用代理：代理量对轨迹几乎是线性的，而 K 个谐波的傅里叶轨迹
        # 让所有关节都落在同一个 2K 维子空间里 ⇒ 代理矩阵必然病态，
        # 条件数会一直显示 10¹² 而**与激励好坏无关**。
        # 真回归矩阵 Y 对 q 是非线性的（含 sin/cos、速度乘积），
        # 才真正反映「这条轨迹能不能把参数激励出来」。
        #
        # ⭐ armctrl 元教训 #28：一个「永远报不出坏消息」（或永远报坏消息）的指标，
        # 比没有指标更危险——它会让你以为自己在监控。
        if self._reg is not None and len(self._rows) < 3000:
            acc = self.data.qacc[self.robot.qvel_idx].copy()
            self._rows.append(self._reg.regressor(q, v, acc))

        if len(self._rows) >= 60 and self._proj is not None:
            # ⭐⭐ 必须先投影到**基参数子空间**再算条件数。
            #
            # ⚠️ 完整参数集里有一部分参数**任何轨迹都辨不出来**
            # （例如固定基座上连杆的某些惯量分量，它们对关节力矩没有任何影响）。
            # 直接对完整 Y 算条件数，得到的是 10¹⁸ 量级——那反映的是
            # **结构性不可辨识**，和「这条轨迹激励得好不好」毫无关系，
            # 而且**不管你怎么调轨迹它都不变**。
            #
            # 投影到可辨识子空间后，条件数才真正随激励质量变化。
            # Panthera：完整 78 维 → 基参数 rank 52（26 个结构性不可辨识）。
            stacked = np.vstack(self._rows[-3000:]) @ self._proj
            sv = np.linalg.svd(stacked, compute_uv=False)
            # ⭐ 全谱 σmax/σmin，不按秩截断——截断版按定义永远报不出坏消息
            #    （元教训 #28 的原型就是这个坑）。
            self._cond = float(sv[0] / max(sv[-1], 1e-15))

        return dict(cond=self._cond,
                    cond_log=float(np.log10(max(self._cond, 1e-15))),
                    base_rank=float(self._rank),
                    n_samples=float(len(self._rows)),
                    tau_sat=self._saturation_pct(),
                    q_margin=self._q_margin, tau=self.tau_last.tolist())

    def overlays(self):
        p, _ = self.robot.fk(self.data.qpos[self.robot.qpos_idx])
        return [Sphere(pos=p, size=0.010, rgba=hex_rgba("#a855f7", 0.8))]


SCENES = [GravityScene, ImpedanceScene, TrackingScene, IdentifyScene]
SCENE_BY_NAME = {c.name: c for c in SCENES}
