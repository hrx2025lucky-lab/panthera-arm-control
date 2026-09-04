"""Slotine-Li 模型自适应控制：模型不准也能把跟踪误差压到零。

① 先说人话
----------
CTC 的全部威力来自"模型准"。可现实是——

* URDF 的惯量是 **CAD 导出的**，不含线束、不含装配误差
* 摩擦和转子惯量在模型里是**占位值**（`_GUESS` 后缀）
* 抓了个不知道多重的东西

模型错了，CTC 的抵消项就是**错误的前馈**。

**自适应控制的想法**：既然不知道参数，就**一边控制一边估**。
而且——⭐ 这是关键——**估计律不是拍脑袋定的，是从稳定性证明里反推出来的**。

② 换成机器人
-----------
先定义一个"打包误差"，叫**滤波误差**：

.. math:: s = \\dot e + \\Lambda e,\\qquad e = q_d - q

⭐ 为什么用它：$s\\to 0$ 是一个一阶稳定微分方程 $\\dot e = -\\Lambda e$，
所以**只要能把 $s$ 压到零，位置误差和速度误差自动都到零**。
六个耦合的二阶问题被降成一个一阶问题。

再定义参考速度/加速度（注意是"参考"不是"期望"）：

.. math:: \\dot q_r = \\dot q_d + \\Lambda e,\\qquad \\ddot q_r = \\ddot q_d + \\Lambda\\dot e

于是 $s = \\dot q_r - \\dot q$。控制律与自适应律：

.. math::
    \\tau &= Y(q,\\dot q,\\dot q_r,\\ddot q_r)\\,\\hat\\pi + K_D s \\\\
    \\dot{\\hat\\pi} &= \\Gamma Y^{\\mathsf T} s

③ 那个自适应律是怎么来的
------------------------
取 Lyapunov 函数（"能量"）$V=\\tfrac12 s^{\\mathsf T}M s+\\tfrac12\\tilde\\pi^{\\mathsf T}\\Gamma^{-1}\\tilde\\pi$，
其中 $\\tilde\\pi=\\hat\\pi-\\pi$ 是参数误差。闭环满足 $M\\dot s+Cs+K_Ds=-Y\\tilde\\pi$。

求导，用上 $\\dot M-2C$ **斜对称**这条性质（这是机器人动力学的结构性质，
不是巧合），得

.. math:: \\dot V = -s^{\\mathsf T}K_D s + \\tilde\\pi^{\\mathsf T}\\big(\\Gamma^{-1}\\dot{\\hat\\pi} - Y^{\\mathsf T}s\\big)

⭐ 括号里那一项符号不定——它可能让 $\\dot V>0$，系统就不稳了。
**自适应律 $\\dot{\\hat\\pi}=\\Gamma Y^{\\mathsf T}s$ 就是为了让它精确等于零而设计的。**
于是 $\\dot V=-s^{\\mathsf T}K_Ds\\le 0$，所有信号有界，
再由 Barbalat 引理得 $s\\to0$。

④ ⚠️⚠️ 跟踪收敛 ≠ 参数收敛
--------------------------
这是本模块最容易被误解的地方，也是它和参数辨识的分界线：

============  ==================================================
保证的        $s\\to0$，即**跟踪误差**渐近收敛到零
**不**保证    $\\hat\\pi\\to\\pi$，即参数估计收敛到真值
============  ==================================================

$\\hat\\pi\\to\\pi$ 还需要轨迹满足**持续激励（PE）条件**——
和参数辨识要求的是同一个条件（见
:mod:`panthera.identification.excitation`）。

⭐ 所以：**想要准的参数，就老老实实做辨识；自适应只保证控得好。**
拿自适应跑出来的 $\\hat\\pi$ 当辨识结果用，是一个很常见的错误。
守护测试 :class:`TestTrackingIsNotParameterConvergence` 把这条钉死。

⑤ 和强化学习的区别
------------------
两者都叫"学习控制"，但：

==============  ==========================  ========================
                自适应控制                   强化学习
==============  ==========================  ========================
更新律来源       Lyapunov 函数反推            奖励函数 + 采样试错
稳定性           **可证**                    无保证
需要离线训练     不需要，上电即可运行          需要大量 rollout
需要模型结构     需要（要写得出回归矩阵）      不需要
==============  ==========================  ========================

这正是四象限对照实验里"传统控制"与"学习控制"两列的分野。
"""

from __future__ import annotations

import numpy as np


def scaled_gains(tau_limit, kd_ref: float = 20.0, lam_ref: float = 5.0):
    """按各关节的**力矩容量**缩放增益，返回 ``(lam, kd)`` 两个向量。

    ⚠️⚠️ **为什么必须这么做**：Panthera 的力矩上限跨度是 5 ~ 20 N·m
    （腕部 5，肩肘 20）。用一个标量 $K_D$，等于要求 5 N·m 的腕关节
    和 20 N·m 的肩关节出一样大的力——腕部必然先饱和。

    `实测`（标量 kd=20，正弦跟踪 20 s）：

    ==========  ======  ======  ======  ======  ======  ======
    关节          J1      J2      J3      J4      J5      J6
    ==========  ======  ======  ======  ======  ======  ======
    力矩上限      10      20      20      10       5       5
    饱和比例      0%      0%      0%     51%     78%     89%
    ==========  ======  ======  ======  ======  ======  ======

    ⭐ 而力矩饱和会**直接破坏 Lyapunov 证明的前提**：
    证明假设施加的力矩就是控制律算出来的 $\\tau$，饱和时两者不等，
    $\\dot V\\le0$ 不再成立——所以"越学越差"不是调参问题，是**理论前提被违反**。

    Args:
        kd_ref / lam_ref: 参考值，对应力矩上限的**中位数**关节。
    """
    tau_limit = np.asarray(tau_limit, dtype=float)
    ratio = tau_limit / np.median(tau_limit)
    return lam_ref * np.ones_like(ratio), kd_ref * ratio


class SlotineLiAdaptiveController:
    """Slotine-Li 模型自适应控制器（Panthera 6 轴）。

    Args:
        reg: :class:`~panthera.identification.regressor.DynamicsRegressor`
        lam: $\\Lambda$，滤波误差增益。决定 $s\\to0$ 之后误差流形的收敛速度。
            标量或逐关节向量。
        kd: $K_D$，鲁棒反馈增益。标量或逐关节向量。
            ⚠️ Panthera 上**必须用向量**，见 :func:`scaled_gains`。
        gamma: $\\Gamma$，自适应增益。⚠️ 太大会震荡，太小学得慢。
        use_base: 是否在**基参数子空间**里自适应。
            ⚠️ 强烈建议 True。完整 78 维里有 26 维结构性不可辨识，
            直接在完整空间自适应等于让 26 个方向随机游走。
            Panthera 基参数维数 **52**。
        pi_init: 参数初值。None = 从零起估（最保守，能看到完整学习过程）。
        proj_bound: ⚠️ **参数投影**的范数上界。PE 不满足时参数会缓慢漂移，
            这个投影保证 $\\hat\\pi$ 有界。代价是引入偏差——
            所以它是"安全阀"，不是"精度手段"。
        normalized: 是否用**归一化**自适应律。⚠️ 默认 True，见下。

    ⑥ ⚠️⚠️ 为什么默认用归一化自适应律
    --------------------------------
    教科书的 $\\dot{\\hat\\pi}=\\Gamma Y^{\\mathsf T}s$ 在**连续时间、无饱和、
    参数无量纲差异**的理想条件下才成立。真实系统上三条全不满足：

    * $\\pi$ 的量纲跨度极大——质量 ~0.3 kg、一次矩 ~0.01 kg·m、
      惯量 ~4×10⁻⁴ kg·m²、摩擦 ~0.2 N·m。**一个标量 $\\Gamma$ 对这样一个
      异质向量是错的**（和标量 $K_D$ 犯的是同一个错，见 :func:`scaled_gains`）
    * 离散化（500 Hz）
    * 力矩饱和

    `实测` 标准律在本系统上**从不收敛**：$\\gamma$ 小则 $\\hat\\beta$ 几乎不动
    （误差比 0.94×），$\\gamma$ 稍大就冲上投影上界 200 并卡死（饱和 93%）。
    中间没有可用窗口。

    归一化律 $\\dot{\\hat\\beta}=\\Gamma Y_b^{\\mathsf T}s/(1+\\lVert Y_b\\rVert^2)$
    把更新量的尺度归一，`实测`（初值取真值的 0.7 倍）：

    ==========  ========  ==========  ==========  ==========  ======
    律          $\\gamma$   前 1/4 误差  后 1/4 误差  改善        ‖β‖末
    ==========  ========  ==========  ==========  ==========  ======
    标准        1e-3      0.00698     0.03257     **0.21×**   0.50
    归一化      10        0.00437     0.00340     1.29×       0.55
    归一化      50        0.00361     0.00112     3.23×       0.61
    归一化      1000      0.00071     0.00019     **3.78×**   0.65
    ==========  ========  ==========  ==========  ==========  ======

    ⭐ 注意 ‖β‖ 全程稳定在 0.55~0.65，**远低于投影上界 200**——
    说明收敛是真的，不是被安全阀按住的。

    ⚠️ 一条元教训：本模块最初用标量 $\\Gamma$ + 标准律，某次跑出了
    "3.44× 改善"的漂亮结果。但那是一个**发散系统的偶然轨迹**——
    位级扰动就会翻成 1.01×。**一个会因 1e-16 扰动而翻转的结论不是结论。**
    发现它的方法是把同一个脚本重跑一遍。
    """

    def __init__(self, reg, lam=10.0, kd=20.0,
                 gamma: float = 500.0, use_base: bool = True,
                 pi_init: np.ndarray | None = None,
                 proj_bound: float = 200.0, normalized: bool = True):
        self.reg = reg
        self.n = reg.nv
        self.lam = np.atleast_1d(np.asarray(lam, dtype=float))
        self.kd = np.atleast_1d(np.asarray(kd, dtype=float))
        self.gamma = float(gamma)
        self.proj_bound = float(proj_bound)
        self.normalized = bool(normalized)

        if use_base:
            self.P, self.rank = reg.base_parameter_projection(samples=300)
        else:
            self.P = np.eye(reg.n_par)
            self.rank = reg.n_par

        pi0 = np.zeros(reg.n_par) if pi_init is None else np.asarray(pi_init)
        self.beta = self.P.T @ pi0
        self._beta0 = self.beta.copy()

    def reset(self) -> None:
        self.beta = self._beta0.copy()

    @property
    def pi_hat(self) -> np.ndarray:
        """⚠️ 这不是辨识结果。见模块文档 ④。"""
        return self.P @ self.beta

    def compute(self, q, qd, q_des, v_des, a_des, dt: float):
        """返回 ``(tau, s)``。"""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        e = np.asarray(q_des) - q
        de = np.asarray(v_des) - qd
        s = de + self.lam * e
        v_ref = np.asarray(v_des) + self.lam * e
        a_ref = np.asarray(a_des) + self.lam * de

        Yb = self.reg.slotine_li_regressor(q, qd, v_ref, a_ref) @ self.P
        tau = Yb @ self.beta + self.kd * s

        # 自适应律 β̂̇ = Γ Ybᵀ s ——⭐ 这一行就是 Lyapunov 反推的结果
        update = Yb.T @ s
        if self.normalized:
            # ⚠️ 归一化：除以 1+‖Yb‖²。见类文档 ⑥。
            update = update / (1.0 + float(np.sum(Yb * Yb)))
        self.beta = self.beta + self.gamma * update * dt

        # ⚠️ 参数投影：PE 不足时防漂移的安全阀
        nrm = float(np.linalg.norm(self.beta))
        if nrm > self.proj_bound:
            self.beta *= self.proj_bound / nrm
        return tau, s
