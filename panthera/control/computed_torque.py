"""计算力矩控制（CTC）：把非线性系统"抵消"成一串独立的二阶弹簧。

① 先说人话
----------
PD 控制是"错多少、拉多少"。问题是机械臂**不是弹簧**：

* 同样的力矩，手臂伸直时转得慢，收起来时转得快（惯量随构型变）
* 动起来会有离心力和科氏力把自己甩偏
* 重力在不同姿态下拉的方向完全不同

PD 对这些一无所知，只能靠增益调大去"硬扛"，代价是超调和振荡。

**CTC 的想法**：既然我知道模型，就先把这些非线性项**算出来抵消掉**，
剩下的部分就变成一个干净的、单位质量的双积分器，再用 PD 收拾它。

② 换成机器人
-----------
机械臂动力学：

.. math:: M(q)\\ddot q + C(q,\\dot q)\\dot q + g(q) = \\tau

令

.. math:: \\tau = M(q)\\,a_q + C(q,\\dot q)\\dot q + g(q)

代进去，$M\\ddot q = M a_q$。只要 $M$ 可逆（机械臂总是），就得到

.. math:: \\ddot q = a_q

⭐ **不管手臂在什么姿态、以什么速度运动，它现在都是 $\\ddot q = a_q$。**
非线性被"反馈线性化"掉了。剩下的 $a_q$ 随便用 PD：

.. math:: a_q = \\ddot q_d + K_d(\\dot q_d-\\dot q) + K_p(q_d-q)

闭环误差方程变成 $\\ddot e + K_d\\dot e + K_p e = 0$——
六个**互相解耦**的二阶系统，阻尼比和自然频率直接由 $K_p,K_d$ 指定。

③ ⚠️ 任务空间版本最容易漏的一项
------------------------------
在任务空间（直接指定 TCP 的笛卡尔轨迹）时，加速度的映射是

.. math:: \\ddot x = J\\ddot q + \\dot J\\dot q

所以从期望的笛卡尔加速度 $a_x$ 反解关节加速度必须**减掉** $\\dot J\\dot q$：

.. math:: a_q = J^{+}\\big(a_x - \\dot J\\dot q\\big)

⚠️ 漏掉这一项代码不会报错、界面看着也正常，只是 CTC 相对 PD 的优势被
系统性削弱。armctrl 迁移时 `实测` RMS 10.71 → 10.19（约 5%），
而且**当时的测试照样全绿**——因为用的是代理量（RMS 比值），
5% 被容差吃掉了。见 :mod:`panthera.tests.test_computed_torque`。

④ ⚠️ CTC 不是万能的
------------------
CTC 的全部威力来自"模型准"。模型错了，抵消项就是**错误的前馈**，
可能比不抵消还糟。这正是本项目要做参数辨识的原因，
也是四象限对照实验的左半边。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CTCGains:
    """CTC 的 PD 增益。

    ⭐ 因为 CTC 已经把系统解耦成 $\\ddot e + K_d\\dot e + K_p e = 0$，
    增益可以直接按二阶系统的标准式来配：

    .. math:: K_p = \\omega_n^2,\\qquad K_d = 2\\zeta\\omega_n

    这是 CTC 相对 PD 的一个实际好处——**增益有物理含义**，
    不用瞎试。见 :func:`from_bandwidth`。
    """

    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self):
        self.kp = np.atleast_1d(np.asarray(self.kp, dtype=float))
        self.kd = np.atleast_1d(np.asarray(self.kd, dtype=float))


def from_bandwidth(n: int, wn: float = 12.0, zeta: float = 1.0) -> CTCGains:
    """按自然频率与阻尼比生成增益。

    Args:
        wn: 自然频率 (rad/s)。⚠️ 上限受采样率约束——
            经验规则 $\\omega_n \\lesssim 2\\pi f_s/10$。
            本项目 $f_s=500\\,$Hz ⇒ $\\omega_n \\lesssim 314$ rad/s，
            但真机上还要留摩擦和通信延迟的余量。
        zeta: 阻尼比。1.0 = 临界阻尼，无超调。
    """
    return CTCGains(kp=np.full(n, wn ** 2), kd=np.full(n, 2.0 * zeta * wn))


class ComputedTorqueController:
    """关节空间 CTC。

    Args:
        robot: :class:`~panthera.core.robot.ArmModel`
        gains: PD 增益
        compensate_coriolis: ⚠️ 关掉它可以量化科氏力的影响。
            低速时科氏力很小，这时 CTC 退化成"重力补偿 + PD"。
    """

    def __init__(self, robot, gains: CTCGains, compensate_coriolis: bool = True):
        self.robot = robot
        self.gains = gains
        self.compensate_coriolis = compensate_coriolis

    def compute(self, q, qd, q_des, v_des, a_des):
        """返回关节力矩。"""
        e = np.asarray(q_des) - np.asarray(q)
        de = np.asarray(v_des) - np.asarray(qd)
        a_q = np.asarray(a_des) + self.gains.kd * de + self.gains.kp * e

        M = self.robot.mass_matrix(q)
        tau = M @ a_q + self.robot.gravity(q)
        if self.compensate_coriolis:
            tau = tau + self.robot.coriolis_times_qd(q, qd)
        return tau


def jacobian_dot_qd(robot, q, qd, h: float = 1e-6) -> np.ndarray:
    """用中心差分算 $\\dot J\\dot q$。

    $\\dot J = \\frac{\\mathrm d}{\\mathrm dt}J(q)$，沿当前速度方向差分：

    .. math:: \\dot J \\approx \\frac{J(q+h\\dot q)-J(q-h\\dot q)}{2h}

    ⭐ 只需要 $\\dot J$ 乘上 $\\dot q$，所以直接沿 $\\dot q$ 方向差分即可，
    不用算出完整的三阶张量。

    ⚠️ 步长 $h$ 太小会被浮点误差吃掉，太大有截断误差。
    $10^{-6}$ 是双精度下的经验平衡点（见守护测试的数值验证）。
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    J_p = robot.jacobian(q + h * qd)
    J_m = robot.jacobian(q - h * qd)
    return ((J_p - J_m) / (2.0 * h)) @ qd


class TaskSpaceCTC:
    """任务空间 CTC：直接指定 TCP 的笛卡尔轨迹。

    ⚠️⚠️ **本类存在的全部意义是那个 $\\dot J\\dot q$ 项。**
    ``include_jdot`` 默认 True；设为 False 会复现 armctrl 迁移时踩的坑，
    守护测试用它来证明该项确实有效果。

    ⚠️ Panthera 是 6 轴，$J$ 是 6×6 方阵，**没有零空间**。
    奇异构型附近 $J^{-1}$ 会爆炸，所以这里用**阻尼最小二乘**（DLS）：

    .. math:: J^{+}_{\\lambda} = J^{\\mathsf T}(JJ^{\\mathsf T}+\\lambda^2 I)^{-1}

    $\\lambda>0$ 牺牲一点精度换取奇异点附近的有界性。
    """

    def __init__(self, robot, gains: CTCGains, damping: float = 1e-3,
                 include_jdot: bool = True):
        self.robot = robot
        self.gains = gains
        self.damping = float(damping)
        self.include_jdot = include_jdot

    def _dls_pinv(self, J: np.ndarray) -> np.ndarray:
        m = J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + self.damping ** 2 * np.eye(m))

    def compute(self, q, qd, x_des, v_des, a_des, x_cur=None):
        """Args 中的 x 是 6 维（位置 3 + 姿态 3），与 :meth:`ArmModel.jacobian` 对齐。"""
        J = self.robot.jacobian(q)
        x = self.robot.tcp_position(q) if x_cur is None else np.asarray(x_cur)
        v = J @ np.asarray(qd)

        n_task = len(np.atleast_1d(x_des))
        e = np.asarray(x_des) - np.asarray(x)[:n_task]
        de = np.asarray(v_des) - v[:n_task]
        a_x = np.asarray(a_des) + self.gains.kd[:n_task] * de + self.gains.kp[:n_task] * e

        # ⭐ 这里就是关键的一行：ẍ = J q̈ + J̇q̇ ⇒ q̈ = J⁺(a_x − J̇q̇)
        if self.include_jdot:
            a_x = a_x - jacobian_dot_qd(self.robot, q, qd)[:n_task]

        a_q = self._dls_pinv(J[:n_task]) @ a_x

        M = self.robot.mass_matrix(q)
        return (M @ a_q + self.robot.gravity(q)
                + self.robot.coriolis_times_qd(q, qd))
