"""广义动量观测器：没有力矩传感器，也能知道"被谁碰了"。

① 先说人话
----------
你想让机械臂在撞到人的时候立刻停下。最直接的办法是装力矩传感器——
但那很贵，Panthera 也没有。

**能不能只用电机电流和编码器，反推出外力？**

朴素做法：$\\tau_{ext}=M\\ddot q+C\\dot q+g-\\tau$。
⚠️ 但这需要 $\\ddot q$，而 $\\ddot q$ 只能靠编码器差分两次得到——
噪声被放大 $1/\\Delta t^2$，500 Hz 下就是 $2.5\\times10^5$ 倍。**完全不可用。**

② 换成机器人：用动量绕开加速度
-----------------------------
定义**广义动量** $p=M(q)\\dot q$。它的导数满足

.. math:: \\dot p = \\tau + C^{\\mathsf T}(q,\\dot q)\\dot q - g(q) + \\tau_{ext}

⭐ **注意右边没有 $\\ddot q$。** 这一步用到了 $\\dot M=C+C^{\\mathsf T}$
（斜对称性质）——又是机器人动力学的结构性质在救命。

De Luca 的构造：

.. math::
    r(t) = K_I\\Big[p(t)-p(0)-\\int_0^t\\big(\\tau+C^{\\mathsf T}\\dot q-g+r\\big)\\,\\mathrm ds\\Big]

代进去可以验证 $r$ 满足

.. math:: \\dot r = K_I(\\tau_{ext}-r)

⭐ 也就是说：**$r$ 就是真实外力矩经过一个一阶低通之后的估计**，
截止频率正好是 $K_I$。

* $p$ 只需要 $q,\\dot q$ —— 编码器直接给
* 积分是**低通**，会把噪声压下去，而不是像微分那样放大

③ ⚠️ $K_I$ 是个折中，不是越大越好
--------------------------------

=========  ==========================================
$K_I$ 大   响应快，但把测量噪声也一起放大
$K_I$ 小   平滑，但碰撞检测有延迟（$\\approx 1/K_I$ 秒）
=========  ==========================================

`理论`：$K_I=25$ ⇒ 时间常数 40 ms ⇒ 碰撞后约 40 ms 才升到 63%。

④ ⚠️⚠️ 摩擦是这套方法的头号敌人
------------------------------
$r$ 估的是"**所有模型没算到的东西**"，不只是外力。摩擦没建模准，
就会被算成一个虚假的外力。

⭐ 所以：**碰撞检测阈值必须由"空跑一遍轨迹时 $r$ 的实际波动"来定**，
不能拍脑袋。见 :meth:`MomentumObserver.calibrate_threshold`。
这也是为什么摩擦辨识（`identification` 模块）是它的前置工作。

⑤ 和 LESO / 扩张状态观测器的关系
-------------------------------
思想同源：都把"未建模量 + 外部作用"打包成一个待估的总扰动，
用动态观测器在线估出来。

区别是动量观测器利用了 $\\dot M-2C$ 斜对称这个**结构性质**，
所以不需要 $\\ddot q$，而且估计量能直接解释成关节外力矩，
再经 $J^{\\mathsf T}$ 的伪逆映射成笛卡尔外力。
"""

from __future__ import annotations

import numpy as np


class MomentumObserver:
    """基于广义动量的外力矩观测器。

    Args:
        robot: :class:`~panthera.core.robot.ArmModel`
        k_i: 观测器增益 (1/s)，等于估计带宽。
        dt: 离散步长。⚠️ 必须与实际控制周期一致，否则积分是错的。
    """

    def __init__(self, robot, k_i: float = 25.0, dt: float = 0.002):
        self.robot = robot
        self.k_i = float(k_i)
        self.dt = float(dt)
        self.n = robot.n
        self.reset()

    def reset(self, q=None, qd=None) -> None:
        self.integral = np.zeros(self.n)
        self.r = np.zeros(self.n)
        if q is not None and qd is not None:
            self.p0 = self.robot.mass_matrix(np.asarray(q)) @ np.asarray(qd)
            self._init = True
        else:
            self.p0 = np.zeros(self.n)
            self._init = False

    def _coriolis_transpose_qd(self, q, qd) -> np.ndarray:
        """计算 $C^{\\mathsf T}\\dot q$。

        用 $\\dot M=C+C^{\\mathsf T}\\Rightarrow C^{\\mathsf T}\\dot q=\\dot M\\dot q-C\\dot q$。
        $\\dot M$ 沿当前速度方向中心差分：
        $\\dot M\\approx[M(q+h\\dot q)-M(q-h\\dot q)]/(2h)$。

        ⭐ 差分的是**质量矩阵**（关于 $q$ 的光滑函数），
        不是含噪声的测量信号——这是它能用的原因。
        """
        h = 1e-6
        M_p = self.robot.mass_matrix(q + h * qd)
        M_m = self.robot.mass_matrix(q - h * qd)
        return ((M_p - M_m) / (2.0 * h)) @ qd - self.robot.coriolis_times_qd(q, qd)

    def update(self, q, qd, tau_cmd) -> np.ndarray:
        """推进一步，返回外力矩估计 $r\\approx\\tau_{ext}$。

        ⚠️ ``tau_cmd`` 必须是**实际施加**的力矩。
        力矩饱和时指令 ≠ 实际，直接把指令喂进来会让 $r$ 把
        "被限幅削掉的那部分"误判成外力。
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if not self._init:
            self.p0 = self.robot.mass_matrix(q) @ qd
            self._init = True

        p = self.robot.mass_matrix(q) @ qd
        beta = self.robot.gravity(q) - self._coriolis_transpose_qd(q, qd)
        self.integral = self.integral + (np.asarray(tau_cmd) - beta + self.r) * self.dt
        self.r = self.k_i * (p - self.p0 - self.integral)
        return self.r.copy()

    def cartesian_force(self, q) -> np.ndarray:
        """把关节残差映射成笛卡尔外力：$F\\approx(J^{\\mathsf T})^{+}r$。

        ⚠️ Panthera 是 6 轴方阵雅可比，奇异构型附近这个伪逆会爆炸。
        报笛卡尔力之前先看 :meth:`ArmModel.condition_number`。
        """
        J = self.robot.jacobian(np.asarray(q))
        return np.linalg.pinv(J.T) @ self.r

    def detect(self, threshold) -> bool:
        """超阈值判定碰撞。threshold 可为标量或逐关节向量。"""
        thr = (np.full(self.n, threshold) if np.isscalar(threshold)
               else np.asarray(threshold))
        return bool(np.any(np.abs(self.r) > thr))

    @staticmethod
    def calibrate_threshold(r_log: np.ndarray, sigma: float = 6.0,
                            floor: float = 0.05) -> np.ndarray:
        """从**无碰撞**的空跑记录里定阈值。

        Args:
            r_log: 形状 (T, n) 的残差记录，必须来自**确认没有外力**的一段运动。
            sigma: 取多少倍标准差。6σ 对应误报率约 $2\\times10^{-9}$/样本。
            floor: 最小阈值，防止某个关节残差恰好很小导致阈值过敏。

        ⭐ **判据必须由数据定，不能拍脑袋。**
        阈值拍小了天天误触发，拍大了撞了也不停——两种错法都很危险。
        """
        r_log = np.atleast_2d(np.asarray(r_log, dtype=float))
        return np.maximum(np.abs(r_log).mean(axis=0)
                          + sigma * r_log.std(axis=0), floor)


class GravityCompensationTeaching:
    """拖动示教（零力控制）：只补重力和摩擦，人手一推就能拖着走。

    ⭐ 这是动量观测器最直接的应用场景，也是**真机到货第一天就能演示**的功能。

    Args:
        friction_comp: 摩擦补偿系数 (0~1)。
            ⚠️ **不要设成 1.0**。摩擦补偿是正反馈——补过头会自激振荡。
            工程上留 20%~30% 余量。
    """

    def __init__(self, robot, friction_comp: float = 0.7,
                 tanh_scale: float = 0.05):
        self.robot = robot
        self.friction_comp = float(friction_comp)
        self.tanh_scale = float(tanh_scale)
        m = robot.model
        self.fc = m.dof_frictionloss[robot.qvel_idx].copy()
        self.fv = m.dof_damping[robot.qvel_idx].copy()

    def compute(self, q, qd) -> np.ndarray:
        """返回拖动示教力矩。"""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = self.robot.gravity(q)
        # ⚠️ 用 tanh 而不是 sign：sign 在零速附近会抖动（chattering）
        tau = tau + self.friction_comp * (
            self.fc * np.tanh(qd / self.tanh_scale) + self.fv * qd)
        return tau
