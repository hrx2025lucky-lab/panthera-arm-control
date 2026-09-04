"""激励轨迹设计：让参数辨识真的辨得出来。

为什么需要设计轨迹
------------------
参数辨识解的是 :math:`\\tau = Y(q,\\dot q,\\ddot q)\\,\\pi`。
方程本身对 :math:`\\pi` 是线性的，看起来只要采够样本就能解——**但不是**。

如果轨迹太单调（比如只让一个关节慢慢来回摆），回归矩阵 :math:`Y` 的很多列
几乎线性相关，最小二乘就变成病态问题：**测量噪声被放大成巨大的参数误差**。

⭐ 这在控制理论里叫**持续激励（PE）条件**：轨迹必须"足够花哨"，
把每个参数都单独激发出来。

衡量指标：**回归矩阵的条件数** :math:`\\kappa`。它是噪声的放大倍数——

.. math::
    \\frac{\\lVert\\Delta\\pi\\rVert}{\\lVert\\pi\\rVert}
    \\;\\lesssim\\; \\kappa \\cdot \\frac{\\lVert\\Delta\\tau\\rVert}{\\lVert\\tau\\rVert}

$\\kappa=10^2$ 时，1% 的力矩噪声带来约 100% 的参数误差；
$\\kappa=10^8$ 时——不用算了，结果毫无意义。

⚠️⚠️ 两个必须避开的陷阱
-----------------------
**① 条件数必须在基参数子空间里算。**

完整参数集里有一部分参数**任何轨迹都辨不出来**（例如固定基座上某些惯量分量，
它们对关节力矩没有任何贡献）。直接对完整 :math:`Y` 算条件数会得到
$10^{18}$ 量级——那反映的是**结构性不可辨识**，与"轨迹好不好"无关，
而且**不管怎么优化都不变**。

⭐ Panthera：完整 78 维 → 基参数 rank **52**（26 维结构性不可辨识）。

**② 不要用 $G=W^{\\mathsf T}W$ 算条件数、也不要用它解最小二乘。**

$\\kappa(G)=\\kappa(W)^2$。`实测`：同一条轨迹上
$\\kappa(W)=10^{2.43}$、$\\kappa(G)=10^{4.85}$——正好是平方（2×2.43=4.86）。

`理论`：这意味着一条 $\\kappa(W)=10^{8}$ 的轨迹（很常见，随手写的激励就这个量级）
会得到 $\\kappa(G)=10^{16}$，**超出双精度约 $10^{16}$ 的分辨极限**，
解出来的参数是纯噪声，而程序不会报任何错。

本模块一律用 SVD 直接作用在 $W$ 上；:class:`~.offline_ls.OfflineBaseLS`
用增量 QR，同样不构造 $G$。
（armctrl 元教训 #26：一个判据给出逻辑上不可能的答案时，先怀疑判据。）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FourierTrajectory:
    """有限傅里叶级数激励轨迹。

    .. math::
        q_i(t) = q_{i,0}
        + \\sum_{k=1}^{K}\\Big[\\frac{a_{ik}}{k\\omega_f}\\sin(k\\omega_f t)
        - \\frac{b_{ik}}{k\\omega_f}\\cos(k\\omega_f t)\\Big]

    ⭐ **为什么选这个形式而不是随便一条曲线**：

    1. **解析可导**——$\\dot q, \\ddot q$ 有闭式，不用数值差分。
       这一点很重要：$\\ddot q$ 靠差分会把噪声放大 $1/\\Delta t^2$，
       而回归矩阵恰恰要用 $\\ddot q$。
    2. **周期性**——可以重复跑很多遍再平均，压掉随机噪声。
    3. **频谱可控**——谐波数 $K$ 直接对应激励的丰富程度。
    4. ⭐ **起止速度可以配平为零**（见 :meth:`zero_boundary`），
       真机上能安全启停，不会一上电就甩出去。

    Attributes:
        q0: 基准构型，形状 (n,)
        a: 正弦系数，形状 (n, K)
        b: 余弦系数，形状 (n, K)
        w_f: 基频 (rad/s)
    """

    q0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    w_f: float

    @property
    def n_joints(self) -> int:
        return len(self.q0)

    @property
    def n_harmonics(self) -> int:
        return self.a.shape[1]

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.w_f

    def zero_boundary(self) -> "FourierTrajectory":
        """把系数调成"起止速度为零"的版本。

        $\\dot q(0)=\\sum_k a_{ik}$，令它为 0 即可。做法是把每个关节的
        $a$ 系数整体减去其均值——⭐ 这样既满足边界条件，又不改变各谐波的相对结构。

        ⚠️ 真机上**务必用这个版本**：$\\dot q(0)\\ne 0$ 意味着一上电就要求
        一个阶跃速度，轻则跟踪误差巨大、重则触发保护。
        """
        a = self.a - self.a.mean(axis=1, keepdims=True)
        return FourierTrajectory(self.q0.copy(), a, self.b.copy(), self.w_f)

    def __call__(self, t: float | np.ndarray):
        """返回 (q, qd, qdd)。t 可以是标量或数组。"""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = np.arange(1, self.n_harmonics + 1)
        wk = k * self.w_f                                   # (K,)
        phase = np.outer(t, wk)                             # (T, K)
        sin_p, cos_p = np.sin(phase), np.cos(phase)

        q = self.q0[None, :] + (sin_p / wk) @ self.a.T - (cos_p / wk) @ self.b.T
        qd = cos_p @ self.a.T + sin_p @ self.b.T
        qdd = (-sin_p * wk) @ self.a.T + (cos_p * wk) @ self.b.T
        if q.shape[0] == 1:
            return q[0], qd[0], qdd[0]
        return q, qd, qdd

    def flat(self) -> np.ndarray:
        """把 a、b 展平成一维向量，供优化器使用。"""
        return np.concatenate([self.a.ravel(), self.b.ravel()])

    def with_flat(self, x: np.ndarray) -> "FourierTrajectory":
        """用展平向量重建轨迹（结构不变，只换系数）。"""
        size = self.a.size
        return FourierTrajectory(
            self.q0.copy(),
            x[:size].reshape(self.a.shape),
            x[size:].reshape(self.b.shape),
            self.w_f)


def random_trajectory(q0: np.ndarray, n_harmonics: int = 5,
                      w_f: float = 0.6, amplitude: float = 0.3,
                      seed: int = 0) -> FourierTrajectory:
    """随机初始化一条傅里叶轨迹，作为优化的起点。"""
    rng = np.random.default_rng(seed)
    n = len(q0)
    a = rng.uniform(-amplitude, amplitude, (n, n_harmonics))
    b = rng.uniform(-amplitude, amplitude, (n, n_harmonics))
    return FourierTrajectory(np.asarray(q0, dtype=float), a, b, w_f).zero_boundary()


@dataclass
class TrajectoryLimits:
    """轨迹必须满足的物理约束。

    ⚠️ 这些不是"最好满足"，是**违反了数据就不能用**：

    * 超关节限位 —— 真机会撞死
    * 超速度上限 —— 真机会报保护
    * ⭐ 超加速度上限 —— 官方 ``Follower.yaml`` 明确给了 2.0 rad/s²
    * ⭐ **力矩饱和** —— 饱和时实际力矩 ≠ 指令力矩，
      回归方程 $\\tau = Y\\pi$ **根本不成立**，那段数据必须丢弃

    ⚠️⚠️ **限值必须来自官方配置文件，不能自己填。**
    本项目最初把 ``qd_max`` 拍成 2.0、完全没有加速度约束，
    结果优化出的"合规"轨迹 `实测` **超速 68%、超加速度 57%**——
    而 ``violation()`` 报的是 0，因为它照着错误的限值检查。

    权威出处：``Panthera-HT_SDK/panthera_python/robot_param/Follower.yaml``::

        joint_limits: lower [-2.4,-0.1,-0.1,-1.6,-1.7,-2.5]
                      upper [ 2.4, 3.2, 4.0, 1.6, 1.7, 2.5]
        velocity_limits:     [1.0]*6
        acceleration_limits: [2.0]*6
        max_torque:          [21,36,36,21,10,10]   # ⚠️ 这是堵转扭矩

    ⚠️ ``max_torque`` 那一行是**堵转扭矩**，不是可持续输出。
    官方示例脚本自己就用了三套不同的限幅（`实测` 统计）：
    ``[10,20,20,10,5,5]``（2 处）、``[15,30,30,15,5,5]``（8 处）、
    ``[21,36,36,21,10,10]``（2 处）。本项目取**最保守的第一套**。
    """

    q_lower: np.ndarray
    q_upper: np.ndarray
    qd_max: np.ndarray
    tau_max: np.ndarray
    #: 加速度上限。None 表示不约束（⚠️ 只应在明确知道自己在做什么时用）
    qdd_max: np.ndarray | None = None
    #: 安全裕度：只用限位的这个比例，给真机留余量
    margin: float = 0.85


#: 官方 ``Follower.yaml`` 的限值。⭐ 用它，不要自己填数。
OFFICIAL_QD_MAX = np.full(6, 1.0)
OFFICIAL_QDD_MAX = np.full(6, 2.0)
#: 官方示例中最保守的一套力矩限幅（另有 15/30 与 21/36 两套，见上）
SDK_TAU_MAX = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])


def official_limits(model, margin: float = 0.85) -> TrajectoryLimits:
    """按官方配置构造约束。⭐ 新代码一律用这个，不要手工拼 TrajectoryLimits。"""
    return TrajectoryLimits(
        q_lower=model.jnt_range[:6, 0].copy(),
        q_upper=model.jnt_range[:6, 1].copy(),
        qd_max=OFFICIAL_QD_MAX.copy(),
        tau_max=SDK_TAU_MAX.copy(),
        qdd_max=OFFICIAL_QDD_MAX.copy(),
        margin=margin)


def violation(traj: FourierTrajectory, limits: TrajectoryLimits,
              n_samples: int = 200) -> float:
    """返回约束违反量（0 表示全部满足）。

    ⭐ 只看位置、速度、加速度——力矩要靠动力学模型算，代价高得多，
    放在评估阶段单独做（见 :func:`evaluate`）。
    """
    t = np.linspace(0.0, traj.period, n_samples)
    q, qd, qdd = traj(t)

    span = limits.q_upper - limits.q_lower
    lo = limits.q_lower + (1 - limits.margin) * span / 2
    hi = limits.q_upper - (1 - limits.margin) * span / 2

    over_lo = np.maximum(lo[None, :] - q, 0.0).sum()
    over_hi = np.maximum(q - hi[None, :], 0.0).sum()
    over_v = np.maximum(np.abs(qd) - limits.margin * limits.qd_max[None, :],
                        0.0).sum()
    total = over_lo + over_hi + over_v
    if limits.qdd_max is not None:
        total += np.maximum(np.abs(qdd)
                            - limits.margin * limits.qdd_max[None, :], 0.0).sum()
    return float(total)


def condition_number(traj: FourierTrajectory, regressor_fn,
                     projection: np.ndarray, n_samples: int = 120) -> float:
    """在**基参数子空间**里算回归矩阵的条件数。

    Args:
        regressor_fn: ``f(q, qd, qdd) -> (n, n_par)`` 的回归矩阵函数
        projection: 基参数投影矩阵 P，形状 (n_par, r)

    ⚠️ 两条不能省的规矩：

    1. **必须先投影**。不投影得到的是结构性病态（$10^{18}$），
       与激励质量无关，优化它毫无意义。
    2. **用 SVD 直接作用在 W 上**，不要构造 $G=W^{\\mathsf T}W$——
       后者把条件数平方，会顶穿双精度（元教训 #26）。
    """
    t = np.linspace(0.0, traj.period, n_samples, endpoint=False)
    q, qd, qdd = traj(t)
    rows = [regressor_fn(q[i], qd[i], qdd[i]) for i in range(len(t))]
    W = np.vstack(rows) @ projection
    sv = np.linalg.svd(W, compute_uv=False)
    if sv[0] <= 0:
        return float("inf")
    return float(sv[0] / max(sv[-1], 1e-300))


def optimize(traj: FourierTrajectory, regressor_fn, projection: np.ndarray,
             limits: TrajectoryLimits, iterations: int = 300,
             n_samples: int = 80, penalty: float = 1e4,
             seed: int = 0, verbose: bool = False):
    """用随机搜索最小化条件数。

    ⭐ **为什么用随机搜索而不是梯度法**：条件数对系数的梯度需要 SVD 的
    导数，实现复杂且数值敏感；而这个问题维度不高（n×K×2，通常几十维）、
    评估一次也不贵，随机搜索足够，而且**不会因为梯度实现出错而悄悄收敛到坏解**。

    ⚠️ 约束用罚函数处理：违反约束的候选直接加一个大惩罚。
    这样保证返回的轨迹**一定是可执行的**——不可执行的最优解没有意义。

    Returns:
        ``(best_traj, history)``，history 是每次改进时的 (iteration, log10κ)
    """
    rng = np.random.default_rng(seed)

    def cost(candidate: FourierTrajectory) -> float:
        bad = violation(candidate, limits)
        if bad > 0:
            return penalty * (1.0 + bad)
        return np.log10(condition_number(candidate, regressor_fn,
                                         projection, n_samples))

    best = traj.zero_boundary()
    best_cost = cost(best)
    history = [(0, best_cost)]
    scale = 0.25 * float(np.abs(best.flat()).mean() + 1e-9)

    for it in range(1, iterations + 1):
        x = best.flat() + rng.normal(0.0, scale, best.flat().shape)
        candidate = best.with_flat(x).zero_boundary()
        c = cost(candidate)
        if c < best_cost:
            best, best_cost = candidate, c
            history.append((it, c))
            if verbose:
                print(f"  iter {it:4d}  log10(κ) = {c:.3f}")
        else:
            # 长期没有改进就缩小步长，避免在好解附近乱跳
            if it % 50 == 0:
                scale *= 0.8
    return best, history


def evaluate(traj: FourierTrajectory, regressor_fn, projection: np.ndarray,
             limits: TrajectoryLimits, rnea_fn=None,
             n_samples: int = 200) -> dict:
    """给出一条轨迹的完整体检报告。

    ⭐ 报结果时必须带工况——单独一个条件数说明不了什么
    （armctrl 元教训 #18：报误差之前先说清楚哪个工况）。
    """
    t = np.linspace(0.0, traj.period, n_samples, endpoint=False)
    q, qd, qdd = traj(t)
    report = {
        "n_harmonics": traj.n_harmonics,
        "period": traj.period,
        "qd_limit_pct": float(
            (np.abs(qd) / limits.qd_max[None, :]).max() * 100.0),
        "cond": condition_number(traj, regressor_fn, projection, n_samples=120),
        "violation": violation(traj, limits, n_samples),
        "q_max_abs": float(np.abs(q).max()),
        "qd_max_abs": float(np.abs(qd).max()),
        "qdd_max_abs": float(np.abs(qdd).max()),
        "qd_start": float(np.abs(qd[0]).max()),
    }
    report["cond_log"] = float(np.log10(max(report["cond"], 1e-300)))
    if limits.qdd_max is not None:
        # ⭐ 直接报"用掉了限值的百分之几"，比报绝对值更容易看出危险
        report["qdd_limit_pct"] = float(
            (np.abs(qdd) / limits.qdd_max[None, :]).max() * 100.0)

    if rnea_fn is not None:
        tau = np.array([rnea_fn(q[i], qd[i], qdd[i]) for i in range(len(t))])
        report["tau_max_abs"] = float(np.abs(tau).max())
        # ⭐ 力矩饱和度：饱和期间的数据不能用于辨识
        report["tau_saturation_pct"] = float(
            (np.abs(tau) > limits.tau_max[None, :]).mean() * 100.0)
    return report
