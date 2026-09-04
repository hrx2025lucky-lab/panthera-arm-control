"""辨识流水线：从一条轨迹到一组参数。

⭐ **为什么这个模块必须存在**：辨识不是"跑个最小二乘"这么一句话。
中间有四个地方会静悄悄地出错，每一个都会给出**看起来很正常的错误答案**：

1. 轨迹没优化 → 条件数大 → 噪声被放大成参数误差
2. 条件数在完整参数空间算 → 得到 $10^{300}$，与激励质量无关
3. 用 $W^{\\mathsf T}W$ 解正规方程 → 条件数平方 → 顶穿双精度
4. ⭐ 拿 $\\hat\\pi$ 和真值逐项比 → **必然对不上**，因为不可辨识方向
   本来就无法确定，但这不代表辨识失败

第 4 条是最容易误判的。正确的判据见 :func:`identify` 的返回值说明。

判据怎么选
----------
⚠️ 不能用"力矩残差小"就宣布辨识成功——残差小只说明**在这条轨迹上**拟合得好，
换条轨迹可能全错（过拟合）。本模块同时给出三个互相独立的判据：

======================  ==========================================
判据                     它能证明什么
======================  ==========================================
``residual``            在**训练轨迹**上拟合得好（必要不充分）
``beta_error``          基参数坐标下的误差——⭐ **这才是主判据**
``holdout_residual``    在**另一条没参与拟合的轨迹**上仍然准
======================  ==========================================

⭐ ``beta_error`` 之所以是主判据：基参数子空间里的分量**唯一确定**，
真值可以精确算出来（$\\beta_{true}=P^{\\mathsf T}\\pi_{true}$，当 $P$ 列正交时），
所以这是一个**独立于拟合过程**的判据（元教训 #10：判据必须独立于被测对象）。
"""

from __future__ import annotations

import numpy as np

from .excitation import FourierTrajectory
from .offline_ls import OfflineBaseLS


def collect(traj: FourierTrajectory, regressor_fn, rnea_fn,
            dt: float = 0.002, periods: float = 1.0,
            noise_std: float = 0.0, seed: int = 0):
    """沿轨迹采样，返回 ``(Y_stack, tau_stack)``。

    Args:
        noise_std: 加在力矩上的高斯噪声标准差 (N·m)。
            ⭐ 真机上力矩测量必有噪声，把它显式建模出来才能看清
            "条件数 = 噪声放大倍数"这件事。

    ⚠️ 这里的 tau 由 RNEA **正向算出**，是"仿真真值"。
    真机上 tau 来自电流环反馈，还要额外考虑：
    力矩常数误差、减速器效率、温漂——所以真机的 ``noise_std`` 远大于这里。
    """
    rng = np.random.default_rng(seed)
    n_steps = int(periods * traj.period / dt)
    t = np.arange(n_steps) * dt
    q, qd, qdd = traj(t)

    Ys, taus = [], []
    for i in range(n_steps):
        Ys.append(regressor_fn(q[i], qd[i], qdd[i]))
        tau = rnea_fn(q[i], qd[i], qdd[i])
        if noise_std > 0:
            tau = tau + rng.normal(0.0, noise_std, tau.shape)
        taus.append(tau)
    return np.vstack(Ys), np.concatenate(taus)


def identify(Y: np.ndarray, tau: np.ndarray, projection: np.ndarray,
             pi_true: np.ndarray | None = None, ridge: float = 0.0) -> dict:
    """求解并给出**可复核的**体检报告。

    Returns:
        字典，含 ``beta``、``pi_hat``、``rank``、``cond``、``residual``；
        给了 ``pi_true`` 时还有：

        * ``beta_error`` —— ⭐ 主判据。基参数坐标下的相对误差
          $\\lVert\\hat\\beta-\\beta_{true}\\rVert/\\lVert\\beta_{true}\\rVert$
        * ``pi_error`` —— ⚠️ **参考值，不是判据**。完整参数空间的误差
          在不可辨识方向上没有意义，这个数偏大是**正常的**。
        * ``torque_error`` —— 力矩预测误差，物理上最直观
    """
    ls = OfflineBaseLS(projection, ridge=ridge)
    ls.add(Y, tau)
    beta, info = ls.solve()
    if beta is None:
        return {"beta": None, **info}

    out = {"beta": beta, "pi_hat": projection @ beta, **info}
    if pi_true is not None:
        pi_true = np.asarray(pi_true, dtype=float)
        beta_true = projection.T @ pi_true
        denom = np.linalg.norm(beta_true)
        out["beta_error"] = float(
            np.linalg.norm(beta - beta_true) / denom) if denom > 0 else float("nan")
        out["pi_error"] = float(
            np.linalg.norm(out["pi_hat"] - pi_true) / np.linalg.norm(pi_true))
        tau_true = Y @ pi_true
        out["torque_error"] = float(
            np.linalg.norm(Y @ out["pi_hat"] - tau_true)
            / max(np.linalg.norm(tau_true), 1e-12))
    return out


def holdout_check(result: dict, traj: FourierTrajectory, regressor_fn,
                  rnea_fn, dt: float = 0.002) -> float:
    """在**另一条轨迹**上检验辨识结果，返回相对力矩误差。

    ⭐ 这是防过拟合的独立判据：训练残差小 + 留出残差也小，
    才能说"这组参数真的描述了这台机器"，而不是"这组参数背下了这段数据"。
    """
    Y, tau = collect(traj, regressor_fn, rnea_fn, dt=dt)
    pred = Y @ result["pi_hat"]
    return float(np.linalg.norm(pred - tau) / max(np.linalg.norm(tau), 1e-12))
