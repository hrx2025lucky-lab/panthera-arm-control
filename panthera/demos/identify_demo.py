"""激励轨迹优化 + 参数辨识的端到端演示。

运行::

    PYTHONPATH=. python -m panthera.demos.identify_demo

⭐ 这个脚本的用途不是"跑出一个结果"，而是**把三条轨迹放在一起对比**，
让"训练集残差不能当判据"这件事直接摆在眼前。
"""

from __future__ import annotations

import numpy as np

from panthera.core.robot import Q_HOME, make_panthera
from panthera.identification import excitation as ex
from panthera.identification import pipeline as pl
from panthera.identification.regressor import DynamicsRegressor

XML = "models/panthera/panthera.xml"
#: 力矩上限取 SDK 示例值，不是 URDF 的堵转扭矩，也不是手册额定扭矩。
#: 三个来源必须分清——见 docs/参数辨识与sim2real.md
TAU_MAX_SDK = np.array([10.0, 20.0, 20.0, 10.0, 5.0, 5.0])
NOISE_STD = 0.02  # N·m，力矩测量噪声


def main() -> None:
    robot = make_panthera()
    reg = DynamicsRegressor(XML, n_arm=6)
    P, _ = reg.base_parameter_projection(samples=400)
    pi_true = reg.true_parameters()
    m = robot.model
    limits = ex.TrajectoryLimits(
        q_lower=m.jnt_range[:6, 0].copy(), q_upper=m.jnt_range[:6, 1].copy(),
        qd_max=np.full(6, 2.0), tau_max=TAU_MAX_SDK)
    q0 = np.array(Q_HOME)

    print(f"完整参数维数 {P.shape[0]} → 基参数 rank {P.shape[1]}"
          f"（{P.shape[0] - P.shape[1]} 维结构性不可辨识）\n")

    # A：几乎不动。这是"随便晃两下就采数据"的典型下场。
    bad = ex.FourierTrajectory(q0, np.zeros((6, 5)), np.zeros((6, 5)), 0.6)
    bad.a[1, 0] = 0.15
    bad = bad.zero_boundary()
    # B：随机傅里叶，没优化过
    mid = ex.random_trajectory(q0, n_harmonics=5, w_f=0.6, amplitude=0.35, seed=3)
    # C：以基参数条件数为目标优化过
    good, hist = ex.optimize(mid, reg.regressor, P, limits,
                             iterations=250, n_samples=60, seed=1)
    # ⚠️ hist[0] 是起点代价。起点若违反约束，这个数是罚函数值而不是 log10κ，
    #    照着念会得到一个荒唐的数字——所以分开报。
    k_mid = np.log10(ex.condition_number(mid, reg.regressor, P, 80))
    infeasible = ex.violation(mid, limits) > 0
    start_desc = (f"起点超限位（violation {ex.violation(mid, limits):.2f}），"
                  f"log10(κ) {k_mid:.2f}" if infeasible
                  else f"起点 log10(κ) {k_mid:.2f}")
    print(f"优化：{start_desc} → log10(κ) {hist[-1][1]:.2f}"
          f"，violation {ex.violation(good, limits):.2f}"
          f"，共改进 {len(hist) - 1} 次\n")

    holdout = ex.random_trajectory(q0, 5, 0.55, 0.3, seed=11)

    print(f"{'轨迹':<18}{'log10κ':>9}{'β误差%':>11}"
          f"{'训练残差%':>12}{'留出残差%':>12}")
    print("-" * 62)
    for name, traj in [("A 单关节慢摆", bad),
                       ("B 随机傅里叶", mid),
                       ("C 优化后", good)]:
        k = np.log10(ex.condition_number(traj, reg.regressor, P, 80))
        Y, tau = pl.collect(traj, reg.regressor, reg.rnea,
                            dt=0.004, noise_std=NOISE_STD, seed=5)
        res = pl.identify(Y, tau, P, pi_true=pi_true)
        ho = pl.holdout_check(res, holdout, reg.regressor, reg.rnea, dt=0.01)
        print(f"{name:<18}{k:>9.2f}{res['beta_error'] * 100:>11.3f}"
              f"{res['torque_error'] * 100:>12.3f}{ho * 100:>12.3f}")

    print("\n⭐ 注意 A 那一行：β 误差最大，训练残差却最小。")
    print("   训练集残差**不是**判据。能抓住 A 的只有 log10κ（事前）")
    print("   和留出残差（事后）。")

    rep = ex.evaluate(good, reg.regressor, P, limits, rnea_fn=reg.rnea)
    print(f"\n优化后轨迹体检：周期 {rep['period']:.1f}s，"
          f"|q|max {rep['q_max_abs']:.2f} rad，"
          f"|q̇|max {rep['qd_max_abs']:.2f} rad/s，"
          f"|τ|max {rep['tau_max_abs']:.2f} N·m，"
          f"饱和 {rep['tau_saturation_pct']:.1f}%，"
          f"起始速度 {rep['qd_start']:.1e} rad/s")


if __name__ == "__main__":
    main()
