#!/usr/bin/env python3
"""上电调试：真机到货第一天照着跑。

⛔ **这是唯一被批准的"第一次给真机发力矩"的方式。**

用法::

    # 全流程（推荐，会在每一步之间停下来等你确认）
    PYTHONPATH=. python scripts/commission.py

    # 只跑某一步
    PYTHONPATH=. python scripts/commission.py --only 3

    # 用假 SDK 先演练一遍（不接硬件，强烈建议先做）
    PYTHONPATH=. python scripts/commission.py --sim

设计原则
--------
⭐ **每一步都必须先跑通官方示例，再上我们的代码。**

这样出问题时能立刻分清是"机器的问题"还是"我们代码的问题"——
否则第一天就会陷入"到底谁错了"的泥潭。官方顺序（ROS2 README §七）：

    1_PD_control → 2_joint_impedance → 3_cartesian_impedance

⭐ **每一步都产出一个数**，而不只是"看起来能动"。
那些数会回填进模型和文档（见每步末尾的「记下来」）。

⚠️ **顺序不能改。** 后面的步骤假设前面的已经通过。

安全约定
--------
* 🔴 **手全程放在急停上**，直到第 6 步结束
* 🔴 工作空间清空，人不要站在手臂正下方或前方
* 🔴 底座必须固定牢（见客服问题 Q4）
* ⚠️ 任何一步出现异常抖动/异响 —— **立刻急停**，不要"再试一次看看"
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from panthera.core.robot import Q_HOME, make_panthera          # noqa: E402
from panthera.driver.real_backend import RealBackend           # noqa: E402
from panthera.driver.safety import SafetyLayer, config_from_backend  # noqa: E402

RESULTS: dict = {}


# ---------------------------------------------------------------- 工具

def banner(step: int, title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  第 {step} 步　{title}")
    print("=" * 66)


def confirm(msg: str) -> bool:
    """⚠️ 每一步之间必须人工确认。不允许无人值守连跑。"""
    ans = input(f"\n  ▶ {msg} [y/N] ").strip().lower()
    return ans == "y"


def abort(reason: str) -> None:
    print(f"\n  ⛔ 中止：{reason}")
    print("  请排查后重新运行本脚本。")
    sys.exit(1)


def make_backend(sim: bool):
    """⭐ ``--sim`` 用假 SDK 演练整个流程，不接硬件。

    强烈建议**先在 sim 下跑一遍**，把流程走熟，再接真机。
    """
    if sim:
        from panthera.driver.fake_sdk import FakePanthera
        from panthera.driver.mujoco_backend import MujocoBackend
        mb = MujocoBackend()
        print("  ⚠️ 【演练模式】使用假 SDK + MuJoCo，不涉及真实硬件。")
        return RealBackend(sdk=FakePanthera(mb), dt=mb.dt, model=mb.robot)
    return RealBackend(dt=0.005, model=make_panthera())


# ---------------------------------------------------------------- 步骤

def step1_official_examples() -> None:
    """第 1 步：先跑官方示例，确认**机器本身**是好的。"""
    banner(1, "先跑官方示例（不用本脚本，手动执行）")
    print("""
  ⭐ 在碰我们的代码之前，先确认机器本身没问题。
     否则后面出错时你分不清是机器还是代码。

  按顺序执行，每个都确认正常再进行下一个：

    cd <Panthera-HT_SDK>/panthera_python/scripts

    1) python 0_robot_get_state.py
       ✓ 6 个关节 + 夹爪的位置/速度/力矩都有读数
       ✓ 手动轻推关节，位置读数跟着变

    2) python 1_Joint_PD_control.py
       ✓ 手臂移动到 [0, 0.7, 0.7, -0.1, 0, 0] 并保持
       ⚠️ 这个构型和我们的 Q_HOME 完全一致

    3) python 2_gravity_compensation_control.py
       ✓ 手臂不下垂，手推能拖动，松手停在原地
       ⚠️ 这是第一次跑纯力矩模式，手放急停上

    4) python 2_gravity_friction_compensation_control.py
       ✓ 比上一个更容易推动（摩擦被补偿了）

  ⚠️ 任何一个不正常，先联系客服，不要继续。
""")
    if not confirm("以上 4 个官方示例都跑通了吗？"):
        abort("请先跑通官方示例")
    RESULTS["official_ok"] = True


def step2_connect(sim: bool):
    """第 2 步：连上我们的 RealBackend，只读不写。"""
    banner(2, "连接并读状态（只读，不发任何力矩）")
    be = make_backend(sim)
    print(f"  关节数 {be.n}，力矩限幅 {be.tau_limit}")

    st = be.read()
    print(f"  q  = {np.round(st.q, 4)}")
    print(f"  q̇  = {np.round(st.qd, 4)}")
    print(f"  τ  = {np.round(st.tau, 4)}")

    if not np.isfinite(st.q).all():
        abort("读到非有限值，检查连接")

    # ⚠️ 和模型的零位比一下——差太多说明零位没标定
    dev = np.abs(st.q - np.array(Q_HOME)).max()
    print(f"\n  与 Q_HOME 的最大偏差：{dev:.4f} rad")
    if dev > 1.0:
        print("  ⚠️ 偏差较大。如果手臂现在不在 Q_HOME 姿态，这是正常的。")

    print("""
  ▶ 请手动轻推每个关节，确认读数跟着变（尤其 J1 和 J6）。
    ⚠️ 如果某个关节推不动或读数不变，立刻停止排查。""")
    if not confirm("6 个关节的读数都正常吗？"):
        abort("状态读取异常")
    RESULTS["read_ok"] = True
    return be


def step3_measure_rate(be) -> None:
    """第 3 步：⭐ 实测控制频率。**这个数决定后面所有增益。**"""
    banner(3, "实测控制频率与抖动（仍然不发力矩）")
    print("""
  ⭐ 为什么这一步必须做：
     官方示例里控制率从 100 Hz 到 500 Hz 都有，阻抗示例注释写 200 Hz。
     那些是**期望值**不是实测值。所有基于 Δt 的推导都依赖真实节拍。

  ⚠️ 我们仿真里测过延迟敏感度，有个陡峭的拐点：
     延迟 4 ms → 跟踪正常；6 ms → 误差涨 15 倍并出现力矩饱和。
""")
    be.periods.clear()
    be._last = None
    n = 2000
    print(f"  正在采样 {n} 次读取…")
    t0 = time.monotonic()
    for _ in range(n):
        be.read()
    wall = time.monotonic() - t0

    s = be.period_stats()
    print(f"""
  实测结果（{s['n']} 个周期，总耗时 {wall:.2f} s）：
    平均    {s['mean_ms']:.3f} ms  →  {s['hz']:.0f} Hz
    标准差  {s['std_ms']:.3f} ms
    最小    {s['min_ms']:.3f} ms
    P99     {s['p99_ms']:.3f} ms
    最大    {s['max_ms']:.3f} ms   ⚠️ 看门狗要按这个定
""")
    RESULTS["rate"] = s

    hz = s["hz"]
    if hz < 100:
        print("  ⚠️ 低于 100 Hz。力矩控制会很吃力，考虑：")
        print("     - 检查是否有其他进程占用 CPU")
        print("     - 确认波特率设置（官方 4 Mbps）")
    elif hz < 180:
        print("  ⚠️ 低于官方阻抗示例的 200 Hz。控制带宽要相应降低。")
    else:
        print("  ✓ 达到或超过官方阻抗示例的 200 Hz。")

    print(f"""
  📝 记下来（要回填进代码和文档）：
     RealBackend(dt={s['mean_ms'] / 1000:.4f})
     SafetyConfig.watchdog_timeout = {max(s['max_ms'] * 2, 20) / 1000:.4f}
        （取实测最大周期的 2 倍，且不小于 20 ms）
""")
    if not confirm("频率可以接受，继续？"):
        abort("控制频率不满足要求")


def step4_single_joint(be, sim: bool) -> None:
    """第 4 步：⛔ **第一次发力矩**。单关节，从零开始。"""
    banner(4, "⛔ 第一次发力矩：单关节，从 0 开始斜坡")
    print("""
  🔴🔴 手放在急停上。现在开始真的会动。

  做法：只给 J1 加力矩，从 0 缓慢升到 1.5 N·m，观察它什么时候开始转。
  ⭐ 选 J1 是因为它绕竖直轴转，**重力不参与**，现象最干净。
  ⚠️ 其余关节保持零力矩（会因重力下垂，属正常——所以要托住或放低位）。
""")
    if not confirm("急停在手上，工作空间已清空，确认开始？"):
        abort("用户取消")

    cfg = config_from_backend(be) if hasattr(be, "model") and be.model is not None \
        else None
    q0 = be.read().q
    tau = np.zeros(be.n)
    moved_at = None

    try:
        for level in np.arange(0.0, 1.55, 0.1):
            tau[:] = 0.0
            tau[0] = level
            for _ in range(int(0.3 / be.dt)):        # 每档保持 0.3 s
                be.send_torque(tau)
                be.step()
            st = be.read()
            d = abs(st.q[0] - q0[0])
            print(f"    τ_J1 = {level:4.1f} N·m   Δq = {d:+.4f} rad"
                  f"   q̇ = {st.qd[0]:+.4f} rad/s")
            if moved_at is None and d > 0.02:
                moved_at = level
                print(f"    ⭐ J1 开始转动，静摩擦力矩约 {level:.1f} N·m")
            if d > 0.5:
                print("    已转过 0.5 rad，停止加力。")
                break
    finally:
        be.send_torque(np.zeros(be.n))

    RESULTS["j1_breakaway"] = moved_at
    print(f"""
  📝 记下来：
     J1 静摩擦（breakaway）力矩 ≈ {moved_at if moved_at else '未观测到'} N·m
     ⭐ 这个数可以直接和模型里的 frictionloss 对比：
        我们 MJCF 里 J1 的 frictionloss = 0.2（占位值，待辨识）
""")
    if moved_at is None:
        print("  ⚠️ 到 1.5 N·m 都没动。可能：底座没固定 / 关节卡住 / 指令没送到。")
        if not confirm("确认要继续吗？"):
            abort("单关节测试未通过")
    elif moved_at > 1.0:
        print("  ⚠️ 静摩擦比模型占位值(0.2)大很多，辨识时要注意。")


def step5_verify_clamp(be) -> None:
    """第 5 步：⭐ 故意超限，确认限幅**真的生效**。"""
    banner(5, "验证力矩限幅确实生效")
    print("""
  ⭐ 我们一直假设"限幅会保护硬件"。这一步验证这个假设。
     ⚠️ 一个没被验证过的安全措施，等于没有。

  做法：下发一个远超限幅的力矩指令，检查实际反馈是否被截断。
  ⚠️ 只对 J1 做，且只持续很短时间。
""")
    if not confirm("确认开始（手仍在急停上）？"):
        abort("用户取消")

    over = np.zeros(be.n)
    over[0] = be.tau_limit[0] * 10          # 10 倍超限
    peak = 0.0
    try:
        for _ in range(int(0.2 / be.dt)):
            be.send_torque(over)
            be.step()
            peak = max(peak, abs(be.read().tau[0]))
    finally:
        be.send_torque(np.zeros(be.n))

    limit = be.tau_limit[0]
    print(f"""
  指令   {over[0]:.1f} N·m
  限幅   {limit:.1f} N·m
  实测峰值 {peak:.2f} N·m
""")
    RESULTS["clamp_peak"] = peak
    if peak > limit * 1.3:
        abort(f"⛔ 限幅未生效！实测 {peak:.2f} 超过限幅 {limit:.1f} 的 1.3 倍。"
              "在解决之前不要跑任何整臂力矩控制。")
    print("  ✓ 限幅生效。")

    if be.dropped:
        print(f"  ⚠️ 有 {be.dropped} 条指令被 SDK 静默丢弃！这需要排查。")


def step6_gravity_comp(be, sim: bool) -> None:
    """第 6 步：整臂重力补偿 + 安全层。⭐ 第一次跑我们自己的完整链路。"""
    banner(6, "整臂重力补偿（经过安全层）")
    print("""
  ⭐ 这是第一次跑完整链路：控制器 → 安全层 → RealBackend → SDK → 电机

  预期：手臂保持不下垂，手推能拖动，松手停住。
  ⚠️ 和官方 2_gravity_compensation_control.py 的现象应该**一致**。
     如果差别很大，说明我们的模型或链路有问题。
""")
    if not confirm("确认开始？"):
        abort("用户取消")

    cfg = config_from_backend(be) if be.model is not None else None
    if cfg is None:
        print("  ⚠️ 无模型，跳过安全层（不推荐）")
        return
    cfg.tau_limit = be.tau_limit.copy()
    if "rate" in RESULTS:
        cfg.watchdog_timeout = max(RESULTS["rate"]["max_ms"] * 2, 20) / 1000
        print(f"  看门狗按实测设为 {cfg.watchdog_timeout * 1000:.1f} ms")

    safety = SafetyLayer(cfg)
    safety.arm()
    print("  运行 10 秒，请用手轻推手臂感受。Ctrl+C 提前结束。")

    n_drop = be.dropped
    t_end = time.monotonic() + 10.0
    try:
        while time.monotonic() < t_end:
            st = be.read()
            tau = be.gravity(st.q)
            tau, status = safety.filter(tau, st.q, st.qd)
            if not status.ok:
                print(f"  ⚠️ 安全层介入：{status.reason}")
                break
            be.send_torque(tau)
            be.step()
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        be.send_torque(np.zeros(be.n))

    print(f"""
  安全层状态：{'已触发故障' if safety.faulted else '正常'}
  被丢弃指令：{be.dropped - n_drop}
""")
    RESULTS["gravity_ok"] = not safety.faulted
    if not confirm("手感和官方示例一致吗（不下垂、能推动）？"):
        print("  ⚠️ 手感不一致可能意味着模型的重力项有偏差。")
        print("     真机辨识之后应该会改善。")


def step7_collision_threshold(be) -> None:
    """第 7 步：标定碰撞检测阈值。⭐ 必须用**空跑数据**定，不能拍脑袋。"""
    banner(7, "标定碰撞检测阈值（空跑，不要碰手臂）")
    from panthera.control.momentum_observer import MomentumObserver
    print("""
  ⭐ 阈值必须由"无外力时残差的实际波动"来定：
     拍小了天天误报警，拍大了撞了也不停。

  ⚠️ 接下来 10 秒**不要碰手臂**，也不要让任何东西碰到它。
""")
    if not confirm("确认手臂周围无接触？"):
        abort("用户取消")

    obs = MomentumObserver(be.robot, k_i=25.0, dt=be.dt)
    st0 = be.read()
    obs.reset(st0.q, st0.qd)
    rs = []
    t_end = time.monotonic() + 10.0
    try:
        while time.monotonic() < t_end:
            st = be.read()
            tau = be.saturate(be.gravity(st.q))
            obs.update(st.q, st.qd, tau)      # ⚠️ 喂实际施加的力矩
            rs.append(obs.r.copy())
            be.send_torque(tau)
            be.step()
    finally:
        be.send_torque(np.zeros(be.n))

    rs = np.array(rs)
    thr = MomentumObserver.calibrate_threshold(rs)
    print(f"""
  残差统计（{len(rs)} 个采样，无外力）：
    均值 {np.round(np.abs(rs).mean(axis=0), 4)}
    最大 {np.round(np.abs(rs).max(axis=0), 4)}

  ⭐ 6σ 阈值：{np.round(thr, 4)}

  📝 记下来：这组阈值只对**当前工况**有效。
     ⚠️ 高速运动时残差会更大（摩擦未建模），换工况要重标。
""")
    RESULTS["collision_threshold"] = thr.tolist()


def summary(sim: bool) -> None:
    banner(8, "汇总")
    if "rate" in RESULTS:
        s = RESULTS["rate"]
        print(f"  实测控制频率   {s['hz']:.0f} Hz（平均 {s['mean_ms']:.3f} ms）")
        print(f"  最大周期       {s['max_ms']:.3f} ms")
    if RESULTS.get("j1_breakaway") is not None:
        print(f"  J1 静摩擦      {RESULTS['j1_breakaway']:.1f} N·m")
    if "clamp_peak" in RESULTS:
        print(f"  限幅验证       峰值 {RESULTS['clamp_peak']:.2f} N·m ✓")
    if "collision_threshold" in RESULTS:
        print(f"  碰撞阈值(6σ)   {np.round(RESULTS['collision_threshold'], 3)}")

    print("""
  ✅ 上电调试完成。接下来可以做：
     1. 参数辨识（激励轨迹已按官方限值收紧）
     2. CTC / 阻抗控制真机验证
     3. 装 D405 之前**先辨识一次**，装完再辨识一次
        ⭐ 白拿一组"末端负载变化"的对照数据

  ⚠️ 把上面的数填进：
     - panthera/driver/real_backend.py  的 dt
     - panthera/driver/safety.py        的 watchdog_timeout
     - docs/官方资料对照审计.md          的实测栏
""")
    if sim:
        print("  ⚠️ 本次是【演练模式】，以上数字来自仿真，不是真机实测。")


def main() -> int:
    ap = argparse.ArgumentParser(description="Panthera-HT 上电调试")
    ap.add_argument("--sim", action="store_true",
                    help="用假 SDK 演练，不接硬件（建议先跑这个）")
    ap.add_argument("--only", type=int, help="只跑某一步（1-7）")
    args = ap.parse_args()

    print(__doc__)
    if not args.sim:
        print("\n  🔴🔴🔴 真机模式。确认：")
        print("     [ ] 底座已固定牢")
        print("     [ ] 急停在手边且可用")
        print("     [ ] 工作空间无人无障碍")
        print("     [ ] 已在 --sim 下演练过一遍")
        if not confirm("以上全部确认？"):
            abort("请先完成安全检查")

    steps = args.only and [args.only] or list(range(1, 8))
    be = None
    try:
        if 1 in steps:
            step1_official_examples()
        if any(s >= 2 for s in steps):
            be = step2_connect(args.sim) if 2 in steps else make_backend(args.sim)
        if 3 in steps:
            step3_measure_rate(be)
        if 4 in steps:
            step4_single_joint(be, args.sim)
        if 5 in steps:
            step5_verify_clamp(be)
        if 6 in steps:
            step6_gravity_comp(be, args.sim)
        if 7 in steps:
            step7_collision_threshold(be)
        summary(args.sim)
    except KeyboardInterrupt:
        print("\n\n  ⚠️ 用户中断")
    finally:
        if be is not None:
            be.close()
            print("  已置零力矩并关闭。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
