#!/usr/bin/env python3
"""连接自检：证明**我们自己的代码**能在这台电脑上和真机对话。

⭐ **这是从电脑控制真机的第 0 步。** 在跑任何会动的东西之前先跑它。

用法::

    # 只读（默认，安全，手臂不会有任何动作）
    PANTHERA_SDK=~/workspace/Roxan_warmup/panthera_official/Panthera-HT_SDK/panthera_python \\
    PYTHONPATH=. python scripts/connect_check.py

    # 加测"读+写"完整往返（⚠️ 会下发零力矩，手臂会变软）
    PYTHONPATH=. python scripts/connect_check.py --with-send

    # 不接硬件先演练
    PYTHONPATH=. python scripts/connect_check.py --sim

为什么需要这一步
----------------
你已经跑通了官方的 ``2_gravity_compensation_control.py``，这证明了
**硬件 + 官方 SDK** 是好的。但那不等于**我们的代码**能用——中间还隔着：

* ``Panthera_lib`` 只能在 SDK 的 ``scripts/`` 目录里导入（我们不在那）；
* Leader / Follower 两套配置，末端定义和限位都不同；
* 我们的 ``RealBackend`` 假设了一批 SDK 方法名，得验证真的存在。

⭐ **把"硬件问题"和"我们代码的问题"分开**，是排查真机故障的第一原则。
这个脚本就是那条分界线：它过了，之后再出问题就一定在我们这边。

它会产出三个数
--------------
1. ⭐⭐ **实测控制频率**——决定所有增益、看门狗阈值、延迟预算。
   官方示例写 ``sleep(0.002)``＝500 Hz，但那是**期望**不是实测。
2. **当前关节角**——顺便确认零位是否正常、是否贴着限位。
3. **周期抖动 p99**——看门狗超时阈值要按它设，不是按均值。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from panthera.driver.real_backend import RealBackend  # noqa: E402

JOINT_NAMES = [f"J{i + 1}" for i in range(6)]


def hr(title: str = "") -> None:
    print("\n" + "=" * 64)
    if title:
        print(f"  {title}")
        print("=" * 64)


def check_serial_ports() -> None:
    """⚠️ 先看串口再连 SDK。

    SDK 连不上时报的错很含糊，而九成原因是串口不在或没权限——
    先把这两件事排除掉，能省掉大量猜测。
    """
    from pathlib import Path
    ports = sorted(Path("/dev").glob("ttyACM*"))
    print(f"  /dev/ttyACM*  发现 {len(ports)} 个：{[p.name for p in ports]}")
    if not ports:
        print("  ✗ 一个都没有。检查 USB 线、通信底板供电、机械臂电源。")
        return
    # Follower 配置用 serial_id: 1 ⇒ /dev/ttyACM1
    target = Path("/dev/ttyACM1")
    if not target.exists():
        print("  ⚠️ /dev/ttyACM1 不存在，而 Follower 配置写的是 serial_id: 1")
        return
    mode = oct(target.stat().st_mode)[-3:]
    ok = mode[-1] in "67"      # others 有读写
    print(f"  /dev/ttyACM1  权限 {mode}  {'✓' if ok else '✗ 需要 sudo chmod 666'}")
    if len(ports) != 7:
        print(f"  ⚠️ 官方文档说正常应看到 7 个设备，现在是 {len(ports)} 个。")


def measure(be: RealBackend, seconds: float, send: bool) -> dict:
    """跑一段定频循环并统计实际周期。

    ⚠️ **不用 ``time.sleep(dt)`` 定频**，用**绝对截止时间**。
    ``sleep(dt)`` 是"干完活再睡 dt"，于是每周期都比 dt 长一点，
    误差会累积成越来越慢的漂移。官方 ``7_gamepad`` 示例用的也是绝对截止时间。
    """
    be.periods.clear()
    t0 = time.monotonic()
    deadline = t0
    n = 0
    zeros = np.zeros(be.n)
    while time.monotonic() - t0 < seconds:
        be.read()
        if send:
            # kp=kd=0、tau=0 ⇒ 电机输出零力矩（手臂变软），
            # 但完整走了一遍"下发 + motor_send_cmd"，才测得到真实往返耗时。
            be.send_torque(zeros)
        n += 1
        deadline += be.dt
        slack = deadline - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        else:
            deadline = time.monotonic()   # 已经超时，不要让欠账累积
    st = be.period_stats()
    st["loops"] = n
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="用假 SDK，不接硬件")
    ap.add_argument("--with-send", action="store_true",
                    help="⚠️ 同时下发零力矩，测完整往返（手臂会变软）")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.002,
                    help="目标周期，默认 2ms=500Hz（官方示例值，待实测校正）")
    args = ap.parse_args()

    print(__doc__.split("为什么需要这一步")[0])

    hr("① 串口")
    if args.sim:
        print("  (--sim，跳过)")
    else:
        check_serial_ports()

    hr("② 连接 SDK")
    if args.sim:
        from panthera.driver.fake_sdk import FakePanthera
        from panthera.driver.mujoco_backend import MujocoBackend
        mb = MujocoBackend()
        be = RealBackend(sdk=FakePanthera(mb), dt=args.dt, model=mb.robot)
        print("  假 SDK ✓（底下挂的是 MuJoCo，所以读回来的数是仿真的）")
    else:
        from panthera.driver.sdk_path import follower_config, sdk_root
        print(f"  SDK 根目录 : {sdk_root()}")
        print(f"  配置文件   : {follower_config().name}  (末端 tool_link, TCP 0.165)")
        print("  正在开串口扫电机……\n")
        try:
            be = RealBackend(dt=args.dt)
        except Exception as e:                      # noqa: BLE001
            print(f"\n  ✗ 连接失败: {type(e).__name__}: {e}")
            print("\n  排查顺序：机械臂电源 → USB 线 → 串口权限 → "
                  "是否有别的程序正占着串口（同一时刻只能有一个）")
            return 1
    print(f"\n  ✓ 电机数 = {be.n}")

    hr("③ 读一帧状态（只读，手臂不会动）")
    s = be.read()
    lo = np.asarray(be.sdk.joint_limits["lower"], dtype=float)
    up = np.asarray(be.sdk.joint_limits["upper"], dtype=float)
    print(f"  {'关节':<5}{'角度(rad)':>11}{'角度(度)':>10}"
          f"{'速度':>9}{'力矩':>9}   限位余量")
    for i in range(be.n):
        m = min(s.q[i] - lo[i], up[i] - s.q[i])
        flag = "  ⚠️ 贴限位" if m < 0.05 else ""
        print(f"  {JOINT_NAMES[i]:<5}{s.q[i]:>11.4f}{np.rad2deg(s.q[i]):>10.2f}"
              f"{s.qd[i]:>9.4f}{s.tau[i]:>9.3f}   {m:>6.3f}{flag}")

    hr("④ 重力力矩（只算不发）")
    g = be.gravity(s.q)
    rated = np.array([6.0, 10.0, 10.0, 6.0, 6.0, 6.0])
    print(f"  {'关节':<5}{'g(q) N·m':>11}{'额定':>8}{'占额定':>9}")
    for i in range(be.n):
        pct = abs(g[i]) / rated[i] * 100
        flag = "  ⚠️ 超额定" if pct > 100 else ""
        print(f"  {JOINT_NAMES[i]:<5}{g[i]:>11.3f}{rated[i]:>8.1f}"
              f"{pct:>8.1f}%{flag}")
    print("\n  ⭐ 这就是官方重力补偿示例每周期下发的东西。"
          "\n     现在我们自己也能算出来了——控制权已经在我们手上。")

    hr(f"⑤ 定频循环 {args.seconds}s"
       f"（{'读+写' if args.with_send else '只读'}，目标 {1 / args.dt:.0f} Hz）")
    if args.with_send and not args.sim:
        print("\n  ⚠️ 将下发零力矩，手臂会变软下垂。确认已托住或处于低位。")
        if input("  继续？(yes/no) ").strip().lower() not in ("y", "yes"):
            print("  已取消。")
            args.with_send = False
    st = measure(be, args.seconds, args.with_send and not args.sim)
    print(f"\n  循环次数   {st['loops']}")
    print(f"  ⭐ 实测频率 {st['hz']:.1f} Hz   （目标 {1 / args.dt:.0f} Hz）")
    print(f"  周期均值   {st['mean_ms']:.3f} ms   标准差 {st['std_ms']:.3f} ms")
    print(f"  最快/最慢  {st['min_ms']:.3f} / {st['max_ms']:.3f} ms")
    print(f"  ⭐ p99     {st['p99_ms']:.3f} ms  ← 看门狗阈值按这个设，不是按均值")
    print(f"  被丢弃指令 {be.dropped}")

    hr("结论")
    hz = st["hz"]
    if hz < 1 / args.dt * 0.9:
        print(f"  ⚠️ 实测 {hz:.0f} Hz 明显低于目标 {1 / args.dt:.0f} Hz。")
        print("     ⇒ 说明瓶颈在通信而不是我们的计算"
              "（关节 CTC 只占 500Hz 预算的 2.3%）。")
        print(f"     ⇒ 把 dt 改成 {1 / hz * 1000:.1f} ms 重跑，"
              "并按这个频率重算所有增益。")
    else:
        print(f"  ✓ 实测 {hz:.0f} Hz 达到目标。")
    print(f"\n  ⭐ 记下来（要回填进文档和 SafetyLayer 配置）：")
    print(f"     实测频率 = {hz:.1f} Hz")
    print(f"     周期 p99 = {st['p99_ms']:.3f} ms")
    print(f"     零位读数 = {np.array2string(s.q, precision=4)}")

    if not args.sim:
        be.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n  已中断（电机进入阻尼模式，手臂会缓慢落下）")
        sys.exit(130)
