#!/usr/bin/env python3
"""把高擎 Panthera-HT 官方 URDF 转成可做力矩控制与 RL 的 MuJoCo MJCF。

为什么需要这一步
----------------
官方 ``panthera_ht_ros_description.urdf``（MIT）是 **CAD 导出**的：连杆几何与
连杆惯量可信，但缺三样"落到真机一定存在、而 CAD 给不出"的东西——

1. **执行器**：URDF 里没有 actuator，直接加载得到 ``nu=0``，力矩控制无从谈起。
2. **转子反射惯量** ``armature``：减速比 :math:`N=36` ⇒ 反射惯量放大 :math:`N^2=1296` 倍。
   典型转子 1e-5 kg·m² 反射后是 **0.013**，而 link2 的 izz 只有 **0.0227**——
   同一量级。它**不是连杆的几何属性**，CAD 导不出来。
3. **关节摩擦** ``damping`` / ``frictionloss``：同理，几何模型里不存在摩擦。

第 2、3 项目前写入的是**占位值**（见 ``ROTOR_INERTIA_GUESS`` / ``FRICTION_GUESS``），
来源是官方 SDK 示例注释里的"建议初始值"，原文写明 **「需要根据实际机器人进行辨识」**。

⚠️⚠️ **不辨识就直接训 RL，等于在一个没有摩擦的世界里训练**，
策略上真机必然发涩。辨识流程见 ``docs/参数辨识与sim2real.md``。

用法
----
::

    python tools/urdf_to_mjcf.py \\
        --urdf  <ROS2 包>/urdf/panthera_ht_ros_description.urdf \\
        --meshes <ROS2 包>/meshes \\
        --out    models/panthera
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
from pathlib import Path

import numpy as np

#: MuJoCo 单个 mesh 的三角面上限
MAX_FACES = 200_000

#: 各关节减速比（参数手册 §2.2.2）
GEAR_RATIO = {"joint1": 36, "joint2": 36, "joint3": 36,
              "joint4": 36, "joint5": 30, "joint6": 30}

#: ⚠️ 占位：电机转子惯量（kg·m²，电机侧）。反射到关节 = 本值 × N²。
#: 这是该尺寸无刷电机的典型量级，**必须由真机辨识替换**。
ROTOR_INERTIA_GUESS = 1.0e-5

#: ⚠️ 占位：库仑摩擦 Fc（N·m）与粘滞摩擦 Fv（N·m·s/rad）。
#: 取自官方 SDK ``2_gravity_friction_compensation_control.py`` 的"建议初始值"。
#: 注意 J2/J3 给的是同一组数，而两者负载差很多——这本身就说明它不是标定结果。
FRICTION_GUESS = {"joint1": (0.20, 0.06), "joint2": (0.15, 0.06),
                  "joint3": (0.15, 0.06), "joint4": (0.15, 0.03),
                  "joint5": (0.04, 0.02), "joint6": (0.04, 0.02)}

#: 力矩限幅（N·m）。⚠️ 三个来源必须分清，本项目取最保守的 SDK 值：
#:   * URDF ``<limit effort>`` = 21/36/36/21/10/10 —— 参数手册的**堵转扭矩**
#:   * 参数手册**额定扭矩**    = 6/10/10/6/6/6     —— 可长期连续输出
#:   * SDK 示例 ``tau_limit``  = 10/20/20/10/5/5   —— 官方给出的工程限幅
TORQUE_LIMIT = {"joint1": 10.0, "joint2": 20.0, "joint3": 20.0,
                "joint4": 10.0, "joint5": 5.0, "joint6": 5.0}

#: TCP 相对 link6 原点的偏置（m）。裸法兰口径，装夹爪后必须重标。
#: ⭐ TCP 必须和官方 SDK 的 ``tool_link`` 一致（Follower.yaml 的
#: ``end_effector_link``），否则和 SDK 的 FK/IK 对不上。
#: ⚠️ 官方有三个不同的末端点：tool_link 0.165（TCP）、
#: bat_center 0.18（电池中心，不是 TCP）、我们最初量的裸法兰 0.1893（已废弃）。
TCP_OFFSET = (0.165, 0.0, 0.0)


# ---------------------------------------------------------------- STL 工具

def _face_count(path: Path) -> int:
    with path.open("rb") as f:
        f.seek(80)
        return struct.unpack("<I", f.read(4))[0]


def _load_stl(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        buf = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return buf[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def _save_stl(path: Path, tris: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            a, b, c = tri
            nz = np.cross(b - a, c - a)
            norm = np.linalg.norm(nz)
            f.write(struct.pack("<3f", *(nz / norm if norm else np.zeros(3))))
            for vertex in tri:
                f.write(struct.pack("<3f", *vertex.astype(float)))
            f.write(b"\0\0")


def decimate(tris: np.ndarray, grid: int) -> np.ndarray:
    """顶点聚类简化：把包围盒切成 grid³ 个格，同格顶点合并、退化面丢弃。

    只用于把超过 MuJoCo 面数上限的 mesh 降下来。碰撞用的是凸包，
    所以这里损失的是**视觉细节**，不影响动力学。
    """
    flat = tris.reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    cell = (hi - lo) / grid
    cell[cell == 0] = 1e-9

    idx = np.floor((tris - lo) / cell).astype(np.int64)
    key = idx[:, :, 0] * grid * grid + idx[:, :, 1] * grid + idx[:, :, 2]
    keep = ((key[:, 0] != key[:, 1]) & (key[:, 1] != key[:, 2])
            & (key[:, 0] != key[:, 2]))
    tris, key = tris[keep], key[keep]

    uniq, inv = np.unique(key.reshape(-1), return_inverse=True)
    acc = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(acc, inv, tris.reshape(-1, 3))
    np.add.at(cnt, inv, 1)
    merged = (acc / cnt[:, None])[inv].reshape(-1, 3, 3)

    _, first = np.unique(np.sort(key, axis=1), axis=0, return_index=True)
    return merged[first]


def prepare_meshes(src: Path, dst: Path) -> list[str]:
    """复制 STL，把超限的简化到 MuJoCo 能吃下的面数。"""
    dst.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    for stl in sorted(src.glob("*.STL")):
        out = dst / stl.name
        shutil.copy(stl, out)
        n = _face_count(out)
        if n <= MAX_FACES:
            continue
        tris = _load_stl(out)
        for grid in (220, 180, 150, 120, 100):
            small = decimate(tris, grid)
            if len(small) <= MAX_FACES * 0.95:
                _save_stl(out, small)
                notes.append(f"{stl.name}: {n} → {len(small)} 面 (grid={grid})")
                break
        else:
            raise RuntimeError(f"{stl.name} 无法简化到 {MAX_FACES} 面以内")
    return notes


# ---------------------------------------------------------------- MJCF 生成

def urdf_to_mjcf(urdf: Path, meshes: Path, out_dir: Path) -> Path:
    """借 MuJoCo 自己的编译器把 URDF 转成规范 MJCF，再补上缺失的物理量。

    ⭐ 刻意不手写 XML 解析：MuJoCo 的 ``mj_saveLastXML`` 保证结果一定能被它自己
    加载，比手工拼 XML 稳得多。
    """
    import mujoco

    # MuJoCo 读 URDF 时 mesh 路径必须能解析，先就地改成绝对路径
    text = urdf.read_text(encoding="utf-8")
    text = re.sub(r'filename="package://[^/]+/meshes/',
                  f'filename="{out_dir.resolve()}/meshes/', text)
    tmp = out_dir / "_tmp_abs.urdf"
    tmp.write_text(text, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(tmp))
    raw = out_dir / "_tmp_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    mjcf = raw.read_text(encoding="utf-8")
    tmp.unlink()
    raw.unlink()

    # 路径改回相对，便于随仓库分发
    mjcf = mjcf.replace(f'file="{out_dir.resolve()}/meshes/', 'file="')
    mjcf = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes"', mjcf)
    mjcf = mjcf.replace('<mujoco model="panthera_ht_ros_description">',
                        '<mujoco model="panthera_ht">')

    # ① 关节：补 armature / damping / frictionloss
    def patch_joint(match: re.Match) -> str:
        name = match.group(1)
        fc, fv = FRICTION_GUESS[name]
        armature = ROTOR_INERTIA_GUESS * GEAR_RATIO[name] ** 2
        return (match.group(0)[:-2]
                + f' armature="{armature:.6f}"'
                + f' damping="{fv:.4f}" frictionloss="{fc:.4f}"/>')

    mjcf = re.sub(r'<joint name="(joint\d)"[^/]*/>', patch_joint, mjcf)

    # ② TCP site（控制点，不是连杆原点）
    tcp = " ".join(str(v) for v in TCP_OFFSET)
    anchor = '<geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link6_ttb"/>'
    mjcf = mjcf.replace(
        anchor,
        f'{anchor}\n                '
        f'<site name="tcp" pos="{tcp}" size="0.008" rgba="0 1 0 0.6"/>')

    # ③ 执行器：纯力矩电机（对应真机 MIT 模式 kp=kd=0）
    #    ⚠️ ctrlrange 与 forcerange 都要设：前者限指令，后者限实际出力。
    #    这是保护真实硬件的最后一道闸，**不要为了仿真方便放开**。
    act = ["\n  <actuator>"]
    for joint, lim in TORQUE_LIMIT.items():
        act.append(f'    <motor name="{joint}_mot" joint="{joint}" '
                   f'ctrlrange="{-lim} {lim}" ctrllimited="true" '
                   f'forcerange="{-lim} {lim}"/>')
    act.append("  </actuator>\n")

    # ④ 传感器：与真机 SDK 的 get_current_pos/vel/torque 一一对应
    sen = ["  <sensor>"]
    for joint in GEAR_RATIO:
        sen += [f'    <jointpos name="{joint}_pos" joint="{joint}"/>',
                f'    <jointvel name="{joint}_vel" joint="{joint}"/>',
                f'    <jointactuatorfrc name="{joint}_frc" joint="{joint}"/>']
    sen.append("  </sensor>\n")

    # ⑤ ⚠️⚠️ 排除底座与 link1 的碰撞。
    #
    # MuJoCo 默认会过滤"父子刚体"之间的碰撞，**但这条规则对 world 不适用**
    # （父体是 world 时不自动过滤）。而 URDF 里 base_link 的碰撞网格和
    # link1 的碰撞网格本来就是贴合的，转过来之后就永久互相穿模。
    #
    # `实测` 后果：base↔link1 恒有 4 个接触点、穿透 2.3 mm，接触摩擦
    # **把 J1 的旋转完全锁死**——施加 5 N·m 一秒钟，J1 只转了 0.0001 rad。
    #
    # ⭐ 这个 bug 是闭环仿真抓出来的。在它之前的 51 项测试全部通过：
    # 因为它们只验"公式算得对"，没有一项真的把回路闭起来积分动力学。
    # 运动学、动力学、回归矩阵、条件数——全都不碰接触求解器。
    excl = ['\n  <contact>',
            '    <exclude body1="world" body2="link1"/>',
            '  </contact>\n']

    mjcf = mjcf.replace("</mujoco>", "\n".join(act + sen + excl) + "</mujoco>")

    dst = out_dir / "panthera.xml"
    dst.write_text(mjcf, encoding="utf-8")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--meshes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for note in prepare_meshes(args.meshes, args.out / "meshes"):
        print(f"  简化 {note}")

    dst = urdf_to_mjcf(args.urdf, args.meshes, args.out)
    print(f"  写出 {dst}")

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(dst))
    print(f"  自检 nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nsensor={model.nsensor}")
    assert model.nu == 6, "执行器数不对，力矩控制会失效"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
