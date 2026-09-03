# panthera-arm-control

高擎 **Panthera-HT** 六轴机械臂的力矩级控制栈，面向**真机部署**与**参数辨识驱动的 sim2real**。

> 本项目是 [`armctrl`](https://github.com/roxan-limx/armctrl) 的真机分支。
> armctrl 用 Franka Panda 在 MuJoCo 里验证算法，结论至多到「**同模型自洽**」；
> 这里换成有真机可跑的 Panthera-HT，目标是把那一档推到「**已在真机验证**」。

---

## 为什么单开一个仓库

armctrl 的 734 项测试和十几篇讲义里，大量实测数字是 **Panda 专属**的
（雅可比 0.4840、可辨识秩 62、7 自由度零空间、接触力分解那四个数……）。
直接把模型换掉，等于把几轮独立审核攒下来的验证结果**一次性作废**。

所以两边并存：**armctrl 保持仿真侧的完整验证资产，本仓库负责真机**。

---

## 与 armctrl 的三个关键差异

| | armctrl（Panda） | 本项目（Panthera-HT） |
|---|---|---|
| 自由度 | 7 ⇒ **有冗余**，零空间维度 1 | **6 ⇒ 零冗余**，零空间维度恒为 0 |
| 力矩限幅 | 仿真，放开到 ±1e6 无所谓 | ⚠️ **代码原样下发真机，限幅是安全闸，不许放开** |
| 模型来源 | `mujoco_menagerie`（第三方标定模型） | 官方 URDF（**CAD 导出**）+ 本仓库补齐传动参数 |

### ⚠️ 6 轴没有零空间

`impedance.py` 里保留了零空间项，但在本项目中它**恒等于零向量**。
保留是为了与 armctrl 逐行对照、以及将来挂冗余臂时能自然生效。

⭐ **不要因为「跑通了」就认为零空间在起作用**——想验证请断言投影矩阵的秩
（`test_model.py::test_no_redundancy` 就是干这个的）。

---

## ⚠️⚠️ 模型里有占位参数，用之前必读

官方 `panthera_ht_ros_description.urdf` 是 **CAD 导出**的。
连杆几何与连杆惯量可信，但缺三样**落到真机一定存在、而 CAD 给不出**的东西：

| 缺什么 | 现状 | 为什么 CAD 给不出 |
|---|---|---|
| **执行器** | ✅ 转换脚本已补 | URDF 本来就不描述执行器，原始模型 `nu=0` |
| **转子反射惯量** `armature` | ⚠️ **占位值** | 它是**传动系统**属性，不是连杆几何属性 |
| **关节摩擦** `damping`/`frictionloss` | ⚠️ **占位值** | 同上，几何模型里不存在摩擦 |

**转子惯量为什么不能忽略**：减速比 $N=36$ ⇒ 反射惯量放大 $N^2 = 1296$ 倍。
典型转子 `1e-5 kg·m²` 反射后是 **0.013**，而 link2 的 `izz` 只有 **0.0227**——**同一量级**。

占位值取自官方 SDK 示例注释里的「建议初始值」，原文写明
**「需要根据实际机器人进行辨识和调整」**。
（一个旁证：示例里 J2 和 J3 给了完全相同的一组数，而两者负载差很多。）

> ⛔ **不辨识就直接训 RL，等于在一个没有摩擦的世界里训练**，策略上真机必然发涩。
> 辨识流程见 [`docs/参数辨识与sim2real.md`](docs/参数辨识与sim2real.md)。

---

## 快速开始

```bash
pip install mujoco numpy
PYTHONPATH=. python -m unittest discover panthera/tests
```

```python
from panthera.core.robot import make_panthera, Q_HOME

robot = make_panthera()
print(robot.n)                  # 6
print(robot.tau_limit)          # [10. 20. 20. 10.  5.  5.]
print(robot.gravity(Q_HOME))    # [0. 0.379 4.216 1.039 0. 0.]
```

### 重新生成模型

模型随仓库分发，正常不需要重生成。需要时：

```bash
git clone https://github.com/HighTorque-Robotics/Panthera-HT_ROS2.git
python tools/urdf_to_mjcf.py \
    --urdf   Panthera-HT_ROS2/src/panthera_ht_ros_description/urdf/panthera_ht_ros_description.urdf \
    --meshes Panthera-HT_ROS2/src/panthera_ht_ros_description/meshes \
    --out    models/panthera
```

---

## 统一后端：同一份控制代码，仿真与真机都能跑

```python
from panthera.driver.mujoco_backend import MujocoBackend
from panthera.core.robot import Q_HOME

with MujocoBackend() as be:          # 换成 RealBackend() 即上真机，控制代码不动
    be.reset(Q_HOME)
    for _ in range(1500):
        s = be.read()
        be.send_torque(be.gravity(s.q))
        be.step()
```

⭐ **为什么要这一层**：如果仿真和真机的接口不同，会出现最难查的一类错误——
控制器搬到真机上因为**接口语义差一点**而行为不同，而你会以为那是 sim2real gap。
判据就脏了。

⚠️ `RealBackend` 目前是骨架。接入前必须先做四件事（见 `driver/mujoco_backend.py`
的 docstring）：实测控制频率、单关节先行、看门狗、验证限幅生效。

---

## ⭐ 最终目标：系统辨识 vs 域随机化的定量对照

这个项目不止是「把算法搬到真机上」。真正的目标是回答一个问题：

> **面对同一个 sim2real gap，模型法（把模型做准）和学习法（让策略鲁棒）
> 各自表现如何？**

|  | CAD 模型（摩擦=0、armature=0） | 辨识后模型 |
|---|---|---|
| **传统控制**（CTC / 阻抗） | A | B |
| **RL + 域随机化** | C | D |

⭐ **关键论点**：「RL 不需要精确模型」只对了一半。准确说是
**不需要精确的点估计，但需要合理的分布**——而辨识给的正是分布的中心与方差。
不辨识就只能把随机化范围拍得很宽，代价是**策略保守、性能下降**。

详见 [`docs/辨识与域随机化对照实验.md`](docs/辨识与域随机化对照实验.md)（实验设计）
与 [`docs/RL_sim2real流程.md`](docs/RL_sim2real流程.md)（八步流程、任务定义、部署架构）。

---

## sim2real 流程

```
辨识参数 ──→ 同步进 IsaacLab(USD) 与 MuJoCo(MJCF)
                  │                    │
            训练(PPO, 数千并行)          │
                  │                    │
               策略 ──── sim2sim 验证 ──┘   ← ⭐ 独立引擎交叉检验
                  │
                真机部署
```

⭐ 中间那步是**两个物理引擎的独立实现**（PhysX vs MuJoCo 求解器）在互验：
策略在 A 能跑、到 B 就废 ⇒ 它过拟合了 A 的数值特性，而不是学到了物理。

⚠️ **前提是两边模型一致**——`armature` 最容易在 URDF→USD 转换里丢掉。
丢了以后 sim2sim 对不上，你会以为是「引擎差异」，实际是「模型没对齐」。
所以有 [`tests/test_model_parity.py`](panthera/tests/test_model_parity.py)：
导出**物理指纹**（质量 / armature / 摩擦 / 限幅 / 减速比），在 Isaac 侧也导一遍逐项 diff。

```bash
PYTHONPATH=. python -m panthera.tests.test_model_parity   # 打印物理指纹
```

---

## 目录

```
panthera/
├── core/            运动学、雅可比、动力学（M, C, g, Λ）
├── control/         阻抗控制（⚠️ 零空间项在 6 轴上恒为零）
├── identification/  回归矩阵 τ = Y(q,q̇,q̈)π、基参数、离线最小二乘
├── driver/          统一后端：backend.py(接口) + mujoco_backend.py
│                   ⚠️ RealBackend 待接入 hightorque_robot
└── tests/           冒烟测试
models/panthera/     MJCF + 网格（由 tools/urdf_to_mjcf.py 生成）
tools/               URDF → MJCF 转换
docs/                参数辨识与 sim2real 方案、新对话交接 prompt
```

---

## 硬件

| | |
|---|---|
| 型号 | 高擎 Panthera-HT 六轴 |
| 负载 / 臂展 / 自重 | 3.5 kg / 860 mm / 4.35 kg |
| 减速比 | J1–J4：**36**；J5–J6：**30** |
| 额定 / 堵转扭矩 | 6–10 N·m / 21–36 N·m |
| 通信 | CAN 1 Mbps，**CAN FD 5 Mbps** |
| 模组控制频率 | 3 kHz |
| 官方 SDK | [`Panthera-HT_SDK`](https://github.com/HighTorque-Robotics/Panthera-HT_SDK)（MIT） |
| 控制协议 | **MIT 模式**（源自 MIT Mini Cheetah）：一帧下发 pos / vel / kp / kd / τ_ff，<br>电机内部按 τ = kp·Δq + kd·Δq̇ + τ_ff 以 3 kHz 执行。<br>⭐ kp=kd=0 即纯力矩（CTC 走这个）；kp 大即位置控制（RL 走这个）。<br>详见 [`docs/RL_sim2real流程.md`](docs/RL_sim2real流程.md) §零 |

⚠️ **力矩限幅有三个来源，别混**：

| 来源 | 值 | 含义 |
|---|---|---|
| URDF `<limit effort>` | 21/36/36/21/10/10 | 参数手册的**堵转扭矩** |
| 参数手册额定 | 6/10/10/6/6/6 | 可**长期连续**输出 |
| ⭐ SDK 示例 `tau_limit` | **10/20/20/10/5/5** | 官方工程限幅，**本项目取这个** |

---

## 许可

代码 MIT。模型派生自高擎 `Panthera-HT_ROS2`（MIT），版权归原作者。
