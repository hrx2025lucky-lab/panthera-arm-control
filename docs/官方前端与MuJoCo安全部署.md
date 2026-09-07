# 官方前端与 MuJoCo 安全部署

## 当前结论

截至 2026-09-07，已部署的是下面这条**纯仿真链**：

```text
官方 Three.js 页面
  + 项目中文控制算法实验台
        │  30 Hz 显示 / 有界 REST 参数命令
        ▼
panthera.host（只监听 127.0.0.1）
        │  500 Hz：读状态 → 参考整形 → 控制律 → 重力优先限幅 → step → 回读
        ▼
MujocoBackend → MuJoCo 3.2.3
```

页面必须区分三个所有权层，不能把左侧模型误称为“MuJoCo 画面”：

| 层 | 当前实现 | 文件位置 |
|---|---|---|
| 三维显示 | 官方 `Panthera-HT_Host` 的 Three.js + URDF/mesh | `../panthera_official/Panthera-HT_Host/Panthera_digital_twin-main/frontend/dist` 与 `arm_description` |
| 中文实验台与 API | 本项目在内存中向官方 HTML 注入 CSS/JS，不覆盖官方文件 | `panthera/host/server.py`、`panthera/host/static/` |
| 物理与闭环控制 | 无头 MuJoCo，以 MJCF 为物理真值并执行 500 Hz 控制循环 | `panthera/host/runtime.py`、`panthera/driver/mujoco_backend.py`、`models/panthera/panthera.xml` |

实验台采用“四角仪表 + 中央模型”布局，不再永久占用整条右侧画布：

- **左上**：运行模式、重力补偿、硬件写入状态和四项闭环摘要；
- **右上**：控制算法、控制律、增益及服务器端运动约束；
- **左下**：MuJoCo 数据流、末端实际位姿、趋势图和可展开的逐轴力矩明细；
- **右下**：六轴目标、仿真扰动、运行提示和“保持当前位置”。

四个仪表之间的透明区域不捕获鼠标，中央 Three.js 模型仍可旋转和缩放。“保持当前位置”
固定在操作卡底部；回读失效时和所有运动/调参控件一起禁用，但仪表仍可查看和整体隐藏。
重复的官方末端位姿及模型信息浮窗已隐藏，避免覆盖监控角。

桌面端可通过每个仪表标题区的 `⠿` 手柄重新摆放，拖动只修改浏览器布局，绝不调用
控制 API。位置按归一化坐标保存在当前浏览器，并始终限制在顶部工具栏下方及视口四边
以内；窗口缩到 `900 px` 以下时自动恢复单列，重新放宽后再恢复桌面位置。左上角 `↺`
只清除布局记录并恢复默认四角，不修改算法、目标或机械臂状态。

这个入口不会 import `RealBackend` 或官方 `hightorque_robot` SDK，不会打开
`/dev/ttyACM*`，也没有真机脚本运行与编码器清零接口。它适合：

- 观察官方 URDF/网格和 MuJoCo 关节状态；
- 在同一中文页面验证 PD+重力、CTC、笛卡尔阻抗、纯重力及模型摩擦补偿；
- 调整有界控制参数、参考速度/加速度和仿真关节扰动；
- 对照目标、参考、实际、误差、原始/限幅命令/实际施加/传感回读力矩及重力分量；
- 后续接入统一的 SpaceMouse、BC、RL 动作执行器；
- 在真机 commissioning 前独立演示项目。

它**不是**“真机已经安全同步”，更不是“真机控制已经验收”。

### 当前显示与控制能力

| 能力 | 当前状态 | 准确含义 |
|---|---|---|
| Web → MuJoCo | 已接通 | 有界关节目标、算法、参数和仿真扰动进入 MuJoCo 闭环 |
| MuJoCo → Web | 已接通 | 步进后 `q/qd/actuator_force` 驱动中央 Three.js 和四角监控 |
| 原生 MuJoCo Viewer 同步 | 未接通 | 独立 Viewer 使用另一份 `MjData`，不能冒充当前闭环画面 |
| 真机 → Web/MuJoCo 镜像 | 未接通 | 尚无带硬件时间戳和新鲜度判据的相干真机状态总线 |
| Web/MuJoCo → 真机 | 禁止 | Host 固定 `hardware_write_enabled=false`，不得直接替换 backend |
| Web 拖动控制 | 未接通 | 当前模型只接受回读；浏览器本地图形拖动不进入控制器 |
| SpaceMouse | 仅离线数学模块 | 尚无 deadman、掉帧超时、控制权租约和运行时执行器 |

以后同时显示真机与预测仿真时，应使用两份独立状态：实色模型由 `q_real` 驱动，半透明
ghost 由 MuJoCo predictor 驱动，并分别记录两条 TCP 轨迹。不能每帧把 `q_real` 覆盖进
同一个 predictor，否则 sim/real 偏差会被人为清零。

## 为什么不用官方默认后端直接接真机

当前 `Panthera-HT_Host` 的官方入口有这些确定边界：

- `backend.sh` 默认选择 live，而不是 Demo；
- live 后端即使浏览器没打开，也会以 200 Hz 下发 `Joint_Pos_Vel` 保持命令；
- 无效初始位置会退成零目标；
- 服务监听 `0.0.0.0`，CORS/WebSocket 允许任意来源；
- 没有鉴权，却提供 move、home、set-zero 和运行 SDK 脚本的接口；
- 网页 `Disconnect` 只断开浏览器，后端控制循环仍在运行；
- `Stop` 是“当前位置保持”，不是急停或失能；
- 没有已验证的 signal/退出/掉线安全状态机。

因此本项目只复用官方静态前端，不复用它的 live 控制后端。

## 启动、检查与停止

### 1. 第一次安装依赖

执行机器：连接显示器的这台 Ubuntu 电脑。  
执行目录：`/home/limx/workspace/Roxan_warmup/panthera_project`。

```bash
/home/limx/venvs/panthera/bin/python -m pip install \
    --target .host_deps -r requirements-host.txt
```

作用：只把 Flask/Socket.IO 依赖装入项目内 `.host_deps`；不修改系统 Python、
Conda 环境或官方 SDK。成功标志是最后出现 `Successfully installed`。网络错误时保留
第一条完整错误，不要改用 `sudo pip`。

### 2. 启动

```bash
PYTHONPATH=. /home/limx/venvs/panthera/bin/python -m panthera.host
```

成功时必须看到：

```text
URL: http://127.0.0.1:5000
runtime_mode=mujoco_sim
hardware_write_enabled=false
gravity_compensation_enabled=true
```

若端口被占用，会出现 `Address already in use`；先确认旧服务是否仍是本项目，不要
直接杀掉未知进程。若缺官方前端，会明确打印缺失的 `frontend/dist` 或
`arm_description` 绝对路径。

### 3. 页面连接

打开 <http://127.0.0.1:5000/>。中文实验台先从 `/api/config` 验证
`runtime_mode=mujoco_sim`、`hardware_write_enabled=false` 和重力补偿不可关闭，再验证
`sample_seq/physics_time` 持续递增、当前 M/P/C 版本与最近 transition 完全一致、完整
重力项未被力矩限幅吃掉；全部通过后才自动连接官方三维视图。官方 Socket URL 被
锁定为当前同源，连接/关节/键盘/文件拖放入口均被禁用。页面上的“已连接”只表示
浏览器连接到本机 MuJoCo 后端，不是连接到真机。

任何一次回读超时、样本不递增、控制循环停止、版本错拍、重力饱和或 fault，页面都会
撤销绿色状态、禁用调参控件并断开三维 Socket；恢复后再按节流自动重连。官方前端关键
DOM id 若发生变化，服务会在启动控制循环前 fail-closed，要求重新审核覆盖层。

页面会隐藏官方键盘连续控制、SDK 脚本和未接入本项目后端的路点区域；关节目标、
算法选择、参数、扰动和闭环回读分布在四角仪表中，中央画布不再为侧栏永久让位。

### 4. 健康检查

```bash
curl -fsS http://127.0.0.1:5000/api/health
```

判断：

- `ok=true`：仿真循环正在运行且未锁存异常；停止或关闭后返回 HTTP 503；
- `hardware_write_enabled=false`：真机写入能力不存在；
- `gravity_compensation_enabled=true`：控制律强制包含 `g(q)`；
- `overrun_count`：Ubuntu 普通线程错过 2 ms 截止时间的累计次数，只能用于诊断，
  不能据此宣称具备真机实时性。

### 5. 停止

回到运行服务的终端按 `Ctrl+C`。这只停止纯 MuJoCo 进程；不会操作真机。

## 控制算法、重力补偿与闭环回读

五种可选择算法都包含 `MujocoBackend.gravity(q)`，网页没有关闭或缩放重力项的入口：

| 中文页面算法 | 仿真控制律 | 当前可调参数 |
|---|---|---|
| PD + 重力补偿 | `g + Kp(q_ref-q) + Kd(qd_ref-qd)` | 有界 `Kp/Kd` 倍率 |
| 计算力矩 CTC | `g + Cqd + M(qdd_ref + Kd edot + Kp e)` | `wn`、`zeta` |
| 笛卡尔阻抗 | `g + Cqd + J^T(K e + D edot)` | 平移/姿态刚度、固定设计阻尼倍率 |
| 纯重力补偿 | `g - D qd` | 小阻尼 |
| 重力 + 摩擦补偿 | `g - tau_passive - D qd` | 残余阻尼 |

目标位置先经过服务端参考生成器；默认最大速度 `0.05 rad/s`（约 `2.9 deg/s`）、
最大加速度 `0.25 rad/s²`，网页可调速度仍硬限制在 `0.01..0.15 rad/s`。网页目标在
模型硬关节限位内保留 `0.01 rad` 软裕量；运行中若新速度上限低于当前参考速度会被
拒绝，算法切换也只允许在参考与仿真机械臂都已静止时进行。输出遵守
`[10, 20, 20, 10, 5, 5] N·m` 逐轴上限，但先保留 `g(q)`，再用剩余 headroom 裁剪
反馈项；模型 `g(q)` 自身越界时不宣称补偿有效。所有参数都在服务端拒绝未知键、
NaN/Inf 和越界值，不能靠修改浏览器滑块绕过。

笛卡尔阻抗当前使用对称的固定阻尼设计 `D=2·倍率·sqrt(K)`；页面明确称“阻尼倍率”，
不把它误称为惯量归一化临界阻尼。自适应惯量阻尼的矩阵实现未完成对称正定验收前不开放。

每个控制周期保存同一 `sample_seq` 的：

```text
algorithm + mode_epoch / param_revision / command_revision
q_before / qd_before
→ q_ref / qd_ref / qdd_ref
→ tau_raw（含 gravity/coriolis/inertia-feedforward/tracking-feedback 分项）
→ tau_limited_command → data.ctrl 实际施加
→ MuJoCo step
→ q_after / qd_after / actuator_force readback
```

末端位置/姿态面板也直接使用这份已验证的 HTTP 后步进状态，不把 Socket 重连期间的旧值
标成新回读。算法、参数、目标、扰动或服务进程版本变化时，累计 RMS 与趋势会立即清空，
避免把两组实验数据画到同一条曲线上。

网页以约 10 Hz 降采样展示，算法本身不在浏览器里运行。自适应控制暂时显示为
“待辨识”且不可点击：在负载辨识、名义参数导入、持续激励和饱和时冻结规则验收前，
不把它伪装成可安全调参的算法。

这里验证的是**MuJoCo 模型内的重力自洽**。当前 `armature`、`damping`、
`frictionloss` 仍含 `_GUESS`，不能据此声称真机重力补偿已正确。真机必须用已确认
新鲜度的实际 `q` 每拍重新计算 `g(q)`，并经过单一安全命令网关。

## “同步真机”必须拆成两个模式

### A. 只读姿态镜像

```text
新鲜 q_real、qd_real → 写入 MuJoCo qpos/qvel → mj_forward → 网页显示
```

这条链不积分动力学、不向真机发命令。重力补偿不是显示所需，但真机若处于力矩
模式，仍必须由独立控制进程负责。当前官方 SDK 的“状态查询”并非可证明的纯被动读，
且现有状态是异步缓存，所以本次没有擅自启用它。

### B. 动力学预测器

```text
一次新鲜 real sample → 同一控制律 → 同一条限幅后 tau
                              ├─→ 真机
                              └─→ MuJoCo predictor
```

只有 B 才能比较同一输入下的 sim/real 发散。它还必须对齐采样时刻、生效时刻、
实际周期、限幅、延迟和初始 `q+qd`；软件数组 `dtau=0` 不能证明电机已同时执行。

## 当前真机禁止项与开闸条件

在以下条件全部留下日志证据前，不运行自研 `connect_check.py`、`commission.py`、
`gravity_hold.py`、tuner real backend、LeRobot record/eval 或 SpaceMouse 真机模式：

1. 通讯板后的 7 个电机都返回正确 ID、版本、mode、fault 与单调更新时间；
2. 每个控制周期只获取一次状态，能识别 stale/drop/out-of-order，且保存 sample age；
3. 只有一个绝对截止时间调度器，记录 sample/compute/send/flush 的墙钟时间；
4. 命令入口拒绝非 6 维、NaN/Inf、越界和过期状态，并记录 SDK 返回值；
5. 固件 watchdog 非零，并在机械支撑、低位姿、人工保护下实测通信丢失行为；
6. `brake`、`set_stop`、正常退出、异常退出、断电各自的物理效果已现场确认；
7. 无相机/无末端负载的重力模型先交叉验证，再逐关节、小幅、斜坡开力矩；
8. 相机或夹爪安装后重新做负载与重力辨识。

目前历史记录的首个故障边界是“USB 通讯板可见，但 7 个电机全部 disconnected / 
版本 0.0.0”，即通讯板到电机 CAN。这个边界恢复前，控制算法没有有效真机输入。

## BC + RL、IsaacLab/MJLab、D405、SpaceMouse 顺序

1. **统一动作契约**：分别保存 `policy_raw[-1,1]^6` 与 `q_target_abs[6] rad`；观测至少
   包含 `q、qd、prev_action、task_reference、stamp`。夹爪未到货前标 optional，不能
   把 6 维数据伪装成 7 维。
2. **统一执行器**：MuJoCo 和真机都采用相同的 q-target→内层 PD+重力→力矩限幅、
   latency 与 decimation。当前真机有 `send_mit`，MuJoCo 公共后端还没有等价接口，
   这是下一项代码任务。
3. **无相机 state-only 任务**：先做轨迹跟踪/reach，固定坐标系、关节顺序、单位、
   频率、终止条件和随机种子。
4. **真正的独立 sim2sim**：推荐 `IsaacLab/PhysX → ONNX → Panthera MuJoCo`。
   MJLab 本身也是 MuJoCo，适合接口回归与快速训练，但 `MJLab → MuJoCo` 不能作为
   两种独立物理引擎的互证。
5. **SpaceMouse 到货**：先在仿真逐轴确认方向、死区、按钮/deadman、丢帧与
   anti-windup，再录制带时间戳的 canonical episode，最后才转 LeRobot 格式。
6. **D405 到货**：先标定 intrinsics/extrinsics，输出
   `{object_pose, stamp, confidence, covariance}`，测延迟、掉帧与位姿误差；策略先吃
   这个状态接口，再逐步升级到图像策略。
7. **学习策略顺序**：scripted/SpaceMouse synthetic BC → BC baseline → IsaacLab PPO
   baseline → BC regularization 或 residual RL。自由在线探索不直接上真机。

## 当前自动验证

- Host 运行时 + HTTP/Socket.IO 安全契约：`44 passed`；
- 连同传统控制闭环测试：`73 passed`；
- 逐项验证五种算法静止时都含 `g(q)`、参数/速度/扰动边界、参考加速度与速度、
  `q_before → tau → q_after` 配对、算法/参数/命令版本绑定、重力优先限幅、状态新鲜度
  门禁、脚本执行和编码器清零禁用；
- `1440×900`、`1184×900`、`1280×720` 与 `430×932` 真实 Chrome 截图均加载官方
  URDF、6 个网格、中文四角仪表和实时闭环摘要；桌面端中央画布保持全宽且四角卡片
  不重叠，窄屏使用纵向仪表并保留 44 px 触控目标；
- 面板移动仅由专用手柄接收 Pointer Events，带 4 px 防抖、视口夹紧、位置记忆、
  `Escape`/取消回滚和方向键移动；布局控件与运动命令门禁相互独立；
- 浏览器页面点击链实际完成：切换 CTC、接受 `wn=9`、速度限制 `0.05 rad/s`、发送
  `J1 +0.02 rad` 的短距离目标，随后 Hold 恢复 `PD + 重力补偿`，回读版本一致且参考
  速度归零；还验证了回读中断时变红、禁控、断开三维 Socket，恢复后自动重连。全过程
  `hardware_write_enabled=false`。

这些都是离线/仿真证据，不覆盖真机 CAN、固件 watchdog、制动、掉电、状态新鲜度
或真实重力精度。
