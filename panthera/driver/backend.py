"""统一的机械臂后端接口：同一份控制代码，既能跑仿真也能跑真机。

为什么需要这一层
----------------
armctrl 是纯仿真项目，控制代码直接操作 ``mjData``。本项目要把**同一份控制律**
下发到真实硬件，如果两边接口不同，就会出现最难查的一类错误：

    仿真里跑得好好的控制器，搬到真机上因为**接口语义差一点**而行为不同，
    而你会以为那是 sim2real gap。

⭐ 于是判据就脏了——你分不清差异来自「物理不一样」还是「代码不一样」。
（armctrl 元教训 #10：判据除了你关心的原因，不能还有别的原因让它变化。）

所以这里定义一个两边都实现的最小接口。**控制代码只依赖这个接口**，
换后端时一行不用改。

与官方 SDK 的对应
-----------------
真机后端包的是高擎 ``hightorque_robot``（预编译 whl），其 Python API 形如::

    robot.get_current_pos()  / get_current_vel()  / get_current_torque()
    robot.pos_vel_tqe_kp_kd(pos, vel, torque, kp, kd)   # MIT 模式
    robot.get_Gravity()
    robot.get_friction_compensation(vel, Fc, Fv, vel_threshold)

本接口的 :meth:`ArmBackend.send_torque` 对应 ``kp=kd=0`` 的纯力矩模式。

⚠️⚠️ 三条安全约定（真机后端必须遵守）
-------------------------------------
1. **力矩限幅永远生效**，不接受"临时放开"。它是保护硬件的最后一道闸。
2. **看门狗**：控制循环若超过 ``watchdog_s`` 没有新指令，后端必须置零力矩。
   真机上一次 Python 卡顿就可能让机械臂保持满力矩顶住。
3. **退出时必须掉电或置零**。官方 SDK 免责声明原话：
   「掉电请扶好机械臂，防止其跌落。」
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmState:
    """一次采样得到的关节状态。

    Attributes:
        q: 关节位置 (rad)，形状 (n,)
        qd: 关节速度 (rad/s)，形状 (n,)
        tau: 关节力矩 (N·m)，形状 (n,)
        stamp: 采样时刻 (s)，单调时钟

    ⚠️ ``tau`` 的可信度在两个后端**不一样**，用它做判据时必须分清：

    * 仿真：来自 ``jointactuatorfrc`` 传感器，是求解器算出的真值
    * 真机：由**电流 × 固定系数**估算（官方 SDK 就是这么做的），
      不含摩擦、不含减速器效率，**是电机侧估计而非关节输出力矩**

    ⭐ 所以真机的 ``tau`` 属于「自报」档，**不能用来自证控制器算得对**。
    做动量观测器时按惯例用**指令力矩**而不是这个读数。
    """

    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    stamp: float


class ArmBackend(abc.ABC):
    """仿真与真机共用的最小后端接口。

    控制算法只依赖这五个方法，因此换后端不需要改任何控制代码。
    """

    #: 关节数
    n: int

    #: 力矩限幅 (N·m)，形状 (n,)。⚠️ 任何实现都不得在运行时放开它。
    tau_limit: np.ndarray

    #: 控制周期 (s)
    dt: float

    @abc.abstractmethod
    def read(self) -> ArmState:
        """读取当前关节状态。"""

    @abc.abstractmethod
    def send_torque(self, tau: np.ndarray) -> None:
        """下发关节力矩 (N·m)。实现必须在内部做限幅，不能指望调用方。"""

    @abc.abstractmethod
    def gravity(self, q: np.ndarray) -> np.ndarray:
        """给定构型下的重力力矩 g(q)。"""

    @abc.abstractmethod
    def step(self) -> None:
        """推进一个控制周期。

        仿真：``mj_step``；真机：等到下一个控制节拍（真实时间）。

        ⭐ 把"推进时间"也放进接口，是为了让同一段控制循环在两边逐字相同。
        否则仿真里写 ``mj_step``、真机里写 ``sleep``，循环结构就分叉了。
        """

    @abc.abstractmethod
    def close(self) -> None:
        """安全收尾：置零力矩 / 掉电。**任何异常路径都必须走到这里。**"""

    # ---- 通用工具，两个后端共享实现 ----

    def saturate(self, tau: np.ndarray) -> np.ndarray:
        """力矩限幅。⚠️ 这是安全边界，不是"建议值"。"""
        return np.clip(tau, -self.tau_limit, self.tau_limit)

    def __enter__(self) -> "ArmBackend":
        return self

    def __exit__(self, *exc) -> None:
        # ⭐ 用 with 语句保证异常时也会掉电——真机上这一条能防事故。
        self.close()
