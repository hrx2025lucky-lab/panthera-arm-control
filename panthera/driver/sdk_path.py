"""定位官方 SDK 的 ``Panthera_lib``，让我们的代码能从任意目录导入它。

⚠️ **为什么需要这个模块**

官方 SDK 的 README 写着：

    注意：Panthera_lib 作为源码提供，需要在 scripts 目录下使用

也就是说官方所有示例都必须 ``cd scripts && python 2_xxx.py``。
``Panthera_lib`` 不是一个装进 site-packages 的包，而是**散在源码树里的一个目录**，
只有当前工作目录恰好是 ``scripts/`` 时 Python 才找得到它。

这对官方示例没问题（它们本来就都躺在那个目录里），但对我们是硬伤：

* 我们的代码在 ``panthera_project/``，跑测试要 ``PYTHONPATH=.``；
* 如果还得 ``cd`` 到 SDK 的 ``scripts/``，那我们自己的包又找不到了；
* 两个"必须是当前目录"的要求直接冲突。

⇒ 所以这里把 SDK 的 ``scripts/`` 目录**显式加进 ``sys.path``**，
解开这个死结。之后无论从哪里启动，``from Panthera_lib import Panthera`` 都能用。

⭐ **顺带解决一个更隐蔽的问题：配置文件的相对路径。**

``Panthera()`` 不传参时会去找 ``scripts/Panthera_lib/../../robot_param/Follower.yaml``，
而这是相对**模块文件**算的（用了 ``os.path.dirname(__file__)``），不是相对 cwd，
所以配置能找到。但 YAML 里的 ``param_file: "../robot_param/motor_param/..."``
是相对 **cwd** 解析的——在别的目录启动就会读不到电机参数。
:func:`sdk_scripts_dir` 因此也提供给调用方去 ``chdir``，或直接用
:func:`follower_config` 拿绝对路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 环境变量：手动指定 SDK 的 ``panthera_python`` 目录，优先级最高
ENV_VAR = "PANTHERA_SDK"

#: 猜测位置。⚠️ 只是"常见摆法"，不是约定——找不到就报错让人显式指定。
_GUESSES = (
    "~/workspace/Roxan_warmup/panthera_official/Panthera-HT_SDK/panthera_python",
    "~/Panthera-HT_SDK/panthera_python",
    "~/Panthera_SDK/panthera_python",
)


def _looks_like_sdk(p: Path) -> bool:
    """判据：``scripts/Panthera_lib/Panthera.py`` 和 ``robot_param/`` 都在。

    ⚠️ 只检查目录名会误判——克隆到一半、或者只有空目录也会通过。
    这里检查的是**实际要 import 的那个文件**。
    """
    return (p / "scripts" / "Panthera_lib" / "Panthera.py").is_file() \
        and (p / "robot_param").is_dir()


def sdk_root() -> Path:
    """返回 SDK 的 ``panthera_python`` 目录。

    Raises:
        FileNotFoundError: 找不到时**抛异常而不是返回 None**。
            ⭐ 返回 None 会让错误推迟到 ``import Panthera_lib`` 那一行，
            报出来的是 ``ModuleNotFoundError: Panthera_lib``——
            看到这个的人第一反应是 "pip install panthera_lib"，然后卡住半小时。
            这里直接把真正的原因和解法写进异常里。
    """
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env).expanduser().resolve()
        if not _looks_like_sdk(p):
            raise FileNotFoundError(
                f"{ENV_VAR}={env} 指向的目录不像 SDK。\n"
                f"期望存在 {p / 'scripts' / 'Panthera_lib' / 'Panthera.py'}")
        return p

    for g in _GUESSES:
        p = Path(g).expanduser()
        if _looks_like_sdk(p):
            return p.resolve()

    raise FileNotFoundError(
        "找不到官方 SDK。请先克隆 Panthera-HT_SDK，然后指定路径：\n"
        f"  export {ENV_VAR}=/path/to/Panthera-HT_SDK/panthera_python\n"
        f"已尝试的位置：{', '.join(_GUESSES)}")


def sdk_scripts_dir() -> Path:
    """``scripts/`` 目录。官方示例都在这里，``Panthera_lib`` 也在这里。"""
    return sdk_root() / "scripts"


def follower_config() -> Path:
    """⭐ Follower 配置的**绝对路径**。

    ⚠️ **为什么默认 Follower 而不是 Leader**

    我们只有一台臂，而 SDK 里两套配置的差别不是"主/从"这么简单：

    ============  ====================  ====================
    项            Follower              Leader
    ============  ====================  ====================
    末端 link     ``tool_link``         ``joint6``  ← 是关节名不是连杆名
    J2/J3 下限    ``-0.1``              ``0.0``
    串口          ``/dev/ttyACM1``      ``/dev/ttyACM2``
    ============  ====================  ====================

    🔴 ``end_effector_link`` 不同意味着 **FK/IK 的结果不一样**——
    Leader 配置算出来的是裸法兰位姿，Follower 算的是加了工具的 TCP。
    这是我们发现的**第四个 TCP 定义**（另外三个：SDK ``tool_link`` 0.165、
    ROS2 ``bat_center`` 0.18、我们早期用的裸法兰 0.1893）。

    ⚠️ Leader 的 ``joint6`` 作为 ``end_effector_link`` 很可能是**笔误**——
    URDF 里 link 才有位姿，joint 名一般取不到 frame。待验证。

    ⇒ 我们统一用 Follower：它是 ``Panthera()`` 的默认值，
    也是官方重力补偿示例实际在用的那套（你上电跑通的就是它）。
    """
    return sdk_root() / "robot_param" / "Follower.yaml"


def ensure_importable() -> Path:
    """把 SDK 的 ``scripts/`` 插到 ``sys.path`` 最前面。

    ⚠️ 插到**最前面**而不是 append：如果环境里碰巧有同名模块，
    我们要的是 SDK 那个。

    幂等——重复调用不会插多次。

    Returns:
        插入的 ``scripts/`` 路径。
    """
    d = sdk_scripts_dir()
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    return d


def load_panthera_class():
    """``ensure_importable()`` + 导入 ``Panthera`` 类。

    ⚠️ 这里**只返回类不实例化**。实例化会真的去开串口、扫电机，
    是一个有副作用的动作，必须由调用方显式决定什么时候做。
    """
    ensure_importable()
    from Panthera_lib import Panthera  # noqa: E402  (路径准备好之后才能导)
    return Panthera
