"""模型资产路径解析。

与 armctrl 的区别
-----------------
armctrl 依赖第三方的 ``mujoco_menagerie``（Franka Panda），那是个不随仓库分发的
外部资产，所以它需要一整套「环境变量 + 候选路径」的搜索逻辑。

本项目的模型是**从高擎官方 URDF 转换而来、随仓库分发**的，路径是确定的，
因此这里不需要任何搜索——只做一次存在性校验，失败时给出重新生成的命令。

模型是怎么来的
--------------
源头是 `Panthera-HT_ROS2 <https://github.com/HighTorque-Robotics/Panthera-HT_ROS2>`_
的 ``panthera_ht_ros_description.urdf``（MIT）。转换过程见 ``tools/urdf_to_mjcf.py``，
它补上了 URDF 里**没有、CAD 也导不出**的三样东西：执行器、转子反射惯量、关节摩擦。

⚠️ 其中转子惯量与摩擦目前是**占位值**，必须由真机辨识回填，
见 ``docs/参数辨识与sim2real.md``。
"""

from __future__ import annotations

from pathlib import Path

#: 仓库根目录（本文件位于 <repo>/panthera/assets.py）
REPO_ROOT = Path(__file__).resolve().parents[1]

#: 模型目录
MODEL_DIR = REPO_ROOT / "models" / "panthera"


def panthera_xml() -> str:
    """Panthera-HT 六轴臂的 MJCF 路径。"""
    return str(MODEL_DIR / "panthera.xml")


def require_panthera_xml() -> str:
    """同 :func:`panthera_xml`，文件缺失时抛出带修复指引的异常。"""
    path = Path(panthera_xml())
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 Panthera 模型：{path}\n"
            f"模型随仓库分发，缺失通常意味着 clone 不完整或被误删。\n"
            f"可用官方 URDF 重新生成：\n"
            f"    git clone https://github.com/HighTorque-Robotics/Panthera-HT_ROS2.git\n"
            f"    python tools/urdf_to_mjcf.py \\\n"
            f"        --urdf Panthera-HT_ROS2/src/panthera_ht_ros_description/"
            f"urdf/panthera_ht_ros_description.urdf \\\n"
            f"        --meshes Panthera-HT_ROS2/src/panthera_ht_ros_description/meshes \\\n"
            f"        --out {MODEL_DIR}"
        )
    return str(path)
