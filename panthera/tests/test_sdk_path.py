"""``sdk_path`` 的守护测试。

⚠️ 这些测试**不碰硬件、不 import Panthera_lib**——只验证"找路"这件事。
真正的导入需要 SDK 在场，那属于 ``connect_check.py`` 的职责。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from panthera.driver import sdk_path


def make_fake_sdk(tmp: Path, *, with_lib: bool = True,
                  with_param: bool = True) -> Path:
    root = tmp / "panthera_python"
    if with_lib:
        d = root / "scripts" / "Panthera_lib"
        d.mkdir(parents=True)
        (d / "Panthera.py").write_text("# fake")
    else:
        (root / "scripts" / "Panthera_lib").mkdir(parents=True)
    if with_param:
        (root / "robot_param").mkdir(parents=True, exist_ok=True)
        (root / "robot_param" / "Follower.yaml").write_text("robot: {}")
    root.mkdir(parents=True, exist_ok=True)
    return root


class TestDiscovery:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        root = make_fake_sdk(tmp_path)
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        assert sdk_path.sdk_root() == root.resolve()

    def test_env_var_pointing_at_wrong_dir_raises(self, tmp_path, monkeypatch):
        """⚠️ 指错了要**当场报错**，而不是悄悄回退去猜。

        回退会造成最糟的一类 bug：你以为在用 A 的配置，实际用的是 B。
        Leader/Follower 的末端定义不同，这种混淆会让 FK 结果整体偏 165mm。
        """
        monkeypatch.setenv(sdk_path.ENV_VAR, str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError, match="不像 SDK"):
            sdk_path.sdk_root()

    def test_empty_lib_dir_is_rejected(self, tmp_path, monkeypatch):
        """⚠️ 目录在但 ``Panthera.py`` 不在 —— 克隆到一半就是这样。

        判据必须落在**真正要 import 的那个文件**上，不能只看目录名。
        """
        root = make_fake_sdk(tmp_path, with_lib=False)
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        with pytest.raises(FileNotFoundError):
            sdk_path.sdk_root()

    def test_missing_robot_param_is_rejected(self, tmp_path, monkeypatch):
        root = make_fake_sdk(tmp_path, with_param=False)
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        with pytest.raises(FileNotFoundError):
            sdk_path.sdk_root()

    def test_error_message_tells_you_how_to_fix_it(self, tmp_path, monkeypatch):
        """⭐ 报错信息里必须有变量名和 export 命令。

        没有的话，看到 ``ModuleNotFoundError: Panthera_lib`` 的人
        第一反应是 ``pip install panthera_lib``，然后卡住。
        """
        monkeypatch.delenv(sdk_path.ENV_VAR, raising=False)
        monkeypatch.setattr(sdk_path, "_GUESSES", (str(tmp_path / "none"),))
        with pytest.raises(FileNotFoundError) as ei:
            sdk_path.sdk_root()
        assert sdk_path.ENV_VAR in str(ei.value)
        assert "export" in str(ei.value)


class TestPathHelpers:
    def test_follower_not_leader(self, tmp_path, monkeypatch):
        """🔴 默认必须是 Follower。

        Leader 的 ``end_effector_link`` 是 ``joint6``（裸法兰），
        Follower 是 ``tool_link``（TCP 0.165）—— 差 165mm。
        我们上电跑通的是 Follower（``Panthera()`` 的默认），必须保持一致。
        """
        root = make_fake_sdk(tmp_path)
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        assert sdk_path.follower_config().name == "Follower.yaml"
        assert sdk_path.follower_config().is_absolute()

    def test_ensure_importable_prepends_and_is_idempotent(
            self, tmp_path, monkeypatch):
        """⚠️ 必须插在 ``sys.path`` **最前面**，且重复调用不重复插。"""
        root = make_fake_sdk(tmp_path)
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        monkeypatch.setattr(sys, "path", list(sys.path))
        d = sdk_path.ensure_importable()
        assert sys.path[0] == str(d)
        before = len(sys.path)
        sdk_path.ensure_importable()
        assert len(sys.path) == before

    def test_load_panthera_class_does_not_instantiate(
            self, tmp_path, monkeypatch):
        """⭐ 只返回类，不实例化。

        实例化会真的开串口、扫电机——有副作用的动作必须由调用方决定时机。
        这里用一个假的 ``Panthera_lib`` 模块验证：如果它被实例化了，
        构造函数会把标志置 True。
        """
        root = make_fake_sdk(tmp_path)
        (root / "scripts" / "Panthera_lib" / "__init__.py").write_text(
            "instantiated = False\n"
            "class Panthera:\n"
            "    def __init__(self, *a, **k):\n"
            "        import Panthera_lib\n"
            "        Panthera_lib.instantiated = True\n")
        monkeypatch.setenv(sdk_path.ENV_VAR, str(root))
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.delitem(sys.modules, "Panthera_lib", raising=False)
        cls = sdk_path.load_panthera_class()
        import Panthera_lib
        assert Panthera_lib.instantiated is False
        assert cls is Panthera_lib.Panthera
        del sys.modules["Panthera_lib"]


class TestRealSdkIfPresent:
    """⭐ 如果本机真的克隆了 SDK，就顺便验一遍真的能找到、能导入。

    ⚠️ 用 ``skipif`` 而不是硬性要求——CI 上没有 SDK 也要能跑测试。
    """

    @staticmethod
    def _present() -> bool:
        try:
            sdk_path.sdk_root()
            return True
        except FileNotFoundError:
            return False

    @pytest.mark.skipif(not _present.__func__(), reason="本机没有克隆 SDK")
    def test_can_import_from_any_cwd(self, tmp_path, monkeypatch):
        """🔴 这是本模块存在的全部理由：**换个目录也能导入**。

        官方 README 写的是"需要在 scripts 目录下使用"。
        这个测试故意 chdir 到别处，证明我们绕开了那个限制。
        """
        monkeypatch.chdir(tmp_path)
        cls = sdk_path.load_panthera_class()
        assert cls.__name__ == "Panthera"

    @pytest.mark.skipif(not _present.__func__(), reason="本机没有克隆 SDK")
    def test_follower_config_actually_exists(self):
        assert sdk_path.follower_config().is_file()
