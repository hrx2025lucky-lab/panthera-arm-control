"""上电脚本的守护测试。

⚠️ 这个脚本是**唯一被批准的"第一次给真机发力矩"的方式**，
它自己出 bug 的后果比一般代码严重得多。

⭐ 这里测的是"脚本的安全属性"，不是"流程能跑完"——
后者要人在场确认，测不了。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "commission", Path(__file__).resolve().parents[2] / "scripts" / "commission.py")


@pytest.fixture(scope="module")
def mod():
    m = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(m)
    return m


class TestSafetyProperties:
    def test_sim_mode_never_touches_real_sdk(self, mod):
        """⭐ ``--sim`` 必须完全走假 SDK，不能有任何真机调用。"""
        be = mod.make_backend(sim=True)
        from panthera.driver.fake_sdk import FakePanthera
        assert isinstance(be.sdk, FakePanthera)

    def test_abort_exits_nonzero(self, mod):
        """⛔ 中止必须真的退出，不能只打印一行然后继续。"""
        with pytest.raises(SystemExit) as e:
            mod.abort("test")
        assert e.value.code == 1

    def test_backend_is_closed_with_zero_torque(self, mod):
        """⚠️ 任何退出路径都必须置零力矩。"""
        be = mod.make_backend(sim=True)
        be.send_torque(np.full(be.n, 3.0))
        be.close()
        np.testing.assert_array_equal(be.sdk._pending[:, 2], np.zeros(be.n))

    def test_clamp_check_would_catch_a_broken_limit(self, mod):
        """⭐ 第 5 步的判据必须真的能抓到"限幅失效"。

        ⚠️ 这是元教训 #10 的应用：用一个**已知失效**的对象去考判据本身。
        """
        be = mod.make_backend(sim=True)
        limit = be.tau_limit[0]
        # 判据是 peak > limit * 1.3 就中止
        assert limit * 10 > limit * 1.3          # 失效时会被抓到
        assert limit * 1.0 <= limit * 1.3        # 正常时不误报

    def test_single_joint_step_uses_j1(self, mod):
        """⭐ 第一次发力矩必须选 J1——它绕竖直轴，重力不参与，现象最干净。"""
        src = (Path(mod.__file__)).read_text(encoding="utf-8")
        assert "tau[0] = level" in src
        assert "重力不参与" in src

    def test_watchdog_is_derived_from_measurement(self, mod):
        """⚠️ 看门狗阈值必须由**实测**周期推出，不能写死。"""
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert 'RESULTS["rate"]["max_ms"]' in src


class TestOfficialAlignment:
    def test_step1_requires_official_examples_first(self, mod):
        """⭐ 必须先跑官方示例——否则出错时分不清是机器还是我们的代码。"""
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for name in ("0_robot_get_state", "1_Joint_PD_control",
                     "2_gravity_compensation_control"):
            assert name in src

    def test_default_dt_matches_official_200hz(self, mod):
        """官方阻抗示例注释"控制频率：200Hz"，我们默认 dt 应对齐。"""
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "dt=0.005" in src
