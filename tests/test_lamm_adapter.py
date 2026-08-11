import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/lamm/run_lamm_facescape.py"
SPEC = importlib.util.spec_from_file_location("run_lamm_facescape", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_official_learning_rate_schedule_endpoints_and_decay():
    values = [
        MODULE.cosine_learning_rate(step, 1500, 1, 1e-3, 1e-6, 1e-8)
        for step in range(1501)
    ]
    assert values[0] == pytest.approx(1e-8)
    assert values[1] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-6)
    assert all(left >= right for left, right in zip(values[1:], values[2:]))


def test_official_manipulation_alpha_curriculum():
    assert MODULE.manipulation_alpha_max(0) == pytest.approx(0.25)
    assert MODULE.manipulation_alpha_max(1) == pytest.approx(0.2575)
    assert MODULE.manipulation_alpha_max(100) == pytest.approx(1.0)
    assert MODULE.manipulation_alpha_max(1500) == pytest.approx(1.0)
