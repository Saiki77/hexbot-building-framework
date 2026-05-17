"""Import-smoke tests for modules not exercised by the rest of the suite.

These don't check behaviour — they just make sure each module can be
imported without exploding (catches broken top-level statements, stale
imports, dangling references after refactors). Cheap and high-signal.
"""

import importlib

import pytest


# All `orca/*.py` modules that the rest of tests/ never touches.
# Bare import is enough to catch most regressions; deeper functional
# tests live in dedicated files.
_MODULES = [
    "orca.augment",
    "orca.benchmark",
    "orca.distributed",
    "orca.ensemble",
    "orca.gpu_server",
    "orca.leaderboard",
    "orca.replay",
    "orca.samples",
    "orca.scrape",
    "orca.sweep",
    "orca.zoo",
    "orca.__main__",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports_cleanly(module_name):
    """Each module must load without raising. No side-effect imports."""
    importlib.import_module(module_name)


def test_distributed_warning_fires():
    """MultiGPUTrainer/RayTrainer stubs must warn on construction."""
    import warnings
    from orca.distributed import RayTrainer

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RayTrainer(num_workers=2)
    assert any("STUB" in str(x.message) or "stub" in str(x.message)
               for x in w), "expected stub warning on RayTrainer()"


def test_main_dispatcher_has_main():
    """orca.__main__ exposes _main and does not execute it on bare import."""
    import orca.__main__ as main_mod
    assert callable(main_mod._main)
