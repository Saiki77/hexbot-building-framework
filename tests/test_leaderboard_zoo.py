"""Tests for the Leaderboard and Model Zoo - features promoted on the README.

These cover the persistence and metadata paths without actually playing games
(which would be slow) or hitting GitHub (network-flaky on CI).
"""

import json
import os
import tempfile

import pytest
import torch


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _make_lb(tmpdir):
    """Build a Leaderboard pointed at a fresh leaderboard.json under tmpdir."""
    from orca.leaderboard import Leaderboard
    return Leaderboard(path=os.path.join(tmpdir, "leaderboard.json"))


def test_leaderboard_init_empty(tmp_path):
    """A new leaderboard at a non-existent path starts empty."""
    lb = _make_lb(str(tmp_path))
    assert lb.entries == []


def test_leaderboard_persist_and_reload(tmp_path):
    """Manually-added entries survive save+reload."""
    from orca.leaderboard import Leaderboard

    path = str(tmp_path / "lb.json")
    lb = Leaderboard(path=path)
    lb.entries.append({
        "name": "test-bot",
        "elo": 1234.5,
        "results": {},
        "total_wins": 10,
        "total_games": 20,
        "overall_win_rate": 0.5,
        "rated_at": 1.0,
    })
    lb._save()

    # Reload from disk
    lb2 = Leaderboard(path=path)
    assert len(lb2.entries) == 1
    assert lb2.entries[0]["name"] == "test-bot"
    assert lb2.entries[0]["elo"] == 1234.5


def test_leaderboard_show_empty_does_not_crash(tmp_path, capsys):
    """show() must not raise on an empty leaderboard."""
    lb = _make_lb(str(tmp_path))
    lb.show()
    # Output is informational; just confirm something was printed
    captured = capsys.readouterr()
    assert isinstance(captured.out, str)


# ---------------------------------------------------------------------------
# Zoo
# ---------------------------------------------------------------------------

def test_zoo_list_empty(tmp_path, monkeypatch):
    """Zoo.list() with no local models and no remote registry returns []."""
    from orca import zoo as zoo_mod
    monkeypatch.setattr(zoo_mod, "MODEL_DIR", str(tmp_path))
    # Bypass remote lookup so we never hit GitHub
    monkeypatch.setattr(zoo_mod.Zoo, "_list_remote",
                        staticmethod(lambda url: []))
    # cwd into a fresh dir so the hardcoded 'orca/checkpoint.pt' /
    # 'pretrained.pt' fallback scans an empty tree.
    monkeypatch.chdir(tmp_path)
    models = zoo_mod.Zoo.list(verbose=False)
    assert models == []


def test_zoo_list_local_with_sidecar(tmp_path, monkeypatch):
    """A local .pt + .json sidecar shows up under source='local'."""
    from orca import zoo as zoo_mod
    monkeypatch.setattr(zoo_mod, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(zoo_mod.Zoo, "_list_remote",
                        staticmethod(lambda url: []))
    monkeypatch.chdir(tmp_path)

    # Drop a stub model + metadata in the fake registry dir
    (tmp_path / "my-bot.pt").write_bytes(b"stub")
    (tmp_path / "my-bot.json").write_text(json.dumps({
        "name": "my-bot",
        "author": "tester",
        "elo": 1450,
        "params": 3909308,
    }))

    models = zoo_mod.Zoo.list(verbose=False)
    names = [m["name"] for m in models]
    assert "my-bot" in names
    entry = next(m for m in models if m["name"] == "my-bot")
    assert entry["source"] == "local"
    assert entry["elo"] == 1450


def test_zoo_package_writes_metadata(tmp_path):
    """Zoo.package() writes both the .pt with zoo_metadata and a sidecar .json."""
    from orca.zoo import Zoo

    # Build a tiny stub checkpoint
    src = tmp_path / "src.pt"
    torch.save({
        "model_state_dict": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "iteration": 7,
    }, str(src))

    dst = tmp_path / "packaged.pt"
    Zoo.package(
        str(src), str(dst),
        name="phoenix-v1", author="tester", elo=1450,
        description="unit test",
    )

    # Repackaged .pt has zoo_metadata
    loaded = torch.load(str(dst), weights_only=False)
    meta = loaded["zoo_metadata"]
    assert meta["name"] == "phoenix-v1"
    assert meta["author"] == "tester"
    assert meta["elo"] == 1450
    assert meta["params"] == 3  # 3-element tensor

    # Sidecar JSON has the same shape
    sidecar = json.loads((tmp_path / "packaged.json").read_text())
    assert sidecar["name"] == "phoenix-v1"
    assert sidecar["elo"] == 1450
