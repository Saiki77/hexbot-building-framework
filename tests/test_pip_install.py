"""Sanity tests that the pip-installed hexbot package exposes its public API.

These run as part of `pytest tests/` and also catch wheel-vs-source drift
(e.g. a function declared in `hexbot.py` but missing from the packaged
wheel). Each test function is auto-discovered by pytest.
"""

import os
import sys

# Add parent directory so this works both from pip install AND from the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Core imports
def test_core_imports():
    from hexbot import HexGame, Bot, Arena, train
    assert HexGame and Bot and Arena and train


def test_analysis_imports():
    from hexbot import (evaluate_moves, find_threats, find_winning_moves,
                        count_lines, rollout, alphabeta)


def test_v3_imports():
    from hexbot import (nn_evaluate, mcts_policy, create_network, encode_state,
                        FastGame, find_forced_move, self_play, augment_sample)


def test_v4_imports():
    from hexbot import (solve, quick_solve, opening_move, OpeningBook,
                        Ensemble, import_games, register_bot, Zoo)


def test_orca_imports():
    from orca import Orca, __version__
    assert Orca is not None
    assert __version__  # any non-empty version string


# Game engine
def test_hexgame_basic():
    from hexbot import HexGame
    g = HexGame()
    g.place(0, 0)
    assert g.total_stones == 1
    assert g.current_player == 1
    g.place(1, 0)
    g.place(1, -1)
    assert g.total_stones == 3
    g.undo()
    assert g.total_stones == 2


def test_hexgame_win():
    from hexbot import HexGame
    g = HexGame()
    # P0 plays on horizontal axis, P1 plays far away
    # Turn structure: P0 plays 1, then alternating 2 each
    g.place(0, 0)                  # P0 move 1 (1 stone)
    g.place(0, 5); g.place(0, 6)   # P1 turn
    g.place(1, 0); g.place(2, 0)   # P0 turn
    g.place(1, 5); g.place(1, 6)   # P1 turn
    g.place(3, 0); g.place(4, 0)   # P0 turn
    g.place(2, 5); g.place(2, 6)   # P1 turn
    g.place(5, 0)                  # P0 wins: 0,0 through 5,0
    assert g.is_over
    assert g.winner == 0


def test_hexgame_clone():
    from hexbot import HexGame
    g = HexGame()
    g.place(0, 0)
    c = g.clone()
    c.place(1, 0)
    assert g.total_stones == 1
    assert c.total_stones == 2


def test_hexgame_search():
    from hexbot import HexGame
    g = HexGame()
    g.place(0, 0)
    r = g.search(depth=4)
    assert 'best_move' in r
    assert 'value' in r
    assert 'nodes' in r


# Bot
def test_bot_heuristic():
    from hexbot import Bot, HexGame
    bot = Bot.heuristic()
    g = HexGame()
    g.place(0, 0)
    move = bot.best_move(g)
    assert isinstance(move, tuple) and len(move) == 2


def test_bot_random():
    from hexbot import Bot, HexGame
    bot = Bot.random()
    g = HexGame()
    g.place(0, 0)
    move = bot.best_move(g)
    assert isinstance(move, tuple)


def test_arena():
    from hexbot import Bot, Arena
    r = Arena(Bot.heuristic(), Bot.random(), num_games=3).play(verbose=False)
    assert r.total_games == 3
    assert r.wins[0] + r.wins[1] + r.draws == 3


# Analysis
def test_evaluate_moves():
    from hexbot import HexGame, evaluate_moves
    g = HexGame()
    g.place(0, 0)
    g.place(1, 0)
    top = evaluate_moves(g, top_n=5)
    assert len(top) > 0


def test_rollout():
    from hexbot import HexGame, rollout
    g = HexGame()
    g.place(0, 0)
    r = rollout(g, num_games=10)
    assert 'p0_wins' in r


# Networks
def test_create_networks():
    from hexbot import create_network
    import torch
    dummy = torch.randn(1, 7, 19, 19)
    for cfg in ['fast', 'standard', 'hex-masked']:
        net = create_network(cfg)
        p, v, t = net(dummy)
        assert p.shape == (1, 361)
        assert v.shape == (1, 1)
        assert t.shape == (1, 4)


def test_encode_state():
    from hexbot import HexGame, encode_state
    g = HexGame()
    g.place(0, 0)
    t, oq, orr = encode_state(g)
    assert t.shape[0] == 7
    assert t.shape[1] == 19
    assert t.shape[2] == 19


# Orca modules
def test_solver():
    from hexbot import HexGame, solve
    g = HexGame()
    g.place(0, 0)
    r = solve(g, max_depth=4)
    assert 'result' in r
    assert 'move' in r


def test_opening_book():
    from orca.openings import OpeningBook, build_default_book
    book = build_default_book()
    assert len(book) > 0


def test_curriculum():
    from orca.curriculum import SkillCurriculum
    c = SkillCurriculum()
    cfg = c.get_config()
    assert cfg['level'] == 1
    assert cfg['sims'] == 30


def test_config():
    from orca.config import BATCH_SIZE, NUM_FILTERS, BOARD_SIZE
    assert isinstance(BATCH_SIZE, int) and BATCH_SIZE > 0
    assert NUM_FILTERS == 128
    assert BOARD_SIZE == 19


def test_sft_parser():
    import json, tempfile
    from orca.sft import import_games
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                     delete=False) as f:
        moves = [[i, 0] for i in range(8)]  # 8 moves, above min_moves=6
        f.write(json.dumps({"moves": moves, "result": 1.0}) + "\n")
        f.write(json.dumps({"moves": moves, "result": -1.0}) + "\n")
        path = f.name
    games = import_games(path)
    os.unlink(path)
    assert len(games) == 2


def test_augment():
    from hexbot import augment_sample
    from orca.data import TrainingSample
    import torch, numpy as np
    s = TrainingSample(
        encoded_state=torch.randn(7, 19, 19),
        policy_target=np.random.dirichlet(np.ones(361)),
        player=0, result=1.0,
    )
    augs = augment_sample(s)
    # 3 grid-safe transforms + up to 4 axial rotations
    assert len(augs) >= 3


def test_plugin_system():
    from hexbot import register_bot, registered_bots
    class TestBot:
        def best_move(self, game):
            return (0, 0)
    register_bot('test-pip', TestBot)
    assert 'test-pip' in registered_bots()


# Profiles + scaffolder + observability (v4.2.x)
def test_profiles_listing():
    from orca.config import PROFILES, get_profile, list_profiles
    assert 'mps-laptop' in list_profiles()
    assert 'cuda-single' in list_profiles()
    p = get_profile('cpu-only')
    assert p is not None and 'batch_size' in p


def test_init_scaffolder():
    import tempfile
    from orca.init import scaffold
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, 'demo-bot')
        rc = scaffold(target, profile='cpu-only')
        assert rc == 0
        assert os.path.exists(os.path.join(target, 'README.md'))
        assert os.path.exists(os.path.join(target, 'train.sh'))
        assert os.path.exists(os.path.join(target, 'plugins.py'))


def test_checkpoint_metadata():
    from orca.train import _make_checkpoint_meta
    meta = _make_checkpoint_meta(arch='standard', iteration=42, elo=1234.5)
    assert meta['schema_version'] == 1
    assert meta['arch'] == 'standard'
    assert meta['iter'] == 42
    assert meta['elo'] == 1234.5
    assert 'git_sha' in meta and 'hexbot_version' in meta


def test_atomic_save_roundtrip():
    import tempfile, torch
    from orca.train import _atomic_torch_save, _make_checkpoint_meta
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        obj = {'state': torch.tensor([1.0]),
               '_hexbot_meta': _make_checkpoint_meta('standard', 1, 1000.0)}
        _atomic_torch_save(obj, path)
        loaded = torch.load(path, weights_only=False)
        assert '_hexbot_meta' in loaded
        assert not os.path.exists(path + '.tmp')
    finally:
        if os.path.exists(path):
            os.unlink(path)
