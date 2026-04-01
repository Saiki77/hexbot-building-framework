#!/usr/bin/env python3
"""
Test that the pip-installed hexbot package works correctly.

Run after: pip install hexbot
    python tests/test_pip_install.py

Tests all major features to verify the package is complete.
"""

import os
import sys

# Add parent directory so this works both from pip install AND from the repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        errors.append((name, traceback.format_exc()))
        failed += 1


# ── Core imports ──
def test_core_imports():
    from hexbot import HexGame, Bot, Arena, train
    assert HexGame and Bot and Arena and train

test("Core imports (HexGame, Bot, Arena, train)", test_core_imports)


def test_analysis_imports():
    from hexbot import (evaluate_moves, find_threats, find_winning_moves,
                        count_lines, rollout, alphabeta)

test("Analysis function imports", test_analysis_imports)


def test_v3_imports():
    from hexbot import (nn_evaluate, mcts_policy, create_network, encode_state,
                        FastGame, find_forced_move, self_play, augment_sample)

test("v3 API imports (nn, mcts, encode)", test_v3_imports)


def test_v4_imports():
    from hexbot import (solve, quick_solve, opening_move, OpeningBook,
                        Ensemble, import_games, register_bot, Zoo)

test("v4 API imports (solver, openings, ensemble, zoo)", test_v4_imports)


def test_orca_imports():
    from orca import Orca, __version__
    assert __version__ == '4.0.0'

test("Orca package import + version", test_orca_imports)


# ── Game engine ──
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

test("HexGame: place, undo, properties", test_hexgame_basic)


def test_hexgame_win():
    from hexbot import HexGame
    g = HexGame()
    # P0 plays on horizontal axis, P1 plays far away
    # Turn structure: P0 plays 1, then alternating 2 each
    g.place(0, 0)   # P0 move 1 (1 stone)
    g.place(0, 5); g.place(0, 6)  # P1 turn
    g.place(1, 0); g.place(2, 0)  # P0 turn
    g.place(1, 5); g.place(1, 6)  # P1 turn
    g.place(3, 0); g.place(4, 0)  # P0 turn
    g.place(2, 5); g.place(2, 6)  # P1 turn
    g.place(5, 0)                  # P0 wins: 0,0 through 5,0
    assert g.is_over
    assert g.winner == 0

test("HexGame: win detection (6 in a row)", test_hexgame_win)


def test_hexgame_clone():
    from hexbot import HexGame
    g = HexGame()
    g.place(0, 0)
    c = g.clone()
    c.place(1, 0)
    assert g.total_stones == 1
    assert c.total_stones == 2

test("HexGame: clone independence", test_hexgame_clone)


def test_hexgame_search():
    from hexbot import HexGame
    g = HexGame()
    g.place(0, 0)
    r = g.search(depth=4)
    assert 'best_move' in r
    assert 'value' in r
    assert 'nodes' in r

test("HexGame: alpha-beta search", test_hexgame_search)


# ── Bot ──
def test_bot_heuristic():
    from hexbot import Bot, HexGame
    bot = Bot.heuristic()
    g = HexGame()
    g.place(0, 0)
    move = bot.best_move(g)
    assert isinstance(move, tuple) and len(move) == 2

test("Bot.heuristic() best_move", test_bot_heuristic)


def test_bot_random():
    from hexbot import Bot, HexGame
    bot = Bot.random()
    g = HexGame()
    g.place(0, 0)
    move = bot.best_move(g)
    assert isinstance(move, tuple)

test("Bot.random() best_move", test_bot_random)


def test_arena():
    from hexbot import Bot, Arena
    r = Arena(Bot.heuristic(), Bot.random(), num_games=3).play(verbose=False)
    assert r.total_games == 3
    assert r.wins[0] + r.wins[1] + r.draws == 3

test("Arena: 3 games heuristic vs random", test_arena)


# ── Analysis ──
def test_evaluate_moves():
    from hexbot import HexGame, evaluate_moves
    g = HexGame()
    g.place(0, 0)
    g.place(1, 0)
    top = evaluate_moves(g, top_n=5)
    assert len(top) > 0

test("evaluate_moves", test_evaluate_moves)


def test_rollout():
    from hexbot import HexGame, rollout
    g = HexGame()
    g.place(0, 0)
    r = rollout(g, num_games=10)
    assert 'p0_wins' in r

test("rollout (10 games)", test_rollout)


# ── Networks ──
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

test("create_network: fast, standard, hex-masked", test_create_networks)


def test_encode_state():
    from hexbot import HexGame, encode_state
    g = HexGame()
    g.place(0, 0)
    t, oq, orr = encode_state(g)
    assert t.shape[0] == 7
    assert t.shape[1] == 19
    assert t.shape[2] == 19

test("encode_state", test_encode_state)


# ── Orca modules ──
def test_solver():
    from hexbot import HexGame, solve
    g = HexGame()
    g.place(0, 0)
    r = solve(g, max_depth=4)
    assert 'result' in r
    assert 'move' in r

test("Endgame solver", test_solver)


def test_opening_book():
    from orca.openings import OpeningBook, build_default_book
    book = build_default_book()
    assert len(book) > 0

test("Opening book (default)", test_opening_book)


def test_curriculum():
    from orca.curriculum import SkillCurriculum
    c = SkillCurriculum()
    cfg = c.get_config()
    assert cfg['level'] == 1
    assert cfg['sims'] == 30

test("Skill curriculum", test_curriculum)


def test_config():
    from orca.config import BATCH_SIZE, NUM_FILTERS, BOARD_SIZE
    assert BATCH_SIZE == 1024
    assert NUM_FILTERS == 128
    assert BOARD_SIZE == 19

test("orca.config values", test_config)


def test_sft_parser():
    import json, tempfile, os
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

test("SFT game parser (JSONL)", test_sft_parser)


def test_augment():
    from hexbot import augment_sample, create_network
    from orca.data import TrainingSample
    import torch, numpy as np
    s = TrainingSample(
        encoded_state=torch.randn(7, 19, 19),
        policy_target=np.random.dirichlet(np.ones(361)),
        player=0, result=1.0,
    )
    augs = augment_sample(s)
    assert len(augs) == 3

test("Hex augmentation (3 transforms)", test_augment)


def test_plugin_system():
    from hexbot import register_bot, registered_bots
    class TestBot:
        def best_move(self, game):
            return (0, 0)
    register_bot('test-pip', TestBot)
    assert 'test-pip' in registered_bots()

test("Plugin system (register_bot)", test_plugin_system)


# ── Summary ──
print()
print("=" * 50)
print(f"  {passed + failed} tests: {passed} passed, {failed} failed")
print("=" * 50)

if errors:
    print("\nFailed tests:")
    for name, tb in errors:
        print(f"\n--- {name} ---")
        print(tb)

sys.exit(0 if failed == 0 else 1)
