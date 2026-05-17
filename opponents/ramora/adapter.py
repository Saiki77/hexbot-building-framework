"""Adapter to play Orca against SealBot (Ramora's C++ engine).

Translates between our CGameState/MCTS interface and SealBot's HexGame
interface. Uses the compiled C++ minimax_cpp module when available for
much stronger play than the pure Python MinimaxBot.

Load priority for the underlying `game` module:

1. `opponents/sealbot/game.py` — preferred. Required if you want to use
   the C++ minimax_cpp module, which does `py::import("game")` and
   compares Player identity with `is`. Skipped if the sealbot directory
   is not checked out (it is gitignored).
2. `opponents/ramora/game.py` — fallback. Same Python interface, works
   with the pure-Python MinimaxBot. The C++ path is disabled in this
   mode (would silently misclassify stones because of the identity
   check), so we go straight to the Python MinimaxBot.

If neither module can be imported, `create_ramora_bot()` raises
`SealBotUnavailable`. Callers in the training pipeline catch this
and log a single warning before falling back to self-play-only mode
for the iteration.
"""

import importlib
import importlib.util
import os
import sys

_HERE = os.path.dirname(__file__)
_OPPONENTS_DIR = os.path.dirname(_HERE)
_sealbot_dir = os.path.join(_OPPONENTS_DIR, 'sealbot')
_sealbot_best = os.path.join(_sealbot_dir, 'best')
_sealbot_current = os.path.join(_sealbot_dir, 'current')


class SealBotUnavailable(ImportError):
    """Raised when neither sealbot nor ramora `game.py` can be loaded."""


def _load_game_module():
    """Return (game_module, source_label) or (None, reason).

    Tries sealbot/game.py first (preferred — enables the C++ path), then
    ramora/game.py (Python-only fallback).
    """
    # Preferred: sealbot's game.py (lets C++ minimax_cpp identity checks pass)
    if os.path.exists(os.path.join(_sealbot_dir, 'game.py')):
        if _sealbot_dir not in sys.path:
            sys.path.insert(0, _sealbot_dir)
        try:
            return importlib.import_module('game'), 'sealbot'
        except Exception:
            pass

    # Fallback: ramora's own game.py (Python MinimaxBot only)
    ramora_game = os.path.join(_HERE, 'game.py')
    if os.path.exists(ramora_game):
        # Load it by file path so we do not clash with any other 'game' module
        spec = importlib.util.spec_from_file_location(
            'opponents.ramora._game', ramora_game)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, 'ramora'

    return None, 'no game.py found in opponents/sealbot or opponents/ramora'


_seal_game, _game_source = _load_game_module()
if _seal_game is not None:
    HexGame = _seal_game.HexGame
    Player = _seal_game.Player
else:
    HexGame = None
    Player = None


def create_ramora_bot(time_limit: float = 1.0):
    """Create a SealBot opponent.

    Prefers the C++ engine when available, falls back to the pure-Python
    MinimaxBot when the C++ module is missing or when ramora's game.py
    is the only one available (identity check would misclassify stones
    against the C++ engine).

    Raises SealBotUnavailable if neither path can be used.
    """
    if _seal_game is None:
        raise SealBotUnavailable(_game_source)

    if _game_source == 'sealbot':
        try:
            if _sealbot_dir not in sys.path:
                sys.path.insert(0, _sealbot_dir)
            for d in [_sealbot_best, _sealbot_current]:
                if d not in sys.path:
                    sys.path.insert(0, d)
            from minimax_cpp import MinimaxBot as SealBot
            bot = SealBot(time_limit)
            print(f"  |  SealBot loaded: depth_limit={bot.max_depth}, "
                  f"time={time_limit}s")
            return bot
        except Exception:
            print("  |  WARNING: SealBot C++ not compiled, falling back "
                  "to Python MinimaxBot")
            print("  |  To compile: cd opponents/sealbot/current && "
                  "python setup.py build_ext --inplace")

    # Ramora-only mode OR sealbot C++ not compiled: Python MinimaxBot
    from opponents.ramora.ai import MinimaxBot
    return MinimaxBot(time_limit=time_limit)


def play_match(orca_search, orca_net, ramora_bot, orca_plays_first=True, max_moves=200):
    """Play a full game between Orca and Ramora.

    Args:
        orca_search: CMCTSSearch or BatchedMCTS instance
        orca_net: HexNet (for encoding)
        ramora_bot: MinimaxBot instance
        orca_plays_first: True = Orca is Player A (goes first)
        max_moves: max total stones before draw

    Returns:
        dict with keys: winner ('orca', 'ramora', 'draw'), moves, num_moves
    """
    if _seal_game is None:
        raise SealBotUnavailable(_game_source)

    from orca.encoding import CGameState

    ramora_game = HexGame()
    orca_game = CGameState(max_total_stones=max_moves)
    move_history = []
    orca_policies = []  # MCTS distributions for Orca's moves
    total_stones = 0

    while not ramora_game.game_over and total_stones < max_moves:
        is_orca_turn = (ramora_game.current_player == Player.A) == orca_plays_first

        if is_orca_turn:
            stones_to_play = ramora_game.moves_left_in_turn
            for _ in range(stones_to_play):
                if ramora_game.game_over or orca_game.is_terminal:
                    break
                # Mostly play best moves (temp=0.15), with light Dirichlet noise
                # for game-to-game variety. This plays strong moves 90%+ of the time
                # but occasionally picks second/third best, creating diverse games
                # without throwing games with random moves.
                policy = orca_search.search(orca_game, temperature=0.15, add_noise=True)
                if not policy:
                    break
                best_move = max(policy, key=policy.get)
                q, r = best_move

                ramora_game.make_move(q, r)
                orca_game.place_stone(q, r)
                move_history.append(('orca', q, r))
                orca_policies.append(policy)  # store full MCTS distribution
                total_stones += 1
        else:
            result = ramora_bot.get_move(ramora_game)
            if not result:
                break
            for m in result:
                if ramora_game.game_over:
                    break
                q, r = m
                ramora_game.make_move(q, r)
                orca_game.place_stone(q, r)
                move_history.append(('ramora', q, r))
                total_stones += 1

    if ramora_game.winner == Player.NONE:
        winner = 'draw'
    elif (ramora_game.winner == Player.A) == orca_plays_first:
        winner = 'orca'
    else:
        winner = 'ramora'

    return {
        'winner': winner,
        'moves': move_history,
        'num_moves': total_stones,
        'ramora_depth': ramora_bot.last_depth,
        'orca_policies': orca_policies,
    }


def evaluate_vs_ramora(orca_search, orca_net, n_games=10, time_limit=1.0):
    """Play n_games against Ramora and return win/loss/draw stats.

    Alternates colors: odd games Orca first, even games Ramora first.

    Returns dict with: wins, losses, draws, win_rate, games (list of results)
    """
    ramora = create_ramora_bot(time_limit=time_limit)
    results = []
    wins = losses = draws = 0

    for i in range(n_games):
        orca_first = (i % 2 == 0)
        result = play_match(orca_search, orca_net, ramora, orca_plays_first=orca_first)
        results.append(result)

        if result['winner'] == 'orca':
            wins += 1
        elif result['winner'] == 'ramora':
            losses += 1
        else:
            draws += 1

        symbol = 'W' if result['winner'] == 'orca' else ('L' if result['winner'] == 'ramora' else 'D')
        color = 'first' if orca_first else 'second'
        print(f"  Game {i+1}/{n_games}: {symbol} ({color}, {result['num_moves']} moves)")

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': round(win_rate, 3),
        'games': results,
    }
