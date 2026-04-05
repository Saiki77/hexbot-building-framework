"""Adapter to play Orca against SealBot (Ramora's C++ engine).

Translates between our CGameState/MCTS interface and SealBot's HexGame interface.
Uses the compiled C++ minimax_cpp module for much stronger play than the
pure Python MinimaxBot.
"""

import os
import sys

# Use SealBot's game.py (same interface as ramora's)
_sealbot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sealbot')
_sealbot_current = os.path.join(_sealbot_dir, 'current')

# Import game types from the local ramora copy (same interface as SealBot's)
from opponents.ramora.game import HexGame, Player


def create_ramora_bot(time_limit: float = 1.0):
    """Create SealBot C++ engine (much stronger than Python MinimaxBot).
    Falls back to Python MinimaxBot if C++ module not compiled."""
    try:
        # SealBot needs its own game.py importable as 'game'
        if _sealbot_dir not in sys.path:
            sys.path.insert(0, _sealbot_dir)
        if _sealbot_current not in sys.path:
            sys.path.insert(0, _sealbot_current)
        from minimax_cpp import MinimaxBot as SealBot
        bot = SealBot(time_limit)
        return bot
    except Exception:
        print("  |  WARNING: SealBot C++ not compiled, falling back to Python MinimaxBot")
        print("  |  To compile: cd opponents/sealbot/current && python setup.py build_ext --inplace")
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
    from orca.encoding import CGameState

    ramora_game = HexGame()
    orca_game = CGameState(max_total_stones=max_moves)
    move_history = []
    total_stones = 0

    while not ramora_game.game_over and total_stones < max_moves:
        is_orca_turn = (ramora_game.current_player == Player.A) == orca_plays_first

        if is_orca_turn:
            # Orca plays one stone at a time
            stones_to_play = ramora_game.moves_left_in_turn
            for _ in range(stones_to_play):
                if ramora_game.game_over or orca_game.is_terminal:
                    break
                policy = orca_search.search(orca_game, temperature=0.1, add_noise=False)
                if not policy:
                    break
                # Pick best move
                best_move = max(policy, key=policy.get)
                q, r = best_move

                ramora_game.make_move(q, r)
                orca_game.place_stone(q, r)
                move_history.append(('orca', q, r))
                total_stones += 1
        else:
            # Ramora plays (returns pair of moves)
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

    # Determine winner
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
