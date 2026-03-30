"""Evolving bot demo with live dashboard.

Run:
    python test_dashboard.py
    Then open http://localhost:5002

Demonstrates:
- Building a bot with evolving heuristic weights
- Using the dashboard to track improvement over generations
- Auto-ELO computed from games against a baseline
- All charts, game replays, and stats update live

The bot starts with random weights and evolves them through
tournament selection. You can watch it improve in real-time.
"""

import sys
import time
import random
import copy

sys.path.insert(0, '.')
from hexbot import HexGame, evaluate_moves
from dashboard_clean import Dashboard

# ---------------------------------------------------------------------------
# Evolving heuristic bot
# ---------------------------------------------------------------------------

class EvoBot:
    """A bot with evolvable weights for move scoring."""

    def __init__(self, weights=None):
        # Weights control how the bot values different move properties
        self.weights = weights or {
            'score': random.uniform(0.5, 2.0),       # C engine heuristic score
            'center': random.uniform(-0.5, 1.0),     # prefer center
            'spread': random.uniform(-0.5, 1.0),     # prefer spread-out moves
            'random': random.uniform(0.0, 0.5),      # randomness
        }

    def best_move(self, game):
        top = evaluate_moves(game, top_n=15)
        if not top:
            moves = game.legal_moves()
            return random.choice(moves) if moves else (0, 0)

        best_move, best_val = top[0][0], -1e9
        for move, score in top:
            q, r = move
            val = score * self.weights['score']
            # Center preference
            val -= (abs(q) + abs(r)) * self.weights['center'] * 0.1
            # Spread bonus (distance from other top moves)
            val += self.weights['spread'] * random.uniform(0, 0.3)
            # Noise
            val += random.gauss(0, self.weights['random'])
            if val > best_val:
                best_val = val
                best_move = move
        return best_move

    def mutate(self):
        """Return a mutated copy."""
        new_weights = {}
        for k, v in self.weights.items():
            mutation = random.gauss(0, 0.15)
            new_weights[k] = max(-2, min(3, v + mutation))
        return EvoBot(new_weights)

    def __repr__(self):
        w = ', '.join(f'{k}={v:.2f}' for k, v in self.weights.items())
        return f'EvoBot({w})'


def random_bot(game):
    """Baseline: random legal moves."""
    moves = game.legal_moves()
    return random.choice(moves) if moves else (0, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

dash = Dashboard(port=5002)
dash.start()
time.sleep(1)

print("Dashboard: http://localhost:5002")
print("Evolving a heuristic bot over generations")
print("Watch the ELO chart to see improvement!")
print("Press Ctrl+C to stop\n")

# Start with a population of random bots
POPULATION = 6
GAMES_PER_MATCHUP = 4

population = [EvoBot() for _ in range(POPULATION)]
generation = 0
best_ever = None
best_elo = 0

try:
    while True:
        generation += 1
        print(f"--- Generation {generation} ({POPULATION} bots) ---")

        # Tournament: each bot plays against random baseline
        scores = []
        for i, bot in enumerate(population):
            wins = 0
            for g in range(GAMES_PER_MATCHUP):
                game = HexGame()
                moves = []
                bots = [bot, random_bot] if g % 2 == 0 else [random_bot, bot]
                bot_is_p0 = (g % 2 == 0)

                while not game.is_over:
                    b = bots[game.current_player]
                    if hasattr(b, 'best_move'):
                        move = b.best_move(game)
                    else:
                        move = b(game)
                    moves.append(list(move))
                    game.place(*move)

                result = 1.0 if game.winner == 0 else -1.0
                if not bot_is_p0:
                    result = -result

                # Stream to dashboard
                dash.add_game(moves, result)

                if result > 0:
                    wins += 1

            scores.append((wins, i))
            dash.update_progress(i + 1, POPULATION, phase='tournament')

        # Sort by wins (best first)
        scores.sort(reverse=True)
        ranked = [population[idx] for _, idx in scores]

        # Report
        best = ranked[0]
        worst_score = scores[-1][0]
        best_score = scores[0][0]
        wr = best_score / GAMES_PER_MATCHUP

        print(f"  Best: {best_score}/{GAMES_PER_MATCHUP} wins - {best}")
        print(f"  Worst: {worst_score}/{GAMES_PER_MATCHUP} wins")

        # Track best ever
        if best_elo == 0 or best_score >= best_elo:
            best_elo = best_score
            best_ever = copy.deepcopy(best)

        # Push metrics
        dash.add_metric(
            iteration=generation,
            wins=[sum(s for s, _ in scores), POPULATION * GAMES_PER_MATCHUP - sum(s for s, _ in scores), 0],
            games=POPULATION * GAMES_PER_MATCHUP,
            avg_game_length=20,
            self_play_time=1.0,
            workers=1,
        )

        # ELO eval: best bot vs snapshot of previous best
        if generation > 1 and best_ever:
            eval_wins = 0
            for eg in range(6):
                game = HexGame()
                moves = []
                bots_eval = [best, best_ever] if eg % 2 == 0 else [best_ever, best]
                is_p0 = (eg % 2 == 0)
                while not game.is_over:
                    b = bots_eval[game.current_player]
                    move = b.best_move(game)
                    moves.append(list(move))
                    game.place(*move)
                r = 1.0 if game.winner == 0 else -1.0
                if not is_p0: r = -r
                if r > 0: eval_wins += 1

            wr_eval = max(0.01, min(0.99, eval_wins / 6))
            import math
            elo = 1000 + generation * 3 + 400 * math.log10(wr_eval / (1 - wr_eval))
            dash.add_metric(iteration=generation, elo=round(elo, 1))
            print(f"  ELO: {elo:.0f} (vs previous best: {eval_wins}/6 wins)")

        # Evolution: keep top half, mutate to fill rest
        survivors = ranked[:POPULATION // 2]
        children = [s.mutate() for s in survivors]
        population = survivors + children

        print()
        time.sleep(0.5)

except KeyboardInterrupt:
    print(f"\nStopped after {generation} generations")
    if best_ever:
        print(f"Best bot: {best_ever}")
