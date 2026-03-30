# Bot Approaches

Six complete strategies for building hexbot bots, from the simplest hand-tuned evaluation to neural network training. Each section is self-contained with a full working code example you can copy and run.

See also: [Getting Started](getting-started.md) | [API Reference](api-reference.md) | [Dashboard Guide](dashboard-guide.md)

---

## Building Your First Bot

The simplest bot is a function that takes a `HexGame` and returns a move `(q, r)`:

```python
from hexbot import HexGame

def my_first_bot(game):
    """Pick the first legal move."""
    return game.legal_moves()[0]
```

This bot is terrible -- it always picks the same move without any strategy. Let's improve it step by step.

### Step 1: Use Heuristic Scoring

The C engine scores every legal move based on how much it extends lines and blocks the opponent:

```python
from hexbot import evaluate_moves

def better_bot(game):
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

This bot already beats random play consistently because it prefers moves that build longer lines.

### Step 2: Add Win/Block Detection

Always take a winning move if available, and always block the opponent's winning move:

```python
from hexbot import evaluate_moves, find_winning_moves

def good_bot(game):
    # Take the win
    wins = find_winning_moves(game, game.current_player)
    if wins:
        return wins[0]

    # Block opponent's win
    blocks = find_winning_moves(game, 1 - game.current_player)
    if blocks:
        return blocks[0]

    # Best heuristic move
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

### Step 3: Add Threat Awareness

Look for moves that create lines of 4 or more (threats that must be answered):

```python
from hexbot import evaluate_moves, find_winning_moves, find_threats

def strong_bot(game):
    player = game.current_player
    opponent = 1 - player

    wins = find_winning_moves(game, player)
    if wins: return wins[0]

    blocks = find_winning_moves(game, opponent)
    if blocks: return blocks[0]

    # Prefer moves that create threats
    threats = find_threats(game, player)
    if threats: return threats[0]

    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]
```

### Step 4: Test Against Built-in Bots

```python
from hexbot import Arena, Bot

# Test against random
result = Arena(strong_bot, Bot.random(), num_games=50).play()
print(f"vs Random: {result.wins[0]}W-{result.wins[1]}L")

# Test against heuristic
result = Arena(strong_bot, Bot.heuristic(), num_games=50).play()
print(f"vs Heuristic: {result.wins[0]}W-{result.wins[1]}L")
```

Any function that takes a game and returns a `(q, r)` move works as a bot. You can also create a class with a `best_move(game)` method.

---

## Approach 1: Hand-Tuned Evaluation

The simplest approach -- score each candidate move using domain knowledge and pick the best one. No machine learning needed. Fast to iterate and easy to understand.

```python
from hexbot import (HexGame, Arena, Bot, evaluate_moves,
                     find_threats, find_winning_moves)

def smart_bot(game):
    player = game.current_player
    opponent = 1 - player

    # Priority 1: Win immediately
    wins = find_winning_moves(game, player)
    if wins:
        return wins[0]

    # Priority 2: Block opponent's winning move
    blocks = find_winning_moves(game, opponent)
    if blocks:
        return blocks[0]

    # Priority 3: Extend our longest threats
    threats = find_threats(game, player)
    if threats:
        return threats[0]

    # Priority 4: Block opponent's threats
    opp_threats = find_threats(game, opponent)
    if opp_threats:
        return opp_threats[0]

    # Priority 5: Best positional move
    moves = evaluate_moves(game, top_n=1)
    return moves[0][0] if moves else game.legal_moves()[0]

# Test it
result = Arena(smart_bot, Bot.heuristic(), num_games=50).play()
```

**Making it stronger:** add preemptive awareness (2-in-a-row setups), colony detection (distant stone groups), and formation recognition (triangles, rhombuses).

---

## Approach 2: Evolutionary Weights

Instead of hand-tuning priorities, let evolution discover them. Create a population of bots with random scoring weights, play tournaments, and breed the winners.

```python
import random
from hexbot import Arena, evaluate_moves, find_winning_moves

class EvoBot:
    def __init__(self, weights=None):
        self.weights = weights or {
            'line_score': random.uniform(0.5, 2.0),
            'center_pull': random.uniform(0.0, 1.0),
            'threat_weight': random.uniform(1.0, 5.0),
            'block_weight': random.uniform(1.0, 5.0),
        }

    def best_move(self, game):
        # Always take wins/blocks
        for p in [game.current_player, 1 - game.current_player]:
            wins = find_winning_moves(game, p)
            if wins: return wins[0]

        # Score moves with evolved weights
        best, best_score = game.legal_moves()[0], -999
        for (q, r), base in evaluate_moves(game, 15):
            score = base * self.weights['line_score']
            score -= (abs(q) + abs(r)) * self.weights['center_pull']

            my_line = game.max_line(q, r, game.current_player)
            opp_line = game.max_line(q, r, 1 - game.current_player)
            if my_line >= 4: score += my_line * self.weights['threat_weight']
            if opp_line >= 4: score += opp_line * self.weights['block_weight']

            if score > best_score:
                best_score, best = score, (q, r)
        return best

    def mutate(self):
        child = EvoBot(dict(self.weights))
        for k in child.weights:
            if random.random() < 0.3:
                child.weights[k] *= random.uniform(0.7, 1.3)
        return child

# Evolution loop
POP = 8
population = [EvoBot() for _ in range(POP)]

for gen in range(10):
    scores = [0] * POP
    for i in range(POP):
        for j in range(i+1, POP):
            r = Arena(population[i], population[j], num_games=6).play(verbose=False)
            scores[i] += r.wins[0]
            scores[j] += r.wins[1]

    ranked = sorted(range(POP), key=lambda i: scores[i], reverse=True)
    survivors = [population[i] for i in ranked[:POP//2]]
    population = survivors + [s.mutate() for s in survivors]

    best = survivors[0]
    print(f"Gen {gen+1}: score={scores[ranked[0]]}, weights={best.weights}")
```

After 10 generations the winning weights typically converge to: high threat weight (4+), moderate line score (1.0-1.5), low center pull (0.1-0.3), and high block weight (3+). This matches human intuition -- threats and blocks matter most.

---

## Approach 3: Minimax with Alpha-Beta

Use depth-first search with the built-in C engine for deep tactical analysis. The C engine includes transposition tables, killer move heuristics, and late move reduction.

```python
from hexbot import HexGame, alphabeta, Arena, Bot

def minimax_bot(game):
    result = alphabeta(game, depth=6)  # 3 full turns ahead
    return result['best_move']

# Alpha-beta returns:
# result['best_move']  - (q, r) best move
# result['value']      - evaluation (-1 to +1, from P0's perspective)
# result['nodes']      - nodes searched

result = Arena(minimax_bot, Bot.heuristic(), num_games=20).play()
```

### Custom Evaluation Function

For a minimax with your own evaluation function:

```python
from hexbot import find_threats

def my_eval(game):
    """Custom position evaluation. Returns float (-1 to +1)."""
    p = game.current_player
    my_threats = len(find_threats(game, p))
    opp_threats = len(find_threats(game, 1 - p))
    return (my_threats - opp_threats) * 0.2

def my_minimax(game, depth):
    if depth == 0 or game.is_over:
        if game.winner == 0: return 1.0, None
        if game.winner == 1: return -1.0, None
        return my_eval(game), None

    best_val = -2.0
    best_move = None
    for q, r, _ in game.scored_moves(12):  # top 12 candidates
        game.place(q, r)
        val, _ = my_minimax(game, depth - 1)
        val = -val  # negamax
        game.undo()
        if val > best_val:
            best_val, best_move = val, (q, r)
    return best_val, best_move
```

The `game.place()` / `game.undo()` pattern lets you explore the game tree without copying the board. This is the same pattern the C engine uses internally (973K place/undo cycles per second).

### Search Depth Guidelines

| Depth | Turns Ahead | Typical Nodes | Typical Time |
|-------|-------------|---------------|--------------|
| 4     | 2           | ~700          | 0.04s        |
| 6     | 3           | ~18K          | 0.11s        |
| 8     | 4           | ~250K         | 0.96s        |
| 10    | 5           | ~950K         | 8.3s         |

---

## Approach 4: Monte Carlo Random Playouts

Estimate move quality by playing random games from each candidate position. The C engine's fast random playout (33K stones/sec) makes this practical.

```python
from hexbot import HexGame, rollout, evaluate_moves, Arena, Bot

def monte_carlo_bot(game):
    best_move = game.legal_moves()[0]
    best_rate = -1

    for (q, r), _ in evaluate_moves(game, 8):  # top 8 candidates
        game.place(q, r)
        result = rollout(game, num_games=100)
        game.undo()

        # Win rate for current player
        p = game.current_player
        rate = result['p0_wins'] if p == 0 else result['p1_wins']
        if rate > best_rate:
            best_rate, best_move = rate, (q, r)

    return best_move

result = Arena(monte_carlo_bot, Bot.heuristic(), num_games=20).play()
```

The strength of this bot scales with the number of rollouts per move. 100 rollouts gives a rough estimate; 1000 gives reliable evaluations. The bottleneck is that random playouts do not understand strategy -- they explore randomly. To improve, bias the rollouts toward better moves using `scored_moves()`.

---

## Approach 5: Neural Network (AlphaZero-Style)

Train a neural network to evaluate positions and guide Monte Carlo Tree Search. This is the most powerful approach but requires PyTorch and compute time.

### Quick Training

```python
from hexbot import Bot, train, Arena

# Train from scratch
bot = train(
    iterations=50,         # training cycles
    games_per_iter=20,     # self-play games per cycle
    sims=50,               # MCTS simulations per move
    network_config='fast', # small network, quick training
)
bot.save('my_nn_bot.pt')

# Test
result = Arena(bot, Bot.heuristic(), num_games=30).play()
```

The network architecture is a ResNet with separate policy (where to play), value (who's winning), and threat (tactical awareness) heads. Training works by having the network play against itself, generating training data from MCTS visit counts, then training on that data.

### Longer Training with a Stronger Network

```python
bot = train(
    iterations=200,
    games_per_iter=50,
    sims=200,
    network_config='standard',  # 3.9M params, 128 filters, 12 res blocks
    lr=0.001,
    checkpoint_every=20,
)
```

### Network Configurations

| Config   | Parameters | Sims | Speed           | Use Case         |
|----------|-----------|------|-----------------|------------------|
| fast     | 700K      | 50   | ~3 games/sec    | Quick prototyping |
| standard | 3.9M      | 50   | ~0.3 games/sec  | Good quality     |
| standard | 3.9M      | 200  | ~0.1 games/sec  | Full quality     |
| large    | 14.5M     | 200  | slower          | Maximum strength |

### Loading a Pre-trained Bot

```python
from hexbot import Bot, Arena

bot = Bot.load('pretrained.pt')
print(bot)  # Bot(mcts, sims=200, 3,909,308 params)

result = Arena(bot, Bot.heuristic(), num_games=20).play()
```

### Custom Network Architecture

The `Bot` class accepts any PyTorch `nn.Module` that takes a `(batch, 7, 19, 19)` tensor and returns `(policy_logits, value, threat_logits)`. See `bot.py` for the default `HexNet` architecture.

---

## Approach 6: Combine Multiple Strategies

The strongest bots often combine approaches. Use fast tactical checks for obvious moves, deep search for complex positions, and heuristics as a fallback.

```python
from hexbot import (evaluate_moves, find_winning_moves, find_threats,
                     alphabeta, Bot, Arena)

def hybrid_bot(game):
    player = game.current_player

    # Layer 1: Instant tactical responses (microseconds)
    wins = find_winning_moves(game, player)
    if wins: return wins[0]
    blocks = find_winning_moves(game, 1 - player)
    if blocks: return blocks[0]

    # Layer 2: Deep search for critical positions (milliseconds)
    result = alphabeta(game, depth=4)
    if abs(result['value']) > 0.8:
        return result['best_move']

    # Layer 3: Threat-based play (microseconds)
    threats = find_threats(game, player)
    if threats: return threats[0]

    # Layer 4: Positional heuristic (microseconds)
    moves = evaluate_moves(game, 1)
    return moves[0][0] if moves else game.legal_moves()[0]

result = Arena(hybrid_bot, Bot.heuristic(), num_games=50).play()
```

### Why Hybrid Works Best

For this game, the recommended combination is:

- **C engine alpha-beta** for tactical situations (forced wins/blocks within 4 turns)
- **MCTS with a neural network** for strategic evaluation
- **Heuristic scoring** as a fast fallback

The neural network + MCTS approach (AlphaZero-style) is the most powerful long-term but requires significant training time. The hybrid approach gives strong play immediately while the network trains.

---

## Testing Any Bot

Every bot can be tested against built-in opponents using the Arena:

```python
from hexbot import Arena, Bot

# Against random
result = Arena(my_bot, Bot.random(), num_games=100).play()
print(f"vs Random: {result.wins[0]}W-{result.wins[1]}L")

# Against heuristic
result = Arena(my_bot, Bot.heuristic(), num_games=100).play()
print(f"vs Heuristic: {result.wins[0]}W-{result.wins[1]}L")

# Against a trained neural network
nn_bot = Bot.load('pretrained.pt')
result = Arena(my_bot, nn_bot, num_games=50).play()
print(f"vs NN: {result.wins[0]}W-{result.wins[1]}L")
```

See the [Dashboard Guide](dashboard-guide.md) for visualizing arena matches and tracking ELO over time.

---

## Next Steps

- [Getting Started](getting-started.md) -- HexGame API fundamentals
- [API Reference](api-reference.md) -- full method signatures
- [Dashboard Guide](dashboard-guide.md) -- live training visualization
