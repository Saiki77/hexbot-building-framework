# Getting Started with hexbot

A hands-on guide to the HexGame API -- placing stones, reading the board, searching ahead, and understanding the coordinate system. By the end you will be comfortable building any bot on top of this engine.

See also: [Bot Approaches](bot-approaches.md) | [API Reference](api-reference.md) | [Dashboard Guide](dashboard-guide.md)

---

## The Coordinate System

The board uses **axial coordinates** `(q, r)` on an infinite hexagonal grid.

| Axis | Direction |
|------|-----------|
| `q`  | Horizontal |
| `r`  | Diagonal |
| `s = -q - r` | Implicit third axis (cube coordinates) |

Every hex has six neighbors. The three win-line directions are `(1,0)`, `(0,1)`, and `(1,-1)`.

---

## Creating a Game

```python
from hexbot import HexGame

game = HexGame()                          # empty board
game = HexGame(max_stones=300)            # custom stone limit
game = HexGame.from_moves([(0,0), (1,0), (1,-1)])  # replay a sequence
game = HexGame.triangle()                 # pre-built triangle opening
```

---

## Turn Structure

Player 0 goes first and places **1 stone** on their opening turn. After that every turn is **2 stones**.

```python
from hexbot import HexGame

game = HexGame()

# Turn 0: Player 0 places 1 stone
game.place(0, 0)
print(game.current_player)    # 1 (turn switches after 1 stone)

# Turn 1: Player 1 places 2 stones
game.place(2, 0)
print(game.current_player)    # 1 (still P1, needs one more)
game.place(2, -1)
print(game.current_player)    # 0 (P1 done, back to P0)

# Turn 2: Player 0 places 2 stones
game.place(1, 0)
game.place(0, 1)
print(game.current_player)    # 1
print(game.total_stones)      # 5
```

The properties `stones_this_turn` and `stones_per_turn` tell you where you are within the current turn. This matters for bots because the two stones in a turn should work together -- the first stone sets up the second.

```python
print(game.stones_this_turn)  # 0 (haven't placed yet this turn)
print(game.stones_per_turn)   # 2 (need to place 2)
game.place(3, 0)
print(game.stones_this_turn)  # 1 (placed 1 of 2)
```

---

## Placing Stones and Undoing Moves

`place()` puts a stone on the board and auto-advances the turn. `undo()` reverses the last placement. Together they let search algorithms explore moves without copying the board.

```python
game = HexGame()
game.place(0, 0)

# Try a move
game.place(1, 0)
print(game.total_stones)      # 2
score_a = game.scored_moves(1)[0][2]

# Undo and try a different move
game.undo()
print(game.total_stones)      # 1

game.place(0, 1)
score_b = game.scored_moves(1)[0][2]
game.undo()

print(f"Move A scored {score_a}, Move B scored {score_b}")
```

This is extremely fast -- 1.4 million place/undo cycles per second on M4 Pro. The C engine stores undo information on a stack with no allocation or garbage collection.

---

## Reading the Board

```python
game = HexGame()
for q, r in [(0,0), (3,0), (3,-1), (1,0), (2,0)]:
    game.place(q, r)

# Who is winning?
print(game.winner)             # None (game not over)
print(game.is_over)            # False

# What moves are available?
moves = game.legal_moves()     # all legal positions
print(f"{len(moves)} legal moves")

# Which moves are best? (C heuristic scoring)
for q, r, score in game.scored_moves(5):
    print(f"  ({q},{r}) score={score}")

# Where are the threats?
print(f"P0 can win: {game.has_winning_move(0)}")
print(f"P1 can win: {game.has_winning_move(1)}")
print(f"Winning cells for P0: {game.count_winning_moves(0)}")
```

---

## Analyzing Specific Cells

You can analyze any cell without placing a stone there:

```python
# How long is the line through (3,0) for Player 0?
line_len = game.max_line(3, 0, player=0)
print(f"Line through (3,0): {line_len}")

# Would placing at (4,0) win for Player 0?
would_win = game.would_win(4, 0, player=0)
print(f"(4,0) wins: {would_win}")
```

These functions check what would happen if a stone were placed, without actually placing it.

---

## Deep Search (Alpha-Beta)

For tactical analysis, the built-in alpha-beta search looks several turns ahead:

```python
result = game.search(depth=8)  # depth 8 = 4 full turns ahead
print(f"Best move: {result['best_move']}")
print(f"Evaluation: {result['value']}")   # -1 to +1
print(f"Nodes searched: {result['nodes']}")
```

The search uses the C engine with transposition tables, killer heuristics, and late move reduction. At depth 8 it searches around 250K positions in about 1 second and can spot forced wins and losses that simpler evaluation misses.

---

## Cloning and Serialization

When you need independent copies of a game:

```python
# Clone: independent deep copy
copy = game.clone()
copy.place(5, 0)             # doesn't affect original
print(game.total_stones)      # unchanged

# Serialize to dict (JSON-compatible)
data = game.to_dict()

# Reconstruct from dict
game2 = HexGame.from_dict(data)

# Reconstruct from move list
game3 = HexGame.from_moves([(0,0), (1,0), (1,-1)])
```

---

## The Zobrist Hash

Every position has a unique 64-bit hash that changes incrementally as stones are placed. This is useful for transposition tables and position caching in your own search algorithms.

```python
game = HexGame()
game.place(0, 0)
hash_a = game.zhash

game.place(1, 0)
game.place(1, -1)
hash_b = game.zhash   # different position, different hash

game.undo()
game.undo()
assert game.zhash == hash_a  # undo restores the hash exactly
```

Two games with identical stone placements in different order may produce the same hash (transposition). This is what makes transposition tables work -- you detect when two different move sequences reach the same position and reuse the evaluation.

---

## Candidate Tracking

The C engine maintains a set of candidate moves around existing stones. You do not manage this yourself -- `legal_moves()` and `scored_moves()` return only positions within range of the current stones. When a stone is placed the candidate set expands; when a stone is undone the candidates shrink back.

`scored_moves(limit)` returns the top N candidates ranked by the C heuristic, which scores line extension potential, blocking value, and proximity. This is the fastest way to get a short list of promising moves for any search algorithm.

---

## Putting It All Together

```python
from hexbot import HexGame, evaluate_moves, find_winning_moves

game = HexGame()

# Opening: Player 0 places center
game.place(0, 0)

# Player 1 responds
game.place(1, 0)
game.place(1, -1)

# Player 0 builds a line
game.place(-1, 1)
game.place(0, 1)

# Check the position
print(game)
print(f"Player {game.current_player} to move")
print(f"Top moves: {evaluate_moves(game, 3)}")
print(f"P0 winning moves: {find_winning_moves(game, 0)}")
print(f"P1 winning moves: {find_winning_moves(game, 1)}")

# Look ahead with alpha-beta
result = game.search(depth=6)
print(f"Best move: {result['best_move']}, eval: {result['value']:.2f}")
```

---

## Next Steps

- [Bot Approaches](bot-approaches.md) -- six complete strategies from hand-tuned to neural networks
- [API Reference](api-reference.md) -- full method signatures and return types
- [Dashboard Guide](dashboard-guide.md) -- visualize training and arena matches
