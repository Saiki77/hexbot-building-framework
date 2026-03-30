"""
Pure Python Hex Connect-6 engine — used internally by the training pipeline.

For the public API, use hexgame.py instead (C-backed, faster, cleaner interface).

This module provides HexGame with full undo support, Zobrist hashing,
and candidate tracking. Axial coordinates (q, r) on an infinite hex grid.
Three win axes: (1,0), (0,1), (1,-1). Win = 6 in a row.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Hex distance & precomputed offsets
# ---------------------------------------------------------------------------

def _hex_distance(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


def _generate_offsets(radius: int) -> tuple:
    offsets = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            d = _hex_distance(dq, dr)
            if 1 <= d <= radius:
                offsets.append((dq, dr))
    return tuple(offsets)


CANDIDATE_OFFSETS: dict[int, tuple] = {}
for _r in (1, 2, 3, 4, 5, 6, 7, 8):
    CANDIDATE_OFFSETS[_r] = _generate_offsets(_r)

# ---------------------------------------------------------------------------
# Zobrist hashing (lazy, for infinite board)
# ---------------------------------------------------------------------------

_zobrist_rng = random.Random(42)
_zobrist_cache_0: dict = {}
_zobrist_cache_1: dict = {}
_zobrist_getrandbits = _zobrist_rng.getrandbits


# ---------------------------------------------------------------------------
# HexGame — the main engine
# ---------------------------------------------------------------------------

class HexGame:
    """Fast hexagonal connect-6 game state for bot training."""

    __slots__ = (
        'stones_0', 'stones_1',  # sets for player 0 and 1
        'occupied',              # union set
        'candidates',            # frontier set
        'turn', 'current_player',
        'stones_this_turn', 'stones_per_turn',
        'winner', 'total_stones', 'max_total_stones',
        '_history',              # list of tuples (compact undo records)
        '_zobrist_hash',
        '_candidate_offsets',
    )

    def __init__(self, candidate_radius: int = 2, max_total_stones: int = 200):
        self.stones_0 = set()
        self.stones_1 = set()
        self.occupied = set()
        self.candidates = set()
        self.turn = 0
        self.current_player = 0
        self.stones_this_turn = 0
        self.stones_per_turn = 1
        self.winner = None
        self.total_stones = 0
        self.max_total_stones = max_total_stones
        self._history = []
        self._zobrist_hash = 0
        self._candidate_offsets = CANDIDATE_OFFSETS[candidate_radius]

    # ----- core: place a single stone ------------------------------------

    def place_stone(self, q: int, r: int) -> None:
        cell = (q, r)
        player = self.current_player
        p_stones = self.stones_0 if player == 0 else self.stones_1
        occ = self.occupied
        cands = self.candidates
        offsets = self._candidate_offsets

        cand_was_present = cell in cands

        p_stones.add(cell)
        occ.add(cell)
        if cand_was_present:
            cands.discard(cell)

        # expand candidates — track additions for undo
        cands_added = []
        cands_add = cands.add
        cands_added_append = cands_added.append
        for dq, dr in offsets:
            nb = (q + dq, r + dr)
            if nb not in occ and nb not in cands:
                cands_add(nb)
                cands_added_append(nb)

        # win check — fully inlined, no function calls
        win = False
        # axis (1, 0)
        count = 1
        nq, nr = q + 1, r
        while (nq, nr) in p_stones:
            count += 1
            if count >= 6: break
            nq += 1
        if count < 6:
            nq, nr = q - 1, r
            while (nq, nr) in p_stones:
                count += 1
                if count >= 6: break
                nq -= 1
        if count >= 6:
            win = True
        else:
            # axis (0, 1)
            count = 1
            nq, nr = q, r + 1
            while (nq, nr) in p_stones:
                count += 1
                if count >= 6: break
                nr += 1
            if count < 6:
                nq, nr = q, r - 1
                while (nq, nr) in p_stones:
                    count += 1
                    if count >= 6: break
                    nr -= 1
            if count >= 6:
                win = True
            else:
                # axis (1, -1)
                count = 1
                nq, nr = q + 1, r - 1
                while (nq, nr) in p_stones:
                    count += 1
                    if count >= 6: break
                    nq += 1; nr -= 1
                if count < 6:
                    nq, nr = q - 1, r + 1
                    while (nq, nr) in p_stones:
                        count += 1
                        if count >= 6: break
                        nq -= 1; nr += 1
                if count >= 6:
                    win = True

        # zobrist — inlined lookup
        zc = _zobrist_cache_0 if player == 0 else _zobrist_cache_1
        zv = zc.get(cell)
        if zv is None:
            zv = _zobrist_getrandbits(64)
            zc[cell] = zv
        zh = self._zobrist_hash ^ zv

        # push undo as plain tuple (avoids object creation overhead)
        self._history.append((
            q, r, player,
            cands_added, cand_was_present,
            self.stones_this_turn, self.stones_per_turn,
            self.turn, self.current_player,
            self.winner, self._zobrist_hash,
        ))

        self._zobrist_hash = zh
        self.total_stones += 1

        if win:
            self.winner = player

        # advance turn
        self.stones_this_turn += 1
        if self.stones_this_turn >= self.stones_per_turn:
            self.turn += 1
            self.current_player = self.turn & 1
            self.stones_this_turn = 0
            self.stones_per_turn = 2

    # ----- undo -----------------------------------------------------------

    def undo(self) -> None:
        rec = self._history.pop()
        q, r, player, cands_added, cand_was_present, \
            stt, spt, turn, cp, winner, zh = rec

        cell = (q, r)
        if player == 0:
            self.stones_0.discard(cell)
        else:
            self.stones_1.discard(cell)
        self.occupied.discard(cell)

        cands = self.candidates
        for nb in cands_added:
            cands.discard(nb)
        if cand_was_present:
            cands.add(cell)

        self.stones_this_turn = stt
        self.stones_per_turn = spt
        self.turn = turn
        self.current_player = cp
        self.winner = winner
        self._zobrist_hash = zh
        self.total_stones -= 1

    # ----- clone ----------------------------------------------------------

    def clone(self) -> HexGame:
        new = HexGame.__new__(HexGame)
        new.stones_0 = self.stones_0.copy()
        new.stones_1 = self.stones_1.copy()
        new.occupied = self.occupied.copy()
        new.candidates = self.candidates.copy()
        new.turn = self.turn
        new.current_player = self.current_player
        new.stones_this_turn = self.stones_this_turn
        new.stones_per_turn = self.stones_per_turn
        new.winner = self.winner
        new.total_stones = self.total_stones
        new.max_total_stones = self.max_total_stones
        new._history = []
        new._zobrist_hash = self._zobrist_hash
        new._candidate_offsets = self._candidate_offsets
        return new

    # ----- legal moves ----------------------------------------------------

    def legal_moves(self) -> List[Tuple[int, int]]:
        if not self.occupied:
            return [(0, 0)]
        return list(self.candidates)

    def legal_moves_set(self) -> set:
        if not self.occupied:
            return {(0, 0)}
        return self.candidates

    def random_move(self, rng_choice) -> Tuple[int, int]:
        """Pick a random legal move. Pass rng.choice (bound method) for speed."""
        if not self.occupied:
            return (0, 0)
        # Convert to tuple for O(1) random access — cheaper than list()
        # when candidates is small, similar when large
        return rng_choice(tuple(self.candidates))

    def play_random_game(self, rng: random.Random) -> int:
        """Play out randomly to terminal. Returns winner (0/1) or -1 for draw.
        Optimized fast path for MCTS rollouts."""
        randint = rng.randint
        place = self.place_stone
        max_stones = self.max_total_stones

        # Handle first move if board is empty
        if not self.occupied:
            place(0, 0)

        while self.winner is None and self.total_stones < max_stones:
            cands = self.candidates
            if not cands:
                break
            moves = tuple(cands)
            move = moves[randint(0, len(moves) - 1)]
            place(move[0], move[1])
        return self.winner if self.winner is not None else -1

    # ----- high-level move (full turn) ------------------------------------

    def make_move(self, cells: List[Tuple[int, int]]) -> None:
        for q, r in cells:
            self.place_stone(q, r)

    def undo_move(self) -> None:
        if not self._history:
            return
        if self.stones_this_turn == 0 and self._history:
            prev_turn = self._history[-1][7]
            while self._history and self._history[-1][7] == prev_turn:
                self.undo()
        else:
            count = self.stones_this_turn
            for _ in range(count):
                self.undo()

    # ----- properties / queries -------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or self.total_stones >= self.max_total_stones

    def result(self) -> float:
        if self.winner == 0:
            return 1.0
        if self.winner == 1:
            return -1.0
        return 0.0

    def result_for(self, player: int) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def state_key(self) -> int:
        return self._zobrist_hash

    @property
    def stones_remaining_this_turn(self) -> int:
        return self.stones_per_turn - self.stones_this_turn

    def get_stones(self, player: int) -> set:
        return self.stones_0 if player == 0 else self.stones_1

    def __repr__(self) -> str:
        return (
            f"HexGame(turn={self.turn}, player={self.current_player}, "
            f"stones={self.total_stones}, "
            f"stt={self.stones_this_turn}/{self.stones_per_turn}, "
            f"winner={self.winner})"
        )


# ---------------------------------------------------------------------------
# Self-test & benchmark
# ---------------------------------------------------------------------------

def _test_win_detection():
    # Horizontal axis (1, 0)
    g = HexGame(candidate_radius=2, max_total_stones=500)
    g.place_stone(0, 0)
    g.place_stone(0, 5); g.place_stone(0, 6)
    g.place_stone(1, 0); g.place_stone(2, 0)
    g.place_stone(0, 7); g.place_stone(0, 8)
    g.place_stone(3, 0); g.place_stone(4, 0)
    g.place_stone(0, 9); g.place_stone(0, 10)
    g.place_stone(5, 0)
    assert g.winner == 0, f"Expected p0 win on (1,0), got {g.winner}"
    print("  PASS: win on axis (1,0)")

    # Axis (0, 1)
    g = HexGame(candidate_radius=2, max_total_stones=500)
    g.place_stone(0, 0)
    g.place_stone(5, 0); g.place_stone(6, 0)
    g.place_stone(0, 1); g.place_stone(0, 2)
    g.place_stone(7, 0); g.place_stone(8, 0)
    g.place_stone(0, 3); g.place_stone(0, 4)
    g.place_stone(9, 0); g.place_stone(10, 0)
    g.place_stone(0, 5)
    assert g.winner == 0, f"Expected p0 win on (0,1), got {g.winner}"
    print("  PASS: win on axis (0,1)")

    # Axis (1, -1)
    g = HexGame(candidate_radius=2, max_total_stones=500)
    g.place_stone(0, 0)
    g.place_stone(0, 5); g.place_stone(0, 6)
    g.place_stone(1, -1); g.place_stone(2, -2)
    g.place_stone(0, 7); g.place_stone(0, 8)
    g.place_stone(3, -3); g.place_stone(4, -4)
    g.place_stone(0, 9); g.place_stone(0, 10)
    g.place_stone(5, -5)
    assert g.winner == 0, f"Expected p0 win on (1,-1), got {g.winner}"
    print("  PASS: win on axis (1,-1)")


def _test_undo():
    g = HexGame(candidate_radius=2)
    g.place_stone(0, 0)
    g.place_stone(1, 0)
    g.place_stone(0, 1)
    g.place_stone(-1, 0)
    g.place_stone(0, -1)

    for _ in range(5):
        g.undo()

    assert len(g.occupied) == 0
    assert g.turn == 0 and g.current_player == 0
    assert g.stones_this_turn == 0 and g.stones_per_turn == 1
    assert g.winner is None
    assert g._zobrist_hash == 0
    print("  PASS: undo restores to empty")


def _test_turn_logic():
    g = HexGame(candidate_radius=2)
    assert g.current_player == 0 and g.stones_per_turn == 1
    g.place_stone(0, 0)
    assert g.current_player == 1 and g.stones_per_turn == 2
    g.place_stone(1, 0)
    assert g.current_player == 1
    g.place_stone(-1, 0)
    assert g.current_player == 0 and g.stones_per_turn == 2
    g.place_stone(0, 1)
    assert g.current_player == 0
    g.place_stone(0, -1)
    assert g.current_player == 1 and g.stones_per_turn == 2
    print("  PASS: turn logic correct")


def _test_clone():
    g = HexGame(candidate_radius=2)
    g.place_stone(0, 0)
    g.place_stone(1, 0); g.place_stone(-1, 0)
    c = g.clone()
    assert c.stones_0 == g.stones_0
    assert c.stones_1 == g.stones_1
    assert c.occupied == g.occupied
    assert c.candidates == g.candidates
    assert c._zobrist_hash == g._zobrist_hash
    assert c.turn == g.turn
    # mutate clone, original unaffected
    c.place_stone(0, 1)
    assert (0, 1) not in g.occupied
    print("  PASS: clone works correctly")


def _benchmark(num_games: int = 10000):
    import time
    rng = random.Random(123)
    total_stones = 0
    t0 = time.perf_counter()

    for _ in range(num_games):
        g = HexGame(candidate_radius=2, max_total_stones=200)
        while not g.is_terminal:
            moves = g.legal_moves()
            if not moves:
                break
            move = moves[rng.randint(0, len(moves) - 1)]
            g.place_stone(move[0], move[1])
        total_stones += g.total_stones

    elapsed = time.perf_counter() - t0
    print(f"  {num_games} games, {total_stones} total stones in {elapsed:.2f}s")
    print(f"  {num_games / elapsed:.0f} games/sec, {total_stones / elapsed:.0f} stones/sec")


def _benchmark_with_undo(num_games: int = 10000):
    import time
    rng = random.Random(123)
    total_ops = 0
    t0 = time.perf_counter()

    for _ in range(num_games):
        g = HexGame(candidate_radius=2, max_total_stones=200)
        while not g.is_terminal:
            moves = g.legal_moves()
            if not moves:
                break
            move = moves[rng.randint(0, len(moves) - 1)]
            g.place_stone(move[0], move[1])
            total_ops += 1
        # undo all
        while g._history:
            g.undo()
            total_ops += 1

    elapsed = time.perf_counter() - t0
    print(f"  {num_games} games with full undo, {total_ops} ops in {elapsed:.2f}s")
    print(f"  {total_ops / elapsed:.0f} ops/sec")


def _benchmark_fast_rollout(num_games: int = 10000):
    import time
    rng = random.Random(123)
    total_stones = 0
    wins = [0, 0, 0]  # p0, p1, draw
    t0 = time.perf_counter()

    for _ in range(num_games):
        g = HexGame(candidate_radius=2, max_total_stones=200)
        result = g.play_random_game(rng)
        total_stones += g.total_stones
        if result == 0:
            wins[0] += 1
        elif result == 1:
            wins[1] += 1
        else:
            wins[2] += 1

    elapsed = time.perf_counter() - t0
    print(f"  {num_games} games, {total_stones} stones in {elapsed:.2f}s")
    print(f"  {num_games / elapsed:.0f} games/sec, {total_stones / elapsed:.0f} stones/sec")
    print(f"  Wins: P0={wins[0]}, P1={wins[1]}, Draw={wins[2]}")


if __name__ == "__main__":
    print("Running tests...")
    _test_turn_logic()
    _test_undo()
    _test_win_detection()
    _test_clone()
    print("\nBenchmark: random playouts (legal_moves)")
    _benchmark(10000)
    print("\nBenchmark: play_random_game (fast rollout)")
    _benchmark_fast_rollout(10000)
    print("\nBenchmark: play + full undo")
    _benchmark_with_undo(5000)
