# Advanced: Speedup Techniques

Deep dive into the performance optimizations used in the hexbot framework
for neural network search, MCTS, and engine integration.

---

## Batched Neural Network Evaluation

When using a neural network for position evaluation inside alpha-beta search,
the naive approach calls Python from C for every leaf node -- about 600
microseconds per call due to ctypes overhead. With thousands of leaves per move,
this dominates search time.

The `BatchedNNAlphaBeta` class eliminates this bottleneck with a 3-phase approach:

```python
from bot import BatchedNNAlphaBeta, create_network

net = create_network('standard')
# ... load weights ...

searcher = BatchedNNAlphaBeta(net, depth=8, nn_depth=5)
policy = searcher.search(game)
best_move = max(policy, key=policy.get)
```

### Phase 1: Collect

C alpha-beta runs to completion using the fast heuristic evaluator at leaf nodes.
Positions that need neural network evaluation are stored in a buffer (up to 2,048
per batch). No Python callbacks occur during this phase -- the C engine runs at
full speed.

### Phase 2: Evaluate

Python reads all collected positions from the buffer, encodes them as tensors,
and runs one batched forward pass through the neural network on GPU/MPS. The
results (value estimates and policy vectors) are injected into the C engine's
value cache.

### Phase 3: Re-search

C alpha-beta runs again from the root. This time, leaf positions hit the NN cache
instead of using the heuristic, producing much better move ordering and pruning.
The search sees deeper and more accurately because the cached NN values are
stronger than heuristic scores.

This gives a 5-10x speedup over the per-node callback approach because there are
zero Python-C transitions during search -- just two bulk data transfers plus two
search invocations.

---

## MCTS with Virtual Loss Batching

The `BatchedMCTS` class uses virtual loss to select multiple leaves simultaneously,
then evaluates them all in one batched neural network forward pass:

```python
from bot import BatchedMCTS, create_network

net = create_network('standard')
mcts = BatchedMCTS(net, num_simulations=200, batch_size=64)
policy = mcts.search(game, temperature=1.0, add_noise=True)
```

### How Virtual Loss Works

1. Select a leaf node using the UCB formula
2. Before evaluating, temporarily increment the node's visit count (virtual loss)
3. Repeat selection -- the virtual loss steers the next selection away from the
   same branch, exploring different parts of the tree
4. After collecting `batch_size` (64) leaves, evaluate them all in one NN call
5. Backpropagate the real values and remove the virtual losses

This reduces the number of neural network forward passes by the batch size factor.
With `batch_size=64`, MCTS makes 64x fewer NN calls compared to evaluating one
leaf at a time. The trade-off is slightly lower search quality (virtual loss is an
approximation), but the wall-clock speedup is dramatic.

### Dirichlet Noise

At the root node, Dirichlet noise is mixed into the prior policy to ensure
exploration:

```
noisy_prior = (1 - epsilon) * nn_prior + epsilon * dirichlet(alpha)
```

With `DIRICHLET_ALPHA=0.3` and `DIRICHLET_EPSILON=0.25`, the root explores
diverse moves while still respecting the network's recommendations.

---

## Alpha-Beta Hybrid in MCTS Root

The `BatchedMCTS` includes a shallow alpha-beta pre-check at the root. Before
running MCTS simulations, it performs a quick depth-4 C engine search (~10ms).

If alpha-beta finds a proven win (value = +/-1.0), MCTS is skipped entirely and
the winning move is returned immediately. This catches forced tactical sequences
-- checkmates, unstoppable threats, forced wins -- that MCTS would need hundreds
of simulations to discover.

This hybrid approach combines the tactical precision of alpha-beta (which is
exact within its search depth) with the strategic depth of MCTS (which evaluates
positions using the neural network).

---

## Transposition Cache

MCTS normally evaluates the same position multiple times when reached through
different move orders. In Connect-6, transpositions are common because placing
stones A then B produces the same board as placing B then A within the same turn.

The `BatchedMCTS` caches neural network evaluations by Zobrist hash, reusing
results for transposed positions. This typically saves 20-40% of NN calls.

### Zobrist Hashing

Every board position has a unique 64-bit hash computed incrementally:

```python
game.place(3, 0)
hash_a = game.zhash  # updated incrementally, O(1)

game.undo()
# hash is restored exactly after undo
```

The hash changes with each `place()` and `undo()` call using XOR operations on
pre-computed random values. Two games with identical stone configurations
(regardless of move order) produce the same hash, enabling transposition detection.

The C engine maintains the Zobrist hash internally, so there is no Python overhead
for hash computation during search.

---

## C Engine Move Ordering

The C engine's `board_get_scored_moves()` function provides fast heuristic move
ordering that benefits any search algorithm. Moves are scored by:

- **Line extension potential** -- how much a move extends existing lines toward 6
- **Blocking value** -- how effectively a move disrupts opponent lines
- **Proximity** -- preference for moves near the action

When used as the first pass before neural network evaluation, this ensures the
most promising moves are searched first. For alpha-beta, good move ordering
improves cutoff rates dramatically. For MCTS, it provides better initial priors
when the NN evaluation is not yet available.

The C engine scores moves in bulk -- scoring all legal moves takes roughly the
same time as scoring one, because the board state is already in cache.

---

## Progressive Widening

In standard MCTS, all legal moves are expanded at each node. On a 19x19 board,
this can mean 300+ children per node, most of which are irrelevant. Progressive
widening limits the branching factor based on visit count:

```
max_children = C * N^alpha
```

Where `N` is the visit count and `C`, `alpha` are tuning constants. Early in
search, only the top few moves (ranked by NN prior or heuristic score) are
considered. As a node accumulates more visits, additional moves are gradually
introduced.

This focuses search effort on the most promising lines while still allowing
exploration of surprising moves as the search deepens.

---

## Quiescence Extensions

Alpha-beta search can produce misleading evaluations when it stops in the middle
of a tactical sequence (the "horizon effect"). Quiescence extensions address this
by continuing the search beyond the nominal depth when the position is "noisy" --
typically when there are immediate threats or forced sequences.

In hexbot, the C engine detects positions with active threats (4-in-a-row,
near-winning configurations) and extends the search to resolve them before
returning a static evaluation. This prevents the engine from being blind to
threats that fall just beyond its search horizon.

---

## ONNX Export for Worker Inference

During self-play, each worker process needs its own copy of the neural network.
Loading full PyTorch models in subprocesses is slow and memory-heavy.

The training pipeline exports the current model to ONNX format at the start of
each iteration:

```python
onnx_path = f"/tmp/hex_model_{iteration}.onnx"
export_onnx(net, onnx_path)
```

V1 workers load this ONNX model using `OnnxPredictor`, which uses ONNX Runtime
for inference. This is lighter than loading a full PyTorch model and avoids
GPU contention between the training process and worker processes.

V2 workers (when the C engine is available) load the PyTorch state dict directly
and run inference on CPU, but the ONNX export is still performed as a fallback.

---

## Device Support

The framework auto-detects the best available device:

| Priority | Device | When Used |
|----------|--------|-----------|
| 1 | CUDA | NVIDIA GPU available (`torch.cuda.is_available()`) |
| 2 | MPS | Apple Silicon (`torch.backends.mps.is_available()`) |
| 3 | CPU | Fallback when no GPU is available |

Override with `--device cuda`, `--device mps`, or `--device cpu`.

### GPU Usage

- **Training** runs on GPU (CUDA or MPS) for fast gradient computation
- **Self-play workers** run on CPU -- the C game engine is CPU-only and already
  very fast (1.4M place/undo cycles per second on M4 Pro)
- **ONNX inference** in V1 workers uses CPU via ONNX Runtime
- **ELO evaluation** runs on the training device

### Memory Considerations

The `ModelVault` stores past generations in fp16 to reduce memory usage. A
standard HexNet (3.9M params) takes ~7.5 MB in fp16, so 200 generations use
about 1.5 GB of RAM.

The replay buffer at full capacity (400K samples) uses approximately 2 GB of RAM
depending on the tensor sizes.
