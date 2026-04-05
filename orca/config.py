"""
Orca configuration - all tunable constants in one place.

These are the defaults used by the Orca training pipeline.
Override by passing arguments to OrcaTrainer or editing this file.
"""

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
BOARD_SIZE = 19
NUM_CHANNELS = 7       # 5 board planes + 2 threat planes
NUM_FILTERS = 128      # filter width (128 = fast, 256 = stronger but slower)
NUM_RES_BLOCKS = 12    # depth of residual tower

# ---------------------------------------------------------------------------
# Search (MCTS)
# ---------------------------------------------------------------------------
C_PUCT = 1.5           # exploration constant
NUM_SIMULATIONS = 400  # MCTS sims per move (curriculum may override)
MCTS_BATCH_SIZE = 64   # positions per batched NN forward pass
DIRICHLET_ALPHA = 0.3  # root noise (0.3 = diverse exploration)
DIRICHLET_EPSILON = 0.15 # fraction of prior replaced by noise (0.15 = subtle, 0.25 = aggressive)
TEMP_THRESHOLD = 20    # moves before switching to greedy play

# ---------------------------------------------------------------------------
# Distant play (colony strategy)
# ---------------------------------------------------------------------------
PLAY_STYLE = 'distant'       # 'distant' or 'close'
C_BLEND_ADJACENT = 0.15      # C heuristic weight for adjacent moves
C_BLEND_DISTANT = 0.05       # C heuristic weight for far moves
DISTANT_EXPLORE_PROB = 0.25  # probability of injecting distant candidates
DISTANT_RANGE = (2, 5)       # min/max distance from nearest stone

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 1024             # training batch size
LEARNING_RATE = 0.001         # Adam optimizer LR
L2_REG = 1e-4                 # weight decay
REPLAY_BUFFER_SIZE = 400_000  # experience replay capacity

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_STEPS = 200     # gradient steps per iteration
DEFAULT_GAMES_PER_ITER = 100  # base games (curriculum may adjust)
CHECKPOINT_EVERY = 1          # save checkpoint every N iterations
MAX_WORKERS = 5               # parallel self-play workers
GAMES_PER_FUTURE = 2          # games per subprocess future
ELO_EVAL_EVERY = 1            # ELO evaluation frequency (iterations)
ELO_EVAL_GAMES = 5            # games per ELO opponent
ELO_EVAL_SIMS = 30            # MCTS sims during ELO games (lower = faster eval)
ELO_MAX_OPPONENTS = 5         # max past versions to play against
ELO_BASELINE_GAMES = 4        # games vs random + heuristic baselines (0 = disable)

# ---------------------------------------------------------------------------
# Model vault
# ---------------------------------------------------------------------------
VAULT_MAX_MODELS = 200        # max stored model snapshots

# ---------------------------------------------------------------------------
# Curriculum (adaptive sim/game scaling)
# ---------------------------------------------------------------------------
CURRICULUM = {
    # time_hours: (sims, games_per_iter)
    0.0: (50, 60),    # fast exploration
    0.5: (100, 50),   # better quality
    1.5: (150, 40),   # deeper search
    3.0: (200, 30),   # full depth
}
PLATEAU_THRESHOLD = 15   # ELO delta to detect plateau
PLATEAU_ITERS = 10       # iterations of stall before boosting sims
PLATEAU_SIM_BOOST = 50   # extra sims on plateau (capped at 400)

# ---------------------------------------------------------------------------
# LR Schedule (CosineAnnealingWarmRestarts)
# ---------------------------------------------------------------------------
COSINE_T0 = 50
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-4

# ---------------------------------------------------------------------------
# Defensive training (blocking reward)
# ---------------------------------------------------------------------------
BLOCKING_PRIORITY_BOOST = 5.0    # priority for moves that block opponent threats
SURVIVAL_PRIORITY_BOOST = 3.0    # priority for surviving an opponent threat turn
USE_AB_HYBRID = True             # set False to disable AB pre-check in MCTS
AB_HYBRID_DEPTH = 4              # depth of AB pre-check (0 = disable)
THREAT_POLICY_BLEND = 0.5        # blend threat spatial map into policy logits (0 = disabled)
RAMORA_GAME_FRACTION = 0.70      # fraction of games played vs SealBot (strong opponent)
RAMORA_TIME_LIMIT = 0.3          # seconds per move for SealBot during training

# ---------------------------------------------------------------------------
# Mixed precision (CUDA only - MPS FP16 is unreliable)
# ---------------------------------------------------------------------------
USE_MIXED_PRECISION = True    # auto FP16 on CUDA, ignored on MPS/CPU
GRAD_CLIP_NORM = 1.0          # gradient clipping (0 = disabled)

# ---------------------------------------------------------------------------
# Experimental: Transformer variant
# ---------------------------------------------------------------------------
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_DROPOUT = 0.1
