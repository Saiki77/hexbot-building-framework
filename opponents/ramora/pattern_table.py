"""Pattern table for N-cell hex window evaluation with symmetry reduction.

Each window is N cells along a hex axis. Cell states: 0=empty, 1=current player, 2=opponent.
Symmetries:
  - Flipping (reverse): same value
  - Piece swap (1↔2): negates value (symmetric game)
"""

WINDOW_LENGTH = 6
NUM_PATTERNS = 3 ** WINDOW_LENGTH


def _int_to_pattern(n, wl):
    pat = []
    for _ in range(wl):
        pat.append(n % 3)
        n //= 3
    return tuple(pat)


def _pattern_to_int(pat):
    n = 0
    for i in range(len(pat) - 1, -1, -1):
        n = n * 3 + pat[i]
    return n


def _swap_pieces(pat):
    """Swap pieces: 1↔2, 0 unchanged."""
    return tuple(({0: 0, 1: 2, 2: 1}[c]) for c in pat)


def build_tables(wl=None, enforce_piece_swap=True):
    """Build the canonical pattern table for a given window length.

    If enforce_piece_swap=True, patterns related by 1↔2 swap share the same
    canonical index with opposite sign, halving the parameter count.

    Returns:
        canon_patterns: list of canonical pattern tuples (the learnable set)
        pattern_map: dict mapping pattern_int -> (canon_index, sign)
    """
    if wl is None:
        wl = WINDOW_LENGTH
    num_patterns = 3 ** wl
    canon_patterns = []
    canon_lookup = {}
    pattern_map = {}

    for i in range(num_patterns):
        if i in pattern_map:
            continue

        pat = _int_to_pattern(i, wl)

        if all(c == 0 for c in pat):
            pattern_map[i] = (-1, 0)
            continue

        # Generate all equivalent patterns
        p_flip = pat[::-1]
        if enforce_piece_swap:
            p_swap = _swap_pieces(pat)
            p_swap_flip = _swap_pieces(p_flip)
            # Positive variants (sign +1): pat, flip
            # Negative variants (sign -1): swap, swap_flip
            pos_variants = [pat, p_flip]
            neg_variants = [p_swap, p_swap_flip]
            # Choose canonical as the min across all variants
            all_variants = pos_variants + neg_variants
            canon = min(all_variants)
            # Is canon from the positive or negative set?
            if canon in pos_variants:
                canon_sign = 1
            else:
                canon_sign = -1
        else:
            pos_variants = [pat, p_flip]
            neg_variants = []
            canon = min(pos_variants)
            canon_sign = 1

        if canon not in canon_lookup:
            canon_lookup[canon] = len(canon_patterns)
            canon_patterns.append(canon)
        cidx = canon_lookup[canon]

        # Check if this pattern is self-symmetric under piece swap
        # (i.e., swap maps it back to itself or its flip → forced zero)
        is_self_symmetric = False
        if enforce_piece_swap:
            pos_set = {p for p in pos_variants}
            neg_set = {p for p in neg_variants}
            if pos_set & neg_set:
                is_self_symmetric = True

        for p in pos_variants:
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                if is_self_symmetric:
                    pattern_map[pi] = (cidx, 0)  # forced zero
                else:
                    pattern_map[pi] = (cidx, canon_sign)

        for p in neg_variants:
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                if is_self_symmetric:
                    pattern_map[pi] = (cidx, 0)
                else:
                    pattern_map[pi] = (cidx, -canon_sign)

    return canon_patterns, pattern_map


def build_arrays(wl=None, enforce_piece_swap=True):
    """Build tables and return flat arrays for fast lookup.

    Returns (canon_patterns, canon_index, canon_sign, num_canon, num_patterns).
    """
    if wl is None:
        wl = WINDOW_LENGTH
    num_patterns = 3 ** wl
    canon_patterns, pattern_map = build_tables(wl, enforce_piece_swap=enforce_piece_swap)
    canon_index = [0] * num_patterns
    canon_sign = [0] * num_patterns
    for pi, (ci, s) in pattern_map.items():
        canon_index[pi] = ci
        canon_sign[pi] = s
    return canon_patterns, canon_index, canon_sign, len(canon_patterns), num_patterns


# Pre-compute default (length 6) on import
CANON_PATTERNS, PATTERN_MAP = build_tables()
NUM_CANON = len(CANON_PATTERNS)

CANON_INDEX = [0] * NUM_PATTERNS
CANON_SIGN = [0] * NUM_PATTERNS
for _pi, (_ci, _s) in PATTERN_MAP.items():
    CANON_INDEX[_pi] = _ci
    CANON_SIGN[_pi] = _s


def pattern_to_int(pat):
    return _pattern_to_int(pat)


if __name__ == "__main__":
    for wl in [6, 7, 8]:
        cp, _, _, nc, np_ = build_arrays(wl)
        print(f"Window length {wl}: {np_} patterns, {nc} canonical")
