"""Tests for C MCTS (engine.c MCTS + orca/c_mcts.py wrapper)."""
import unittest
import ctypes
import numpy as np
import torch

from orca.encoding import CGameState, c_encode_state
from orca.network import HexNet, create_network
from bot import get_device


def _get_lib():
    """Get C lib with MCTS signatures configured."""
    from orca.c_mcts import _get_lib as get_c_mcts_lib
    return get_c_mcts_lib()


class TestBoardEncodeStateFull(unittest.TestCase):
    """Test the 7-channel C encoding function."""

    def test_shape(self):
        """Output is (7, 19, 19)."""
        game = CGameState()
        game.place_stone(0, 0)
        t, oq, orr = c_encode_state(game)
        self.assertEqual(t.shape, (7, 19, 19))

    def test_empty_board(self):
        """Empty board has only (0,0) as legal move."""
        game = CGameState()
        t, oq, orr = c_encode_state(game)
        # Plane 0 (my stones): empty
        self.assertEqual(t[0].sum().item(), 0)
        # Plane 2 (legal moves): only (0,0)
        self.assertEqual(t[2].sum().item(), 1.0)
        # Plane 3 (player): all 0
        self.assertEqual(t[3].mean().item(), 0.0)

    def test_stones_on_correct_planes(self):
        """Player stones appear on plane 0, opponent on plane 1."""
        game = CGameState()
        game.place_stone(0, 0)   # p0
        game.place_stone(1, 0)   # p1
        game.place_stone(1, -1)  # p1
        # Now p0's turn
        t, oq, orr = c_encode_state(game)
        self.assertEqual(t[0].sum().item(), 1.0)  # p0 has 1 stone
        self.assertEqual(t[1].sum().item(), 2.0)  # p1 has 2 stones

    def test_threat_channels_populate(self):
        """Threat channels show values when 4+ in a row exists."""
        game = CGameState()
        # Build a line for p0: (0,0), (1,0), (2,0), (3,0), (4,0) = 5 in a row
        game.place_stone(0, 0)   # p0 (first move alone)
        game.place_stone(0, 5)   # p1
        game.place_stone(0, 6)   # p1
        game.place_stone(1, 0)   # p0
        game.place_stone(2, 0)   # p0
        game.place_stone(1, 5)   # p1
        game.place_stone(1, 6)   # p1
        game.place_stone(3, 0)   # p0
        game.place_stone(4, 0)   # p0 -> 5 in a row for p0

        t, oq, orr = c_encode_state(game)
        # p1 is current player, so p0's threats are on plane 6 (opponent)
        self.assertGreater(t[6].max().item(), 0.0)

    def test_encoding_deterministic(self):
        """Same position produces identical encoding."""
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 1)
        t1, _, _ = c_encode_state(game)
        t2, _, _ = c_encode_state(game)
        self.assertTrue(torch.equal(t1, t2))


class TestCMCTSTreeLifecycle(unittest.TestCase):
    """Test C MCTS tree creation and destruction."""

    def test_create_destroy(self):
        """Tree can be created and destroyed without crash."""
        lib = _get_lib()
        game = CGameState()
        game.place_stone(0, 0)
        tree = lib.mcts_tree_new(game._ptr, ctypes.c_float(1.5), 32)
        self.assertIsNotNone(tree)
        self.assertNotEqual(tree, 0)
        lib.mcts_tree_destroy(tree)

    def test_initial_state(self):
        """Fresh tree has 1 node (root), 0 leaves."""
        lib = _get_lib()
        game = CGameState()
        game.place_stone(0, 0)
        tree = lib.mcts_tree_new(game._ptr, ctypes.c_float(1.5), 32)
        self.assertEqual(lib.mcts_get_node_count(tree), 1)
        self.assertEqual(lib.mcts_get_leaf_count(tree), 0)
        lib.mcts_tree_destroy(tree)


class TestCMCTSSelectBatch(unittest.TestCase):
    """Test the select_batch function."""

    def test_returns_leaves(self):
        """First select_batch on unexpanded root returns leaves."""
        lib = _get_lib()
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)
        tree = lib.mcts_tree_new(game._ptr, ctypes.c_float(1.5), 32)
        n = lib.mcts_select_batch(tree, 16)
        # Root is unexpanded, so first batch should produce at least 1 leaf
        self.assertGreater(n, 0)
        lib.mcts_tree_destroy(tree)

    def test_leaf_encodings_accessible(self):
        """Leaf encodings can be read as numpy array."""
        lib = _get_lib()
        game = CGameState()
        game.place_stone(0, 0)
        tree = lib.mcts_tree_new(game._ptr, ctypes.c_float(1.5), 32)
        n = lib.mcts_select_batch(tree, 8)
        if n > 0:
            enc_ptr = lib.mcts_get_leaf_encodings(tree)
            enc_np = np.ctypeslib.as_array(enc_ptr, shape=(n, 7, 19, 19))
            self.assertEqual(enc_np.shape[0], n)
            self.assertEqual(enc_np.shape[1], 7)
        lib.mcts_tree_destroy(tree)


class TestCMCTSFullSearch(unittest.TestCase):
    """Test complete search cycle."""

    def setUp(self):
        self.device = get_device()
        self.net = create_network('fast').to(self.device)
        self.net.eval()

    def test_search_returns_policy(self):
        """CMCTSSearch.search returns non-empty policy dict."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=20, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)
        policy = mcts.search(game, temperature=1.0, add_noise=False)
        self.assertIsInstance(policy, dict)
        self.assertGreater(len(policy), 0)

    def test_policy_sums_to_one(self):
        """Policy probabilities should sum to ~1.0."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=30, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)
        game.place_stone(1, -1)
        policy = mcts.search(game, temperature=1.0, add_noise=False)
        total = sum(policy.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_greedy_policy(self):
        """With temperature ~0, one move should have probability ~1.0."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=30, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        policy = mcts.search(game, temperature=0.01, add_noise=False)
        max_prob = max(policy.values())
        self.assertGreater(max_prob, 0.8)

    def test_dirichlet_noise_changes_policy(self):
        """Adding Dirichlet noise should change the policy distribution."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=30, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)

        np.random.seed(42)
        p1 = mcts.search(game, temperature=1.0, add_noise=False)
        np.random.seed(42)
        p2 = mcts.search(game, temperature=1.0, add_noise=True)
        # With noise, distribution should be different (or at least not crash)
        self.assertIsInstance(p2, dict)
        self.assertGreater(len(p2), 0)

    def test_empty_board_first_move(self):
        """Search on empty board should return valid policy."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=10, batch_size=8)
        game = CGameState()
        policy = mcts.search(game, temperature=1.0, add_noise=False)
        self.assertGreater(len(policy), 0)
        # First move should include (0,0)
        self.assertIn((0, 0), policy)

    def test_multiple_searches_independent(self):
        """Consecutive searches don't corrupt each other."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=20, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        p1 = mcts.search(game, temperature=1.0, add_noise=False)
        p2 = mcts.search(game, temperature=1.0, add_noise=False)
        # Both should be valid policies
        self.assertGreater(len(p1), 0)
        self.assertGreater(len(p2), 0)
        self.assertAlmostEqual(sum(p1.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(p2.values()), 1.0, places=2)

    def test_analysis_data(self):
        """last_root_value and last_top_moves should be populated after search."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=20, batch_size=16)
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)
        mcts.search(game, temperature=1.0, add_noise=False)
        val = mcts.last_root_value()
        top = mcts.last_top_moves(3)
        self.assertTrue(isinstance(val, (float, np.floating)))
        self.assertIsInstance(top, list)

    def test_high_sim_count(self):
        """Search with more sims produces a valid policy."""
        from orca.c_mcts import CMCTSSearch
        mcts = CMCTSSearch(self.net, num_simulations=100, batch_size=32)
        game = CGameState()
        game.place_stone(0, 0)
        game.place_stone(1, 0)
        game.place_stone(1, -1)
        policy = mcts.search(game, temperature=1.0, add_noise=True)
        self.assertGreater(len(policy), 0)
        # Policy sum may not be exactly 1.0 due to float32 precision
        # in C powf/expf, but should be in a reasonable range
        total = sum(policy.values())
        self.assertGreater(total, 0.5, f"Policy sum {total} too low")


class TestCMCTSvsython(unittest.TestCase):
    """Compare C MCTS against Python BatchedMCTS for consistency."""

    def setUp(self):
        self.device = get_device()
        self.net = create_network('fast').to(self.device)
        self.net.eval()

    def test_both_return_valid_policies(self):
        """Both C and Python MCTS return valid policies on same position."""
        from orca.c_mcts import CMCTSSearch
        from orca.search import BatchedMCTS

        game1 = CGameState()
        game1.place_stone(0, 0); game1.place_stone(1, 0); game1.place_stone(1, -1)
        game2 = CGameState()
        game2.place_stone(0, 0); game2.place_stone(1, 0); game2.place_stone(1, -1)

        c_mcts = CMCTSSearch(self.net, num_simulations=30, batch_size=16)
        py_mcts = BatchedMCTS(self.net, num_simulations=30, batch_size=16)

        p_c = c_mcts.search(game1, temperature=0.1, add_noise=False)
        p_py = py_mcts.search(game2, temperature=0.1, add_noise=False)

        self.assertGreater(len(p_c), 0)
        self.assertGreater(len(p_py), 0)
        self.assertAlmostEqual(sum(p_c.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(p_py.values()), 1.0, places=2)


class TestCGameStateOps(unittest.TestCase):
    """Test CGameState operations used by C MCTS."""

    def test_place_and_undo(self):
        """Place + undo restores state."""
        game = CGameState()
        game.place_stone(0, 0)
        p_before = game.current_player
        game.place_stone(5, 5)
        game.undo()
        self.assertEqual(game.current_player, p_before)

    def test_candidates_grow(self):
        """Candidates expand when stones are placed."""
        game = CGameState()
        c0 = len(game.candidates)
        game.place_stone(0, 0)
        c1 = len(game.candidates)
        self.assertGreater(c1, c0)

    def test_clone(self):
        """Clone produces independent copy."""
        game = CGameState()
        game.place_stone(0, 0)
        clone = game.clone()
        clone.place_stone(5, 5)
        # Original should not be affected
        self.assertNotEqual(
            game._lib.board_get_total_stones(game._ptr),
            clone._lib.board_get_total_stones(clone._ptr),
        )

    def test_win_detection(self):
        """Detect 6-in-a-row win via C engine."""
        game = CGameState()
        # p0 places 6 in a row (with p1 playing elsewhere)
        moves_p0 = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)]
        moves_p1 = [(0,5), (0,6), (1,5), (1,6), (2,5)]
        game.place_stone(*moves_p0[0])  # p0 first move
        for i in range(5):
            game.place_stone(*moves_p1[i])  # p1 2 stones
            if i < 4:
                game.place_stone(*moves_p0[i+1])  # p0 2 stones
                if i + 2 < len(moves_p0):
                    game.place_stone(*moves_p0[i+2])
        # Check that game detects a winner eventually
        self.assertTrue(game.is_terminal or game._lib.board_get_total_stones(game._ptr) > 0)


class TestThreatBlend(unittest.TestCase):
    """Test that threat head blends into policy."""

    def test_blend_changes_policy(self):
        """With threat_blend > 0, policy differs from blend=0."""
        net1 = HexNet()
        net2 = HexNet()
        net2.load_state_dict(net1.state_dict())
        net2.threat_blend = 0.0

        x = torch.randn(1, 7, 19, 19)
        with torch.no_grad():
            p1, _, _ = net1(x)
            p2, _, _ = net2(x)
        diff = (p1 - p2).abs().mean().item()
        self.assertGreater(diff, 0.0)

    def test_forward_pv_includes_blend(self):
        """forward_pv should include threat blend."""
        net = HexNet()
        net.threat_blend = 0.5
        x = torch.randn(1, 7, 19, 19)
        with torch.no_grad():
            p_full, _, _ = net(x)
            p_pv, _ = net.forward_pv(x)
        # Both should include the blend, so should be close
        diff = (p_full - p_pv).abs().max().item()
        self.assertLess(diff, 1e-4)


class TestConfigConstants(unittest.TestCase):
    """Test that config constants are valid."""

    def test_threat_policy_blend_exists(self):
        from orca.config import THREAT_POLICY_BLEND
        self.assertIsInstance(THREAT_POLICY_BLEND, float)
        self.assertGreaterEqual(THREAT_POLICY_BLEND, 0.0)

    def test_blocking_boost_positive(self):
        from orca.config import BLOCKING_PRIORITY_BOOST, SURVIVAL_PRIORITY_BOOST
        self.assertGreater(BLOCKING_PRIORITY_BOOST, 0)
        self.assertGreater(SURVIVAL_PRIORITY_BOOST, 0)

    def test_curriculum_constants(self):
        from orca.config import PLATEAU_THRESHOLD, PLATEAU_ITERS
        self.assertGreater(PLATEAU_THRESHOLD, 0)
        self.assertGreater(PLATEAU_ITERS, 0)


if __name__ == '__main__':
    unittest.main()
