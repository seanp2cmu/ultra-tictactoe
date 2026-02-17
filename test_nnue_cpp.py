"""Test C++ NNUE engine: features, model inference, search consistency."""
import time
import struct
import tempfile
import numpy as np
import torch

from uttt_cpp import Board as BoardCpp
import nnue_cpp

from nnue.core.model import NNUE
from nnue.core.features import extract_features as py_extract_features, NUM_FEATURES
from nnue.cpp.export_weights import export_weights


def make_test_board(moves=None):
    """Create a C++ Board with some moves played."""
    b = BoardCpp()
    if moves:
        for r, c in moves:
            b.make_move(r, c)
    return b


# ─── Test 1: Feature extraction consistency ─────────────────────

def test_features():
    """Compare Python vs C++ feature extraction."""
    print("=" * 60)
    print("Test 1: Feature extraction consistency")
    print("=" * 60)

    test_cases = [
        ("Empty board", []),
        ("One move", [(4, 4)]),
        ("Several moves", [(4, 4), (3, 3), (0, 0), (0, 1)]),
        ("Many moves", [(4, 4), (3, 3), (0, 0), (0, 1), (0, 4), (3, 4),
                        (6, 3), (0, 3), (0, 8)]),
    ]

    passed = 0
    for name, moves in test_cases:
        b = make_test_board(moves)

        # C++ features
        cpp_stm, cpp_nstm = nnue_cpp.extract_features(b)

        # Python features (need same board)
        py_stm, py_nstm = py_extract_features(b)

        cpp_stm_set = set(cpp_stm)
        cpp_nstm_set = set(cpp_nstm)
        py_stm_set = set(py_stm)
        py_nstm_set = set(py_nstm)

        match = (cpp_stm_set == py_stm_set) and (cpp_nstm_set == py_nstm_set)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {name}: STM={len(cpp_stm)} feats, NSTM={len(cpp_nstm)} feats")

        if not match:
            print(f"    STM diff: C++only={cpp_stm_set - py_stm_set}, Pyonly={py_stm_set - cpp_stm_set}")
            print(f"    NSTM diff: C++only={cpp_nstm_set - py_nstm_set}, Pyonly={py_nstm_set - cpp_nstm_set}")
        else:
            passed += 1

    print(f"  → {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


# ─── Test 2: Model inference consistency ─────────────────────────

def test_model_inference():
    """Compare Python vs C++ model evaluation."""
    print("=" * 60)
    print("Test 2: Model inference consistency")
    print("=" * 60)

    # Create a Python model with random weights
    torch.manual_seed(42)
    py_model = NNUE(accumulator_size=256, hidden1_size=32, hidden2_size=32)
    py_model.eval()

    # Export to binary
    with tempfile.NamedTemporaryFile(suffix='.nnue', delete=False) as f:
        weight_path = f.name
    export_weights(py_model, weight_path)

    # Load in C++
    cpp_model = nnue_cpp.NNUEModel()
    cpp_model.load(weight_path)
    print(f"  C++ model loaded: acc={cpp_model.accumulator_size}, h1={cpp_model.hidden1_size}, h2={cpp_model.hidden2_size}")

    test_cases = [
        ("Empty board", []),
        ("One move", [(4, 4)]),
        ("Several moves", [(4, 4), (3, 3), (0, 0), (0, 1)]),
        ("Many moves", [(4, 4), (3, 3), (0, 0), (0, 1), (0, 4), (3, 4),
                        (6, 3), (0, 3), (0, 8)]),
    ]

    max_err = 0.0
    passed = 0
    for name, moves in test_cases:
        b = make_test_board(moves)

        # Python evaluation
        py_val = py_model.evaluate(b)

        # C++ evaluation
        cpp_val = cpp_model.evaluate_board(b)

        err = abs(py_val - cpp_val)
        max_err = max(max_err, err)
        ok = err < 1e-5
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: py={py_val:.6f}, cpp={cpp_val:.6f}, diff={err:.2e}")
        if ok:
            passed += 1

    print(f"  → {passed}/{len(test_cases)} passed (max error: {max_err:.2e})\n")
    return passed == len(test_cases)


# ─── Test 3: Search engine ───────────────────────────────────────

def test_search():
    """Test C++ search engine produces valid moves."""
    print("=" * 60)
    print("Test 3: Search engine (valid moves + depth)")
    print("=" * 60)

    torch.manual_seed(42)
    py_model = NNUE()
    py_model.eval()

    with tempfile.NamedTemporaryFile(suffix='.nnue', delete=False) as f:
        weight_path = f.name
    export_weights(py_model, weight_path)

    cpp_model = nnue_cpp.NNUEModel()
    cpp_model.load(weight_path)
    engine = nnue_cpp.NNUESearchEngine(cpp_model, 16)

    test_cases = [
        ("Empty board d4", [], 4),
        ("After (4,4) d4", [(4, 4)], 4),
        ("Several moves d6", [(4, 4), (3, 3), (0, 0), (0, 1)], 6),
    ]

    passed = 0
    for name, moves, depth in test_cases:
        b = make_test_board(moves)
        legal = set(b.get_legal_moves())

        result = engine.search(b, depth)
        move = (result.best_r, result.best_c)
        valid = move in legal
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {name}: move=({result.best_r},{result.best_c}), "
              f"score={result.score:.4f}, depth={result.depth}, "
              f"nodes={result.nodes}, time={result.time_ms:.1f}ms")
        if valid:
            passed += 1

    print(f"  → {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


# ─── Test 4: Benchmark Python vs C++ ────────────────────────────

def benchmark():
    """Compare Python vs C++ search speed."""
    print("=" * 60)
    print("Test 4: Benchmark Python vs C++ search")
    print("=" * 60)

    torch.manual_seed(42)
    py_model = NNUE()
    py_model.eval()

    with tempfile.NamedTemporaryFile(suffix='.nnue', delete=False) as f:
        weight_path = f.name
    export_weights(py_model, weight_path)

    cpp_model = nnue_cpp.NNUEModel()
    cpp_model.load(weight_path)
    cpp_engine = nnue_cpp.NNUESearchEngine(cpp_model, 16)

    b = make_test_board([(4, 4), (3, 3), (0, 0), (0, 1)])

    for depth in [4, 6, 8, 10]:
        cpp_engine.clear()
        result = cpp_engine.search(b, depth)
        nps = int(result.nodes / (result.time_ms / 1000)) if result.time_ms > 0 else 0
        print(f"  Depth {depth}: {result.nodes:>8} nodes, {result.time_ms:>8.1f}ms, {nps:>8} nps")

    print()


# ─── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_pass = True
    all_pass &= test_features()
    all_pass &= test_model_inference()
    all_pass &= test_search()
    benchmark()

    if all_pass:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
