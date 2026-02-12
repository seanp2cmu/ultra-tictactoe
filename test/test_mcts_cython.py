"""Benchmark: Python Node vs Cython NodeCy."""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
from game import Board
from ai.mcts import Node
from ai.mcts.node_cy import NodeCy


def benchmark_node_operations(NodeClass, name: str, iterations: int = 1000):
    """Benchmark basic node operations."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    board = Board()
    
    # 1. Node creation
    t0 = time.perf_counter()
    for _ in range(iterations):
        node = NodeClass(board)
    create_time = time.perf_counter() - t0
    print(f"Node creation ({iterations}x): {create_time*1000:.2f}ms")
    
    # 2. Expand
    node = NodeClass(board)
    action_probs = {i: 1.0/81 for i in range(81)}
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        node = NodeClass(board)
        node.expand(action_probs)
    expand_time = time.perf_counter() - t0
    print(f"Expand ({iterations}x): {expand_time*1000:.2f}ms")
    
    # 3. Select child (after expand)
    node = NodeClass(board)
    node.expand(action_probs)
    # Add some visits
    for child in node.children.values():
        child.visits = np.random.randint(1, 100)
        child.value_sum = np.random.random() * child.visits
    node.visits = sum(c.visits for c in node.children.values())
    
    t0 = time.perf_counter()
    for _ in range(iterations * 10):
        node.select_child(1.0)
    select_time = time.perf_counter() - t0
    print(f"Select child ({iterations*10}x): {select_time*1000:.2f}ms")
    
    # 4. Update
    node = NodeClass(board)
    t0 = time.perf_counter()
    for _ in range(iterations * 10):
        node.update(0.5)
    update_time = time.perf_counter() - t0
    print(f"Update ({iterations*10}x): {update_time*1000:.2f}ms")
    
    # 5. is_terminal check
    t0 = time.perf_counter()
    for _ in range(iterations * 10):
        node.is_terminal()
    terminal_time = time.perf_counter() - t0
    print(f"is_terminal ({iterations*10}x): {terminal_time*1000:.2f}ms")
    
    return {
        'create': create_time,
        'expand': expand_time,
        'select': select_time,
        'update': update_time,
        'terminal': terminal_time,
    }


def benchmark_mcts_simulation(NodeClass, name: str, num_simulations: int = 100):
    """Simulate MCTS-like tree traversal."""
    print(f"\n{'='*50}")
    print(f"MCTS Simulation: {name} ({num_simulations} sims)")
    print(f"{'='*50}")
    
    board = Board()
    root = NodeClass(board)
    action_probs = {i: 1.0/81 for i in range(81)}
    
    t0 = time.perf_counter()
    
    for _ in range(num_simulations):
        # Selection
        node = root
        path = [node]
        
        while node.is_expanded() and not node.is_terminal():
            _, node = node.select_child(1.0)
            path.append(node)
        
        # Expansion
        if not node.is_terminal() and not node.is_expanded():
            node.expand(action_probs)
        
        # Backup
        value = np.random.random() * 2 - 1
        for n in reversed(path):
            n.update(value)
            value = -value
    
    total_time = time.perf_counter() - t0
    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"Per simulation: {total_time/num_simulations*1000:.3f}ms")
    print(f"Tree size: {count_nodes(root)} nodes")
    
    return total_time


def count_nodes(node):
    """Count total nodes in tree."""
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


if __name__ == '__main__':
    print("="*60)
    print("MCTS Cython Benchmark")
    print("="*60)
    
    # Basic operations
    py_times = benchmark_node_operations(Node, "Python Node")
    cy_times = benchmark_node_operations(NodeCy, "Cython NodeCy")
    
    print("\n" + "="*60)
    print("SPEEDUP (Cython vs Python)")
    print("="*60)
    for op in py_times:
        speedup = py_times[op] / cy_times[op] if cy_times[op] > 0 else 0
        print(f"  {op}: {speedup:.2f}x")
    
    # MCTS simulation
    print("\n")
    py_sim = benchmark_mcts_simulation(Node, "Python Node", 500)
    cy_sim = benchmark_mcts_simulation(NodeCy, "Cython NodeCy", 500)
    
    print("\n" + "="*60)
    print(f"MCTS SPEEDUP: {py_sim/cy_sim:.2f}x")
    print("="*60)
