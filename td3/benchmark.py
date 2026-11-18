import time
import matplotlib.pyplot as plt
from game import create_board
import algorithms

def benchmark(board, algorithm_name, **kwargs):
    start = time.time()

    if algorithm_name == "minimax":
        res = algorithms.minimax(board, **kwargs)
    elif algorithm_name == "alphabeta":
        res = algorithms.alphabeta(board, **kwargs)
    elif algorithm_name == "mcts":
        res = algorithms.mcts(board, **kwargs)
    else:
        raise ValueError(f"Algorithme inconnu")

    execution_time = time.time() - start

    return {
        "algorithm": algorithm_name,
        "move": res.get("move"),
        "score": res.get("score"),
        "nodes": res.get("nodes_explored"),
        "time": execution_time
    }

board = create_board()

result_minmax = [benchmark(board, "minimax", depth=d) for d in range(1, 6)]
result_alphabeta = [benchmark(board, "alphabeta", depth=d) for d in range(1, 6)]
result_mcts = [benchmark(board, "mcts", num_simulations=d, seed=42) for d in [100, 500, 1000, 2000]]

# Graphiques
d_val = list(range(1, 6))
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(d_val, [r['nodes'] for r in result_minmax], marker='1', label='Minimax')
plt.plot(d_val, [r['nodes'] for r in result_alphabeta], marker='2', label='Alpha-Beta')
plt.xlabel('depth')
plt.ylabel('Nœuds explorés')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([100, 500, 1000, 2000], [r['time'] for r in result_mcts], color='green')
plt.xlabel('Simulations MCTS')
plt.ylabel('Temps')
plt.grid(True)

plt.tight_layout()
plt.savefig('benchmark_results.png')
plt.show()

plt.savefig("graphique.pdf", format="pdf", bbox_inches="tight")