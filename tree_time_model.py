import json
import numpy as np
from scipy.optimize import nnls

L_SET = [256, 320, 400, 500, 625, 768, 896, 1024, 1152, 1296, 1458, 1600, 1792, 2048, 2176, 2448, 2754, 3000]

MIN_N_DATA_POINTS = 16
MAX_N_DATA_POINTS = 256

def _membound_sum(chain_lens, bound):
    return sum(max(bound - l, 0) for l in chain_lens)

def _extract_stats(stats, L):
    return stats["n_tree_tokens"], stats["sum_prefix_len"], stats["sum_depth"], _membound_sum(stats["chain_lens"], L)

class TreeTimeModel:
    def __init__(self):
        # T = a * n_tree_tokens + b * sum_prefix_len + c * sum_depth + d * sum_membound
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.L = None
        self.data = []

    def fit(self):
        result = []

        for L in L_SET:
            X = []
            Y = []

            for stats in self.data:
                T = stats["time"]
                n_tree_tokens, sum_prefix_len, sum_depth, sum_membound = _extract_stats(stats, L)

                X.append([n_tree_tokens, sum_prefix_len, sum_depth, sum_membound])
                Y.append(T)

            X = np.array(X)
            Y = np.array(Y)
            
            coeffs, rnorm = nnls(X, Y)

            T_pred = X @ coeffs
            mse = np.mean((T_pred - Y) ** 2)

            result.append((mse, coeffs, L))
            # print(f"Fit with L={L}: mse={mse:.6e}, coeffs={coeffs}")

        result.sort(key=lambda x: x[0])
        best_mse, best_coeffs, best_L = result[0]
        self.a, self.b, self.c, self.d = best_coeffs
        self.L = best_L

        return best_mse
    
    def add_data(self, data):
        self.data.extend(data)
        if len(self.data) > MAX_N_DATA_POINTS:
            self.data = self.data[-MAX_N_DATA_POINTS:]
        if len(self.data) >= MIN_N_DATA_POINTS:
            self.fit()

    def pred(self, stats):
        if self.a is None:
            return stats["n_tree_tokens"]
        n_tree_tokens, sum_prefix_len, sum_depth, sum_membound = _extract_stats(stats, self.L)
        return (self.a * n_tree_tokens +
                self.b * sum_prefix_len +
                self.c * sum_depth +
                self.d * sum_membound)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", type=str, required=True)
    args = parser.parse_args()

    datas = []
    with open(args.stats_file, "r") as f:
        for line in f:
            stats = json.loads(line)
            datas.append(stats)

    train_data = datas
    test_data = datas

    time_model = TreeTimeModel()
    time_model.add_data(train_data)

    print(f"Fitted time model:")
    print(f"  a (n_tree_tokens)  = {time_model.a:.6e}")
    print(f"  b (sum_prefix_len) = {time_model.b:.6e}")
    print(f"  c (sum_depth)      = {time_model.c:.6e}")
    print(f"  d (sum_membound)   = {time_model.d:.6e}")
    print(f"  L (membound)       = {time_model.L}")

    avg_err = 0.0
    for i, stats in enumerate(test_data):
        T_true = stats["time"]
        T_pred = time_model.pred(stats)
        err = abs(T_true - T_pred) / T_true * 100
        avg_err += err
        print(f"Test sample {i}: True time = {T_true:.2f}, Pred time = {T_pred:.2f}, Error = {err:.2f}%")

    avg_err /= len(test_data)
    print(f"Average relative error on test set: {avg_err:.2f}%")

"""
python tree_time_model.py --stats-file stats/Qwen3-8B-K8-DFS-backward.jsonl
"""