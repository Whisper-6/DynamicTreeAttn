import json
import numpy as np
from scipy.optimize import nnls

class TreeTimeModel:
    MIN_N_DATA_POINTS = 16
    MAX_N_DATA_POINTS = 1024

    def __init__(self):
        # T = c_0 * n_leaf_sequences + c_1 * n_tree_tokens + c_2 * n_f1_tokens + c_3 * sum_prefix_len + c_4 * sum_depth
        self.coeffs = None
        self.data = []

    def fit(self):
        X, Y = [], []
        for stats in self.data:
            # X.append([0, stats["n_tree_tokens"], 0, 0, 0])
            X.append([
                stats["n_leaf_sequences"],
                stats["n_tree_tokens"],
                stats.get("n_f1_tokens", 0),
                stats["sum_prefix_len"],
                stats["sum_depth"]
            ])
            Y.append(stats["time"])

        X, Y = np.array(X), np.array(Y)
        self.coeffs, _ = nnls(X, Y)

        T_pred = X @ self.coeffs
        mse = np.mean((T_pred - Y) ** 2)
        return mse
    
    def add_data(self, data):
        self.data.extend(data)
        if len(self.data) > self.MAX_N_DATA_POINTS:
            self.data = self.data[-self.MAX_N_DATA_POINTS:]
        if len(self.data) >= self.MIN_N_DATA_POINTS:
            self.fit()

    def pred(self, stats):
        if self.coeffs is None:
            return stats["n_tree_tokens"]
        return (self.coeffs[0] * stats["n_leaf_sequences"] +
                self.coeffs[1] * stats["n_tree_tokens"] +
                self.coeffs[2] * stats.get("n_f1_tokens", 0) +
                self.coeffs[3] * stats["sum_prefix_len"] +
                self.coeffs[4] * stats["sum_depth"])

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
    print(f"  c_0 (n_leaf_sequences) = {time_model.coeffs[0]:.6e}")
    print(f"  c_1 (n_tree_tokens)    = {time_model.coeffs[1]:.6e}")
    print(f"  c_2 (n_f1_tokens)      = {time_model.coeffs[2]:.6e}")
    print(f"  c_3 (sum_prefix_len)   = {time_model.coeffs[3]:.6e}")
    print(f"  c_4 (sum_depth)        = {time_model.coeffs[4]:.6e}")

    avg_err = 0.0
    for i, stats in enumerate(test_data):
        T_true = stats["time"]
        T_pred = time_model.pred(stats)
        err = abs(T_true - T_pred) / T_true * 100
        avg_err += err
        # print(f"Test sample {i}: True time = {T_true:.2f}, Pred time = {T_pred:.2f}, Error = {err:.2f}%")

    avg_err /= len(test_data)
    print(f"Average relative error on test set: {avg_err:.2f}%")

"""
python tree_time_model.py --stats-file stats/Qwen3-4B-K8-woTM-backward.jsonl
"""