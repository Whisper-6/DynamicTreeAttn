import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", type=str, required=True)
    args = parser.parse_args()

    datas = []
    with open(args.stats_file, "r") as f:
        for line in f:
            stats = json.loads(line)
            name = stats["name"]
            time = stats["time"]
            datas.append((name, time))

    time_set = {}
    total_bin_time = 0.0

    # name 的格式形如 xxx_binx
    for name, time in datas:
        prefix, binx = name.rsplit("_bin", 1)
        if prefix not in time_set:
            time_set[prefix] = 0.0
        total_bin_time += time
        time_set[prefix] = max(time_set[prefix], time)

    total_call_time = 0.0
    for prefix, time in time_set.items():
        total_call_time += time

    print(f"Total call time: {total_call_time:.4f} s")
    print(f"Total bin time: {total_bin_time:.4f} s")


"""
python calc_time.py --stats-file stats/Qwen3-1.7B-K8-DFS-TM-backward.jsonl
"""