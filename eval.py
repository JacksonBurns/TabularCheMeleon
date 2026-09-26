import datetime
import json
import sys

import torch
import pandas as pd
import polaris as po
from polaris.utils.types import TargetType

from inference import TabularCheMeleonRegressor
from featurize import CheMeleonEmbedder

def get_leaderboard_snippet(benchmark_name, current_perf, leaderboard_df, metric):
    """Filters the leaderboard for a specific benchmark and appends the new model."""
    subset = leaderboard_df[leaderboard_df['Benchmark_ID'] == benchmark_name].copy()
    # Add your model as a new row
    new_entry = {'Name': 'TabularCheMeleon', metric: current_perf}
    # Append and sort by score
    lower_is_better = False
    if metric in {'mean_absolute_error', 'mean_squared_error'}:
        lower_is_better = True
    return pd.concat([subset, pd.DataFrame([new_entry])], ignore_index=True).sort_values(by=metric, ascending=lower_is_better).reset_index(drop=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python eval.py /path/to/reg_model.ckpt /path/to/leaderboard.csv")
        exit(1)

    reg_ckpt = sys.argv[1]
    leaderboard_path = sys.argv[2]

    get_chemeleon_embeddings = CheMeleonEmbedder(device="cuda", type="autoencoded")
    reg = TabularCheMeleonRegressor(checkpoint_path=reg_ckpt, device="cuda")

    leaderboard_df = pd.read_csv(leaderboard_path)

    output_file = open("eval_results.md", "w")
    output_file.write(
        f"""# Eval Results

timestamp: {datetime.datetime.now()}
checkpoint: {reg_ckpt}
"""
    )
    performance_dict = {}
    ranks = []
    polaris_benchmarks = (
        "polaris/pkis2-ret-wt-reg-v2",
        "polaris/pkis2-kit-wt-reg-v2",
        "polaris/pkis2-egfr-wt-reg-v2",
        "polaris/adme-fang-solu-1",
        "polaris/adme-fang-rppb-1",
        "polaris/adme-fang-hppb-1",
        "polaris/adme-fang-perm-1",
        "polaris/adme-fang-rclint-1",
        "polaris/adme-fang-hclint-1",
        "tdcommons/lipophilicity-astrazeneca",
        "tdcommons/ppbr-az",
        "tdcommons/clearance-hepatocyte-az",
        "tdcommons/half-life-obach",
        "tdcommons/clearance-microsome-az",
        "tdcommons/vdss-lombardo",
        "tdcommons/caco2-wang",
        "tdcommons/ld50-zhu",
    )
    for benchmark_name in polaris_benchmarks:
        benchmark = po.load_benchmark(benchmark_name)
        smiles_col = list(benchmark.input_cols)[0]
        target_cols = list(benchmark.target_cols)
        train, test = benchmark.get_train_test_split()
        train_df, test_df = train.as_dataframe(), test.as_dataframe()
        task_type = benchmark.target_types[target_cols[0]]

        train_smiles_list = train_df[smiles_col].to_list()
        train_labels_list = train_df[target_cols[0]].to_list()

        # Select checkpoint based on task type
        if task_type != TargetType.REGRESSION:
            print(f"Skipping benchmark {benchmark_name} as it is not a regression task.")
            continue

        predictions = reg.predict_regression(
            x_context=get_chemeleon_embeddings(train_smiles_list),
            y_context=torch.tensor(train_labels_list, dtype=torch.float32),
            x_query=get_chemeleon_embeddings(test_df[smiles_col].to_list())
        )[0]
        torch.cuda.empty_cache()

        results = benchmark.evaluate(predictions.numpy(force=True).flatten()).results
        performance = results.query(
            f"Metric == '{benchmark.main_metric.label}'"
        )["Score"].values[0]

        lb_snippet = get_leaderboard_snippet(benchmark_name, performance, leaderboard_df, benchmark.main_metric.label)
        if lb_snippet.shape[0] >= 5:
            ranks.append(lb_snippet.query("Name == 'TabularCheMeleon'").index[0] + 1)  # +1 for 1-based rank

        output_file.write(
            f"""
## `{benchmark_name}`

### Model Performance
{results.to_markdown()}

### Leaderboard Comparison
{lb_snippet[['Name', benchmark.main_metric.label]].to_markdown(index=False)}
"""
        )
        performance_dict[benchmark_name] = {benchmark.main_metric.label: performance}
    output_file.write(
        f"""
# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries ({len(ranks)}): {sum(ranks) / len(ranks):.2f}

results_dict = {json.dumps(performance_dict, indent=4)}
"""
    )
