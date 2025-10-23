"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: Evaluation
-----------------------------------------------------------------------------------------------------------------------
Process:
1. Reads models from config toml to display results
2. Consolidate results
3. Summarize algorithms and best performances

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import json
import Global
from collections import defaultdict


def create_comparison_table(model_list, output_dir):
    results_data = defaultdict(list)

    print("\n--- Model Comparison Table ---")

    for model_name in model_list:
        results_path = Global.get_project_path(f"{output_dir}/{model_name}_results.json", levels_up_to_root=0)

        try:
            with open(results_path, 'r') as f:
                metrics = json.load(f)

            results_data['Model'].append(model_name)
            results_data['MAE'].append(metrics.get('MAE'))
            results_data['RMSE'].append(metrics.get('RMSE'))
            results_data['MAPE'].append(metrics.get('MAPE'))

        except FileNotFoundError:
            print(f"[WARNING] Results file not found for {model_name}. Skipping...")
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON for {model_name}. Skipping...")

    if not results_data:
        print("No model results were loaded for comparison")
        return None

    comparison_df = pd.DataFrame(results_data)

    comparison_df = comparison_df.sort_values(by='RMSE', ascending=True).set_index('Model')

    print(comparison_df.to_markdown(floatfmt=".4f"))

    return comparison_df


def main():
    config = Global.load_config()
    if not config:
        return

    model_list = config["model"]["algo"]["algo_list"]
    output_dir = config["model"]["data_prep"]["path"]

    comparison_df = create_comparison_table(model_list, output_dir)

    if not comparison_df.empty:
        best_model_name = comparison_df.index[0]
        best_rmse = comparison_df.iloc[0]['RMSE']
        print(f"\n[SUMMARY] The best performing model based on RMSE is: {best_model_name} (RMSE: {best_rmse:.4f})")

    print("\n--- Post-Processing Complete ---")


if __name__ == '__main__':
    main()