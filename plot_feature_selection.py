import json
import os
import matplotlib.pyplot as plt
import numpy as np

files = {
    "all_features_classification": "results/classification_all_features.json",
    "corr_selected_features_classification": "results/classification_corr_selected_features.json",
    "all_features_regression": "results/regression_all_features.json",
    "corr_selected_features_regression": "results/regression_corr_selected_features.json"
}

data = {}
for key, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[key] = json.load(f)

metrics_dict = {
    "classification": ["Macro F1", "Hamming Loss"],
    "regression": ["R2", "RMSE"]
}

datasets = ["regression", "classification"]

# selct metrics
def get_metric_values(metric_index, dataset_type):
    models = list(data[f"all_features_{dataset_type}"].keys())
    values = []
    for model in models:
        values.append([
            data[f"all_features_{dataset_type}"][model][metric_index][1],
            data[f"corr_selected_features_{dataset_type}"][model][metric_index][1]
        ])
    return np.array(values), models

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Regression and Classification Results", fontsize=18)

width = 0.35

for row, dataset in enumerate(datasets):
    metrics = metrics_dict[dataset]
    for col, metric in enumerate(metrics):
        values, models = get_metric_values(col, dataset)
        x = np.arange(len(models))
        ax = axes[row, col]
        bars1 = ax.bar(x - width/2, values[:,0], width)
        bars2 = ax.bar(x + width/2, values[:,1], width)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_title(f"{dataset.capitalize()} - {metric}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)


handles, labels = bars1, ["all_features", "corr_selected_features"]
fig.legend(handles=[bars1, bars2], labels=["all_features", "corr_selected_features"], loc='upper right', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93])

plot_path = os.path.join("plots", "regression_classification_comparison.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()