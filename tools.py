import csv
import json
from typing import List
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


def plot_all_predictions(results: List) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=120)
    axes = axes.flatten()
    for idx, (name, y_test, y_pred) in enumerate(results):
        axes[idx].scatter(y_test, y_pred, alpha=0.6)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[idx].set_xlabel("Real IMDb")
        axes[idx].set_ylabel("Predicted IMDb")
        axes[idx].set_title(f"{name}")
    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/all_predictions.png')
    plt.show()
    plt.close()


def plot_feature_importances(models, X) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plotted = False

    for idx, (name, pipeline) in enumerate(models.items()):
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices_desc = importances.argsort()[::-1]
            indices_asc = importances.argsort()
            top_n = min(10, len(X.columns))

            # Top 10 most important
            shown_indices_top = indices_desc[:top_n]
            axes[0].plot(range(top_n), importances[shown_indices_top], marker='o', label=name)
            axes[0].set_xticks(range(top_n))
            axes[0].set_xticklabels(X.columns[shown_indices_top], rotation=60)
            axes[0].set_title('Top 10 Feature Importances')
            axes[0].set_ylabel('Importance')

            # Top 10 least
            shown_indices_least = indices_asc[:top_n]
            axes[1].plot(range(top_n), importances[shown_indices_least], marker='o', label=name)
            axes[1].set_xticks(range(top_n))
            axes[1].set_xticklabels(X.columns[shown_indices_least], rotation=60)
            axes[1].set_title('Least 10 Feature Importances')
            axes[1].set_ylabel('Importance')
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/feature_importance.png')
    plt.show()
    plt.close()


def plot_metrics(metrics, task: str) -> None:
    models = list(metrics.keys())
    num_metrics = len(metrics[models[0]])
    metric_names = [name for name, _ in metrics[models[0]]]

    first_metric_values = [metrics[m][0][1] for m in models]
    sorted_indices = np.argsort(first_metric_values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]

    data = []
    for i in range(num_metrics):
        metric_values = [metrics[m][i][1] for m in sorted_models]
        data.append(metric_values)

    x = np.arange(len(sorted_models))

    fig, axes = plt.subplots(num_metrics, 1, figsize=(6, 3*num_metrics), sharex=False)
    if num_metrics == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(x, data[i], color="steelblue", width=0.3)
        ax.set_ylabel(metric_names[i])
        ax.set_title(metric_names[i])
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_models)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Metrics Across Models", fontsize=16)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(f'plots/{task}_metrics_plot.png')
    plt.show()
    plt.close()


def plot_corr(corr, task, y):
    corr_df = corr.to_frame(name='correlation')
    plt.figure(figsize=(18,6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.plot("correlation")
    plt.xlabel(y)
    plt.title(f'Correlation with {y}', fontsize=12)
    plot_path = os.path.join("plots", f"{task}_corr.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results, task, experiment):
    log_dir = os.path.join('results')
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{task}_{experiment}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

