from typing import List
import matplotlib.pyplot as plt


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
    plt.show()


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

            plotted = True

    if plotted:
        axes[0].legend()
        axes[1].legend()
    else:
        print("No models with feature_importances_ to plot.")

    plt.tight_layout()
    plt.show()

