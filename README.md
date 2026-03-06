# Movie Analysis: Multi-label Classification & Regression

This repository contains project that explores movie metadata through two distinct lenses: **Multi-label Classification** to predict multiple movie genres for a single film and **Regression** to estimate IMDB scores.

### Data
It contains movie information like title, actors, genre, IMDB score. It contains several object columns.

![data](imgs/data.png)

### 0. Preprocessing of data
In preparation stage this steps has been done:
- Dropping nans (only a few of them where here). 
- Casting columns from object
- Filling values (median - outlier resistance, mean-when data are symmetrical - human high)
- Removing duplicates
- Dropping odd values like (Color: Green and Yellow, which was assigned to only one movie)
- Dropping outliers
- One hot encoding of categorical columns (like: language, country, content_rating)
- One hot encoding of multilabel columns (like: genres, keywords) - selected top k categories, assigned OTHER category to the rest

## Data Pipeline

* **Feature Engineering:** Processing the `genres` string (e.g., `Action|Sci-Fi`) into a binary matrix.
* **Text Processing:** Basic NLP techniques applied to `plot_keywords`.
* **Scaling:** `StandardScaler` applied to numerical features like `budget`, `gross`, and `facebook_likes` for Neural Network stability.


### 1. Multi-label Classification (Genre Prediction)
Unlike standard classification, this task handles overlapping labels:
* **Approach:** One-vs-Rest (OvR) strategy for classical models and multi-output architectures for Neural Networks.
* **Classic ML:** Logistic Regression, Decision Tree, Random Forest, and XGBoost.
* **Deep Learning:** * **Output Layer:** `Sigmoid` activation function to allow multiple independent probabilities.
    * **Loss Function:** `Binary Crossentropy` applied across all label outputs.
    * **Preprocessing:** MultiLabelBinarizer for transforming the `genres` column.

<div align="center">
   <img src="plots/classification_metrics_plot.png" width="700px" alt="Classification Metrics Plot">
</div>

### 2. Regression (IMDB Score Prediction)
Predicting the exact numerical rating:
* **Classic ML:** Linear Regression, Decision Tree, Random Forest Regressor, and XGBoost.
* **Deep Learning:** * **Architecture:** Multi-layer Perceptron (MLP) with `ReLU` activations.
    * **Output Layer:** Linear activation for continuous value prediction.
    * **Loss Function:** Mean Squared Error (MSE).

      
<div align="center">
   <img src="plots/all_predictions.png" width="700px" alt="Regression Predictions vs Real">
   <img src="plots/regression_metrics_plot.png" width="700px" alt="Regression Metrics Plot">
</div>


## Evaluation Metrics

* **Classification:** Hamming Loss and F1-Score Macro
* **Regression:** Root Mean Absolute Error (RMAE) and $R^2$ Score.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/metrych-creator/cinemas.git](https://github.com/metrych-creator/cinemas.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```
3.  **Run the analysis:**
    Execute the `preprocessing.ipynb` notebook to see data preprocessing steps.
    Execute the `classical_ML.ipynb` notebook to see classic ML methods like: Decision Tree, training, and evaluation steps.
    Execute the `classification_network.ipynb` notebook to see multilabel classification neural network training, and evaluation steps.
    Execute the `regression_network.ipynb` notebook to see regression neural network training, and evaluation steps.
---
