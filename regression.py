import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, root_mean_squared_error, hamming_loss
from tools import plot_all_predictions, plot_corr, plot_feature_importances, plot_metrics, save_results
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import os

df = pd.read_pickle('cinemas_cleaned.pkl')
df = df[df.select_dtypes(include=['number']).columns.tolist()]

experiment = 'corr_selected_features'


def run_models(X: pd.DataFrame, y, models, param_grids, task='regression', show_best_params:bool=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    best_models = {}
    results = []
    metrics = {}

    for name, model in models.items():
        if task == 'regression':
            selector = SelectKBest(score_func = f_regression, k = 'all')
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        elif task == 'classification':
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', MultiOutputClassifier(model))
            ])
        else:
            raise ValueError("task must be 'regression' or 'classification'")

        grid = param_grids.get(name, {})

        if grid:
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                cv=3,
                scoring='neg_root_mean_squared_error' if task=='regression' else 'f1_macro',
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            if show_best_params:
                print(f"{name} best params: {search.best_params_}")
            
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe

        best_models[name] = best_model
        y_pred = best_model.predict(X_test)

        if task == 'regression':
            metrics[name]=[('R^2', round(r2_score(y_test, y_pred), 2)), 
                           ('RMSE', round(root_mean_squared_error(y_test, y_pred), 2))]
        else:
            metrics[name] = [('Macro F1', round(f1_score(y_test, y_pred, average='macro', zero_division=0), 2)),
                             ('Hamming Loss' , round(hamming_loss(y_test, y_pred), 2))]

        results.append((name, y_test, y_pred))

    return best_models, results, metrics



# ------------------------------------------------------ regression ------------------------------------------------------
task = 'regression'
y_reg = df['imdb_score']
# X_reg = df.drop(columns=['imdb_score'])
X_reg = df.loc[:, ['cast_total_facebook_likes','movie_facebook_likes','num_voted_users','duration','actor_1_facebook_likes',
'actor_2_facebook_likes','num_critic_for_reviews','color', 'genres_Drama', 'genres_Horror', 'gross', 'title_year']]

# corr
# strong_corr = corr_series[corr_series.abs() > 0.15]
# print("Strong correlations (>0.15): ", strong_corr)
corr_series = X_reg.apply(lambda col: col.corr(y_reg))

models_reg = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
}

param_grids_reg = {
    "Decision Tree": {"model__max_depth": [3, 5, 10, None],
                      "model__min_samples_split": [2, 5, 10, 20, 50]},
    "Random Forest": {"model__n_estimators": [50, 100, 200],
                      "model__max_depth": [3, 5, 10, None],
                      "model__min_samples_split": [2, 5, 10]},
    "XGBoost": {"model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 10],
                "model__learning_rate": [0.01, 0.1, 0.2]},
    "Linear Regression": {}
}

best_models_reg, results_reg, metrics_reg = run_models(X_reg, y_reg, models_reg, param_grids_reg, task='regression', show_best_params=False)
plot_all_predictions(results_reg)
plot_metrics(metrics_reg, task)
print(metrics_reg)
save_results(metrics_reg, task, experiment)
plot_corr(corr_series, task, 'IMDB Score')


# ------------------------------------------------------ classification ------------------------------------------------------
task = 'classification'
genres = [col for col in df.columns if col.startswith('genres_')]
y_class = df[genres].copy()
# X_class = df.drop(columns=genres)
X_class = df.loc[:, ['content_rating_PG','content_rating_R','gross','facenumber_in_poster','keywords_other','duration','keywords_vegetarian','num_critic_for_reviews',
                 'keywords_alien','keywords_battle','imdb_score','budget','keywords_party','keywords_dog','keywords_singer','keywords_wedding']]

# corr
corrs = {}
for col_x in X_class.columns:
    feature_corrs = []
    for col_y in y_class.columns:
        corr = X_class[col_x].corr(y_class[col_y])
        feature_corrs.append(corr)
    corrs[col_x] = np.nanmean(feature_corrs)
corr_series = pd.Series(corrs).sort_values(key=np.abs, ascending=False)
# strong_corr = corr_series[corr_series.abs() > 0.01]
# print("Strong correlations (>0.01):", strong_corr)

models_class = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
}

param_grids_class = {
    "Decision Tree": {"model__estimator__max_depth": [3, 5, 10, None],
                      "model__estimator__min_samples_split": [2, 5, 10, 20, 50]},
    "Random Forest": {"model__estimator__n_estimators": [50, 100, 200],
                      "model__estimator__max_depth": [3, 5, 10, None],
                      "model__estimator__min_samples_split": [2, 5, 10]},
    "XGBoost": {"model__estimator__n_estimators": [50, 100, 200],
                "model__estimator__max_depth": [3, 5, 10],
                "model__estimator__learning_rate": [0.01, 0.1, 0.2]},
    "Logistic Regression": {}
}

best_models_class, results_class, metrics_class = run_models(X_class, y_class, models_class, param_grids_class, task=task, show_best_params=False)
plot_metrics(metrics_class, task)
print(metrics_class)
save_results(metrics_class, task, experiment)
plot_corr(corr_series, task, 'Avg Genres')