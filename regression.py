import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from tools import plot_all_predictions, plot_feature_importances



df = pd.read_pickle('cinemas_cleaned.pkl')
df = df[df.select_dtypes(include=['number']).columns.tolist()]

y = df['imdb_score']
X = df.drop(columns=['imdb_score'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models_reg = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42)
}

models_class = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42)
}

param_grids = {
    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10, 20, 50]
    },
    "Random Forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 10],
        "model__learning_rate": [0.01, 0.1, 0.2]
    },
    "Linear Regression": {}
}

best_models = {}
results = []

for name, model in models_reg.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    grid = param_grids[name]
    if grid:
        search = GridSearchCV(pipe, grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"{name} best params: {search.best_params_}")
    else:
        pipe.fit(X_train, y_train)
        best_model = pipe
    
    best_models[name] = best_model
    y_pred = best_model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.2f}")
    
    results.append((name, y_test, y_pred))


X_train_df = pd.DataFrame(X_train, columns=df.drop(columns=['imdb_score']).columns)
plot_all_predictions(results)
plot_feature_importances(best_models, X_train_df)
