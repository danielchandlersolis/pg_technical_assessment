"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: Training
-----------------------------------------------------------------------------------------------------------------------
Process:
1. Reads information in config toml file regarding data, algorithms, 
2. Runs functions for data split into train/test/validation stored in Global.py
3. Executes the model training based on hyperparameters defined and stored model artifact and results

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import joblib
import json

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
import Global


def get_model_and_params(model_name):

    if model_name == 'LinearRegression':
        return LinearRegression(), {}
    elif model_name == 'Ridge':
        return Ridge(), {'alpha': [0.1, 1.0, 10.0]}
    elif model_name == 'Lasso':
        return Lasso(random_state=42), {'alpha': [0.01, 0.1, 1.0]}
    elif model_name == 'kNN':

        return KNeighborsRegressor(), {'n_neighbors': [5, 7, 9, 11]}
    elif model_name == 'SVR':

        return SVR(), {
            'C': [1.0, 10.0, 50.0, 100.0],
            'kernel': ['rbf'],
            'gamma': ['scale', 0.01, 0.05, 0.1]
        }
    elif model_name == 'KRR':

        return KernelRidge(), {
            'alpha': [0.5, 5.0, 50.0],
            'kernel': ['rbf'],
            'gamma': [0.01, 0.1, 1.0, 'scale']
        }
    elif model_name == 'RandomForestRegressor':
        return RandomForestRegressor(random_state=10), {
            'n_estimators': [50, 100, 500, 1000],
            'max_depth': [5, 10, 15]
        }
    elif model_name == 'MLPRegressor':
        return MLPRegressor(random_state=10, max_iter=100), {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
        }
    elif model_name == 'XGBoost':
        return XGBRegressor(random_state=10, eval_metric='rmse'), {
            'n_estimators': [50, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0]
        }
    elif model_name == 'CatBoost':

        return CatBoostRegressor(random_state=10, verbose=False), {
            'iterations': [800, 1200, '1500', '2000'],
            'learning_rate': [0.03, 0.05, 0.07],
            'depth': [6, 8, 10]
        }
    else:

        return None, None


def train_and_save_model(model_name, X_train, y_train, X_test, y_test, output_dir, config):

    estimator, param_grid = get_model_and_params(model_name)
    if estimator is None:
        print(f"Skipping model: {model_name} (Definition not found)")
        return

    print(f"\n[TRAINING] Starting training for {model_name}...")

    if "catboost" in model_name.lower():
        categorical_cols = [col for col in config["model"]["features"]["categorical"] if col in X_train.columns]
        print(f"[INFO] CatBoost categorical features: {categorical_cols}")

        from catboost import Pool
        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

        estimator.fit(train_pool, eval_set=test_pool, verbose=False)
        y_pred_test = estimator.predict(test_pool)

    else:
        print(f"[TRAINING] Starting Grid Search for {model_name}...")
        optimizer = GridSearchCV(
            estimator,
            param_grid,
            scoring=config["model"]["training"]["optimization_metric"],
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        optimizer.fit(X_train, y_train)
        estimator = optimizer.best_estimator_
        print(f"[TRAINING] Best parameters found: {optimizer.best_params_}")
        y_pred_test = estimator.predict(X_test)

    test_metrics = Global.calculate_all_metrics(y_test, y_pred_test)
    print(f"[TRAINING] Test Metrics: {test_metrics}")

    model_path = Global.get_project_path(f"{output_dir}/{model_name}_best_model.joblib", levels_up_to_root=0)
    joblib.dump(estimator, model_path)
    print(f"[TRAINING] Model saved to: {model_path}")

    results_path = Global.get_project_path(f"{output_dir}/{model_name}_results.json", levels_up_to_root=0)
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f)
    print(f"[TRAINING] Results saved to: {results_path}")

    eval_df = X_test.copy()
    eval_df["Actual"] = y_test
    eval_df["Predicted"] = y_pred_test
    eval_df["Model"] = model_name

    eval_path = Global.get_project_path(f"{output_dir}/{model_name}_predictions_vs_actuals.csv", 0)
    eval_df.to_csv(eval_path, index=False)

    print(f"[TRAINING] Evaluation DataFrame saved to: {eval_path}")


def main():

    config = Global.load_config()
    if not config:
        return

    data_dir = config["model"]["data_prep"]["path"]
    output_dir = config["model"]["data_prep"]["path"]
    Global.os.makedirs(Global.get_project_path(output_dir, 0), exist_ok=True)

    print("Data Loading and Splitting")

    for model_name in config["model"]["algo"]["algo_list"]:
        print(f"\nLoading data for training: {model_name}")

        X_path = Global.get_project_path(f"{data_dir}/X_{model_name}_train.csv", levels_up_to_root=0)
        y_path = Global.get_project_path(f"{data_dir}/y_{model_name}_train.csv", levels_up_to_root=0)

        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path).iloc[:, 0]
        except FileNotFoundError:
            print(f"Skipping {model_name}: Data files not found. Ensure the preprocessing step was run.")
            continue

        X_train, X_test, X_val, y_train, y_test, y_val = Global.split_data(X, y, config["model"]["training"])

        print(f"Data Split Shapes (Train/Test/Val): {X_train.shape[0]}/{X_test.shape[0]}/{X_val.shape[0]}")

        train_and_save_model(model_name, X_train, y_train, X_test, y_test, output_dir, config)

    print("\nTraining Pipeline Complete")



if __name__ == '__main__':
    main()