"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: contain common functions
-----------------------------------------------------------------------------------------------------------------------
List of functions:
    get_project_path        : Helps to identify any specific path level in the project 
    load_config             : Loads the config toml file and returns it as a dict; error if not found
    log                     : Generate process logs
    initialize_database     : Initialize SQLite database
    resolve_sql_path        : Project path for sql queries
    execute_sql_file        : Execute all statements in the SQL file specified
    log_existing_tables     : Return list of tables created in SQLite database
    random_choice           : Selecting random values from a list
    generate_delivery_person: Generate dummy records for delivery_persons table
    generate_restaurant     : Generate dummy records for restaurants table
    generate_delivery       : Generate dummy records for deliveries table
    generate_order          : Generate dummy records for orders table
    insert_records          : Insert dummy records into tables
    run_select_query        : Execute a select query and print the results if required, returning columns and rows
    run_select_file         : Read a sql file, split into individual statements, and execute each select query

Functions related to preprocessing, modeling, and evaluation:
    add_engineered_features
    impute_features
    one_hot_encode_features
    drop_columns
    scale_all_features
    apply_pca
    prepare_data_for_model
    process_all_models
    split_data
    mean_absolute_percentage_error
    calculate_all_metrics
     
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import sqlite3
import tomllib
import logging
import sys
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


def get_project_path(relative_file_path, levels_up_to_root):
    """
    -------------------------------------------------------------------------------------------------------------------
    Helps to identify any specific path level in the project
    ------------------------------------------------------------------------------------------------------------------
    """
    script_dir = Path(__file__).resolve().parent

    project_root = script_dir
    for _ in range(levels_up_to_root):
        project_root = project_root.parent

    absolute_path = project_root / relative_file_path

    return absolute_path


def load_config(config_filename = "config.toml"):
    """
    -------------------------------------------------------------------------------------------------------------------
    Loads the config toml file and returns it as a dict; error if not found
    -------------------------------------------------------------------------------------------------------------------
    """
    cwd_path = Path(__file__).resolve().parent / config_filename
    if cwd_path.exists():
        config_path = cwd_path
    else:
        raise FileNotFoundError(f"Config file not found in: {cwd_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def log(script_name, config):
    """
    -------------------------------------------------------------------------------------------------------------------
    Generate process logs
    -------------------------------------------------------------------------------------------------------------------
    """
    log_dir = Path(config["log_info"]["log_dir"])
    level = config["log_info"]["level"]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}_{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, level, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def initialize_database(db_path):
    """
    -------------------------------------------------------------------------------------------------------------------
    Initialize SQLite database
    -------------------------------------------------------------------------------------------------------------------
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    return conn


def resolve_sql_path(sql_file_path):
    """
    -------------------------------------------------------------------------------------------------------------------
    Project path for sql queries
    -------------------------------------------------------------------------------------------------------------------
    """
    sql_path = Path(sql_file_path)

    if sql_path.is_absolute():
        return sql_path.resolve()

    script_dir = Path(__file__).resolve().parent

    return (script_dir / sql_path).resolve()


def execute_sql_file(conn, path, placeholders):
    """
    -------------------------------------------------------------------------------------------------------------------
    Execute all statements in the sql file specified
    -------------------------------------------------------------------------------------------------------------------
    """
    sql_path = resolve_sql_path(path)

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_script = sql_path.read_text(encoding="utf-8")

    if placeholders:
        for key, value in placeholders.items():
            sql_script = sql_script.replace(f"{{{{ {key} }}}}", value)

    cursor = conn.cursor()
    cursor.executescript(sql_script)
    conn.commit()
    cursor.close()


def log_existing_tables(conn, logger):
    """
    -------------------------------------------------------------------------------------------------------------------
    Return list of tables created in sqlite database
    -------------------------------------------------------------------------------------------------------------------
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Existing tables in database: {tables}")
    return tables


def random_choice(options):
    """
    -------------------------------------------------------------------------------------------------------------------
    Selecting random values from a list
    -------------------------------------------------------------------------------------------------------------------
    """
    return random.choice(options)


def generate_delivery_person(person_id, regions, fake):
    """
    -------------------------------------------------------------------------------------------------------------------
    Generate dummy records for delivery_persons table
    -------------------------------------------------------------------------------------------------------------------
    """
    return {
        "delivery_person_id": person_id,
        "name": fake.name(),
        "region": random_choice(regions),
        "hired_date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
        "is_active": fake.boolean(chance_of_getting_true=90)
    }


def generate_restaurant(restaurant_id, areas, cuisine_types, cfg, fake):
    """
    -------------------------------------------------------------------------------------------------------------------
    Generate dummy records for restaurants table
    -------------------------------------------------------------------------------------------------------------------
    """
    return {
        "restaurant_id": restaurant_id,
        "area": random_choice(areas),
        "name": fake.company(),
        "cuisine_type": random_choice(cuisine_types),
        "avg_preparation_time_min": fake.pyfloat(
            min_value=cfg["sql"]["general"]["avg_preparation_time_min"]["min"],
            max_value=cfg["sql"]["general"]["avg_preparation_time_min"]["max"],
            right_digits=2
        )
    }


def generate_delivery(delivery_id, delivery_person_ids, areas, weather_conditions, traffic_conditions, cfg, fake):
    """
    -------------------------------------------------------------------------------------------------------------------
    Generate dummy records for deliveries table
    -------------------------------------------------------------------------------------------------------------------
    """
    return {
        "delivery_id": delivery_id,
        "delivery_person_id": random_choice(delivery_person_ids),
        "restaurant_area": random_choice(areas),
        "customer_area": random_choice(areas),
        "delivery_distance_km": fake.pyfloat(
            min_value=cfg["sql"]["general"]["delivery_distance_km"]["min"],
            max_value=cfg["sql"]["general"]["delivery_distance_km"]["max"],
            right_digits=2
        ),
        "delivery_time_min": fake.pyint(
            min_value=cfg["sql"]["general"]["delivery_time_min"]["min"],
            max_value=cfg["sql"]["general"]["delivery_time_min"]["max"]
        ),
        "order_placed_at": fake.date_time_this_year().isoformat(sep=' '),
        "weather_condition": random_choice(weather_conditions),
        "traffic_condition": random_choice(traffic_conditions),
        "delivery_rating": fake.random_int(
            min=cfg["sql"]["general"]["delivery_rating"]["min"],
            max=cfg["sql"]["general"]["delivery_rating"]["max"]
        )
    }


def generate_order(order_id, delivery_ids, restaurant_ids, customer_ids, cfg, fake):
    """
    -------------------------------------------------------------------------------------------------------------------
    Generate dummy records for orders table
    -------------------------------------------------------------------------------------------------------------------
    """
    return {
        "order_id": order_id,
        "delivery_id": random_choice(delivery_ids),
        "restaurant_id": random_choice(restaurant_ids),
        "customer_id": random_choice(customer_ids),
        "order_value": fake.pyfloat(
            min_value=cfg["sql"]["general"]["order_value"]["min"],
            max_value=cfg["sql"]["general"]["order_value"]["max"],
            right_digits=2
        ),
        "items_count": fake.pyint(
            min_value=cfg["sql"]["general"]["items_count"]["min"],
            max_value=cfg["sql"]["general"]["items_count"]["max"]
        )
    }


def insert_records(conn, table_name, records):
    """
    -------------------------------------------------------------------------------------------------------------------
    Insert dummy records into tables
    -------------------------------------------------------------------------------------------------------------------
    """
    if not records:
        return

    columns = records[0].keys()
    placeholders = ":" + ", :".join(columns)
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    cur = conn.cursor()
    cur.executemany(sql, records)
    conn.commit()
    cur.close()


def run_select_query(conn, query, print_results= True):
    """
    -------------------------------------------------------------------------------------------------------------------
    Execute a select query and print the results if required, returning columns and rows
    -------------------------------------------------------------------------------------------------------------------
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    if print_results:
        print("\n--- Executing Query ---")
        print(query)
        print("\n--- Query Result ---")
        print(columns)
        for row in rows:
            print(row)

    return columns, rows


def run_select_file(conn, sql_file_path, print_results= True):
    """
    -------------------------------------------------------------------------------------------------------------------
    Read a sql file, split into individual statements, and execute each select query
    -------------------------------------------------------------------------------------------------------------------
    """
    sql_path = resolve_sql_path(sql_file_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    queries = [q.strip() for q in sql_text.split(";") if q.strip()]

    results = []

    for query in queries:
        columns, rows = run_select_query(conn, query, print_results=print_results)
        results.append((query, columns, rows))

    return results


def add_engineered_features(df):
    """
    -------------------------------------------------------------------------------------------------------------------
    Adds engineered features
    -------------------------------------------------------------------------------------------------------------------
    """
    df = df.copy()

    if "Time_of_Day" in df.columns:
        df["Is_Peak_Hour"] = df["Time_of_Day"].isin(["Morning", "Evening"]).astype(int)

    if "Weather" in df.columns:
        df["Is_Adverse_Weather"] = df["Weather"].isin(["Rainy", "Snowy", "Foggy"]).astype(int)

    if "Traffic_Level" in df.columns:
        traffic_map = {"Low": 1, "Medium": 2, "High": 3}
        df["Traffic_Level_Num"] = df["Traffic_Level"].map(traffic_map)

    if "Weather" in df.columns:
        weather_map = {"Clear": 0, "Windy": 1, "Foggy": 2, "Rainy": 3, "Snowy": 4}
        df["Weather_Severity_Num"] = df["Weather"].map(weather_map)

    if {"Distance_km", "Traffic_Level_Num"}.issubset(df.columns):
        df["Distance_Traffic"] = df["Distance_km"] * df["Traffic_Level_Num"]

    if {"Distance_km", "Weather_Severity_Num"}.issubset(df.columns):
        df["Distance_Weather"] = df["Distance_km"] * df["Weather_Severity_Num"]

    if {"Distance_km", "Traffic_Level_Num", "Weather_Severity_Num"}.issubset(df.columns):
        df["Distance_Traffic_Weather"] = (
            df["Distance_km"] * df["Traffic_Level_Num"] * df["Weather_Severity_Num"]
        )

    if {"Courier_Experience_yrs", "Traffic_Level_Num"}.issubset(df.columns):
        df["Experience_Traffic"] = df["Courier_Experience_yrs"] * df["Traffic_Level_Num"]

    if {"Courier_Experience_yrs", "Is_Adverse_Weather"}.issubset(df.columns):
        df["Experience_Weather"] = df["Courier_Experience_yrs"] * df["Is_Adverse_Weather"]

    if {"Courier_Experience_yrs", "Distance_km", "Traffic_Level_Num", "Weather_Severity_Num"}.issubset(df.columns):
        df["Experience_Adj_Factor"] = (
            df["Courier_Experience_yrs"] * df["Distance_km"] * df["Traffic_Level_Num"] * df["Weather_Severity_Num"]
        )


    return df



def impute_features(df):
    """
    -------------------------------------------------------------------------------------------------------------------
    Imputes missing values in specified columns of a df
    -------------------------------------------------------------------------------------------------------------------
    """
    df_imputed = df.copy()

    config = load_config()

    categorical = config["model"]["features"]["categorical"]

    for col in categorical:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(config["model"]["data_prep"]["categorical_filler_nulls"])

    numerical_cols = df_imputed.select_dtypes(include=np.number).columns

    for col in numerical_cols:
        if df_imputed[col].isnull().any():
            median_value = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_value)

    return df_imputed


def one_hot_encode_features(df):
    """
    -------------------------------------------------------------------------------------------------------------------
    Performs one-hot encoding with drop_first=True for all categorical columns defined
    -------------------------------------------------------------------------------------------------------------------
    """
    config = load_config()
    categorical = config["model"]["features"]["categorical"]

    df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True, dtype=int)

    return df_encoded


def drop_columns(df, columns_to_drop):
    """
    -------------------------------------------------------------------------------------------------------------------
    Drops specified columns from a pandas
    -------------------------------------------------------------------------------------------------------------------
    """
    if isinstance(columns_to_drop, str):
        columns_list = [columns_to_drop]
    elif isinstance(columns_to_drop, list):
        columns_list = columns_to_drop
    else:
        raise ValueError("columns_to_drop must be a string or a list of strings.")

    df_dropped = df.drop(columns=columns_list, axis=1, errors='ignore')

    return df_dropped


def scale_all_features(X_encoded, scaler_type='StandardScaler'):
    """
    -------------------------------------------------------------------------------------------------------------------
    Applies scaling to all features in the encoded df using the specified scaler type,
    matching against values defined in config.toml
    -------------------------------------------------------------------------------------------------------------------
    """
    config = load_config()
    if not config:
        print("ERROR: Could not load configuration. Defaulting to StandardScaler.")
        scaler = StandardScaler()
    else:
        config_standard_key = config.get("model", {}).get("data_prep", {}).get("scaler_types", {}).get("StandardScaler", "StandardScaler")
        config_robust_key = config.get("model", {}).get("data_prep", {}).get("scaler_types", {}).get("RobustScaler", "RobustScaler")

        if scaler_type == config_standard_key:
            scaler = StandardScaler()
        elif scaler_type == config_robust_key:
            scaler = RobustScaler()
        else:
            print(f"Warning: Unsupported scaler type '{scaler_type}' requested. Returning unscaled data.")
            return X_encoded

    X_scaled_array = scaler.fit_transform(X_encoded)

    X_scaled = pd.DataFrame(X_scaled_array, columns=X_encoded.columns, index=X_encoded.index)

    return X_scaled


def apply_pca(df_scaled, variance_threshold=0.90):
    """
    -------------------------------------------------------------------------------------------------------------------
    Applies PCA to the scaled features, retain a given variance percentage
    -------------------------------------------------------------------------------------------------------------------
    """
    pca = PCA()
    pca.fit(df_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(df_scaled)

    pca_columns = [f'PC_{i + 1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, index=df_scaled.index, columns=pca_columns)

    return df_pca, n_components


def prepare_data_for_model(df, model_name):
    """
    -------------------------------------------------------------------------------------------------------------------
    Prepares the DataFrame based on the requirements of the specified single model,
    including conditional scaling, encoding, and PCA
    -------------------------------------------------------------------------------------------------------------------
    """

    df_processed = impute_features(df)
    config = load_config()

    target = config["model"]["features"]["target"]
    drop = config["model"]["features"]["drop"]

    if target not in df_processed.columns:
        raise ValueError(f"DataFrame must contain target variable '{target}'.")

    X = df_processed.drop(columns=[target, drop], errors='ignore')
    y = df_processed[target]

    scaling_and_pca_models = ['LinearRegression', 'Ridge', 'Lasso', 'KRR', 'SVR', 'MLPRegressor']
    robust_scaler_models = ['kNN']
    tree_based_models = ['RandomForestRegressor', 'XGBoost', 'CatBoost']

    if "catboost" in model_name.lower():
        print(f"Skipping one-hot encoding for {model_name} (CatBoost handles categorical features natively).")
        X_encoded = X.copy()
    else:
        print(f"Applying one-hot encoding for {model_name}...")
        X_encoded = one_hot_encode_features(X)

    if model_name in scaling_and_pca_models:
        scaler_type = 'RobustScaler' if model_name in robust_scaler_models else 'StandardScaler'
        print(f"Applying {scaler_type} to all features for {model_name}...")

        X_scaled = scale_all_features(X_encoded, scaler_type=scaler_type)

        initial_features = X_encoded.shape[1]
        print(f"Applying PCA for dimensionality reduction. Starting features: {initial_features}.")

        X_final, n_components = apply_pca(X_scaled)
        print(f"Features reduced to {n_components} components (retaining 90% variance).")

    elif model_name in tree_based_models:
        print(f"No scaling or PCA required for tree-based model: {model_name}.")
        X_final = X_encoded

    elif "catboost" in model_name.lower():
        print(f"No scaling or PCA required for CatBoost.")
        X_final = X_encoded

    else:
        print(f"No preprocessing rules found for {model_name}. Returning encoded data.")
        X_final = X_encoded

    return X_final, y


def process_all_models(df, model_list, output_path):
    """
    -------------------------------------------------------------------------------------------------------------------
    Wrapper function to process the input specifically per model
    Applies model-specific preprocessing and saves the resulting df
    -------------------------------------------------------------------------------------------------------------------
    """
    if not isinstance(model_list, list):
        raise TypeError(f"Expected a list of model names, but received type {type(model_list)}.")

    config = load_config()
    training_config = config["model"]["training"]

    absolute_output_dir = get_project_path(output_path, levels_up_to_root=0)
    os.makedirs(absolute_output_dir, exist_ok=True)
    print(f"Data will be saved to: {absolute_output_dir}")

    all_prepared_data = {}

    print("\n--- Starting Batch Data Preparation ---")
    for model_name in model_list:
        if not isinstance(model_name, str):
            print(f"Skipping non-string item in list: {model_name}")
            continue

        print(f"\nProcessing data for: {model_name}")
        X_processed, y = prepare_data_for_model(df.copy(), model_name)

        all_prepared_data[model_name] = (X_processed, y)

        X_train, X_test, X_val, y_train, y_test, y_val = split_data(
            X_processed, y, training_config
        )

        if model_name.lower().startswith("CatBoost"):
            categorical_cols = config["model"]["features"]["categorical"]

            for col in categorical_cols:
                if col in X_train.columns:
                    X_train[col] = X_train[col].astype("category")
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype("category")
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype("category")

            print(f"Preserved categorical dtypes for CatBoost model: {categorical_cols}")

        X_train_path = os.path.join(absolute_output_dir, f"X_{model_name}_train.csv")
        y_train_path = os.path.join(absolute_output_dir, f"y_{model_name}_train.csv")
        X_test_path  = os.path.join(absolute_output_dir, f"X_{model_name}_test.csv")
        y_test_path  = os.path.join(absolute_output_dir, f"y_{model_name}_test.csv")
        X_val_path   = os.path.join(absolute_output_dir, f"X_{model_name}_val.csv")
        y_val_path   = os.path.join(absolute_output_dir, f"y_{model_name}_val.csv")

        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False, header=True)
        X_test.to_csv(X_test_path, index=False)
        y_test.to_csv(y_test_path, index=False, header=True)
        X_val.to_csv(X_val_path, index=False)
        y_val.to_csv(y_val_path, index=False, header=True)

        print(f"Train, test, and validation data saved for {model_name}.")

    print("\n--- Batch Data Preparation Complete ---")
    return all_prepared_data



def split_data(X, y, config):
    """
    -------------------------------------------------------------------------------------------------------------------
    Splits data into train, test, and validation sets based on config ratios
    -------------------------------------------------------------------------------------------------------------------
    """
    test_val_ratio = config['test_ratio'] + config['val_ratio']

    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y,
        test_size=test_val_ratio,
        random_state=config['random_state']
    )

    val_fraction_of_pool = config['val_ratio'] / test_val_ratio

    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val,
        test_size=val_fraction_of_pool,
        random_state=config['random_state']
    )

    return X_train, X_test, X_val, y_train, y_test, y_val


def mean_absolute_percentage_error(y_true, y_pred):
    """
    -------------------------------------------------------------------------------------------------------------------
    Calculates Mean Absolute Percentage Error (MAPE)
    -------------------------------------------------------------------------------------------------------------------
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0

    if non_zero_indices.sum() == 0:
        return 0.0

    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100


def calculate_all_metrics(y_true, y_pred):
    """
    -------------------------------------------------------------------------------------------------------------------
    Calculates and returns a dictionary of key metrics
    -------------------------------------------------------------------------------------------------------------------
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }