"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: Inference
-----------------------------------------------------------------------------------------------------------------------
Process:
1. Reads the best model as defined in config toml
2. Load model artifact .joblib
3. Load validation data
4. Perform inferences
5. Calculate error metrics (RMSE, MSE, MAE, MAPE)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import Global
import pandas as pd
import joblib
import os


def main():

    script_name = "Inference"
    config = Global.load_config()
    logger = Global.log(script_name, config)

    logger.info(f"Starting {script_name} script for inferences over validation")

    MODEL_NAME = config["model"]["algo"]["best_model"]

    MODEL_PATH = Global.get_project_path(config["model"]["save"]["path"], 0)
    DATA_PATH = Global.get_project_path(config["model"]["data_prep"]["path"], 0)

    model_filename = f"{MODEL_NAME}_best_model.joblib"
    X_val_filename = f"X_{MODEL_NAME}_val.csv"
    y_val_filename = f"y_{MODEL_NAME}_val.csv"

    model_file_path = os.path.join(MODEL_PATH, model_filename)
    try:
        best_model = joblib.load(model_file_path)
        logger.info(f"Successfully loaded best model: {model_filename} using joblib.")
    except FileNotFoundError:
        logger.error(
            f"Trained model file not found: {model_file_path}. Please ensure the training script has been run and "
            f"saved the model with the .joblib extension.")
        return
    except Exception as e:
        logger.error(f"Error loading the model file with joblib: {e}")
        return

    X_val_path = os.path.join(DATA_PATH, X_val_filename)
    y_val_path = os.path.join(DATA_PATH, y_val_filename)

    try:
        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path).squeeze()
        logger.info(f"Successfully loaded validation data. X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    except FileNotFoundError:
        logger.error(
            f"Validation data files not found. Ensure Preprocessing runs correctly and saves: {X_val_path} and {y_val_path}")
        return
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return

    logger.info(f"Starting inference using {MODEL_NAME} model...")
    y_pred = best_model.predict(X_val)

    metrics = Global.calculate_all_metrics(y_val, y_pred)

    print("\n" + "=" * 50)
    print(f"FINAL MODEL PERFORMANCE ON VALIDATION DATA ({MODEL_NAME})")
    print("=" * 50)
    for metric, value in metrics.items():
        logger.info(f"Final Validation Metric {metric}: {value}")
    print("=" * 50 + "\n")

    logger.info("Inference Complete")


if __name__ == "__main__":
    main()