"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: Preprocessing of data - Data preparation
-----------------------------------------------------------------------------------------------------------------------
Process:
1. Reads dataset as defined in config toml file
2. Runs functions for data processing stored in Global.py
3. Generates df for model training

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import Global
import pandas as pd


def main():
    script_name = "Preprocessing"

    config = Global.load_config()
    logger = Global.log(script_name, config)

    logger.info("Starting Preprocessing main script")

    try:

        path = Global.get_project_path(config["model"]["input_data"]["path"], 0)

        logger.info("Loading data from path")

        df = pd.read_csv(path)

        logger.info("Applying model-specific data preparation and saving data splits (X and Y)")

        model_list_to_test = config["model"]["algo"]["algo_list"]
        output = config["model"]["data_prep"]["path"]

        Global.process_all_models(df, model_list_to_test, output)


    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()



