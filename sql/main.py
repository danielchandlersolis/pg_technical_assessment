"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: run SQL queries process end to end
-----------------------------------------------------------------------------------------------------------------------
1. Create tables create_tables.sql
2. Insert dummy data into create_tables.sql
3. Run queries sql_queries.sql and print query + results in terminal

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from Global import load_config, log, initialize_database, execute_sql_file, log_existing_tables, run_select_file
import sql.insert_data as insert_data

def main ():

    script_name = "SQL_main"

    config = load_config()

    logger = log(script_name, config)

    logger.info("Starting SQL main script")

    try:
       logger.info("1. Creating tables")
       db_path = config["sql"]["db_path"]
       create_tables = config["sql"]["create_tables"]

       create_db = initialize_database(db_path)

       execute_sql_file(create_db, create_tables, placeholders=config["sql"]["tables"])

       log_existing_tables(create_db, logger)

       logger.info("1. Tables created")

       logger.info("2. Inserting dummy records into tables")

       insert_data.main(create_db, config)

       logger.info("2. Dummy records inserted")

       logger.info("3. Running sql_queries.sql")

       queries_file = config["sql"]["queries"]

       run_select_file(create_db, queries_file, print_results=config["sql"]["print_results"])

       logger.info("3. Queries were run and results were retrieved")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()


