"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Purpose: Insert dummy data into the tables 'deliveries', 'delivery_persons', 'restaurants', 'orders'
-----------------------------------------------------------------------------------------------------------------------
Process:
        Using faker to generate dummy data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from faker import Faker
from Global import (load_config, initialize_database, generate_delivery_person, generate_restaurant, generate_delivery,
                    generate_order, insert_records)


def main(conn, config):
    """
    -------------------------------------------------------------------------------------------------------------------
    Insert dummy data into the tables 'deliveries', 'delivery_persons', 'restaurants', 'orders'
    -------------------------------------------------------------------------------------------------------------------
    """
    fake = Faker()

    # Delivery personnel metadata
    delivery_persons = [generate_delivery_person(i, config["sql"]["general"]["regions"], fake)
                        for i in range(1, config["sql"]["general"]["num_delivery_persons"]+1)]
    insert_records(conn, config["sql"]["tables"]["delivery_persons"], delivery_persons)

    # Restaurant metadata
    restaurants = [generate_restaurant(f"pa_{i}", config["sql"]["general"]["areas"],
                                       config["sql"]["general"]["cuisine_types"], config, fake)
                   for i in range(1, config["sql"]["general"]["num_restaurants"]+1)]
    insert_records(conn, config["sql"]["tables"]["restaurants"], restaurants)

    # Delivery-level data
    delivery_ids = [f"pa_{i}" for i in range(1, config["sql"]["general"]["num_deliveries"]+1)]
    delivery_person_ids = [p["delivery_person_id"] for p in delivery_persons]
    deliveries = [generate_delivery(did, delivery_person_ids, config["sql"]["general"]["areas"],
                                   config["sql"]["general"]["weather_conditions"],
                                   config["sql"]["general"]["traffic_conditions"], config, fake)
                  for did in delivery_ids]
    insert_records(conn, config["sql"]["tables"]["deliveries"], deliveries)

    # Orders table
    restaurant_ids = [r["restaurant_id"] for r in restaurants]
    customer_ids = [f"pa_{i}" for i in range(1, config["sql"]["general"]["num_orders"]+1)]
    orders = [generate_order(i, delivery_ids, restaurant_ids, customer_ids, config, fake)
              for i in range(1, config["sql"]["general"]["num_orders"]+1)]
    insert_records(conn, config["sql"]["tables"]["orders"], orders)

if __name__ == "__main__":
    config = load_config()
    conn = initialize_database(config["sql"]["db_path"])

    main(conn, config)