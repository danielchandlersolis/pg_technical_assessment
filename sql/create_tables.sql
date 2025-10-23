/**********************************************************************************************************************
  Purpose: create tables for PART I: SQL
  ---------------------------------------------------------------------------------------------------------------------
  Tables created:
  1. deliveries
  2. delivery_persons
  3. restaurants
  4. orders
 *********************************************************************************************************************/

--Reference tables and data types

-- -- Delivery-level data
-- deliveries (
-- delivery_id VARCHAR,
-- delivery_person_id VARCHAR,
-- restaurant_area VARCHAR,
-- customer_area VARCHAR,
-- delivery_distance_km FLOAT,
-- delivery_time_min INT,
-- order_placed_at TIMESTAMP,
-- weather_condition VARCHAR,
-- traffic_condition VARCHAR,
-- delivery_rating FLOAT
-- )
--
-- -- Delivery personnel metadata
-- delivery_persons (
-- delivery_person_id INT,
-- name VARCHAR,
-- region VARCHAR,
-- hired_date DATE,
-- is_active BOOLEAN
-- )
--
-- -- Restaurant metadata
-- restaurants (
-- restaurant_id VARCHAR,
-- area VARCHAR,
-- name VARCHAR,
-- cuisine_type VARCHAR,
-- avg_preparation_time_min FLOAT
-- )
--
-- -- Orders table
-- orders (
-- order_id INT,
-- delivery_id VARCHAR,
-- restaurant_id VARCHAR,
-- customer_id VARCHAR,
-- order_value FLOAT,
-- items_count INT
-- )

DROP TABLE IF EXISTS {{ deliveries }};
DROP TABLE IF EXISTS {{ delivery_persons }};
DROP TABLE IF EXISTS {{ restaurants }};
DROP TABLE IF EXISTS {{ orders }};

-- Delivery-level data
CREATE TABLE {{ deliveries }} (
    delivery_id TEXT PRIMARY KEY,
    delivery_person_id INTEGER,
    restaurant_area TEXT,
    customer_area TEXT,
    delivery_distance_km REAL,
    delivery_time_min INTEGER,
    order_placed_at TIMESTAMP,
    weather_condition TEXT,
    traffic_condition TEXT,
    delivery_rating REAL
);

-- Delivery-level data
CREATE TABLE {{ delivery_persons }} (
    delivery_person_id INTEGER PRIMARY KEY,
    name TEXT,
    region TEXT,
    hired_date DATE,
    is_active BOOLEAN
);

-- Restaurant metadata
CREATE TABLE {{ restaurants }} (
    restaurant_id TEXT PRIMARY KEY,
    area TEXT,
    name TEXT,
    cuisine_type TEXT,
    avg_preparation_time_min REAL
);

-- Orders table
CREATE TABLE {{ orders }} (
    order_id INTEGER PRIMARY KEY,
    delivery_id TEXT,
    restaurant_id TEXT,
    customer_id TEXT,
    order_value REAL,
    items_count INTEGER
);