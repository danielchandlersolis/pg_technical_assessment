/**********************************************************************************************************************
  Purpose: Design queries to answer the following questions, you can create hypothetical tables if you want
  ---------------------------------------------------------------------------------------------------------------------
Questions:
    1. Top 5 customer areas with highest average delivery time in the last 30 days

    2. Average delivery time per traffic condition, by restaurant area and cuisine type

    3. Top 10 delivery people with the fastest average delivery time, considering only
      those with at least 50 deliveries and who are still active

    4. The most profitable restaurant area in the last 3 months, defined as the area with the
      highest total order value

    5. Identify whether any delivery people show an increasing trend in average delivery time

  Comments:
  - Hypothetical tables with dummy data were created, please run sql/main.py to generate end to end
  including sql_queries results printed in terminal
  - Queries are based on SQLite syntax, documentation: https://sqlite.org/lang_datefunc.html
  - Query #2 deliveries table is recommended to have restaurant_id instead of restaurant_area, for join purposes
  - Query #5 was done comparing avg delivery time of last 30 days vs. last month
 *********************************************************************************************************************/

-- 1. Top 5 customer areas with highest average delivery time in the last 30 days
SELECT customer_area,
       ROUND(AVG(delivery_time_min), 2) AS avg_delivery_time
FROM deliveries
WHERE order_placed_at >= datetime('now', '-30 days')
GROUP BY customer_area
ORDER BY avg_delivery_time DESC
LIMIT 5;

-- 2. Average delivery time per traffic condition, by restaurant area and cuisine type
SELECT d.restaurant_area,
       r.cuisine_type,
       d.traffic_condition,
       ROUND(AVG(d.delivery_time_min), 2) AS avg_delivery_time
FROM deliveries d
INNER JOIN restaurants r ON d.restaurant_area = r.area
GROUP BY d.restaurant_area, r.cuisine_type, d.traffic_condition
ORDER BY d.restaurant_area, r.cuisine_type, d.traffic_condition;

-- 3. Top 10 delivery people with the fastest average delivery time, considering only those with at least 50
-- deliveries and who are still active
SELECT d.delivery_person_id,
       dp.name,
       ROUND(AVG(d.delivery_time_min), 2) AS avg_delivery_time,
       COUNT(d.delivery_id) AS total_deliveries
FROM deliveries d
INNER JOIN delivery_persons dp ON d.delivery_person_id = dp.delivery_person_id
WHERE dp.is_active = TRUE
GROUP BY d.delivery_person_id
HAVING COUNT(d.delivery_id) >= 50
ORDER BY avg_delivery_time ASC
LIMIT 10;

--  4. The most profitable restaurant area in the last 3 months, defined as the area with the
--  highest total order value
SELECT r.area AS restaurant_area,
       ROUND(SUM(o.order_value), 2) AS total_order_value
FROM orders o
INNER JOIN restaurants r ON o.restaurant_id = r.restaurant_id
INNER JOIN deliveries d ON o.delivery_id = d.delivery_id
WHERE d.order_placed_at >= datetime('now', '-3 months')
GROUP BY r.area
ORDER BY total_order_value DESC
LIMIT 1;

-- 5. Identify whether any delivery people show an increasing trend in average delivery time
SELECT dp.delivery_person_id,
       dp.name,
       dp.region,
       r.avg_recent,
       p.avg_previous,
       ROUND(r.avg_recent - p.avg_previous, 2) AS delta,
       ROUND(
           CASE WHEN p.avg_previous != 0 THEN ((r.avg_recent/p.avg_previous)-1) * 100 ELSE NULL END, 2
       ) AS delta_pct
FROM delivery_persons dp
LEFT JOIN (
    SELECT delivery_person_id, ROUND(AVG(delivery_time_min),2) AS avg_recent
    FROM deliveries
    WHERE order_placed_at >= datetime('now','-30 days')
    GROUP BY delivery_person_id
) r ON dp.delivery_person_id = r.delivery_person_id
LEFT JOIN (
    SELECT delivery_person_id, ROUND(AVG(delivery_time_min),2) AS avg_previous
    FROM deliveries
    WHERE order_placed_at >= datetime('now','-60 days')
      AND order_placed_at < datetime('now','-30 days')
    GROUP BY delivery_person_id
) p ON dp.delivery_person_id = p.delivery_person_id
WHERE (r.avg_recent - p.avg_previous) > 0 and dp.is_active = TRUE
ORDER BY delta DESC;

-- EXTRA - What is the average delivery time for the top 5 most profitable restaurant areas, in the last 3 months?
SELECT
    r.area,
    ROUND(AVG(d.delivery_time_min), 2) AS average_delivery_time,
    SUM(o.order_value) AS total_order_value
FROM deliveries d
LEFT JOIN orders o
    ON d.delivery_id = o.delivery_id
INNER JOIN restaurants r
    ON o.restaurant_id = r.restaurant_id
WHERE d.order_placed_at >= datetime('now', '-3 months')
GROUP BY r.area
ORDER BY total_order_value DESC
LIMIT 5;

-- EXTRA - What is the average distance by order for each restaurant area in the last 3 months?
SELECT
    r.area AS restaurant_area,
    ROUND(AVG(d.delivery_distance_km), 2) AS avg_distance_km
FROM deliveries d
LEFT JOIN orders o
    ON d.delivery_id = o.delivery_id
LEFT JOIN restaurants r
    ON o.restaurant_id = r.restaurant_id
WHERE d.order_placed_at >= datetime('now', '-3 months')
GROUP BY r.area
ORDER BY avg_distance_km DESC;

-- EXTRA - What are the days of the week having the most activity (peak days) by restaurant area in the last 3 months?
SELECT
    r.area AS restaurant_area,
    CASE strftime('%w', d.order_placed_at)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS weekday_name,
    COUNT(o.order_id) AS total_orders
FROM deliveries d
LEFT JOIN orders o
    ON d.delivery_id = o.delivery_id
LEFT JOIN restaurants r
    ON o.restaurant_id = r.restaurant_id
WHERE d.order_placed_at >= datetime('now', '-3 months')
GROUP BY r.area, strftime('%w', d.order_placed_at)
ORDER BY r.area, total_orders DESC;
