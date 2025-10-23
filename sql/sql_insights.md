# Part I: SQL
## Additional analysis and questions useful for the business problem

---

### 1. Additional analysis

About the sql queries that were run and future work:
1. List of worst offenders in terms of customer areas, which might have various reasons including weather, traffic disruptions, order time and day of the week (peak hours, off-peak hours),
potential delays in receiving instruction for picking-up the order, restaurants could be getting worse in their 
average preparation time therefore impacting the total time, among other reasons that could be of interest for further studying.
2. Historical traffic conditions and its impact on the average delivery time by restaurant area. The cuisine type could tell
more on specific variations by restaurant due to the business nature (degree of complexity in dishes among fast food, traditional dishes or gourmet, operational constraints). Would be interesting to see how the trends
have developed over the time. It could bring more insights on worst offenders when doing the breakdown by restaurant.
3. The list of best performers could tell us more when doing a distribution analysis of the average delivery time and enriching
the analysis with the areas where they operated, restaurants served, weather and traffic conditions at the time of delivery,
time of the day, week day and with that being able to reach an understanding on average delivery times expected at a given time of the operation
under certain circumstances.
4. About the most profitable restaurant areas information, this is crucial for the platform as these areas are key for growing the business. It can be enriched with the average delivery time as this could unveil
service levels for the most profitable areas.
5. As follow-up from the previous, identifying if some delivery partners are increasing their average delivery time is relevant when comparing against
other delivery partners, as it is the best interest of the platform to maintain a low delivery time in general but more specifically on crucial areas for profit. It could lead to optimizing order allocation.

For a better understanding of delivery times, it might be helpful to test a sample data of the following:
1. Motion data (sensor) from delivery-partner phones, to enrich the analysis and understanding of different states, 
as mentioned in [1]
2. Delivery-partner mean of transportation, whether bike, car, or other as this might correlate with the average 
delivery times
3. Coordinates (latitude, longitude) to enrich customer area and restaurant area with specific locations

#### Reference:

[1] https://www.uber.com/en-CH/blog/uber-eats-trip-optimization/

---

### 2. Questions
Given the information in tables 'deliveries', 'delivery_persons', 'restaurants', 'orders', and the initial questions
that were suggested, the below questions could bring more understanding of the business problem, by restaurant area:

#### 1. What is the average delivery time for the top 5 most profitable restaurant areas, in the last 3 months?

```sql
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
```

#### 2. What is the average distance by order for each restaurant area, in the last 3 months?

```sql
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
```

#### 3. What are the days of the week having the most activity (peak days) by restaurant area in the last 3 months?

```sql
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
```
