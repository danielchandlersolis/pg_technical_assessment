# Model Notes

---

* #### The strategy was to first preprocess the data and prepare it for each of the models in this experimentation.
* #### Data was divided in 3: training 70%, testing 15% and validation 15% with an stochastic sampling using fixed seed for reproducibility.
* #### Data preparation functions are in Global.py file as these functions were executed from preprocessing.py
* #### The following models were tested for this challenge and certain techniques to scale the numerical values were applied depending on the model:
* #### StandardScaler and PCA models = ['LinearRegression', 'Ridge', 'Lasso', 'KRR', 'SVR', 'MLPRegressor'] serving as baselines
* #### Robust scaler applied = ['kNN']
* #### Tree based models = ['RandomForest', 'XGBoost', 'CatBoost'] as decision trees algorithms.
* #### Except for <mark>CatBoost</mark>, which recommends to not use one-hot encoding during preprocessing [1], the <mark>one-hot encoding</mark> technique was applied, always dropping one of the categories.
* #### The missing values in categorical columns were changed to <mark>Unknown</mark> 
* #### The missing value from <mark>courier experience years</mark> was imputed with median.
* #### The following extra features were engineered but didn't prove to add predictive power:

```python
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
    
 ```

* #### About the optimization metric, <mark>Root Mean Squared Error</mark> was selected; main decision driver is due to large errors are heavily penalized.
* #### <mark>Hyperparameter optimization</mark> with GridSearchCV was experimented with different combinations.

```python
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
```
* #### After executing training.py and evaluating the overall results for each round when running Evaluation.py, the <mark>top 3 algorithms</mark> based on lowest RMSE were: 1. CatBoost, 2. SVR (Support Vector Regression) and 3. XGBoost.
| Algorithm         | RMSE  |
|-------------------|-------|
| Catboost          | 14.88 |
| SVR               | 15.06 |
| XGBoost           | 15.08 |
| Ridge             | 15.36 |
| Linear Regression | 15.37 |
| Lasso             | 15.37 |
| KRR               | 15.69 |
| Random Forest     | 15.88 |
| kNN               | 16.10 |
| MLP               | 35.35 |
* #### Other models tested e.g. Ridge, Linear Regression, Lasso, KRR (Kernel Ridge regression), Random Forest Regressor, kNN (k-Nearest Neighbors) were not so far from top 3 algorithms. Worst performer was Multi-layer Perceptron regressor.

---

# Next steps

* ## Experiment with model ensembling and see if this boosts the performance. Two techniques to try are hill climbing and stacking, as mentioned in [2]
* ## Experiment by creating a model depending on the weather conditions, as it is known from the EDA how the average delivery time increases for Snowy weather compared to Clear.


[1] https://catboost.ai/docs/en/concepts/parameter-tuning

[2] https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/







