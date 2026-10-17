"""
Week 4 Project: Predictive Modeling and Optimization in Logistics Systems
Predicts delivery duration using a simulated dataset, evaluates models, and
turns predictions into practical dispatch/route recommendations.

Install:
    pip install pandas numpy scikit-learn

Run:
    python logistics_delivery_prediction.py
Optional:
    python logistics_delivery_prediction.py --data simulated_logistics_data.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def make_simulated_data(n=300, seed=42):
    """Create a reproducible hypothetical dataset when no CSV is supplied."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(2, 80, n)
    traffic = rng.choice(["Low", "Medium", "High"], n, p=[.35, .4, .25])
    weather = rng.choice(["Clear", "Rain", "Storm"], n, p=[.65, .28, .07])
    vehicle = rng.choice(["Bike", "Van", "Truck"], n)
    weight = rng.uniform(.2, 25, n)
    hour = rng.integers(6, 23, n)
    day = rng.integers(0, 7, n)
    stops = rng.integers(1, 13, n)
    traffic_effect = pd.Series(traffic).map({"Low": 0, "Medium": 12, "High": 28}).to_numpy()
    weather_effect = pd.Series(weather).map({"Clear": 0, "Rain": 10, "Storm": 24}).to_numpy()
    vehicle_effect = pd.Series(vehicle).map({"Bike": 0, "Van": 4, "Truck": 8}).to_numpy()
    rush = np.isin(hour, [8, 9, 17, 18, 19]) * 12
    weekend = np.isin(day, [5, 6]) * -2
    target = (14 + distance*1.05 + traffic_effect + weather_effect + vehicle_effect
              + weight*.35 + stops*2.1 + rush + weekend
              + rng.normal(0, 8, n))
    target = np.maximum(target, 12)
    return pd.DataFrame({
        "distance_km": distance, "traffic_level": traffic, "weather": weather,
        "vehicle_type": vehicle, "package_weight_kg": weight,
        "dispatch_hour": hour, "day_of_week": day,
        "stops_before_delivery": stops, "delivery_time_min": target
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Optional CSV path")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = make_simulated_data()

    target = "delivery_time_min"
    features = [
        "distance_km", "traffic_level", "weather", "vehicle_type",
        "package_weight_kg", "dispatch_hour", "day_of_week",
        "stops_before_delivery"
    ]
    X, y = df[features], df[target]
    numeric = ["distance_km", "package_weight_kg", "dispatch_hour",
               "day_of_week", "stops_before_delivery"]
    categorical = ["traffic_level", "weather", "vehicle_type"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        fitted[name] = pipe
        results.append((name, mae, rmse, r2))
        print(f"{name:18s} | MAE={mae:6.2f} min | RMSE={rmse:6.2f} min | R2={r2:.3f}")

    best_name = min(results, key=lambda row: row[2])[0]
    best_model = fitted[best_name]
    print(f"\\nSelected model by test RMSE: {best_name}")

    # Cross-validation on the training set (negative MAE is converted to positive).
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    )
    print(f"5-fold CV MAE: {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f} minutes")

    # Example operational prioritization: rank shipments by predicted duration.
    sample = X_test.copy()
    sample["predicted_delivery_min"] = best_model.predict(X_test)
    sample["priority"] = pd.cut(
        sample["predicted_delivery_min"],
        bins=[-np.inf, 45, 75, np.inf],
        labels=["Standard", "Monitor", "High attention"]
    )
    print("\\nFirst 10 predicted shipments:")
    print(sample[["predicted_delivery_min", "priority"]].head(10).to_string(index=False))

    # Illustrative optimization rules (not an automated route solver):
    print("\\nRecommended actions:")
    print("- High predicted duration: review traffic and dispatch earlier where feasible.")
    print("- Storm/high-traffic jobs: add a realistic buffer and notify customers.")
    print("- High stop-count routes: rebalance stops across nearby vehicles.")
    print("- Compare predicted time with actual time weekly and retrain on fresh data.")

if __name__ == "__main__":
    main()
