# modelling_files/modelling.py (The "Resume Run" Final Version)

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.sklearn

def run_training(n_estimators, max_depth):
    run_id = os.environ.get("MLFLOW_RUN_ID")


    with mlflow.start_run(run_id=run_id):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, 'preprocessing', 'MedicalCost_preprocessing', 'insurance_processed.csv')
        
        df = pd.read_csv(csv_path)
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Latih model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Log metrik (kembali pakai cara simpel)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log model (kembali pakai cara simpel, tanpa run_id)
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model", 
            input_example=X_train.head()
        )
        
    print(f"Run dengan n_estimators={n_estimators} dan max_depth={max_depth} selesai.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    run_training(n_estimators=args.n_estimators, max_depth=args.max_depth)