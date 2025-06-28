import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.sklearn

def train_with_tuning():

    # Dapatkan path absolut dari skrip ini
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'preprocessing', 'MedicalCost_preprocessing', 'insurance_processed.csv')

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Medical Cost Prediction")


    df = pd.read_csv(csv_path)
    mlflow_dataset = mlflow.data.from_pandas(df, source='insurance_processed.csv')

    
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators_list = [50, 100, 150]
    max_depth_list = [5, 10, None] 

    for n in n_estimators_list:
        for depth in max_depth_list:
            with mlflow.start_run(run_name=f"RF_n_estimators_{n}_max_depth_{depth}"):

                mlflow.log_input(mlflow_dataset, context="training")
                
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", depth if depth is not None else "None")

                model = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", r2)

                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="random_forest_model", 
                    input_example=X_train.head()
                )

if __name__ == '__main__':
    train_with_tuning()