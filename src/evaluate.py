import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import numpy as np

def evaluate(model_path, test_data_path, metrics_path):
    # Загрузка модели и данных
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Предсказание
    preds = model.predict(X)
    
    # Расчет метрик
    metrics = {
        'mae': mean_absolute_error(y, preds),
        'mse': mean_squared_error(y, preds),
        'rmse': np.sqrt(mean_squared_error(y, preds)),
        'r2': r2_score(y, preds)
    }
    
    # Сохранение метрик
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test_data, args.metrics)