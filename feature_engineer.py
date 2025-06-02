import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Отделение целевой переменной
    target = 'Price'
    if target in df.columns:
        y = df[target]
        X = df.drop(target, axis=1)
    else:
        X = df
    
    # Масштабирование числовых признаков
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Возвращаем целевую переменную
    if target in df.columns:
        X[target] = y
    
    X.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    engineer_features(args.input, args.output)