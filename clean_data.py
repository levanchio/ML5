import pandas as pd
import re
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(input_path, output_path):
    # Создаем директории при необходимости
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Загрузка данных
    df = pd.read_csv(input_path)
    
    # 1. Удаление пропусков и дубликатов
    df = df.dropna().drop_duplicates()
    
    # 2. Обработка цены
    if 'Price' in df.columns:
        df['Price'] = df['Price'].apply(lambda x: float(re.sub(r'[^\d.]', '', x.split('per')[0])))
    
    # 3. Обработка числовых признаков
    if 'ABV' in df.columns:
        df['ABV'] = df['ABV'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    if 'Capacity' in df.columns:
        df['Capacity'] = df['Capacity'].str.extract(r'(\d+)').astype(float)
    
    # 4. Принудительное преобразование всех оставшихся колонок
    for col in df.columns:
        # Пропускаем уже числовые
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Пробуем преобразовать в число
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except:
            pass
        
        # Для категориальных - Label Encoding
        if df[col].nunique() < 100:  # Не будем кодировать колонки с >100 уникальными значениями
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df = df.drop(col, axis=1)
    
    # 5. Гарантированное удаление всех нечисловых колонок
    df = df.select_dtypes(include=['number'])
    
    # Проверка, что Price остался (целевая переменная)
    if 'Price' not in df.columns:
        raise ValueError("Целевая переменная 'Price' была удалена в процессе очистки!")
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    clean_data(args.input, args.output)