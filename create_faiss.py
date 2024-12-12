import re
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_tskv_line(line):
    """Строка -> словарь"""
    data = {}
    for item in line.split('\t'):
        if '=' in item:
            key, value = item.split('=', 1)
            data[key.strip()] = value.strip()
    return data

def load_tskv_file(file_path):
    """TSKV -> DF"""
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_data = [parse_tskv_line(line) for line in file]
    df = pd.DataFrame(parsed_data)
    
    required_columns = ['address', 'name_ru', 'rating', 'rubrics', 'text']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Столбец '{col}' отсутствует в DataFrame. Доступные столбцы: {df.columns}")

    df.columns = df.columns.str.strip()
    df['text'] = df['text'].str.strip()
    print(f"Загружено {len(df)} записей из {file_path}")
    return df

def reduce_dataset_size(df, required_batch_count=500, batch_size=32):
    """Уменьшение батчей"""
    total_rows = len(df)
    total_batches = total_rows // batch_size 
    if total_batches < required_batch_count:
        print(f"Общее количество батчей ({total_batches}) уже меньше, чем необходимо ({required_batch_count}).")
        return df
    reduced_size = required_batch_count * batch_size
    reduced_df = df.sample(n=reduced_size, random_state=42)
    print(f"Уменьшен размер датасета до {len(reduced_df)} строк для {required_batch_count} батчей (batch_size={batch_size})")
    return reduced_df

def create_faiss_index(embeddings):
    """Создания индекса"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print(f"Добавлено {index.ntotal} векторов в FAISS индекс")
    return index

def create_faiss_db(file_path, output_path, batch_count=500, batch_size=32):
    """Создание БД"""

    df = load_tskv_file(file_path)
    df = reduce_dataset_size(df, required_batch_count=batch_count, batch_size=batch_size)
    print("Начата генерация эмбеддингов для текстов...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    index = create_faiss_index(embeddings)
    faiss.write_index(index, output_path)
    print(f"FAISS индекс успешно сохранён в {output_path}")

if __name__ == "__main__":
    input_path = 'geo-reviews-dataset-2023.tskv'
    output_path = 'faiss_db/review_index.faiss'
    create_faiss_db(input_path, output_path, batch_count=500, batch_size=32)