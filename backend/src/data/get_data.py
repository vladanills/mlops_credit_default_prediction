"""
Загрузка данных из файла
"""
import pandas as pd
import pyarrow.parquet as pq


def get_dataset(df_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param df_path: путь до данных
    :return: датасет в формате parquet
    """
    return pd.read_parquet(df_path)
