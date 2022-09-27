"""
Программа: предсказание из файла
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Предсказать")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data.groupby("customer_ID").head(1).head(5)
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_[["customer_ID", "predict"]].head())
