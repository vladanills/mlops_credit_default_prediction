"""
Программа: Модель для прогнозирования того, будут ли страхователи (клиенты),
которые преобретали страховку в прошлом году, заинтересованы в страховании
транспортных средств в данной страховой компании
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = (
    "C:/Users/farfr/PycharmProjects/credit-default-pred-mlops/config/params.yml"
)


@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {"message": "Hello, it is my first project!"}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn main:app --host=0.0.0.0 --port=8000 --reload
