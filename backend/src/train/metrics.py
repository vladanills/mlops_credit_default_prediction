"""
Программа:получение метрик
"""
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import pandas as pd
import numpy as np
import json
import yaml


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисление метрики соревнования
    """

    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=["target"])

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=["prediction"])

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )

        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
        return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={"target": "prediction"})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    d = top_four_percent_captured(y_true, y_pred)
    g = normalized_weighted_gini(y_true, y_pred)

    return 0.5 * (g + d)


class CatBoostEvalMetricCustom(object):
    """
    eval metric для catbost
    """

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        score = amex_metric(target, preds)
        return score, 0


def create_dict_metrics(
    y_test: np.ndarray, y_predict: np.ndarray, y_probability: np.ndarray
) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_probability: предсказанные вероятности
    :return: словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probability[:, 1]), 4),
        "precision": round(precision_score(y_test, y_predict), 4),
        "recall": round(recall_score(y_test, y_predict), 4),
        "f1": round(f1_score(y_test, y_predict), 4),
        "logloss": round(log_loss(y_test, y_probability), 4),
        "amex": round(amex_metric(y_test, y_probability[:, 1]), 4),
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=data_y.values,
        y_predict=model.predict(data_x.values),
        y_probability=model.predict_proba(data_x.values),
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
