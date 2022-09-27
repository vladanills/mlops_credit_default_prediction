"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "logloss": 0}

    # Train
    with st.spinner(
        "Модель подбирает параметры... Можешь попить чаю. Время ожидания до 20 минут :) "
    ):
        output = requests.post(endpoint, timeout=8000)
    st.success("Ура, мы нашли параметры!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
    roc_auc.metric(
        "ROC-AUC",
        new_metrics["roc_auc"],
        f"{new_metrics['roc_auc']-old_metrics['roc_auc']:.3f}",
    )
    precision.metric(
        "Precision",
        new_metrics["precision"],
        f"{new_metrics['precision']-old_metrics['precision']:.3f}",
    )
    recall.metric(
        "Recall",
        new_metrics["recall"],
        f"{new_metrics['recall']-old_metrics['recall']:.3f}",
    )
    f1_metric.metric(
        "F1 score", new_metrics["f1"], f"{new_metrics['f1']-old_metrics['f1']:.3f}"
    )
    logloss.metric(
        "Logloss",
        new_metrics["logloss"],
        f"{new_metrics['logloss']-old_metrics['logloss']:.3f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
