"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc


def pie_plot_f(data: pd.DataFrame) -> matplotlib.figure.Figure:
    """

    :param data: датасет
    :return: поле рисунка
    """
    delinquency_features = [i for i in data.columns if i.startswith("D_")]
    spend_features = [i for i in data.columns if i.startswith("S_")]
    payment_features = [i for i in data.columns if i.startswith("P_")]
    balance_features = [i for i in data.columns if i.startswith("B_")]
    risk_features = [i for i in data.columns if i.startswith("R_")]

    values = [
        len(delinquency_features),
        len(spend_features),
        len(payment_features),
        len(balance_features),
        len(risk_features),
    ]
    labels = ["delinquency", "spend", "payment", "balance", "risk"]

    fig = plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%.1f%%")
    plt.title("Variables", fontsize=14)

    return fig


def pie_plot_t(data: pd.DataFrame) -> matplotlib.figure.Figure:
    """

    :param data: датасет
    :return: поле рисунка
    """
    target = data.target.value_counts(normalize=True).mul(100)
    fig = plt.figure(figsize=(5, 5))
    plt.pie(target, autopct="%.1f%%", labels=target.index)
    plt.title("Target", fontsize=14)

    return fig


def main_stats(dataset: pd.DataFrame, features: list):
    """
    Главные статистики группы признаков.
    :param data: датасет
    :param features: список признаков
    :return:
    """
    data = dataset[features]
    fig = (
        data.describe()
        .T.style.bar(subset=["max"], color="red")
        .bar(
            subset=[
                "mean",
            ],
            color="blue",
        )
    )

    return fig


def eda_feat_targ(dataset: pd.DataFrame, features: list) -> matplotlib.figure.Figure:
    """
    Разведочный анализ группы признаков.
    :param data: датасет
    :param features: список признаков
    :return: поле рисунка
    """
    data_merged = dataset[features + ["target"]]
    corr = data_merged.loc[:300000, features + ["target"]].corr(method="pearson")[
        ["target"]
    ]

    cor_with_t = []
    cor_with_t = [index for index in corr.index if corr.loc[index][0] >= 0.3]
    cor_with_t.remove("target")
    if not cor_with_t:
        cor_with_t = cor_with_t + features
    if len(cor_with_t) == 1:
        fig = plt.figure()
        sns.kdeplot(
            data=data_merged[:100000], common_norm=False, x=cor_with_t[0], hue="target"
        )
    else:
        fig = plt.figure(figsize=(20, 5 * round(len(cor_with_t) / 2)))
        nrows = round(len(cor_with_t) / 2)
        ncols = 2
        cor_with_t = cor_with_t[: ncols * nrows]
        for l in range(len(cor_with_t)):
            ax = fig.add_subplot(nrows, ncols, l + 1)
            sns.kdeplot(
                data=data_merged[:100000],
                common_norm=False,
                x=cor_with_t[l],
                hue="target",
                ax=ax,
            )
    return fig
