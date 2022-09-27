"""
Программа: Предобработка данных
"""

import json
import gc
import warnings
import pandas as pd
import numpy as np
from shaphypetune import BoostRFE
from lightgbm import LGBMClassifier
from ..data.get_data import get_dataset

warnings.filterwarnings("ignore")


def drop_na(data: pd.DataFrame):
    """
    Удаление признаков со значительном
    количество пропущенных значений => 0.5
    :param data: датасет
    :return: датасет
    """
    return data.dropna(thresh=int(0.5 * len(data)), axis=1)


def drop_corr_features(data: pd.DataFrame):
    """
    Удаление сильнокоррелирующих признаков > 0.95
    :param data: датасет
    :return: датасет
    """
    corr = data.corr(method="pearson")
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    data.drop(columns=to_drop, inplace=True)
    del corr, upper_tri
    gc.collect()
    return data


def new_s_feature(data: pd.DataFrame):
    """
    Добавление нового признака,
    разница в днях между
    первым и последним обращением
    :param data: датасет
    :return: датасет
    """
    last_day = data.groupby("customer_ID")["S_2"].max()
    first_day = data.groupby("customer_ID")["S_2"].min()
    temp = last_day - first_day
    temp = temp.reset_index()
    temp.columns = ["customer_ID", "S_2_diff"]
    data = data.merge(temp, on="customer_ID")
    data.drop(columns=["S_2"], inplace=True)
    data["S_2"] = data.S_2_diff.dt.days
    data.drop(columns=["S_2_diff"], inplace=True)
    del temp
    gc.collect()
    return data


def fill_na(data: pd.DataFrame):
    """
    Заполнение пропусков
    :param data: датасет
    :return: датасет
    """
    data.fillna(method="ffill", inplace=True)
    data.fillna(method="bfill", inplace=True)


def get_difference(data: pd.DataFrame, num_features: list) -> pd.DataFrame:
    """
    Находим разницы в значение признака между месяцами
    :param data: датасет
    :param num_features: признаки, для которых нужно найти разницу
    :return:
    """
    df1 = []
    customer_ids = []
    for customer_id, df in data.groupby(["customer_ID"]):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(
        df1, columns=[col + "_diff1" for col in df[num_features].columns]
    )
    df1["customer_ID"] = customer_ids

    del diff_df1
    gc.collect()

    return df1


def agg_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Аггрегируем данные. Для числовых находим среднее, мин, макс и разницу между первым и последним элементом
    Для категориальных: количество, последний элемент и количество уникальных значений
    params:
    - df: pd.DataFrame - набор данных
    return: pd.DataFrame
    """
    cols = list(df.columns)
    cat_feat = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_68",
    ]
    for col in cat_feat + ["customer_ID", "S_2"]:
        cols.remove(col)
    num_feat = cols

    num_agg = df.groupby("customer_ID")[num_feat].agg(
        ["mean", "first", "min", "max", "last"]
    )
    num_agg.columns = ["_".join(x) for x in num_agg.columns]

    for col in num_feat:
        num_agg[col + "_diff"] = num_agg[col + "_last"] - num_agg[col + "_first"]
        num_agg.drop(columns=[col + "_first"], inplace=True)
    num_agg.reset_index(inplace=True)

    df_diff = get_difference(df, num_feat)

    cat_agg = df.groupby("customer_ID")[cat_feat].agg(["count", "last", "nunique"])
    cat_agg.columns = ["_".join(x) for x in cat_agg.columns]
    cat_agg.reset_index(inplace=True)

    df = num_agg.merge(cat_agg, how="inner", on="customer_ID").merge(
        df_diff, how="inner", on="customer_ID"
    )

    # print('shape', df.shape)
    del num_agg, cat_agg, df_diff
    gc.collect()
    return df


def change_type(df: pd.DataFrame):
    """

    :param df: датасет
    :return:
    """
    cat_feat = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_68",
    ]
    cat_feat = [x + "_last" for x in cat_feat]
    df[list(set(cat_feat) - {"D_63_last", "D_64_last"})] = df[
        list(set(cat_feat) - {"D_63_last", "D_64_last"})
    ].astype(int)
    return df


def merge(data: pd.DataFrame, target: pd.DataFrame):
    """
    Соединение данных с таргетом
    :param data: датасет
    :param target: таргет
    :return: датасет
    """
    return data.merge(target, on="customer_ID")


def feature_selection(data: pd.DataFrame):
    """
    Произведем рекурсивный отбор признаков
    :param data:
    :return:
    """
    data = data.reset_index()
    X = data.drop(columns=["customer_ID", "target"])
    y = data["target"]
    selector = BoostRFE(LGBMClassifier(), min_features_to_select=40, step=30)
    selector.fit(X, y, verbose=3)
    data = data.loc[
        :, ["customer_ID"] + list(selector.transform(X).columns) + ["target"]
    ]
    data.set_index("customer_ID", inplace=True)

    return data


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(data: pd.DataFrame, **kwargs):
    """
    Пайплайн по предобработке тренирвочных данных
    :param data: датасет
    :return: датасет
    """

    data = drop_na(data)
    data = drop_corr_features(data)
    data = new_s_feature(data)
    fill_na(data)

    data_grouped = agg_data(data)
    data_grouped.set_index("customer_ID", inplace=True)
    data_grouped["S_2"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .tail(1)
        .set_index("customer_ID")
    )

    del data
    gc.collect()

    data_grouped = change_type(data_grouped)
    target = get_dataset(df_path=kwargs["target_path"])
    data = merge(data_grouped, target)

    del data_grouped
    gc.collect()

    fill_na(data)

    data = feature_selection(data)

    #   data.to_parquet(path='C:/Users/farfr/PycharmProjects/credit-default-pred-mlops/data/processed/train.parquet')
    return data


def pipeline_preprocess_test(data: pd.DataFrame, **kwargs):
    """
    Пайплайн по предобработке тестовых данных
    :param data: датасет
    :return: датасет
    """

    data = drop_na(data)
    data = drop_corr_features(data)
    data = new_s_feature(data)
    fill_na(data)

    data_grouped = agg_data(data)
    data_grouped.set_index("customer_ID", inplace=True)
    data_grouped["S_2"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .tail(1)
        .set_index("customer_ID")
    )

    del data
    gc.collect()

    data_grouped = change_type(data_grouped)
    fill_na(data_grouped)

    data = data_grouped.loc[:, list(kwargs["main_columns"])]

    del data_grouped
    gc.collect()

    return data
