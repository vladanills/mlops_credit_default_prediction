"""
Программа: Тренировка данных
"""

import optuna
from catboost import CatBoostClassifier, Pool

from optuna import Study

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data
from ..train.metrics import amex_metric
from ..train.metrics import CatBoostEvalMetricCustom
from ..train.metrics import save_metrics


def objective(
    trial,
    cat_feat: list,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_folds: int = 5,
    random_state: int = 10,
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param cat_feat: список категориальных признаков
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [1500]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_uniform("l2_leaf_reg", 1e-5, 1e2),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [2.7971]),
        "eval_metric": CatBoostEvalMetricCustom(),
        "random_state": random_state,
    }

    cv_folds = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_folds.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        train_data = Pool(data=x_train, label=y_train, cat_features=cat_feat)
        eval_data = Pool(data=x_test, label=y_test, cat_features=cat_feat)

        model = CatBoostClassifier(**param_grid)
        model.fit(train_data, eval_set=eval_data, early_stopping_rounds=100, verbose=0)
        predict = model.predict_proba(x_test.values)
        cv_predicts[idx] = amex_metric(y_test.values, predict[:, 1])
    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [CatBoostClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )
    cat_feat = x_train.select_dtypes("category").columns.tolist()
    study = optuna.create_study(direction="maximize", study_name="CAT")
    function = lambda trial: objective(
        trial, cat_feat, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
    random_state: int = 10,
) -> CatBoostClassifier:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :param random_state: random_state
    :return: CatBoostClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # training optimal params
    clf = CatBoostClassifier(
        **study.best_params,
        eval_metric=CatBoostEvalMetricCustom(),
        random_state=random_state
    )
    clf.fit(x_train, y_train)

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf
