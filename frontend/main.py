"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
import streamlit.components.v1 as components
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import pie_plot_f, pie_plot_t, main_stats, eda_feat_targ
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """

    st.image(
        "https://miro.medium.com/max/1400/1*uZyt9Z189siaNsAlIDtjEg.jpeg",
        width=600,
    )

    st.title("Credit default Prediction")

    components.html(
        """
        <b style='font-size:30px'><span  style='color:#4B4B4B'>1 |</span><span style='color:#016CC9'> Описание проекта</span></b>
        """,
        height=50,
    )
    st.write(
        """
        American Express — глобальная интегрированная платежная компания. Являясь крупнейшим эмитентом платежных карт в мире, они предоставляют клиентам доступ к продуктам, информации и опыту, которые обогащают жизнь и способствуют успеху в бизнесе.
        """
    )
    st.write(
        """Целью этой задачи является прогнозирование вероятности того, что клиент не выплатит сумму остатка по кредитной карте в будущем, на основе его ежемесячного профиля клиента."""
    )
    st.write(
        """Задача состоит в том, чтобы предсказать для каждого customer_ID вероятность невыполнения платежа в будущем (target = 1)."""
    )
    components.html(
        """
        <b style='font-size:30px'><span  style='color:#4B4B4B'>2 |</span><span style='color:#016CC9'> Описание полей</span></b>
        """,
        height=50,
    )
    st.markdown(
        """
            Набор данных содержит агрегированные характеристики профиля для каждого клиента на каждую дату выписки. Функции анонимизированы и нормализованы и делятся на следующие общие категории:
               - D_* = переменные просроченной задолженности
               - S_* = Расходные переменные
               - P_* = Платежные переменные
               - B_* = Балансовые переменные
               - R_* = Переменные риска
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Разведочный анализ данных️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(df_path=config["preprocessing"]["train_path"])
    delinquency_features = [i for i in data.columns if i.startswith("D_")]
    spend_features = [i for i in data.columns if i.startswith("S_")]
    payment_features = [i for i in data.columns if i.startswith("P_")]
    balance_features = [i for i in data.columns if i.startswith("B_")]
    risk_features = [i for i in data.columns if i.startswith("R_")]

    t = get_dataset(df_path=config["preprocessing"]["target_path"])
    st.write(data.head())

    # plotting with checkbox
    feature_response = st.sidebar.checkbox("Соотношение признаков")
    stats = st.sidebar.checkbox("Главные статистики")
    target = st.sidebar.checkbox("Соотношение значений таргета")
    feat_targ = st.sidebar.checkbox("Отношение признаки/таргет")

    if feature_response:
        st.pyplot(pie_plot_f(data=data))
    if stats:
        if st.button("Delinquency features"):
            st.write(main_stats(data, delinquency_features))
        if st.button("Spend features"):
            st.write(main_stats(data, spend_features))
        if st.button("Payment features"):
            st.write(main_stats(data, payment_features))
        if st.button("Balance features"):
            st.write(main_stats(data, balance_features))
        if st.button("Risk features"):
            st.write(main_stats(data, risk_features))
    if target:
        st.pyplot(pie_plot_t(data=t))
    if feat_targ:
        data_merged = data.merge(t, on="customer_ID")
        if st.button("Delinquency features"):
            st.pyplot(eda_feat_targ(data_merged, delinquency_features))
        if st.button("Spend features"):
            st.pyplot(eda_feat_targ(data_merged, spend_features))
        if st.button("Payment features"):
            st.pyplot(eda_feat_targ(data_merged, payment_features))
        if st.button("Balance features"):
            st.pyplot(eda_feat_targ(data_merged, balance_features))
        if st.button("Risk features"):
            st.pyplot(eda_feat_targ(data_merged, risk_features))


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model CatBoost")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Начать тренировку"):
        start_training(config=config, endpoint=endpoint)


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Предсказание")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader("", type=["parquet"], accept_multiple_files=False)
    # проверка загружен ли файл
    if upload_file:
        dataset_parquet_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_parquet_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Разведочный анализ данных": exploratory,
        "Тренировка": training,
        "Предсказание из файла": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()

# streamlit run main.py
