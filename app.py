import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from model import (
    open_data,
    preprocess_data,
    predict_on_input,
    load_threshold,
    load_model,
)

from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)


st.set_page_config(layout="wide")

colors = ["#ADCACB", "#FEE3A2"]
sns.set_palette(colors)

cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


# Preload content
def preload_content():
    """Preload content used in the web app."""

    # Load images
    images = {
        "bank": Image.open("data/bank.jpeg"),
        "auto": Image.open("data/auto.png"),
        "age": Image.open("data/age.png"),
        "children": Image.open("data/children.png"),
        "confusion_matrix": Image.open("data/confusion_matrix.png"),
        "credit": Image.open("data/credit.png"),
        "feature_importance": Image.open("data/feature_importance.png"),
        "heatmap": Image.open("data/heatmap.png"),
        "gender": Image.open("data/gender.png"),
        "martial_status": Image.open("data/martial_status.png"),
        "phik": Image.open("data/phik.png"),
        "target": Image.open("data/target.png"),
        "total_loans": Image.open("data/total_loans.png"),
        "work_fl": Image.open("data/work_fl.png"),
        "pens_fl": Image.open("data/pens_fl.png"),
        'dependant': Image.open('data/dependant.png'),
        'cross': Image.open('data/cross.png')
    }
    return images

# Streamlit UI
st.title("Анализ склонности клиентов банка к отклику на предложения")
st.subheader('Исследуем данные клиентов, предсказываем отклик, оцениваем важность факторов')

# Display bank image as cover image
images = preload_content()  # Grab the images dictionary
st.image(images["bank"], use_column_width=False)

tab1, tab2, tab3 = st.tabs(["📊 Анализ", "📈 О модели", "🔍 Предсказать"])
import streamlit as st

with tab1:
    st.header("Данные")
    df = open_data()
    st.dataframe(df.head(7))

    col1, col2 = st.columns(2)

    with col1:
        st.text("• AGE — возраст клиента")
        st.text("• CHILD_TOTAL — количество детей клиента")
        st.text("• DEPENDANTS — количество иждивенцев клиента")
        st.text("• OWN AUTO — автомобили в собственности")
        st.text("• PERSONAL_INCOME — личный доход клиента (в рублях)")
        st.text("• TOTAL_LOANS — количество кредитов клиента")
        st.text("• CLOSED_LOANS — количество погашенных кредитов клиента")

    with col2:
        st.text("• WORK_YEARS — количество рабочих лет клиента")
        st.text("• GENDER — пол клиента")
        st.text("• MARITAL_STATUS — семейное положение")
        st.text("• SOCSTATUS_WORK_FL — работает (1) или нет (0)")
        st.text("• SOCSTATUS_PENS_FL — пенсионер (1) или нет (0)")
        st.text("• FL_PRESENCE_FL — наличие квартиры (1 — да, 0 — нет)")
        st.text("• TARGET — отклик на маркетинговую кампанию (1 — да, 0 — нет)")
        
    st.header("Анализ")

    # Создаём контейнер для сетки
    grid = []
    images_with_captions = [
        (
            images["target"],
            "Целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было). Таргет несбалансирован, что логично, так как обычно люди отказываются от банковских предложений",
        ),
        (images["gender"], "Пол клиента (1 — мужчина, 0 — женщина). В данных банка больше клиентов-мужчин"),
        (images["age"], "Возраст клиента. Можно отметить, что клиенты, соглашающиеся на банковских предложения, в среднем моложе"),
        (images["martial_status"], "Семейное положение. Больше клиентов в состоят в браке"),
        (images["work_fl"], "Работает/не работает. Большая часть клиентов банка работают, а клиенты без работы редко соглашаются на банковские предложения"),
        (images["pens_fl"], "Пенсионер/не пенсионер. Большая часть клиентов банка не являются пенсионерами, а клиенты-пенсионеры редко соглашаются на банковские предложения"),
        (images["children"], "Количество детей. В основном, клиенты имеют до 3 детей"),
        (images["dependant"], "Количество иждивенцев. В основном, клиенты имеют до 3 иждивенцев"),
        (images["auto"], "Автомобили в собственности. Большинство клиентов не имеют авто в собственности"),
        (images["total_loans"], "Кредиты клиента. Большинство клиентов имеют до 3 кредитов"),
    ]

    # Группируем в строки по 2 изображения
    for i in range(0, len(images_with_captions), 2):
        row = st.columns(2)  # Создаём строку с двумя колонками

        # Добавляем два изображения в строку
        for j in range(2):
            if i + j < len(images_with_captions):  # Чтобы не выйти за пределы списка
                with row[j]:  # Берём нужную колонку
                    st.image(
                        images_with_captions[i + j][0],
                        caption=images_with_captions[i + j][1],
                        width=650,
                    )

    # Центрированные доп. графики
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.image(
        images["cross"],
        caption="График попарных зависимостей с отображением таргета. 1 — отклик был зарегистрирован, 0 — отклика не было",
        width=1300,
    )
    st.image(images["phik"], caption="Heatmap для phik корреляция всех признаков, Можно заметить наивысшую корреляцию таргета с возрастом клиентов, рабочим статусом, статусом пенсионера, а также с личным доходом и количеством рабочих лет", width=1300)

with tab2:
    st.header("Модель")
    st.subheader("Описание модели")
    st.write(
        """
    Для предсказания отклика клиентов используется **LogisticRegression** – линейная модель классификации. Логистическая регрессия применяет функцию логистической регрессии для прогнозирования вероятности, что клиент согласится на предложение. Это классическая модель для задач бинарной классификации.

    Перед обучением данные проходят предварительную обработку с помощью **ColumnTransformer**:
    - Числовые признаки стандартизируются с помощью **StandardScaler**, что приводит их к нормальному распределению с нулевым средним и единичной дисперсией. Это помогает LogisticRegression лучше работать, так как она чувствительна к масштабу данных.
    - Категориальные признаки кодируются методом **One-Hot Encoding**.

    """
    )

    st.subheader("Оптимизация порога")
    st.write(
        """
    Целью модели является **максимизация предсказаний класса 1** – клиентов, которые с наибольшей вероятностью согласятся на предложение. Это позволяет **оптимизировать маркетинговые затраты**:
    - **Фокусироваться на клиентах, которые с высокой вероятностью дадут положительный отклик**.
    - **Минимизировать потери времени и ресурсов на заведомо отказавших клиентов**.
    
    Для выбора оптимального порога классификации (`threshold`) использовалась **F2-мера** – метрика, которая придаёт больший вес **полноте (recall)**, что особенно важно для выявления потенциальных клиентов. Таким образом, модель старается минимизировать пропуск клиентов, готовых согласиться на предложение.
    """
    )

    st.subheader("Влияние threshold и Confusion Matrix")
    st.write(
    """
    Confusion Matrix помогает понять, насколько точно модель классифицирует отклики клиентов, разделяя их на четыре категории:

    - **True Positive (TP)**: Клиенты, которые действительно согласились на предложение, и модель правильно предсказала этот отклик.
    - **True Negative (TN)**: Клиенты, которые не согласились на предложение, и модель правильно предсказала отказ.
    - **False Positive (FP)**: Клиенты, которые не согласились на предложение, но модель ошибочно предсказала отклик.
    - **False Negative (FN)**: Клиенты, которые согласились на предложение, но модель ошибочно предсказала отказ.

    С помощью **порога (threshold)** можно настраивать баланс между этими категориями. Чем выше порог, тем меньше клиентов будет предсказано как откликнувшиеся (потому что модель будет более строгой в принятии решения). Это может быть полезно, если хочется минимизировать количество ложных положительных срабатываний.

    ### Настройка порога
    Вы можете изменять порог с помощью ползунка, чтобы увидеть, как это влияет на конфьюжн матрицу и соответствующие метрики. Это позволяет выбрать оптимальное значение порога, которое лучше всего соответствует вашей задаче.
    """)

    # Ваши данные и функции
    X, y, numerical, categorical, preprocessor = preprocess_data(df)
    model = load_model()
    y_pred_proba = expit(model.decision_function(X))  # Преобразуем логиты в вероятности

    # Добавляем слайдер для выбора порога
    threshold = st.slider("Выбери порог для меток", 0.0, 1.0, 0.38, 0.01)
    y_pred_thresholded = (y_pred_proba >= threshold).astype(int)

    # Строим Confusion Matrix
    cm = confusion_matrix(y, y_pred_thresholded)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap, 
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        ax=ax,
    )

    # Уменьшаем шрифт
    ax.set_xlabel("Предсказанный класс", fontsize=8)
    ax.set_ylabel("Истинный класс", fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=10)

    # Настроим размер шрифта для аннотированных чисел
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(7)


    # Размещение графика и метрик рядом
    col1, col2 = st.columns([2, 1])  # 2: для графика, 1: для метрик
    with col1:
        st.pyplot(fig, use_container_width=False)  # График Confusion Matrix

    with col2:
        # Вычисление метрик
        accuracy = accuracy_score(y, y_pred_thresholded)
        recall = recall_score(y, y_pred_thresholded)
        f2 = fbeta_score(y, y_pred_thresholded, beta=2)
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Отображение метрик в виде таблицы
        st.subheader("Метрики модели")
        st.write(f"**Threshold:** {threshold:.2f} - порог")
        st.write(f"**Accuracy:** {accuracy:.2f} – доля правильных предсказаний")
        st.write(
            f"**Recall:** {recall:.2f} – насколько хорошо модель находит откликнувшихся клиентов"
        )
        st.write(f"**F2-score:** {f2:.2f} – ключевая метрика, учитывающая важность полноты")
        st.write(
            f"**ROC-AUC:** {roc_auc:.2f} – качество модели на разных порогах классификации"
        )

    st.subheader("Финальные метрики модели")
    st.write(
        """
    После подбора трешхолда по **F2-мере**, модель достигла следующих показателей на тестовой выборке:
    - **Threshold:** 0.38 - оптимальный порог
    - **Accuracy:** 0.36 – доля правильных предсказаний
    - **Recall:** 0.89 – насколько хорошо модель находит откликнувшихся клиентов
    - **F2-score:** 0.44 – ключевая метрика, учитывающая важность полноты
    - **ROC-AUC:** 0.66 – качество модели на разных порогах классификации
    """
    )

    st.subheader("Feature Importance")

    st.write(
        """
        Важность признаков (Feature Importance) показывает, какие признаки наиболее влияют на прогноз модели. Это позволяет понять, какие факторы наиболее важны для принятия решения моделью.

        - **Положительные коэффициенты** означают, что увеличение значения признака способствует предсказанию положительного отклика.
        - **Отрицательные коэффициенты** означают, что увеличение значения признака способствует предсказанию отрицательного отклика.

        Вы можете просмотреть важность признаков в таблице ниже, где признаки отсортированы по степени влияния на модель.
        """
    )

    st.image(
        images["feature_importance"],
        caption="График влияния признаков на решение модели",
        width=1300,
    )

with tab3:
    # Загрузить модель и порог из файлов
    model = load_model("data/model.pkl")
    best_threshold = load_threshold("data/best_threshold.txt")

    # Функция предсказания
    def predict_on_input(df, model, threshold):
        """Возвращает предсказание и вероятность с использованием оптимального порога"""
        proba = expit(model.decision_function(df))[0]
        pred = int(proba >= threshold)
        return pred, proba

    # Оформление Streamlit
    with st.form(key="prediction_form"):
        st.header("Предсказать")
        st.write("### Даст ли клиент положительный ответ на предложение банка?")

        # Форма ввода данных
        input_data = {}

        categories = {
            "Пол": ["Мужской", "Женский"],
            "Семейный статус": [
                "Состою в браке",
                "Не состоял(а) в браке",
                "Разведен(а)",
                "Вдовец/Вдова",
                "Гражданский брак",
            ],
            "Рабочий статус": ["Да", "Нет"],
            "Статус пенсионера": ["Да", "Нет"],
            "Квартира в собственности": ["Да", "Нет"],
        }

        # Словарь для категориальных данных
        categorical_mapping = {
            "Пол": {"Мужской": 1, "Женский": 0},
            "Рабочий статус": {"Да": 1, "Нет": 0},
            "Статус пенсионера": {"Да": 1, "Нет": 0},
            "Квартира в собственности": {"Да": 1, "Нет": 0},
            "Семейный статус": {
                "Состою в браке": "Состою в браке",
                "Не состоял(а) в браке": "Не состоял(а) в браке",
                "Разведен(а)": "Разведен(а)",
                "Вдовец/Вдова": "Вдовец/Вдова",
                "Гражданский брак": "Гражданский брак",
            },
        }

        for field, options in categories.items():
            input_data[field] = st.selectbox(field, options)

        # Словарь для числовых данных
        fields = {
            "Возраст": (18, 100),
            "Количество детей": (0, 10),
            "Количество иждивенцев": (0, 10),
            "Авто в собственности": (0, 5),
            "Количество рабочих лет": (0, 50),
            "Личный доход": (0, 500000),
            "Количество кредитов": (0, 50),
            "Количество закрытых кредитов": (0, 50),
        }

        for field, (min_val, max_val) in fields.items():
            input_data[field] = st.number_input(
                field, min_value=min_val, max_value=max_val, value=min_val
            )

        # Когда пользователь нажимает кнопку
        submit_button = st.form_submit_button(label="Предсказать")

        if submit_button:
            # Преобразование категориальных значений в числовые
            transformed_input = {
                "GENDER": categorical_mapping["Пол"][input_data["Пол"]],
                "MARITAL_STATUS": categorical_mapping["Семейный статус"][
                    input_data["Семейный статус"]
                ],
                "SOCSTATUS_WORK_FL": categorical_mapping["Рабочий статус"][
                    input_data["Рабочий статус"]
                ],
                "SOCSTATUS_PENS_FL": categorical_mapping["Статус пенсионера"][
                    input_data["Статус пенсионера"]
                ],
                "FL_PRESENCE_FL": categorical_mapping["Квартира в собственности"][
                    input_data["Квартира в собственности"]
                ],
                "AGE": input_data["Возраст"],
                "CHILD_TOTAL": input_data["Количество детей"],
                "DEPENDANTS": input_data["Количество иждивенцев"],
                "OWN_AUTO": input_data["Авто в собственности"],
                "WORK_YEARS": input_data["Количество рабочих лет"],
                "PERSONAL_INCOME": input_data["Личный доход"],
                "TOTAL_LOANS": input_data["Количество кредитов"],
                "CLOSED_LOANS": input_data["Количество закрытых кредитов"],
            }

            input_df = pd.DataFrame([transformed_input])

            # Предсказание с вероятностью
            pred, proba = predict_on_input(input_df, model, best_threshold)

            # Условие для изменения цвета фона
            if pred == 1:
                st.success(
                    f"Предсказание: {'Да' if pred == 1 else 'Нет'} (вероятность: {proba:.2f})"
                )
            else:
                st.markdown(
                    f'<div style="background-color: #E9A098; padding: 10px; color: white;">'
                    f'Предсказание: {"Да" if pred == 1 else "Нет"} (уверенность: {proba:.2f})'
                    f"</div>",
                    unsafe_allow_html=True,
                )
