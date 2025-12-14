import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = 'data.adult.csv'  # загружаем датасет

# чтобы было красиво и пользователю былл удобнр взаимодействовать я перевела все категориальные признаки на русский
WORKCLASS_LABELS = {
    "Federal-gov":      "В федеральное правительство",
    "Local-gov":        "В местное правительство",
    "Private":          "На частную компанию",
    "Self-emp-inc":     "Самозанятый (с юр. лицом)",
    "Self-emp-not-inc": "Самозанятый (без юр. лица)",
    "State-gov":        "Правительство",
    "Without-pay":      "Без оплаты / волонтёр",
    "Never-worked":     "Никогда не работал(а)",
}
EDUCATION_LABELS = {
    "Preschool":   "Дошкольное образование",
    "1st-4th":     "1–4 классы",
    "5th-6th":     "5–6 классы",
    "7th-8th":     "7–8 классы",
    "9th":         "9 классов",
    "10th":        "10 классов",
    "11th":        "11 классов",
    "12th":        "12 классов",
    "HS-grad":     "Средняя школа",
    "Some-college": "Незаконченное высшее",
    "Assoc-acdm":  "Ассоц. степень (академ.)",
    "Assoc-voc":   "Ассоц. степень (проф.)",
    "Bachelors":   "Бакалавр",
    "Masters":     "Магистр",
    "Prof-school": "Проф. школа",
    "Doctorate":   "Доктор наук",
}
MARITAL_LABELS = {
    "Divorced":             "Разведён(а)",
    "Married-AF-spouse":    "Женат/замужем (вооружённые силы)",
    "Married-civ-spouse":   "Женат/замужем (гражданский брак)",
    "Married-spouse-absent":"Женат/замужем, супруг(а) отсутствует",
    "Never-married":        "Никогда не был(а) в браке",
    "Separated":            "В разводе / раздельно",
    "Widowed":              "Вдовец / вдова",
}
OCCUPATION_LABELS = {
    "Handlers-cleaners":  "Грузчики / уборщики",
    "Machine-op-inspct":  "Оператор / инспектор машин",
    "Other-service":      "Сфера услуг (прочее)",
    "Priv-house-serv":    "Домашний персонал",
    "Prof-specialty":     "Профессиональная специальность",
    "Protective-serv":    "Охранные службы",
    "Sales":              "Продажи",
    "Tech-support":       "Техническая поддержка",
    "Transport-moving":   "Транспорт / перевозки",
    "Exec-managerial":    "Руководство / менеджмент",
    "Craft-repair":       "Ремесло / ремонт",
    "Adm-clerical":       "Административная работа",
    "Farming-fishing":    "Сельское хозяйство / рыболовство",
    "Armed-Forces":       "Вооружённые силы",
}
RELATION_LABELS = {
    "Husband":       "Муж",
    "Wife":          "Жена",
    "Not-in-family": "Не в семье",
    "Other-relative":"Другой родственник",
    "Own-child":     "Собственный ребёнок",
    "Unmarried":     "Не в браке",
}
SEX_LABELS = {
    "Female": "Женщина",
    "Male":   "Мужчина",
}
RACE_LABELS = {
    "White":              "Я РУССКИЙ",
    "Black":              "Чёрный",
    "Asian-Pac-Islander": "Азиат / тихоокеанский островитянин",
    "Amer-Indian-Eskimo": "Индейец / эскимос",
    "Other":              "Другая раса",
}
#тут загружаем и предобрабатываем данные
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.replace("?", np.nan).dropna()
    target_col = df.columns[-1]
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int)

    feature_cols = [
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    X = df[feature_cols].copy()
    return X, y


# а тут уже обучается сама модель
@st.cache_resource
def train_best_model(X: pd.DataFrame, y: pd.Series):
    numeric_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=12,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X, y)
    return model, numeric_features, categorical_features

def get_options(X_all: pd.DataFrame, col: str):
    return sorted(X_all[col].unique().tolist())


# а тут уже наш фронтэнд для пользователя
def user_input_form(X_all: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🧾 Введите данные о себе")

    age = st.sidebar.slider("Возраст", min_value=17, max_value=90, value=25, step=1)

    workclass = st.sidebar.selectbox(
        "Тип работы",
        options=get_options(X_all, "workclass"),
        format_func=lambda x: WORKCLASS_LABELS.get(x, x),
    )

    education = st.sidebar.selectbox(
        "Образование",
        options=get_options(X_all, "education"),
        format_func=lambda x: EDUCATION_LABELS.get(x, x),
    )

    marital_status = st.sidebar.selectbox(
        "Семейное положение",
        options=get_options(X_all, "marital-status"),
        format_func=lambda x: MARITAL_LABELS.get(x, x),
    )

    occupation = st.sidebar.selectbox(
        "Род деятельности",
        options=get_options(X_all, "occupation"),
        format_func=lambda x: OCCUPATION_LABELS.get(x, x),
    )

    relationship = st.sidebar.selectbox(
        "Отношение в семье",
        options=get_options(X_all, "relationship"),
        format_func=lambda x: RELATION_LABELS.get(x, x),
    )

    race = st.sidebar.selectbox(
        "Раса",
        options=get_options(X_all, "race"),
        format_func=lambda x: RACE_LABELS.get(x, x),
    )

    sex = st.sidebar.radio(
        "Пол",
        options=get_options(X_all, "sex"),
        format_func=lambda x: SEX_LABELS.get(x, x),
        horizontal=True,
    )

    capital_gain = st.sidebar.number_input(
        "Ваш пассивный доход / прибыль от капитала",
        min_value=0,
        max_value=100000,
        value=0,
        step=100,
    )

    capital_loss = st.sidebar.number_input(
        "Ваши капитальные потери на работе",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
    )

    hours_per_week = st.sidebar.slider(
        "Часов работы в неделю",
        min_value=1,
        max_value=99,
        value=40,
        step=1,
    )

    data = {
        "age": age,
        "workclass": workclass,         
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
    }

    return pd.DataFrame([data])


# основная фукнция
def main():
    st.set_page_config(
        page_title="Новогодний предсказатель вашей вероятности смерти в нищите",
        page_icon="🎄",
        layout="centered"
    )
    st.title("🎄 Новогодний предсказатель вашей вероятности смерти в нищите")
    st.markdown(
        """
        Сейчас небольшая моя моделька машинного обучения оценит,
        звонит вам жизнь в богатстве или смерть в нищете.

        ⚠️ *Всё, что вы увидите ниже — моя домашка по питону, а не финансовая консультация.*
        """
    )
    X_all, y_all = load_data(DATA_PATH)
    model, numeric_features, categorical_features = train_best_model(X_all, y_all)
    user_df = user_input_form(X_all)

    st.subheader("Ваши итоговые данные для модели")
    st.dataframe(user_df, use_container_width=True)

    st.markdown("---")

    if st.button("Кто звонит"):
        proba = model.predict_proba(user_df)[0, 1]
        pred = int(model.predict(user_df)[0])
        proba_percent = proba * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Вероятность дохода > 50k",
                value=f"{proba_percent:.1f} %",
            )
            st.progress(min(max(proba, 0.0), 1.0))

        with col2:
            if pred == 1:
                st.success("✅ Ура, вам звонит жизнь в богатстве!")
            else:
                st.warning("⚠️ Блин, это смерть в нищете...")
        st.markdown("### Пу пу пу...")

        if proba < 0.2:
            st.write(
                """
                🥲 **Смерть в нищите:**  
                с днём бич-пакета.  
                вы живёте на одну стипу..?
                """
            )
        elif proba < 0.5:
            st.write(
                """
                😐 **Среднячок:**  
                можно сказать, вы эталон среднего класса, поздравляю.  
                но модель не уверена, что вы не скатитесь в нищету в любой момент.  
                50 на 50, как говорится.
                """
            )
        elif proba < 0.8:
            st.write(
                """
                😎 **Смерть в богатстве:**  
                ура, вы можете купить себе больше, чем один бич-пакет.  
                поделитесь.  
                ибо я живу на одну стипуху.
                """
            )
        else:
            st.balloons()
            st.snow()
            st.write(
                """
                🤑 **Жизнь в богатстве:**  
                почему вы с таким доходом читаете мою домашку?  
                идите инвестируйте в крипту и нефть.
                """
            )
        st.markdown(
            """
            ---  
            🤖  
            Если модель ошиблась — тем хуже для модели, а не для вас.  
            """
        )
    else:
        st.info("👈 Заполните параметры в сайдбаре и нажмите кнопку, чтобы узнать судьбу своего кошелька.")
    st.markdown(
        """
        <div style="text-align:center; color:grey; font-size:0.8rem; margin-top:2rem;">
        Сделано на Streamlit, с новым годом 🎄<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()