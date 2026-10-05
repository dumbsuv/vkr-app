"""Прототип приложения: влияние социально-экономических факторов на преступность в регионах РФ.

Запуск из корня репозитория:  streamlit run app/app.py
Данные: data/processed/panel_test.csv (создаёт ноутбук notebooks/01_test_data.ipynb).

Это черновик для показа консультанту: модели обучаются прямо в приложении на пробной таблице,
окончательный выбор факторов и модели будет на этапе 5 плана.
"""
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "panel_test.csv"
TARGET = "crime_rate"

# Понятные названия факторов для интерфейса
FEATURE_NAMES = {
    "unemployment": "Безработица, %",
    "wage_rel": "Зарплата к медиане регионов",
    "poverty": "Доля бедных, %",
    "grp_rel": "ВРП на душу к медиане регионов",
    "urban": "Доля городского населения, %",
    "migration": "Миграционный прирост на 10 тыс.",
    "students": "Студенты вузов на 10 тыс.",
    "higher_edu_share": "Доля с высшим образованием, %",
    "unsolved_share": "Доля нераскрытых преступлений, %",
}

st.set_page_config(page_title="Факторы преступности в регионах РФ", layout="wide")


# ---------- Загрузка данных ----------
@st.cache_data
def load_data():
    """Читает таблицу «регион × год» и убирает строки с пропусками."""
    panel = pd.read_csv(DATA_PATH)
    return panel.dropna(subset=[TARGET] + list(FEATURE_NAMES)).reset_index(drop=True)


data = load_data()

# ---------- Боковая панель: настройки модели ----------
st.sidebar.header("Настройки модели")

features = st.sidebar.multiselect(
    "Факторы", options=list(FEATURE_NAMES), default=list(FEATURE_NAMES),
    format_func=FEATURE_NAMES.get,
)
model_type = st.sidebar.radio("Метод", ["Полиномиальная регрессия", "Случайный лес"])
degree = st.sidebar.slider("Степень полинома", min_value=1, max_value=6, value=2,
                           disabled=model_type != "Полиномиальная регрессия")
use_ridge = st.sidebar.checkbox("Регуляризация (Ridge)", value=True,
                                disabled=model_type != "Полиномиальная регрессия")
alpha = st.sidebar.select_slider("Сила регуляризации alpha", options=[0.1, 1, 10, 100, 1000], value=10,
                                 disabled=not use_ridge or model_type != "Полиномиальная регрессия")
split_type = st.sidebar.radio("Способ проверки", ["По годам (тест 2020–2022)", "Случайно 70/30", "По регионам (30 % регионов)"])

if not features:
    st.warning("Выберите хотя бы один фактор.")
    st.stop()


# ---------- Модель ----------
def make_model():
    """Собирает модель по настройкам из боковой панели."""
    if model_type == "Случайный лес":
        return RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=-1)
    final = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler(), final)


def split_rows():
    """Возвращает номера строк обучающей и тестовой выборок."""
    idx = np.arange(len(data))
    if split_type.startswith("По годам"):
        return idx[data["year"] <= 2019], idx[data["year"] >= 2020]
    if split_type.startswith("Случайно"):
        return train_test_split(idx, test_size=0.3, random_state=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    return next(gss.split(data, groups=data["region"]))


@st.cache_resource
def train(features_key, model_key, split_key):
    """Обучает модель; кэш по настройкам, чтобы не переобучать при каждом клике."""
    tr, te = split_rows()
    X, y = data[list(features_key)], data[TARGET]
    model = make_model().fit(X.iloc[tr], y.iloc[tr])
    return model, tr, te


model, tr, te = train(tuple(features), (model_type, degree, use_ridge, alpha), split_type)
X, y = data[features], data[TARGET]
pred_tr, pred_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
r2_tr, r2_te = r2_score(y.iloc[tr], pred_tr), r2_score(y.iloc[te], pred_te)

# ---------- Заголовок и метрики ----------
st.title("Влияние социально-экономических факторов на преступность в регионах РФ")
st.caption("Прототип. Данные: Генпрокуратура и Росстат (обработка «Если быть точным»), переписи 2010 и 2020 гг.; "
           "85 регионов, 2011–2022 гг. Y — зарегистрированные преступления на 100 тыс. жителей.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² на тесте", f"{r2_te:.3f}", help="Ориентир консультанта около 0,85")
c2.metric("R² на обучении", f"{r2_tr:.3f}")
c3.metric("Средняя ошибка (MAE)", f"{mean_absolute_error(y.iloc[te], pred_te):.0f}", help="преступлений на 100 тыс.")
if model_type == "Полиномиальная регрессия":
    n_feat = comb(len(features) + degree, degree) - 1
    c4.metric("Признаков в модели", n_feat, help=f"строк в обучении: {len(tr)}")
if r2_tr - r2_te > 0.15:
    st.warning(f"Разрыв между обучением и тестом {r2_tr - r2_te:.2f}: признак переобучения.")

tab_region, tab_model, tab_whatif = st.tabs(["Регион", "Качество модели", "Сценарий «что, если»"])

# ---------- Вкладка 1: регион ----------
with tab_region:
    region = st.selectbox("Регион", sorted(data["region"].unique()),
                          index=sorted(data["region"].unique()).index("Москва"))
    reg = data[data["region"] == region].sort_values("year")
    reg_pred = model.predict(reg[features])
    chart = pd.DataFrame({"год": reg["year"], "факт": reg[TARGET].values, "модель": reg_pred})
    fig = px.line(chart.melt("год", var_name="ряд", value_name="преступлений на 100 тыс."),
                  x="год", y="преступлений на 100 тыс.", color="ряд", markers=True,
                  title=f"{region}: уровень преступности")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(reg[["year", TARGET] + features].rename(columns={"year": "год", TARGET: "преступлений на 100 тыс.",
                                                                  **FEATURE_NAMES}).round(2),
                 hide_index=True, width="stretch")

# ---------- Вкладка 2: качество модели ----------
with tab_model:
    left, right = st.columns(2)
    with left:
        fact_pred = pd.DataFrame({"факт": y.iloc[te].values, "прогноз": pred_te,
                                  "регион": data["region"].iloc[te].values, "год": data["year"].iloc[te].values})
        fig = px.scatter(fact_pred, x="факт", y="прогноз", hover_data=["регион", "год"],
                         title="Прогноз и факт на тестовой выборке")
        lo, hi = fact_pred[["факт", "прогноз"]].min().min(), fact_pred[["факт", "прогноз"]].max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, width="stretch")
    with right:
        if model_type == "Полиномиальная регрессия":
            rows = []
            for d in range(1, 7):
                final = Ridge(alpha=alpha) if use_ridge else LinearRegression()
                m = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), StandardScaler(), final)
                m.fit(X.iloc[tr], y.iloc[tr])
                rows += [{"степень": d, "выборка": "обучение", "R²": r2_score(y.iloc[tr], m.predict(X.iloc[tr]))},
                         {"степень": d, "выборка": "тест", "R²": max(r2_score(y.iloc[te], m.predict(X.iloc[te])), -1)}]
            fig = px.line(pd.DataFrame(rows), x="степень", y="R²", color="выборка", markers=True,
                          title="R² в зависимости от степени (тест ниже −1 обрезан)")
            fig.add_hline(y=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85")
            fig.add_vline(x=degree, line_color="orange")
            fig.update_yaxes(range=[-1.05, 1.05])
            st.plotly_chart(fig, width="stretch")
        else:
            imp = pd.Series(model.feature_importances_, index=[FEATURE_NAMES[f] for f in features]).sort_values()
            fig = px.bar(imp, orientation="h", title="Важность факторов в случайном лесе")
            fig.update_layout(showlegend=False, xaxis_title="важность", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

# ---------- Вкладка 3: сценарий «что, если» ----------
with tab_whatif:
    st.write("Выберите регион и год, затем измените факторы. Прогноз ориентировочный: модель показывает связь, "
             "причинность она не доказывает.")
    col_a, col_b = st.columns(2)
    w_region = col_a.selectbox("Регион", sorted(data["region"].unique()), key="w_region")
    years = sorted(data.loc[data["region"] == w_region, "year"])
    w_year = col_b.selectbox("Базовый год", years, index=len(years) - 1)
    base = data[(data["region"] == w_region) & (data["year"] == w_year)][features].iloc[[0]]

    scenario = base.copy()
    cols = st.columns(3)
    for i, f in enumerate(features):
        change = cols[i % 3].slider(f"{FEATURE_NAMES[f]}: изменение, %", -50, 50, 0, step=5, key=f"s_{f}")
        scenario[f] = base[f] * (1 + change / 100)

    base_pred, new_pred = model.predict(base)[0], model.predict(scenario)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Прогноз при текущих значениях", f"{base_pred:.0f}")
    m2.metric("Прогноз по сценарию", f"{new_pred:.0f}", delta=f"{new_pred - base_pred:+.0f}", delta_color="inverse")
    m3.metric("Факт в базовом году", f"{data.loc[(data['region'] == w_region) & (data['year'] == w_year), TARGET].iloc[0]:.0f}")
