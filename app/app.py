"""Приложение: влияние социально-экономических факторов на преступность в регионах РФ.

Запуск из корня репозитория:  streamlit run app/app.py
Данные: data/processed/panel.csv (очищенная таблица, ноутбук 03) и data/processed/features.json
(итоговые факторы, ноутбук 04).

Прототип интерфейса методики: модели обучаются прямо в приложении.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "panel.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.json"
TARGET = "crime_rate"

FEATURES_INFO = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
BASE = FEATURES_INFO["base"]            # 7 социально-экономических факторов
CONTROL = FEATURES_INFO["control"]      # доля нераскрытых — включается переключателем
FEATURE_NAMES = FEATURES_INFO["names"]

ALPHAS = [0.1, 1, 10, 100, 1000]
# Регион по умолчанию для демонстрации: лучше всего описывается гибридной моделью
# (средняя ошибка 5,8 %, корреляция оценок с фактом по годам 0,93; перекрёстная проверка на 5 частей)
DEMO_REGION = "Иркутская область"
SPLITS = ["Случайно 70/30", "По годам (тест 2020–2022)", "По регионам (30 % регионов)"]

# Методы и их краткие описания для подсказок
METHODS = {
    "Линейная регрессия": "Прямая зависимость: Y = b₀ + b₁·X₁ + … Самый простой и понятный метод.",
    "Lasso": "Линейная регрессия со штрафом, который может обнулить коэффициенты лишних факторов.",
    "Ridge": "Линейная регрессия со штрафом, который уменьшает слишком большие коэффициенты.",
    "ElasticNet": "Смесь Lasso и Ridge: оба штрафа сразу.",
    "Полиномиальная регрессия": "Формула с квадратами, кубами и произведениями факторов: учитывает нелинейные "
                                "зависимости.",
    "k ближайших соседей (kNN)": "Оценка — среднее уровня преступности у k самых похожих по факторам строк.",
    "Случайный лес": "Среднее множества деревьев решений вида «безработица больше 7 %? да/нет → …».",
    "Гибридная модель (лес + kNN)": "Ансамбль методов: оценка — среднее оценок случайного леса и k ближайших "
                                    "соседей. Состав выбран в ноутбуке 05 перебором всех сочетаний методов.",
}

st.set_page_config(page_title="Факторы преступности в регионах РФ", layout="wide")


# ---------- Данные ----------
@st.cache_data
def load_data():
    """Читает очищенную таблицу «регион × год»."""
    return pd.read_csv(DATA_PATH)


data = load_data()


# ---------- Модели ----------
def make_model(method, p):
    """Собирает модель по названию метода и словарю параметров p."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # масштабирование факторов перед моделью
    if method == "Линейная регрессия":
        return scaled(LinearRegression())
    if method == "Lasso":
        return scaled(Lasso(alpha=p["alpha"], max_iter=50000))
    if method == "Ridge":
        return scaled(Ridge(alpha=p["alpha"]))
    if method == "ElasticNet":
        return scaled(ElasticNet(alpha=p["alpha"], l1_ratio=0.5, max_iter=50000))
    if method == "Полиномиальная регрессия":
        final = Ridge(alpha=p["alpha"]) if p["use_ridge"] else LinearRegression()
        return make_pipeline(StandardScaler(), PolynomialFeatures(p["degree"], include_bias=False),
                             StandardScaler(), final)
    if method == "k ближайших соседей (kNN)":
        return scaled(KNeighborsRegressor(n_neighbors=p["k"]))
    if method == "Гибридная модель (лес + kNN)":
        return VotingRegressor([("лес", RandomForestRegressor(n_estimators=p["trees"], random_state=1, n_jobs=-1)),
                                ("kNN", scaled(KNeighborsRegressor(n_neighbors=p["k"])))])
    return RandomForestRegressor(n_estimators=p["trees"], random_state=1, n_jobs=-1)


# Параметры по умолчанию для вкладки «Сравнение методов» (начальные значения из плана, файл 03)
DEFAULTS = {
    "Линейная регрессия": {},
    "Lasso": {"alpha": 0.1},
    "Ridge": {"alpha": 1},
    "ElasticNet": {"alpha": 0.1},
    "Полиномиальная регрессия": {"degree": 2, "use_ridge": True, "alpha": 10},
    "k ближайших соседей (kNN)": {"k": 5},
    "Случайный лес": {"trees": 300},
    "Гибридная модель (лес + kNN)": {"k": 5, "trees": 300},
}


def split_rows(split_type):
    """Номера строк обучающей и тестовой выборок."""
    idx = np.arange(len(data))
    if split_type.startswith("По годам"):
        return idx[data["year"] <= 2019], idx[data["year"] >= 2020]
    if split_type.startswith("Случайно"):
        return train_test_split(idx, test_size=0.3, random_state=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    return next(gss.split(data, groups=data["region"]))


def freeze(p):
    """Словарь параметров → кортеж (нужно для запоминания результатов)."""
    return tuple(sorted(p.items()))


@st.cache_resource
def train(features, method, params, split_type):
    """Обучает модель; результат запоминается, чтобы не переобучать при каждом клике."""
    tr, te = split_rows(split_type)
    model = make_model(method, dict(params))
    model.fit(data.loc[tr, list(features)], data.loc[tr, TARGET])
    return model, tr, te


@st.cache_data
def compare_methods(features, split_type):
    """Обучает все методы с параметрами по умолчанию и возвращает таблицу метрик."""
    rows = []
    for method, p in DEFAULTS.items():
        model, tr, te = train(features, method, freeze(p), split_type)
        X, y = data[list(features)], data[TARGET]
        p_tr, p_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
        label = method + (f", степень {p['degree']}" if "degree" in p else "")
        rows.append({"метод": label, "R² на тесте": r2_score(y.iloc[te], p_te),
                     "R² на обучении": r2_score(y.iloc[tr], p_tr),
                     "средняя ошибка": mean_absolute_error(y.iloc[te], p_te)})
    return pd.DataFrame(rows).sort_values("R² на тесте", ascending=False).reset_index(drop=True)


@st.cache_data
def factor_effects(features, method, params, split_type):
    """Важность (падение R² при перемешивании фактора) и направление (эффект роста фактора на 10 %)."""
    model, tr, te = train(features, method, params, split_type)
    X_te, y_te = data.loc[te, list(features)], data.loc[te, TARGET]
    perm = permutation_importance(model, X_te, y_te, scoring="r2", n_repeats=5, random_state=1)
    base = model.predict(X_te)
    rows = []
    for i, f in enumerate(features):
        shifted = X_te.copy()
        shifted[f] = shifted[f] * 1.1
        rows.append({"фактор": FEATURE_NAMES[f], "важность": perm.importances_mean[i],
                     "эффект +10 %": float(np.mean(model.predict(shifted) - base))})
    return pd.DataFrame(rows)


# Сетка подбора для каждого метода: какие параметры перебирать
GRIDS = {
    "Lasso": [{"alpha": a} for a in ALPHAS],
    "Ridge": [{"alpha": a} for a in ALPHAS],
    "ElasticNet": [{"alpha": a} for a in ALPHAS],
    "Полиномиальная регрессия": [{"degree": d, "use_ridge": True, "alpha": a} for d in range(1, 5) for a in ALPHAS],
    "k ближайших соседей (kNN)": [{"k": k} for k in [1, 3, 5, 7, 10, 15, 20, 30]],
    "Случайный лес": [{"trees": t} for t in [50, 100, 300, 500]],
    "Гибридная модель (лес + kNN)": [{"k": k, "trees": t} for k in [3, 5, 7, 10] for t in [100, 300]],
}


def tune(features, method, split_type):
    """Перебирает параметры метода с перекрёстной проверкой на обучающей выборке (5 частей).
    Тестовая выборка в подборе не участвует, иначе её оценка станет завышенной."""
    tr, _ = split_rows(split_type)
    X, y = data.loc[tr, list(features)], data.loc[tr, TARGET]
    if split_type.startswith("По регионам"):
        cv, groups = GroupKFold(n_splits=5), data.loc[tr, "region"]
    else:
        cv, groups = KFold(n_splits=5, shuffle=True, random_state=1), None
    rows = []
    for p in GRIDS[method]:
        s = cross_val_score(make_model(method, p), X, y, cv=cv, groups=groups, scoring="r2").mean()
        rows.append({**p, "R² на проверке": s})
    return pd.DataFrame(rows).sort_values("R² на проверке", ascending=False).reset_index(drop=True)


def current_features():
    return BASE + (CONTROL if st.session_state.get("use_control", True) else [])


def on_tune():
    """Кнопка подбора: ставит лучшие параметры в боковую панель."""
    method, split_type = st.session_state.method, st.session_state.split_type
    table = tune(tuple(current_features()), method, split_type)
    best = table.iloc[0]
    for key in ["degree", "k", "trees"]:
        if key in best and not pd.isna(best[key]):
            st.session_state[key] = int(best[key])
    if "alpha" in best and not pd.isna(best["alpha"]):
        st.session_state.alpha = min(ALPHAS, key=lambda a: abs(a - float(best["alpha"])))
    if method == "Полиномиальная регрессия":
        st.session_state.use_ridge = True
    st.session_state.tune_table = table
    st.session_state.tune_for = (tuple(current_features()), method, split_type)


# ---------- Боковая панель ----------
for key, value in {"degree": 2, "alpha": 10, "k": 5, "trees": 300, "use_ridge": True}.items():
    st.session_state.setdefault(key, value)

st.sidebar.header("Настройки модели")

method = st.sidebar.selectbox("Метод", list(METHODS), index=len(METHODS) - 1, key="method",
                              help="Каким способом строится связь факторов с уровнем преступности.")
st.sidebar.caption(METHODS[method])

use_control = st.sidebar.checkbox(
    "Учитывать долю нераскрытых преступлений", value=True, key="use_control",
    help="Доля нераскрытых — показатель работы полиции, к социально-экономическим он не относится. "
         "Без него оценки социально-экономических факторов могут смещаться; переключатель позволяет сравнить "
         "модели с ним и без него.",
)
features = current_features()
with st.sidebar.expander(f"Факторы модели ({len(features)})"):
    st.markdown("\n".join(f"- {FEATURE_NAMES[f]}" for f in features))
    st.caption("Отобраны в ноутбуке 04: доходы и ВРП исключены как дублирующие зарплату.")

params = {}
if method == "Полиномиальная регрессия":
    params["degree"] = st.sidebar.slider(
        "Степень полинома", 1, 6, key="degree",
        help="1 — прямая линия; 2 — добавляются квадраты и произведения факторов; чем выше степень, "
             "тем гибче формула и тем выше риск переобучения.")
    params["use_ridge"] = st.sidebar.checkbox(
        "Регуляризация (Ridge)", key="use_ridge",
        help="Штраф за слишком большие коэффициенты: не даёт формуле изгибаться под каждую точку.")
if method in ("Lasso", "Ridge", "ElasticNet") or (method == "Полиномиальная регрессия" and params["use_ridge"]):
    params["alpha"] = st.sidebar.select_slider(
        "Сила регуляризации alpha", options=ALPHAS, key="alpha",
        help="Насколько сильно сдерживать модель. Мало (0,1) — модель почти свободна и легко переобучается; "
             "много (1000) — модель осторожная и простая, может недоучиться.")
elif method == "Полиномиальная регрессия":
    params["alpha"] = None
if method in ("k ближайших соседей (kNN)", "Гибридная модель (лес + kNN)"):
    params["k"] = st.sidebar.slider(
        "Число соседей k", 1, 30, key="k",
        help="Сколько самых похожих строк усреднять. Мало соседей — модель повторяет отдельные строки и "
             "переобучается; много — оценки сглаживаются и становятся грубее.")
if method in ("Случайный лес", "Гибридная модель (лес + kNN)"):
    params["trees"] = st.sidebar.select_slider(
        "Число деревьев", options=[50, 100, 300, 500], key="trees",
        help="Больше деревьев — устойчивее результат, но дольше расчёт.")

split_type = st.sidebar.radio(
    "Способ проверки", SPLITS, key="split_type",
    help="Какие данные спрятать от модели при обучении, чтобы потом проверить, как она их угадывает. "
         "Основной способ — случайно 70/30, стандартная процедура оценки моделей. По годам — угадать 2020–2022 годы; "
         "по регионам — угадать регионы, которых модель не видела.")

if method != "Линейная регрессия":
    st.sidebar.button("Подобрать лучшие параметры", on_click=on_tune, width="stretch",
                      help="Перебирает параметры метода и ставит лучшие. Проверка идёт только на обучающих "
                           "данных, тестовые в подборе не участвуют.")

model, tr, te = train(tuple(features), method, freeze(params), split_type)
X, y = data[features], data[TARGET]
pred_tr, pred_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
r2_tr, r2_te = r2_score(y.iloc[tr], pred_tr), r2_score(y.iloc[te], pred_te)
mae_te = mean_absolute_error(y.iloc[te], pred_te)
num = lambda v, d: f"{v:,.{d}f}".replace(",", " ").replace(".", ",")  # 12 345,67

# ---------- Заголовок ----------
st.title("Влияние социально-экономических факторов на преступность в регионах РФ")
st.caption("Данные: Генпрокуратура и Росстат (обработка «Если быть точным»), переписи 2010 и 2020 гг.; "
           f"85 регионов, 2011–2022 гг., {len(data)} строк после очистки.")

with st.expander("Что делает приложение и как его читать", expanded=False):
    st.markdown(f"""
**Модель** — это формула. На вход она получает показатели региона за год (безработица, зарплата и другие,
список слева), на выходе выдаёт число: **сколько преступлений на 100 тыс. жителей должно быть** при таких
показателях. Это число называется **оценкой модели**.

**Как модель проверяется.** Часть данных прячется от модели (тестовая выборка). Модель учится на остальном
(обучающая выборка), а потом угадывает спрятанное. Что именно прятать, задаёт «Способ проверки» слева.
Сейчас: **{split_type.lower()}**; в обучении {len(tr)} строк, в тесте {len(te)}.

**Когда модель обучается.** При любом изменении настроек слева модель обучается заново на тех же данных.
Ползунки на вкладке «Что, если» модель не переобучают: они только меняют показатели, которые подаются
в уже обученную модель.

**Вкладки:**
- «Сравнение методов» — все восемь методов на одних данных: какой точнее;
- «Регион» — реальный уровень преступности в регионе по годам и оценка модели;
- «Качество модели» — насколько точно выбранная модель угадывает спрятанные данные;
- «Влияние факторов» — какие показатели важнее для модели и в какую сторону они тянут оценку;
- «Что, если» — как изменится оценка модели, если изменить показатели региона.

Модель показывает **связь** показателей с преступностью; причинность она не доказывает.
""")

# ---------- Метрики ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² на тесте", f"{r2_te:.3f}",
          help="Главная цифра: какую долю различий в уровне преступности модель угадывает на спрятанных данных. "
               "1 — угадывает всё; 0 — не лучше, чем всем назвать среднее; меньше 0 — хуже среднего. "
               "Целевой уровень около 0,85, допустимый — от 0,8.")
c2.metric("R² на обучении", f"{r2_tr:.3f}",
          help="То же на данных, на которых модель училась. Обычно выше, чем на тесте.")
c3.metric("Средняя ошибка на тесте", f"{mae_te:.0f}",
          help=f"На сколько преступлений на 100 тыс. жителей модель в среднем ошибается. "
               f"Для сравнения: средний уровень в данных {y.mean():.0f}.")
if method == "Полиномиальная регрессия":
    c4.metric("Признаков в модели", comb(len(features) + params["degree"], params["degree"]) - 1,
              help=f"Сколько комбинаций факторов (квадраты, произведения) строит полином. Строк в обучении: "
                   f"{len(tr)}. Если признаков больше, чем строк, модель может просто запомнить данные.")

if r2_te < 0:
    st.error(f"R² на тесте {num(r2_te, 2)}: модель на спрятанных данных хуже, чем простое среднее. Обычно это "
             f"значит, что формула слишком сложная для такого объёма данных и на незнакомых данных выдаёт "
             f"нереальные значения (самая низкая оценка на тесте: {num(pred_te.min(), 0)} преступлений на 100 тыс., "
             f"хотя меньше нуля их быть не может). Попробуйте более простые настройки или кнопку "
             f"«Подобрать лучшие параметры».")
elif r2_tr - r2_te > 0.15:
    st.warning(f"Разрыв между обучением и тестом {num(r2_tr - r2_te, 2)}: признак переобучения. На знакомых данных "
               f"модель точнее, чем на новых, значит, часть данных она запомнила. Порог 0,15 взят из методических "
               f"правил работы.")

if st.session_state.get("tune_for") == (tuple(features), method, split_type):
    with st.expander("Результат подбора параметров", expanded=True):
        st.write(f"Подбор для метода «{method}» и способа проверки «{split_type}». Каждая строка — вариант "
                 "параметров; «R² на проверке» — среднее по пяти проверкам внутри обучающей выборки (тестовые "
                 "данные в подборе не участвуют). Лучший вариант (первая строка) выставлен слева; его итоговое "
                 "качество — «R² на тесте» выше.")
        st.dataframe(st.session_state.tune_table.head(10).round(3), hide_index=True, width="stretch")

tab_compare, tab_region, tab_model, tab_factors, tab_whatif = st.tabs(
    ["Сравнение методов", "Регион", "Качество модели", "Влияние факторов", "Что, если"])

# ---------- Вкладка: сравнение методов ----------
with tab_compare:
    st.caption(f"Все методы обучены на одних и тех же данных ({len(features)} факторов, способ проверки "
               f"«{split_type}») с начальными параметрами из плана работы. Параметры выбранного слева метода "
               "можно менять и подбирать отдельно.")
    table = compare_methods(tuple(features), split_type)
    fig = px.bar(table.sort_values("R² на тесте"), x="R² на тесте", y="метод", orientation="h",
                 title="R² на тестовой выборке по методам", text_auto=".3f")
    fig.add_vline(x=0.85, line_dash="dash", line_color="grey")
    fig.add_vline(x=0.8, line_dash="dot", line_color="grey")
    fig.update_layout(yaxis_title="", xaxis_range=[min(0, table["R² на тесте"].min()), 1])
    st.plotly_chart(fig, width="stretch")
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("Чем длиннее столбец, тем точнее метод на спрятанных данных. Пунктирные линии: штрихи — целевой "
               "уровень 0,85, точки — допустимый уровень 0,8. Если «R² на обучении» намного выше «R² на тесте», "
               "метод переобучается.")

# ---------- Вкладка: регион ----------
regions = sorted(data["region"].unique())
with tab_region:
    st.caption("Две линии по годам для выбранного региона: **факт** — реальный уровень преступности, "
               "**модель** — оценка модели по показателям региона в этом году. Чем ближе линии, тем лучше "
               "модель описывает регион.")
    region = st.selectbox("Регион", regions, index=regions.index(DEMO_REGION))
    reg = data[data["region"] == region].sort_values("year")
    chart = pd.DataFrame({"год": reg["year"], "факт": reg[TARGET].values, "модель": model.predict(reg[features])})
    fig = px.line(chart.melt("год", var_name="ряд", value_name="преступлений на 100 тыс."),
                  x="год", y="преступлений на 100 тыс.", color="ряд", markers=True,
                  title=f"{region}: уровень преступности, факт и оценка модели")
    st.plotly_chart(fig, width="stretch")
    st.caption("Таблица: показатели региона по годам, на которых модель строит оценку.")
    st.dataframe(reg[["year", TARGET] + features].rename(
        columns={"year": "год", TARGET: "преступлений на 100 тыс.", **FEATURE_NAMES}).round(2),
        hide_index=True, width="stretch")

# ---------- Вкладка: качество модели ----------
with tab_model:
    left, right = st.columns(2)
    with left:
        fact_pred = pd.DataFrame({"факт": y.iloc[te].values, "оценка модели": pred_te,
                                  "регион": data["region"].iloc[te].values, "год": data["year"].iloc[te].values})
        fig = px.scatter(fact_pred, x="факт", y="оценка модели", hover_data=["регион", "год"],
                         title="Факт и оценка модели на тестовой выборке")
        lo, hi = y.min(), y.max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
        if pred_te.min() < 0 or pred_te.max() > 2 * hi:
            fig.update_yaxes(range=[0, 1.3 * hi])
            st.caption("Часть оценок вышла далеко за реальные значения и не видна на графике.")
        st.plotly_chart(fig, width="stretch")
        st.caption("Каждая точка — один регион в один год из **спрятанных** данных. По горизонтали — реальный "
                   "уровень преступности, по вертикали — оценка модели. Точка на пунктирной диагонали — модель "
                   "угадала точно; чем дальше от диагонали, тем больше ошибка. Нажмите на точку, чтобы увидеть "
                   "регион и год.")
    with right:
        # Как меняется качество при изменении главного параметра выбранного метода
        curve = {"Полиномиальная регрессия": ("degree", list(range(1, 7)), "степень полинома"),
                 "Lasso": ("alpha", ALPHAS, "alpha"), "Ridge": ("alpha", ALPHAS, "alpha"),
                 "ElasticNet": ("alpha", ALPHAS, "alpha"),
                 "k ближайших соседей (kNN)": ("k", [1, 3, 5, 7, 10, 15, 20, 30], "число соседей k"),
                 "Случайный лес": ("trees", [50, 100, 300, 500], "число деревьев"),
                 "Гибридная модель (лес + kNN)": ("k", [1, 3, 5, 7, 10, 15, 20, 30], "число соседей k")}.get(method)
        if curve:
            key, values, label = curve
            rows = []
            for v in values:
                m, _, _ = train(tuple(features), method, freeze({**params, key: v}), split_type)
                rows += [{label: v, "выборка": "обучение", "R²": r2_score(y.iloc[tr], m.predict(X.iloc[tr]))},
                         {label: v, "выборка": "тест", "R²": max(r2_score(y.iloc[te], m.predict(X.iloc[te])), -1)}]
            fig = px.line(pd.DataFrame(rows), x=label, y="R²", color="выборка", markers=True,
                          title=f"R² при разных значениях: {label}", log_x=(key == "alpha"))
            fig.add_hline(y=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85")
            # на логарифмической оси положение линии задаётся в логарифмах
            fig.add_vline(x=np.log10(params[key]) if key == "alpha" else params[key], line_color="orange")
            fig.update_yaxes(range=[-1.05, 1.05])
            st.plotly_chart(fig, width="stretch")
            st.caption("Как меняется качество, если менять главный параметр метода при остальных настройках. "
                       "Линия «обучение» показывает, насколько модель подстраивается под знакомые данные; линия "
                       "«тест» — насколько она угадывает новые. Если «обучение» растёт, а «тест» падает, это "
                       "переобучение. Оранжевая вертикаль — текущее значение, пунктир — ориентир 0,85. "
                       "Значения ниже −1 показаны как −1.")
        else:
            st.info("У линейной регрессии нет параметров для настройки.")

# ---------- Вкладка: влияние факторов ----------
with tab_factors:
    effects = factor_effects(tuple(features), method, freeze(params), split_type)
    left, right = st.columns(2)
    with left:
        fig = px.bar(effects.sort_values("важность"), x="важность", y="фактор", orientation="h",
                     title="Важность факторов для точности модели")
        fig.update_layout(yaxis_title="", xaxis_title="падение R² на тесте")
        st.plotly_chart(fig, width="stretch")
        st.caption("Значения одного фактора по очереди перемешиваются между строками (так его связь с "
                   "преступностью разрушается), и оценивается, насколько ухудшилась точность. Чем длиннее столбец, "
                   "тем сильнее модель опирается на этот фактор. Направление связи этот график не показывает.")
    with right:
        eff = effects.sort_values("эффект +10 %")
        eff["направление"] = np.where(eff["эффект +10 %"] > 0, "оценка растёт", "оценка снижается")
        fig = px.bar(eff, x="эффект +10 %", y="фактор", color="направление", orientation="h",
                     color_discrete_map={"оценка растёт": "#d62728", "оценка снижается": "#2ca02c"},
                     title="Что будет с оценкой, если фактор вырастет на 10 %")
        fig.update_layout(yaxis_title="", xaxis_title="изменение, преступлений на 100 тыс.")
        st.plotly_chart(fig, width="stretch")
        st.caption("Для каждой строки тестовой выборки один фактор увеличивается на 10 % и оценивается, на сколько "
                   "в среднем изменилась оценка модели. Красный — оценка преступности растёт, зелёный — "
                   "снижается. Это связь при прочих равных внутри модели; причинность она не доказывает. "
                   "При низком R² на тесте выводы по этому графику ненадёжны.")

# ---------- Вкладка: сценарий «что, если» ----------
with tab_whatif:
    st.markdown("Выберите регион и год. Модель оценит уровень преступности в этом регионе **в этом же году**, "
                "если бы его показатели были другими. Это не прогноз на будущее: меняются условия, год остаётся "
                "тем же.")
    col_a, col_b = st.columns(2)
    w_region = col_a.selectbox("Регион", regions, index=regions.index(DEMO_REGION), key="w_region")
    years = sorted(data.loc[data["region"] == w_region, "year"])
    w_year = col_b.selectbox("Год", years, index=len(years) - 1)
    row = data[(data["region"] == w_region) & (data["year"] == w_year)]
    base = row[features]

    st.caption("Ползунки: на сколько процентов изменить показатель относительно его реального значения "
               "в выбранном году.")
    scenario = base.copy()
    cols = st.columns(3)
    for i, f in enumerate(features):
        change = cols[i % 3].slider(f"{FEATURE_NAMES[f]} (сейчас {base[f].iloc[0]:.2f}), %", -50, 50, 0,
                                    step=5, key=f"s_{f}")
        scenario[f] = base[f] * (1 + change / 100)

    train_min, train_max = X.iloc[tr].min(), X.iloc[tr].max()
    outside = [FEATURE_NAMES[f] for f in features if not train_min[f] <= scenario[f].iloc[0] <= train_max[f]]
    if outside:
        st.warning("Значения вышли за пределы того, что встречалось в обучающих данных: " + ", ".join(outside)
                   + ". Модель таких регионов не видела, её оценка здесь ненадёжна.")

    base_pred, new_pred = model.predict(base)[0], model.predict(scenario)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Факт", f"{row[TARGET].iloc[0]:.0f}",
              help="Реальный уровень преступности в регионе в выбранном году, на 100 тыс. жителей.")
    m2.metric("Оценка модели при реальных показателях", f"{base_pred:.0f}",
              help="Что модель вычисляет по реальным показателям региона в этом году. Разница с фактом — "
                   "ошибка модели для этого региона.")
    m3.metric("Оценка модели по сценарию", f"{new_pred:.0f}", delta=f"{new_pred - base_pred:+.0f}",
              delta_color="inverse",
              help="Что модель вычисляет с изменёнными показателями. Стрелка — разница с оценкой при реальных "
                   "показателях (красный — больше преступлений, зелёный — меньше).")
    st.caption("Все значения — преступлений на 100 тыс. жителей. Оценки ориентировочные: модель показывает "
               "связь показателей с преступностью, причинность она не доказывает.")
