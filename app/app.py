"""Приложение: влияние социально-экономических факторов на преступность в регионах РФ.

Запуск из корня репозитория:  streamlit run app/app.py
Данные: data/processed/panel.csv (очищенная таблица, ноутбук 03), data/processed/features.json
(итоговые факторы, ноутбук 04), data/processed/crime_categories.csv (уровни по видам преступлений, ноутбук 07).

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
PARAMS_PATH = ROOT / "data" / "processed" / "final_params.json"  # параметры, подобранные в ноутбуке 06
FINAL_PATH = ROOT / "data" / "processed" / "final_comparison.csv"  # результаты сравнения из ноутбука 06
CAT_PATH = ROOT / "data" / "processed" / "crime_categories.csv"  # уровни по видам преступлений, ноутбук 07
CAT_COMPARE_PATH = ROOT / "data" / "processed" / "category_comparison.csv"  # точность по видам, ноутбук 07
CAT_FACTORS_PATH = ROOT / "data" / "processed" / "category_factors.csv"  # важность и направление по видам
LIMITS_PATH = ROOT / "data" / "processed" / "scenario_limits.json"  # пределы сдвига для подбора, ноутбук 08

# Виды преступлений: название в интерфейсе → (ключ, столбец с уровнем на 100 тыс. жителей)
CATEGORIES = {
    "Все преступления": ("all", "rate_all"),
    "Тяжкие и особо тяжкие": ("grave", "rate_grave"),
    "Небольшой и средней тяжести": ("minor", "rate_minor"),
    "Экономической направленности": ("econ", "rate_econ"),
    "Связанные с наркотиками": ("drugs", "rate_drugs"),
    "Убийства и покушения на убийство": ("murder", "rate_murder"),
    "Кражи": ("theft", "rate_theft"),
    "Грабежи и разбои": ("robbery", "rate_robbery"),
}
# Полные названия видов для заголовков и пояснений
CATEGORY_PHRASE = {
    "Все преступления": "все преступления",
    "Тяжкие и особо тяжкие": "тяжкие и особо тяжкие преступления",
    "Небольшой и средней тяжести": "преступления небольшой и средней тяжести",
    "Экономической направленности": "преступления экономической направленности",
    "Связанные с наркотиками": "преступления, связанные с незаконным оборотом наркотиков",
    "Убийства и покушения на убийство": "убийства и покушения на убийство",
    "Кражи": "кражи",
    "Грабежи и разбои": "грабежи и разбои",
}
CATEGORY_HELP = ("Какие преступления считать. «Все преступления» — основной показатель работы. Остальные виды "
                 "позволяют проверить, одинаково ли факторы связаны с разными преступлениями. Модели для каждого "
                 "вида обучаются отдельно. Кражи, грабежи и разбои — по данным МВД из ЕМИСС, остальные виды — "
                 "по данным Генпрокуратуры.")

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
                                    "соседей. Состав выбран перебором всех сочетаний методов: это сочетание точнее остальных.",
}

st.set_page_config(page_title="Факторы преступности в регионах РФ", layout="wide")

# Цвета графиков. Акцентный синий совпадает с цветом элементов (файл .streamlit/config.toml);
# оранжевый — второй ряд. Пара проверена на различимость при нарушениях цветового зрения,
# на белом и на тёмном фоне. Красный и зелёный оставлены только для смысла «выше» и «ниже».
ACCENT, SECOND, NEUTRAL = "#2F74E0", "#DD6B1F", "#8A8F98"
UP, DOWN = "#D03B3B", "#2F9E44"
px.defaults.color_discrete_sequence = [ACCENT, SECOND]


# ---------- Данные ----------
@st.cache_data
def load_final():
    """Готовые результаты окончательного сравнения методов (вложенная проверка считается несколько минут,
    поэтому в приложении не пересчитывается)."""
    return pd.read_csv(FINAL_PATH)


@st.cache_data
def load_data():
    """Читает очищенную таблицу «регион × год» и добавляет уровни по видам преступлений."""
    panel = pd.read_csv(DATA_PATH)
    return panel.merge(pd.read_csv(CAT_PATH), on=["region", "year"], how="left")


@st.cache_data
def load_categories():
    """Готовые результаты ноутбука 07: точность моделей, важность и направление факторов по видам."""
    return pd.read_csv(CAT_COMPARE_PATH), pd.read_csv(CAT_FACTORS_PATH, index_col=0)


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
        return scaled(ElasticNet(alpha=p["alpha"], l1_ratio=p.get("l1_ratio", 0.5), max_iter=50000))
    if method == "Полиномиальная регрессия":
        final = Ridge(alpha=p["alpha"]) if p["use_ridge"] else LinearRegression()
        return make_pipeline(StandardScaler(), PolynomialFeatures(p["degree"], include_bias=False),
                             StandardScaler(), final)
    knn = lambda: scaled(KNeighborsRegressor(n_neighbors=p["k"], weights=p.get("weights", "uniform")))
    forest = lambda: RandomForestRegressor(n_estimators=p["trees"], max_features=p.get("max_features", 1.0),
                                           random_state=1, n_jobs=-1)
    if method == "k ближайших соседей (kNN)":
        return knn()
    if method == "Гибридная модель (лес + kNN)":
        return VotingRegressor([("лес", forest()), ("kNN", knn())])
    return forest()


# Параметры для вкладки «Сравнение методов»: подобраны в ноутбуке 06 перебором по сетке
# с перекрёстной проверкой на обучающей выборке (70 %)
FINAL = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
KNN_P = {"k": FINAL["k ближайших соседей"]["n_neighbors"], "weights": FINAL["k ближайших соседей"]["weights"]}
RF_P = {"trees": FINAL["Случайный лес"]["n_estimators"], "max_features": FINAL["Случайный лес"]["max_features"]}
DEFAULTS = {
    "Линейная регрессия": {},
    "Lasso": {"alpha": FINAL["Lasso"]["alpha"]},
    "Ridge": {"alpha": FINAL["Ridge"]["alpha"]},
    "ElasticNet": {"alpha": FINAL["ElasticNet"]["alpha"], "l1_ratio": FINAL["ElasticNet"]["l1_ratio"]},
    "Полиномиальная регрессия": {"degree": FINAL["Полиномиальная регрессия"]["degree"], "use_ridge": True,
                                 "alpha": FINAL["Полиномиальная регрессия"]["alpha"]},
    "k ближайших соседей (kNN)": KNN_P,
    "Случайный лес": RF_P,
    "Гибридная модель (лес + kNN)": {**KNN_P, **RF_P},
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


num = lambda v, d: (f"{v:,.{d}f}".replace(",", "\u00a0").replace(".", ",")  # 12 345,67, пробел неразрывный
                    .replace("-", "−"))


@st.cache_resource
def train(features, method, params, split_type, target):
    """Обучает модель для столбца target; результат запоминается, чтобы не переобучать при каждом клике."""
    tr, te = split_rows(split_type)
    model = make_model(method, dict(params))
    model.fit(data.loc[tr, list(features)], data.loc[tr, target])
    return model, tr, te


@st.cache_data
def compare_methods(features, split_type, target):
    """Обучает все методы с параметрами по умолчанию и возвращает таблицу метрик."""
    rows = []
    for method, p in DEFAULTS.items():
        model, tr, te = train(features, method, freeze(p), split_type, target)
        X, y = data[list(features)], data[target]
        p_tr, p_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
        label = method + (f", степень {p['degree']}" if "degree" in p else "")
        rows.append({"метод": label, "R² на тесте": r2_score(y.iloc[te], p_te),
                     "R² на обучении": r2_score(y.iloc[tr], p_tr),
                     "средняя ошибка": mean_absolute_error(y.iloc[te], p_te)})
    return pd.DataFrame(rows).sort_values("R² на тесте", ascending=False).reset_index(drop=True)


@st.cache_data
def factor_effects(features, method, params, split_type, target):
    """Важность (падение R² при перемешивании фактора) и направление (эффект роста фактора на 10 %)."""
    model, tr, te = train(features, method, params, split_type, target)
    X_te, y_te = data.loc[te, list(features)], data.loc[te, target]
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
    "k ближайших соседей (kNN)": [{"k": k, "weights": w} for k in [1, 2, 3, 5, 7, 10, 15]
                                  for w in ["uniform", "distance"]],
    "Случайный лес": [{"trees": 300, "max_features": m} for m in [0.33, 0.66, 1.0]],
    "Гибридная модель (лес + kNN)": [{"k": k, "weights": "distance", "trees": 300, "max_features": m}
                                     for k in [2, 3, 5] for m in [0.33, 0.66, 1.0]],
}


def tune(features, method, split_type, target):
    """Перебирает параметры метода с перекрёстной проверкой на обучающей выборке (5 частей).
    Тестовая выборка в подборе не участвует, иначе её оценка станет завышенной."""
    tr, _ = split_rows(split_type)
    X, y = data.loc[tr, list(features)], data.loc[tr, target]
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


def current_target():
    return CATEGORIES[st.session_state.get("category", "Все преступления")][1]


def on_tune():
    """Кнопка подбора: ставит лучшие параметры в боковую панель."""
    method, split_type, target = st.session_state.method, st.session_state.split_type, current_target()
    table = tune(tuple(current_features()), method, split_type, target)
    best = table.iloc[0]
    for key in ["degree", "k", "trees"]:
        if key in best and not pd.isna(best[key]):
            st.session_state[key] = int(best[key])
    if "weights" in best and not pd.isna(best["weights"]):
        st.session_state.knn_distance = best["weights"] == "distance"
    if "max_features" in best and not pd.isna(best["max_features"]):
        st.session_state.max_features = float(best["max_features"])
    if "alpha" in best and not pd.isna(best["alpha"]):
        st.session_state.alpha = min(ALPHAS, key=lambda a: abs(a - float(best["alpha"])))
    if method == "Полиномиальная регрессия":
        st.session_state.use_ridge = True
    st.session_state.tune_table = table
    st.session_state.tune_for = (tuple(current_features()), method, split_type, target)


# ---------- Сценарий «что, если» по усреднённой зависимости ----------
# Показатели в процентах не могут превысить 100
PERCENT_FEATURES = [f for f, name in FEATURE_NAMES.items() if name.endswith("%")]


@st.cache_data
def scenario_effects(features, method, params, split_type, target, changes):
    """Как в среднем меняется оценка модели, если изменить факторы на заданные проценты.

    changes — изменения факторов в процентах (в том же порядке, что features).
    Шаг 1: все строки тестовой выборки сдвигаются на одни и те же проценты.
    Шаг 2: сравнивается средняя оценка модели до и после сдвига.
    Усреднение по многим регионам сглаживает ступенчатый отклик леса и kNN в отдельной точке.
    Возвращает общее изменение в процентах и изменение от каждого фактора по отдельности."""
    model, tr, te = train(features, method, params, split_type, target)
    X_te = data.loc[te, list(features)]
    base = model.predict(X_te).mean()
    changes = np.array(changes, dtype=float)
    # Первый сценарий — все изменения сразу, дальше — каждое изменение отдельно
    scenarios = [changes] + [np.where(np.arange(len(features)) == i, c, 0.0) for i, c in enumerate(changes)]
    frames = []
    for ch in scenarios:
        shifted = X_te * (1 + ch / 100)
        for f in features:
            if f in PERCENT_FEATURES:
                shifted[f] = shifted[f].clip(upper=100)
        frames.append(shifted)
    pred = model.predict(pd.concat(frames)).reshape(len(scenarios), len(X_te)).mean(axis=1)
    rel = (pred / base - 1) * 100
    return float(rel[0]), dict(zip(features, rel[1:].tolist()))


# ---------- Подбор сценария с ограничениями (ноутбук 08) ----------
# Управляемые факторы и пределы сдвига за 1 и 3 года (90-й процентиль наблюдавшихся изменений)
LIMITS = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
MANAGED = list(LIMITS["1"])
SELECTION_NOTE = (
    "Подбор меняет только безработицу и долю бедных: эти вопросы входят в полномочия органов власти субъекта "
    "(занятость населения и социальная помощь малоимущим, закон № 414-ФЗ, ст. 44). Оба показателя можно только "
    "снижать, и не сильнее, чем они менялись у 90 % регионов за выбранный срок: безработица — до "
    f"{LIMITS['1']['unemployment']['limit']} % за год и {LIMITS['3']['unemployment']['limit']} % за три года, доля "
    f"бедных — до {LIMITS['1']['poverty']['limit']} % и {LIMITS['3']['poverty']['limit']} %. Значения не выходят за "
    "пределы того, что встречалось в регионах. Остальные показатели либо не зависят напрямую от решений региона "
    "(образование, городское население, миграция), либо их связь с преступностью в данных объясняется "
    "особенностями отдельных регионов (зарплата).")


def select_scenario(features, method, params, split_type, target, horizon, region_values):
    """Подбор: для каждого управляемого фактора перебираются сдвиги −5, −10, … до предела горизонта
    (и не ниже минимума фактора в данных); выбирается сдвиг с наибольшим снижением оценки.
    Возвращает таблицу по факторам (от сильного эффекта к слабому) и совместный эффект в процентах."""
    rows = []
    for f in MANAGED:
        limit = LIMITS[horizon][f]["limit"]
        floor_pct = (data[f].min() / region_values[f] - 1) * 100  # правило: не ниже минимума в данных
        limit = min(limit, int(-floor_pct // 5 * 5))
        best_pct, best_eff = 0, 0.0
        for pct in range(-5, -limit - 1, -5):
            changes = tuple(pct if g == f else 0 for g in features)
            _, single = scenario_effects(features, method, params, split_type, target, changes)
            if single[f] < best_eff:
                best_pct, best_eff = pct, single[f]
        rows.append({"код": f, "показатель": FEATURE_NAMES[f], "изменение показателя, %": best_pct,
                     "изменение уровня преступности, %": best_eff})
    table = pd.DataFrame(rows).sort_values("изменение уровня преступности, %").reset_index(drop=True)
    chosen = {r["код"]: r["изменение показателя, %"] for _, r in table.iterrows() if r["изменение показателя, %"] < 0}
    total = 0.0
    if chosen:
        total, _ = scenario_effects(features, method, params, split_type, target,
                                    tuple(chosen.get(g, 0) for g in features))
    return table, total


def selection_block(features, method, params, split_type, target, region_values, level, slider_prefix, key):
    """Блок «Подобрать сценарий»: выбор срока, кнопка, таблица результата и перенос в ползунки."""
    with st.container(border=True):
        st.markdown("**Подобрать сценарий.** Какое реалистичное снижение безработицы и бедности сильнее всего "
                    "связано со снижением уровня преступности в регионе.")
        horizon = st.radio("Срок сценария", ["1", "3"], format_func=lambda h: "1 год" if h == "1" else "3 года",
                           horizontal=True, key=f"{key}_horizon",
                           help="За какой срок нужно достичь изменений. От срока зависит, насколько сильно "
                                "показатель может реально измениться.")
        if st.button("Подобрать", key=f"{key}_run", type="primary"):
            st.session_state[f"{key}_result"] = (horizon, target) + select_scenario(
                features, method, params, split_type, target, horizon, region_values)
        result = st.session_state.get(f"{key}_result")
        if result and result[0] == horizon and result[1] == target:
            _, _, table, total = result
            used = table[table["изменение показателя, %"] < 0]
            if used.empty:
                st.info("Для выбранного вида преступлений снижение безработицы и бедности в допустимых пределах "
                        "не связано со снижением оценки.")
                return
            new_level = level * (1 + total / 100)
            accusative = {"unemployment": "безработицу", "poverty": "долю бедных"}
            parts = [f"{accusative[r['код']]} на {abs(r['изменение показателя, %'])} %" for _, r in used.iterrows()]
            st.markdown(f"Сценарий на {'1 год' if horizon == '1' else '3 года'}: снизить " + " и ".join(parts)
                        + f". С этими условиями связано снижение уровня преступности примерно на "
                          f"**{num(abs(total), 1)} %**: с {num(level, 0)} до {num(new_level, 0)} на 100 тыс. жителей.")
            shown = table.drop(columns="код").copy()
            shown["изменение показателя, %"] = shown["изменение показателя, %"].map(lambda v: num(v, 0))
            shown["изменение уровня преступности, %"] = shown["изменение уровня преступности, %"].map(lambda v: num(v, 1))
            st.dataframe(shown, hide_index=True, width="stretch")
            st.caption("Строки упорядочены от более сильной связи к более слабой; в каждой строке — эффект "
                       "изменения одного показателя. Если показатель в сценарий не вошёл (0 %), его снижение в "
                       "допустимых пределах не связано со снижением оценки.")

            def apply():
                for f in features:
                    st.session_state[f"{slider_prefix}{f}"] = 0
                for _, r in used.iterrows():
                    st.session_state[f"{slider_prefix}{r['код']}"] = int(r["изменение показателя, %"])

            st.button("Перенести в ползунки", key=f"{key}_apply", on_click=apply,
                      help="Выставляет подобранные изменения на ползунках ниже; остальные ползунки — в ноль.")
        with st.expander("Как устроен подбор"):
            st.markdown(SELECTION_NOTE + " Для каждого показателя перебираются снижения на 5, 10, 15 % и далее до "
                        "предела и выбирается то, с которым связано наибольшее снижение преступности. Результат "
                        "показывает связь в данных и не является прогнозом последствий мер.")


def effects_chart(single, features):
    """Столбцы: на сколько процентов меняется уровень преступности от каждого изменения по отдельности."""
    eff = pd.DataFrame({"фактор": [FEATURE_NAMES[f] for f in features], "изменение, %": [single[f] for f in features]})
    eff = eff[eff["изменение, %"].abs() > 0.05].sort_values("изменение, %")
    eff["направление"] = np.where(eff["изменение, %"] > 0, "преступность выше", "преступность ниже")
    fig = px.bar(eff, x="изменение, %", y="фактор", color="направление", orientation="h", text_auto=".1f",
                 color_discrete_map={"преступность выше": UP, "преступность ниже": DOWN},
                 title="Вклад каждого изменения по отдельности")
    fig.update_layout(yaxis_title="", xaxis_title="изменение уровня преступности, %", legend_title="",
                      height=160 + 60 * len(eff))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


WAGE_NOTE = ("Почему рост зарплаты связан с ростом преступности. Самые высокие зарплаты относительно других "
             "регионов — на Севере и Дальнем Востоке (Чукотский АО, Ямало-Ненецкий АО, Магаданская область, "
             "Камчатский край, Сахалинская область), и уровень преступности там в основном выше типичного. "
             "Модель отражает эту связь в данных; из неё не следует, что повышение зарплат вызывает рост "
             "преступности.")

# ---------- Режим работы ----------
SIMPLE, ADVANCED = "Для органов власти", "Для аналитика"
mode = st.radio("Режим", [SIMPLE, ADVANCED], horizontal=True, key="mode", label_visibility="collapsed",
                help="«Для органов власти» — обзор региона и сценарии без технических настроек. "
                     "«Для аналитика» — выбор метода, настройки и проверка качества моделей.")
regions = sorted(data["region"].unique())

if mode == SIMPLE:
    # Модель для простого режима зафиксирована: гибридная, все факторы, параметры из ноутбука 06
    s_features = tuple(BASE + CONTROL)
    s_method = "Гибридная модель (лес + kNN)"
    s_params = freeze(DEFAULTS[s_method])
    s_split = SPLITS[0]

    st.title("Преступность в регионах России и социально-экономические условия")
    st.markdown("Здесь можно посмотреть уровень преступности в регионе, сравнить регион с остальными и оценить, "
                "как с уровнем преступности связаны занятость, доходы, образование и другие условия. "
                f"Расчёты основаны на данных по 85 регионам за {data['year'].min()}–{data['year'].max()} гг.")

    col_r, col_c = st.columns(2)
    s_region = col_r.selectbox("Выберите регион", regions, index=regions.index(DEMO_REGION), key="s_region")
    category = col_c.selectbox("Вид преступлений", list(CATEGORIES), key="category", help=CATEGORY_HELP)
    target = CATEGORIES[category][1]
    reg = data[data["region"] == s_region].sort_values("year")
    last, first = reg.iloc[-1], reg.iloc[0]
    year = int(last["year"])
    same_year = data[data["year"] == year]
    level = last[target]
    median = same_year[target].median()
    rank = int((same_year[target] > level).sum()) + 1  # 1 — самый высокий уровень

    # --- 1. Уровень преступности ---
    st.header(f"1. Уровень преступности в {year} году" + ("" if category == "Все преступления"
                                                            else f": {CATEGORY_PHRASE[category]}"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Преступлений на 100 тыс. жителей", num(level, 0),
              help="Число зарегистрированных преступлений в расчёте на 100 тысяч жителей региона. "
                   "Так можно сравнивать регионы с разной численностью населения.")
    c2.metric("Место среди регионов", f"{rank} из {len(same_year)}",
              help="1-е место — самый высокий уровень преступности, последнее — самый низкий.")
    c3.metric(f"Изменение с {int(first['year'])} года", f"{(level / first[target] - 1) * 100:+.0f} %".replace("-", "−"))
    compare = "выше" if level > median else "ниже"
    st.markdown(f"Уровень преступности в регионе **{compare}** типичного для регионов России "
                f"({num(median, 0)} на 100 тыс. жителей в {year} году).")
    med_by_year = data.groupby("year")[target].median()
    chart = pd.DataFrame({"год": reg["year"].values, s_region: reg[target].values,
                          "типичный уровень по регионам": med_by_year.loc[reg["year"]].values})
    fig = px.line(chart.melt("год", var_name="ряд", value_name="преступлений на 100 тыс."),
                  x="год", y="преступлений на 100 тыс.", color="ряд", markers=True,
                  title="Уровень преступности по годам",
                  color_discrete_map={s_region: ACCENT, "типичный уровень по регионам": NEUTRAL})
    fig.update_layout(legend_title="", yaxis_rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption("Типичный уровень — медиана: у половины регионов уровень выше, у половины ниже.")

    # --- 2. Показатели региона ---
    st.header("2. Условия в регионе по сравнению с другими регионами")
    rows = []
    for f in s_features:
        value = last[f]
        share = (same_year[f] < value).mean() * 100
        rows.append({"показатель": FEATURE_NAMES[f], "в регионе": num(value, 2),
                     "типичное значение": num(same_year[f].median(), 2),
                     "положение": f"выше, чем у {share:.0f} % регионов"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Данные за {year} год. «Зарплата к медиане»: 1 — типичная зарплата для регионов в этом году, "
               "1,5 — в полтора раза выше типичной. Типичное значение — медиана по регионам.")

    # --- 3. Что, если ---
    st.header("3. Что, если изменить условия")
    st.markdown("Передвиньте ползунки: на сколько процентов изменится показатель. Расчёт покажет, как такое "
                "изменение в среднем связано с уровнем преступности в регионах России, и применит это к "
                "выбранному региону.")

    def reset_sliders():
        for f in s_features:
            st.session_state[f"p_{f}"] = 0

    selection_block(s_features, s_method, s_params, s_split, target, last, level, "p_", "s_sel")

    groups = {"Экономика": ["unemployment", "wage_rel", "poverty"],
              "Население и образование": ["urban", "migration", "students", "higher_edu_share"],
              "Работа правоохранительных органов": ["unsolved_share"]}
    changes = {}
    for col, (title, group) in zip(st.columns(3), groups.items()):
        col.markdown(f"**{title}**")
        for f in group:
            hint = ("Значение отрицательное: из региона уезжает больше людей, чем приезжает. Рост на 10 % "
                    "означает, что отток усилится на 10 %." if last[f] < 0 else None)
            changes[f] = col.slider(f"{FEATURE_NAMES[f]}, сейчас {num(last[f], 2)}", -30, 30, 0, step=5,
                                    key=f"p_{f}", format="%d %%", help=hint)
    st.button("Вернуть все ползунки к нулю", on_click=reset_sliders)

    if all(v == 0 for v in changes.values()):
        st.info("Сдвиньте один или несколько ползунков, чтобы увидеть результат.")
    else:
        total, single = scenario_effects(s_features, s_method, s_params, s_split, target,
                                         tuple(changes[f] for f in s_features))
        new_level = level * (1 + total / 100)
        m1, m2 = st.columns(2)
        m1.metric(f"Сейчас ({year})", num(level, 0))
        # в поле delta нужен обычный дефис: по нему Streamlit определяет знак и цвет стрелки
        delta = f"{num(new_level - level, 0)} ({num(total, 1)} %)".replace("−", "-")
        m2.metric("По сценарию", num(new_level, 0), delta=delta,
                  delta_color="inverse")
        direction = "снизиться" if total < 0 else "вырасти"
        st.markdown(f"При заданных изменениях уровень преступности в регионе может **{direction} примерно на "
                    f"{num(abs(total), 1)} %**: с {num(level, 0)} до {num(new_level, 0)} преступлений на "
                    "100 тыс. жителей.")
        st.plotly_chart(effects_chart(single, s_features), width="stretch")
        scenario_values = {f: last[f] * (1 + changes[f] / 100) for f in s_features}
        outside = [FEATURE_NAMES[f] for f in s_features
                   if not data[f].min() <= scenario_values[f] <= data[f].max()]
        if outside:
            st.warning("Значения выходят за пределы того, что встречалось в регионах России за 2011–2022 гг.: "
                       + ", ".join(outside) + ". Для таких значений расчёт ненадёжен.")
    with st.expander("Как читать результат"):
        st.markdown(f"""
- Расчёт показывает **связь** условий жизни с уровнем преступности по данным всех регионов. Он не доказывает,
  что изменение показателя само по себе изменит уровень преступности: на преступность влияют и другие причины.
- Изменение считается так: у всех регионов из проверочной части данных показатели сдвигаются на выбранные
  проценты, и сравнивается средний уровень преступности, который вычисляет модель, до и после сдвига.
  Полученный процент применяется к выбранному региону.
- {WAGE_NOTE}
""")

    # --- 4. Точность ---
    model_s, tr_s, te_s = train(s_features, s_method, s_params, s_split, target)
    X_s, y_s = data[list(s_features)], data[target]
    pred_s = model_s.predict(X_s.iloc[te_s])
    r2_s, mae_s = r2_score(y_s.iloc[te_s], pred_s), mean_absolute_error(y_s.iloc[te_s], pred_s)
    st.header("4. Насколько точны расчёты")
    st.markdown(f"Модель проверена на данных, которые не использовались при её построении. Она объясняет около "
                f"**{r2_s * 100:.0f} %** различий в уровне преступности между регионами и годами и ошибается "
                f"в среднем на **{num(mae_s, 0)}** преступлений на 100 тыс. жителей (около "
                f"{mae_s / y_s.mean() * 100:.0f} % среднего уровня). Подробная проверка моделей — в режиме "
                f"«{ADVANCED}».")
    if r2_s < 0.8:
        st.warning(f"Для показателя «{CATEGORY_PHRASE[category]}» модель описывает различия хуже, чем для всех преступлений "
                   "(допустимый уровень точности — 80 %). Результаты сценария здесь следует считать грубыми. "
                   "Причины — в режиме «Для аналитика», вкладка «Виды преступлений».")
    st.caption("Данные: Генпрокуратура и Росстат (обработка «Если быть точным»), переписи 2010 и 2020 гг. "
               "Данные о преступности по регионам доступны по 2022 год.")
    st.stop()

# ---------- Боковая панель ----------
for key, value in {"degree": DEFAULTS["Полиномиальная регрессия"]["degree"], "alpha": 10, "k": KNN_P["k"],
                   "trees": RF_P["trees"], "use_ridge": True, "knn_distance": KNN_P["weights"] == "distance",
                   "max_features": RF_P["max_features"]}.items():
    st.session_state.setdefault(key, value)

st.sidebar.header("Настройки модели")

category = st.sidebar.selectbox("Вид преступлений", list(CATEGORIES), key="category", help=CATEGORY_HELP)
target = CATEGORIES[category][1]

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
    st.caption("Факторы отобраны по корреляциям, мультиколлинеарности и вкладу в точность: доходы и ВРП "
               "исключены как дублирующие зарплату.")

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
if method in ("k ближайших соседей (kNN)", "Гибридная модель (лес + kNN)"):
    params["weights"] = "distance" if st.sidebar.checkbox(
        "Ближние соседи весят больше", key="knn_distance",
        help="Если включено, вклад соседа в оценку тем больше, чем он ближе по факторам.") else "uniform"
if method in ("Случайный лес", "Гибридная модель (лес + kNN)"):
    params["trees"] = st.sidebar.select_slider(
        "Число деревьев", options=[50, 100, 300, 500], key="trees",
        help="Больше деревьев — устойчивее результат, но дольше расчёт.")
    params["max_features"] = st.sidebar.select_slider(
        "Доля факторов для каждого дерева", options=[0.33, 0.66, 1.0], key="max_features",
        help="Какая часть факторов доступна дереву при каждом разбиении. Меньше — деревья разнообразнее.")

split_type = st.sidebar.radio(
    "Способ проверки", SPLITS, key="split_type",
    help="Какие данные спрятать от модели при обучении, чтобы потом проверить, как она их угадывает. "
         "Основной способ — случайно 70/30, стандартная процедура оценки моделей. По годам — угадать 2020–2022 годы; "
         "по регионам — угадать регионы, которых модель не видела.")

if method != "Линейная регрессия":
    st.sidebar.button("Подобрать лучшие параметры", on_click=on_tune, width="stretch",
                      help="Перебирает параметры метода и ставит лучшие. Проверка идёт только на обучающих "
                           "данных, тестовые в подборе не участвуют.")

model, tr, te = train(tuple(features), method, freeze(params), split_type, target)
X, y = data[features], data[target]
pred_tr, pred_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
r2_tr, r2_te = r2_score(y.iloc[tr], pred_tr), r2_score(y.iloc[te], pred_te)
mae_te = mean_absolute_error(y.iloc[te], pred_te)

# ---------- Заголовок ----------
st.title("Влияние социально-экономических факторов на преступность в регионах РФ")
st.caption("Данные: Генпрокуратура и Росстат (обработка «Если быть точным»), переписи 2010 и 2020 гг.; "
           f"85 регионов, 2011–2022 гг., {len(data)} строк после очистки. Показатель: **{CATEGORY_PHRASE[category]}** "
           "на 100 тыс. жителей.")

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
- «Что, если» — как изменится оценка модели, если изменить показатели региона;
- «Виды преступлений» — точность моделей и роль факторов для разных видов преступлений.

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

if st.session_state.get("tune_for") == (tuple(features), method, split_type, target):
    with st.expander("Результат подбора параметров", expanded=True):
        st.write(f"Подбор для метода «{method}» и способа проверки «{split_type}». Каждая строка — вариант "
                 "параметров; «R² на проверке» — среднее по пяти проверкам внутри обучающей выборки (тестовые "
                 "данные в подборе не участвуют). Лучший вариант (первая строка) выставлен слева; его итоговое "
                 "качество — «R² на тесте» выше.")
        st.dataframe(st.session_state.tune_table.head(10).round(3), hide_index=True, width="stretch")

tab_compare, tab_region, tab_model, tab_factors, tab_whatif, tab_categories = st.tabs(
    ["Сравнение методов", "Регион", "Качество модели", "Влияние факторов", "Что, если", "Виды преступлений"])

# ---------- Вкладка: сравнение методов ----------
with tab_compare:
    st.caption(f"Все методы обучены на одних и тех же данных ({len(features)} факторов, способ проверки "
               f"«{split_type}») с параметрами, подобранными перебором по сетке на обучающей выборке. "
               "Параметры выбранного слева метода можно менять и подбирать отдельно.")
    table = compare_methods(tuple(features), split_type, target)
    fig = px.bar(table.sort_values("R² на тесте"), x="R² на тесте", y="метод", orientation="h",
                 title="R² на тестовой выборке по методам", text_auto=".3f")
    fig.add_vline(x=0.85, line_dash="dash", line_color="grey")
    fig.add_vline(x=0.8, line_dash="dot", line_color="grey")
    fig.update_layout(yaxis_title="", xaxis_range=[min(0, table["R² на тесте"].min()), 1])
    st.plotly_chart(fig, width="stretch")
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("Чем длиннее столбец, тем точнее метод на спрятанных данных. Пунктирные линии: штрихи — целевой "
               "уровень 0,85, точки — допустимый уровень 0,8. Если «R² на обучении» намного выше «R² на тесте», "
               "метод переобучается. Исключение — kNN с весом по расстоянию (и гибрид с ним): на обучающей выборке "
               "каждая строка оказывается собственным ближайшим соседом, поэтому R² на обучении близок к 1 при "
               "любых данных; для этого метода показателен только R² на тесте.")

    # --- Устойчивость результата: четыре способа проверки ---
    st.subheader("Насколько устойчив результат")
    if category != "Все преступления":
        st.info("Этот раздел относится ко всем преступлениям. Проверка гибридной модели для выбранного вида — "
                "на вкладке «Виды преступлений».")
    st.markdown("Одно деление на обучающую и тестовую выборки может оказаться удачным или неудачным случайно. "
                "Поэтому каждый метод проверен четырьмя способами (все 8 факторов, параметры подобраны на "
                "обучающих данных):")
    st.markdown("- **случайно 70/30** — основной способ;\n"
                "- **вложенная проверка 5 × 5** — данные пять раз делятся на пять частей, каждая по очереди "
                "становится тестовой, а подбор параметров повторяется внутри каждого раунда; показано среднее и "
                "разброс;\n"
                "- **по годам** — обучение на 2011–2019 гг., проверка на 2020–2022 гг.;\n"
                "- **по регионам** — проверка на регионах, которых не было в обучении.")
    final = load_final()
    checks = {"R² тест 70/30": "случайно 70/30", "R² вложенная, среднее": "вложенная 5 × 5",
              "R² по годам": "по годам", "R² по регионам": "по регионам"}
    order = final.sort_values("R² вложенная, среднее")["метод"].tolist()  # лучший метод — вверху графика
    long = final.melt(id_vars=["метод"], value_vars=list(checks), var_name="проверка", value_name="R²")
    long["проверка"] = long["проверка"].map(checks)
    long["значение"] = long["R²"].map(lambda v: num(v, 3))
    long["R² на графике"] = long["R²"].clip(lower=0)  # значения ниже 0 показываются у левого края
    spread = final.set_index("метод")["R² вложенная, разброс"]
    long["разброс"] = np.where(long["проверка"] == "вложенная 5 × 5", long["метод"].map(spread), 0)
    fig = px.scatter(long, x="R² на графике", y="метод", color="проверка", error_x="разброс",
                     category_orders={"метод": order[::-1], "проверка": list(checks.values())},
                     color_discrete_sequence=[ACCENT, SECOND, "#199E70", "#8A6FE0"],
                     hover_data={"значение": True, "R² на графике": False, "разброс": False},
                     title="R² методов при четырёх способах проверки")
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.add_vline(x=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85",
                  annotation_position="top left")
    fig.update_layout(xaxis_range=[-0.03, 1.0], xaxis_title="R² на проверочных данных", yaxis_title="",
                      legend_title="способ проверки", height=520)
    fig.update_yaxes(ticktext=[f"<b>{m}</b>" if m.startswith("Гибридная") else m for m in order[::-1]],
                     tickvals=order[::-1])
    st.plotly_chart(fig, width="stretch")
    worst = final.loc[final["R² по регионам"].idxmin()]
    st.caption("Точка — R² метода при одном способе проверки, чем правее, тем точнее. Горизонтальная черта у "
               "вложенной проверки — разброс между раундами. Методы упорядочены по результату вложенной "
               f"проверки. Значения ниже 0 (модель хуже простого среднего) показаны у левого края (например, {worst['метод'].lower()} по "
               f"регионам: {num(worst['R² по регионам'], 2)}); точные значения — при наведении и в таблице ниже.")
    hyb = final[final["метод"].str.startswith("Гибридная")].iloc[0]
    st.markdown(f"**Вывод.** Гибридная модель точнее остальных методов на основном делении "
                f"(R² = {num(hyb['R² тест 70/30'], 2)}), при вложенной проверке "
                f"({num(hyb['R² вложенная, среднее'], 2)} ± {num(hyb['R² вложенная, разброс'], 2)}) и при "
                f"проверке по годам ({num(hyb['R² по годам'], 2)}): результат устойчив и не зависит от удачного "
                f"деления. Линейные методы объясняют меньше половины различий: связь факторов с преступностью "
                f"нелинейна. Для регионов, которых не было в обучении, точность у всех методов низкая "
                f"(у гибридной модели {num(hyb['R² по регионам'], 2)}), поэтому модель предназначена для "
                f"регионов, представленных в данных.")
    with st.expander("Таблица с точными значениями"):
        shown = final.replace({"участники с параметрами из шага 2": "лес и kNN с параметрами из строк ниже"})
        shown = shown.rename(columns={"параметры": "подобранные параметры", "MAE тест": "средняя ошибка 70/30"})
        st.dataframe(shown.round(3), hide_index=True, width="stretch")
        st.caption("Средняя ошибка — преступлений на 100 тыс. жителей. Последний столбец: та же проверка 70/30 "
                   "без доли нераскрытых преступлений среди факторов.")

# ---------- Вкладка: регион ----------
with tab_region:
    st.caption("Две линии по годам для выбранного региона: **факт** — реальный уровень преступности, "
               "**модель** — оценка модели по показателям региона в этом году. Чем ближе линии, тем лучше "
               "модель описывает регион.")
    region = st.selectbox("Регион", regions, index=regions.index(DEMO_REGION))
    reg = data[data["region"] == region].sort_values("year")
    chart = pd.DataFrame({"год": reg["year"], "факт": reg[target].values, "модель": model.predict(reg[features])})
    fig = px.line(chart.melt("год", var_name="ряд", value_name="преступлений на 100 тыс."),
                  x="год", y="преступлений на 100 тыс.", color="ряд", markers=True,
                  title=f"{region}: уровень преступности, факт и оценка модели")
    st.plotly_chart(fig, width="stretch")
    st.caption("Таблица: показатели региона по годам, на которых модель строит оценку.")
    st.dataframe(reg[["year", target] + features].rename(
        columns={"year": "год", target: "преступлений на 100 тыс.", **FEATURE_NAMES}).round(2),
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
                m, _, _ = train(tuple(features), method, freeze({**params, key: v}), split_type, target)
                rows += [{label: v, "выборка": "обучение", "R²": r2_score(y.iloc[tr], m.predict(X.iloc[tr]))},
                         {label: v, "выборка": "тест", "R²": max(r2_score(y.iloc[te], m.predict(X.iloc[te])), -1)}]
            fig = px.line(pd.DataFrame(rows), x=label, y="R²", color="выборка", markers=True,
                          title=f"R² при разных значениях: {label}", log_x=(key == "alpha"))
            fig.add_hline(y=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85")
            # на логарифмической оси положение линии задаётся в логарифмах
            fig.add_vline(x=np.log10(params[key]) if key == "alpha" else params[key], line_color=ACCENT)
            fig.update_yaxes(range=[-1.05, 1.05])
            st.plotly_chart(fig, width="stretch")
            st.caption("Как меняется качество, если менять главный параметр метода при остальных настройках. "
                       "Линия «обучение» показывает, насколько модель подстраивается под знакомые данные; линия "
                       "«тест» — насколько она угадывает новые. Если «обучение» растёт, а «тест» падает, это "
                       "переобучение. Синяя вертикаль — текущее значение, пунктир — ориентир 0,85. "
                       "Значения ниже −1 показаны как −1.")
        else:
            st.info("У линейной регрессии нет параметров для настройки.")

# ---------- Вкладка: влияние факторов ----------
with tab_factors:
    effects = factor_effects(tuple(features), method, freeze(params), split_type, target)
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
                     color_discrete_map={"оценка растёт": UP, "оценка снижается": DOWN},
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

    selection_block(tuple(features), method, freeze(params), split_type, target, row.iloc[0],
                    row[target].iloc[0], "s_", "a_sel")
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

    changes = tuple(int(st.session_state[f"s_{f}"]) for f in features)
    fact = row[target].iloc[0]
    base_pred, new_pred = model.predict(base)[0], model.predict(scenario)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Факт", f"{fact:.0f}",
              help="Реальный уровень преступности в регионе в выбранном году, на 100 тыс. жителей.")
    if any(changes):
        total, single = scenario_effects(tuple(features), method, freeze(params), split_type, target, changes)
        new_avg = fact * (1 + total / 100)
        m2.metric("Оценка по сценарию", f"{new_avg:.0f}", delta=f"{new_avg - fact:+.0f} ({total:+.1f} %)",
                  delta_color="inverse",
                  help="Основная оценка. Все строки тестовой выборки сдвигаются на выбранные проценты; "
                       "относительное изменение средней оценки модели применяется к факту региона.")
    else:
        m2.metric("Оценка по сценарию", f"{fact:.0f}", help="Сдвиньте ползунки, чтобы получить сценарий.")
    m3.metric("Оценка модели в точке", f"{new_pred:.0f}", delta=f"{new_pred - base_pred:+.0f}",
              delta_color="inverse",
              help=f"Справочно: оценка модели для одной строки с изменёнными показателями; при реальных "
                   f"показателях — {base_pred:.0f}. У леса и kNN отклик в одной точке ступенчатый: оценка "
                   f"берётся по похожим строкам обучающей выборки, и при любом сдвиге похожими становятся "
                   f"другие регионы и годы.")
    if any(changes):
        st.plotly_chart(effects_chart(single, features), width="stretch")
    st.caption("Почему основная оценка считается по всей тестовой выборке. Лес и kNN оценивают уровень по похожим "
               "строкам обучающих данных, поэтому их отклик в одной точке ступенчатый и может расти при сдвиге "
               "фактора в любую сторону (особенно если строка сама есть в обучающих данных). Усреднение по "
               "многим регионам сглаживает ступеньки и показывает устойчивое направление связи. "
               + WAGE_NOTE)
    st.caption("Все значения — преступлений на 100 тыс. жителей. Оценки ориентировочные: модель показывает "
               "связь показателей с преступностью, причинность она не доказывает.")

# ---------- Вкладка: виды преступлений ----------
with tab_categories:
    st.markdown("Одинаково ли социально-экономические условия связаны с разными видами преступлений? Для каждого "
                "вида гибридная модель обучена отдельно (8 факторов, параметры подобраны на обучающих данных) и "
                "проверена теми же способами, что и для всех преступлений. Результаты рассчитаны заранее.")
    cat_cmp, cat_fac = load_categories()
    key_to_name = {v[0]: k for k, v in CATEGORIES.items()}
    cat_cmp["вид"] = list(CATEGORIES)  # порядок строк в таблице ноутбука 07 совпадает с CATEGORIES
    checks = {"R² тест 70/30": "случайно 70/30", "R² 5 частей": "перекрёстная, 5 частей", "R² по годам": "по годам",
              "R² по регионам": "по регионам"}
    long = cat_cmp.melt(id_vars=["вид"], value_vars=list(checks), var_name="проверка", value_name="R²")
    long["проверка"] = long["проверка"].map(checks)
    long["значение"] = long["R²"].map(lambda v: num(v, 3))
    long["R² на графике"] = long["R²"].clip(lower=0)
    fig = px.scatter(long, x="R² на графике", y="вид", color="проверка",
                     category_orders={"вид": list(CATEGORIES), "проверка": list(checks.values())},
                     color_discrete_sequence=[ACCENT, SECOND, "#199E70", "#8A6FE0"],
                     hover_data={"значение": True, "R² на графике": False},
                     title="Точность гибридной модели по видам преступлений")
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.add_vline(x=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85",
                  annotation_position="top left")
    fig.add_vline(x=0.8, line_dash="dot", line_color="grey")
    fig.update_layout(xaxis_range=[-0.03, 1.0], xaxis_title="R² на проверочных данных", yaxis_title="",
                      legend_title="способ проверки", height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption("Значения ниже 0 показаны у левого края; точные значения — при наведении и в таблице ниже.")
    shown = cat_cmp[["вид", "R² тест 70/30", "средняя ошибка", "средняя ошибка, % среднего", "R² 5 частей",
                     "R² по годам", "R² по регионам", "R² линейная 70/30", "R² 70/30 без доли нераскрытых"]]
    st.dataframe(shown.round(3), hide_index=True, width="stretch")
    econ = cat_cmp.set_index("вид").loc["Экономической направленности"]
    econ = cat_cmp.set_index("вид").loc["Экономической направленности"]
    rob = cat_cmp.set_index("вид").loc["Грабежи и разбои"]
    st.markdown(f"**Вывод.** Для всех преступлений, преступлений небольшой и средней тяжести, краж и убийств модель "
                f"достигает ориентира 0,85; для грабежей и разбоев — уровня ориентира при случайном делении, для тяжких "
                f"преступлений — допустимого уровня 0,8. Хуже всего описываются экономические преступления "
                f"(R² = {num(econ['R² тест 70/30'], 2)}): их уровень за 2011–2022 гг. снизился почти вдвое во всех "
                f"регионах сразу, а различия между регионами объясняют только треть разброса. Грабежи и разбои за тот "
                f"же период сократились почти вчетверо, поэтому модель, обученная на ранних годах, плохо оценивает "
                f"поздние (R² по годам {num(rob['R² по годам'], 2)}). Такие общие для страны изменения условия "
                f"отдельного региона не описывают.")

    st.subheader("Роль факторов по видам преступлений")
    short = ["Все", "Тяжкие и<br>особо тяжкие", "Небольшой<br>и средней<br>тяжести", "Экономи-<br>ческие",
             "Наркотики", "Убийства", "Кражи", "Грабежи<br>и разбои"]
    imp = cat_fac[[f"важность_{v[0]}" for v in CATEGORIES.values()]]
    imp.columns, imp.index = short, [FEATURE_NAMES[f] for f in imp.index]
    eff = cat_fac[[f"эффект_{v[0]}" for v in CATEGORIES.values()]]
    eff.columns, eff.index = short, [FEATURE_NAMES[f] for f in eff.index]
    fig = px.imshow(imp.round(0), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    title="Доля фактора в важности модели, %")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="", height=360,
                      margin=dict(t=135, b=10))
    fig.update_xaxes(side="top", tickangle=0)
    st.plotly_chart(fig, width="stretch")
    st.caption("Важность — насколько падает точность модели, если значения фактора перемешать. В каждом столбце "
               "сумма равна 100 %: так виды с разной точностью можно сравнивать.")
    fig = px.imshow(eff.round(1), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0, zmin=-6, zmax=6,
                    title="Изменение средней оценки модели при росте фактора на 10 %, %")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="", height=360,
                      margin=dict(t=135, b=10))
    fig.update_xaxes(side="top", tickangle=0)
    st.plotly_chart(fig, width="stretch")
    st.caption("Красный — оценка растёт, синий — снижается. Для миграционного прироста рост на 10 % у регионов "
               "с оттоком населения означает усиление оттока.")
    st.markdown("**Что различается.** Зарплата относительно других регионов связана с ростом всех видов, кроме "
                "экономических; сильнее всего — убийств и краж. Доля людей с высшим образованием связана со снижением "
                "почти всех видов, сильнее всего — грабежей и разбоев, убийств и краж. Безработица и доля бедных "
                "сильнее связаны с кражами, грабежами и разбоями, чем с тяжкими преступлениями. Для экономических "
                "преступлений главный фактор — число студентов: их больше в крупных экономических центрах, где выше "
                "и хозяйственная активность. Для краж самый важный фактор — доля нераскрытых, но эта связь отчасти "
                "арифметическая: кражи составляют большую часть нераскрытых преступлений.")
    st.caption("Доля нераскрытых во всех моделях — общая по всем преступлениям. Расчёт: ноутбук 07. Данные о кражах, "
               "грабежах и разбоях: ЕМИСС, показатель 36225 (МВД России).")
