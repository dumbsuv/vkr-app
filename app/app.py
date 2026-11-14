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
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, cross_val_predict, cross_val_score,
                                     train_test_split)
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
BASELINE_PATH = ROOT / "data" / "processed" / "baseline_comparison.csv"  # простые оценки без факторов, ноутбук 09
CONTRIB_PATH = ROOT / "data" / "processed" / "factor_contribution.csv"  # вклад факторов, ноутбук 09

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

# Названия показателей простыми словами (режим «Для органов власти»)
PLAIN = {
    "unemployment": "Безработица",
    "wage_rel": "Зарплата по сравнению с типичной для регионов",
    "poverty": "Доля бедных (доходы ниже границы бедности)",
    "urban": "Доля городского населения",
    "migration": "Миграционный прирост (приехавшие минус уехавшие)",
    "students": "Студенты вузов",
    "higher_edu_share": "Жители с высшим образованием",
    "unsolved_share": "Доля нераскрытых преступлений",
}
# Какое значение показателя лучше для жизни в регионе: −1 — чем меньше, тем лучше; 1 — чем больше, тем лучше.
# Для остальных показателей оценка «лучше» или «хуже» не даётся.
BETTER = {"unemployment": -1, "poverty": -1, "unsolved_share": -1, "wage_rel": 1, "higher_edu_share": 1}


def value_text(f, v):
    """Значение показателя с единицей измерения, как его удобно читать."""
    if f == "wage_rel":  # 1,36 → «136 % типичной зарплаты»
        return f"{v * 100:.0f} % типичной зарплаты".replace("-", "−")
    if f in ("migration", "students"):
        return f"{v:.0f} на 10 тыс. жителей".replace("-", "−")
    return f"{v:.1f} %".replace(".", ",").replace("-", "−")


def compare_text(f, value, median):
    """Сравнение значения региона с типичным (медианой по регионам того же года)."""
    if abs(value - median) <= 0.03 * abs(median):
        return "около типичного"
    higher = value > median
    if f in BETTER:
        return "лучше типичного" if higher == (BETTER[f] > 0) else "хуже типичного"
    return "выше типичного" if higher else "ниже типичного"

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
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
def stamp(*paths):
    """Время изменения и размер файлов. Передаётся в функции чтения как аргумент version: Streamlit запоминает
    результат по аргументам, поэтому после обновления файла данные перечитываются заново."""
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


@st.cache_data
def load_final(version):
    """Готовые результаты окончательного сравнения методов (вложенная проверка считается несколько минут,
    поэтому в приложении не пересчитывается)."""
    return pd.read_csv(FINAL_PATH)


@st.cache_data
def load_baseline(version):
    """Готовые результаты ноутбука 09: простые оценки без факторов и вклад факторов."""
    return pd.read_csv(BASELINE_PATH), pd.read_csv(CONTRIB_PATH)


@st.cache_data
def load_data(version):
    """Читает очищенную таблицу «регион × год» и добавляет уровни по видам преступлений."""
    panel = pd.read_csv(DATA_PATH)
    return panel.merge(pd.read_csv(CAT_PATH), on=["region", "year"], how="left")


@st.cache_data
def load_categories(version):
    """Готовые результаты ноутбука 07: точность моделей, важность и направление факторов по видам."""
    return pd.read_csv(CAT_COMPARE_PATH), pd.read_csv(CAT_FACTORS_PATH, index_col=0)


# Если файлы данных обновились (например, после загрузки новой версии в репозиторий), запомненные
# результаты (таблицы и обученные модели) очищаются: иначе приложение продолжило бы работать со старыми данными.
DATA_FILES = [DATA_PATH, CAT_PATH, FEATURES_PATH, PARAMS_PATH, FINAL_PATH, CAT_COMPARE_PATH, CAT_FACTORS_PATH, LIMITS_PATH,
              BASELINE_PATH, CONTRIB_PATH]


@st.cache_resource
def data_version_box():
    return {"version": None}


current_version = stamp(*DATA_FILES)
if data_version_box()["version"] not in (None, current_version):
    st.cache_data.clear()
    st.cache_resource.clear()
data_version_box()["version"] = current_version

data = load_data(stamp(DATA_PATH, CAT_PATH))


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


def show(fig):
    """Выводит график. Числа на осях и подписях пишутся по-русски: десятичная запятая, пробел между разрядами."""
    fig.update_layout(separators=",\u00a0")
    st.plotly_chart(fig, width="stretch")


def ru_table(df, digits=3):
    """Копия таблицы для показа: дробные числа превращаются в текст с десятичной запятой."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else num(v, digits))
    return out


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
    "ElasticNet": [{"alpha": a, "l1_ratio": FINAL["ElasticNet"]["l1_ratio"]} for a in ALPHAS],
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
    "(занятость населения и социальная помощь малоимущим, закон №\u00a0414-ФЗ, ст. 44). Оба показателя можно только "
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


def selection_block(features, method, params, split_type, target, region_values, slider_prefix, key):
    """Блок «Подобрать сценарий»: выбор срока и кнопка, которая сразу выставляет ползунки."""
    place = (target, region_values["region"], int(region_values["year"]))

    def pick():
        """Нажатие кнопки: подбор выполняется, найденные изменения переносятся на ползунки."""
        horizon = st.session_state[f"{key}_horizon"]
        table, total = select_scenario(features, method, params, split_type, target, horizon, region_values)
        for f in features:
            st.session_state[f"{slider_prefix}{f}"] = 0
        for _, r in table.iterrows():
            st.session_state[f"{slider_prefix}{r['код']}"] = int(r["изменение показателя, %"])
        st.session_state[f"{key}_result"] = (horizon,) + place + (table, total)

    with st.container(border=True):
        st.markdown("**Подобрать автоматически.** Расчёт сам найдёт выполнимое снижение безработицы и доли бедных, "
                    "с которым связано наибольшее снижение уровня преступности, и выставит ползунки.")
        horizon = st.radio("За какой срок", ["1", "3"], index=1,
                           format_func=lambda h: "1 год" if h == "1" else "3 года",
                           horizontal=True, key=f"{key}_horizon",
                           help="От срока зависит, насколько сильно показатель может реально измениться: "
                                "за три года больше, чем за год.")
        st.button("Подобрать", key=f"{key}_run", type="primary", on_click=pick)
        result = st.session_state.get(f"{key}_result")
        if result and result[:4] == (horizon,) + place:
            table, total = result[4], result[5]
            used = table[table["изменение показателя, %"] < 0]
            if used.empty:
                st.info("Для выбранного вида преступлений снижение безработицы и бедности в допустимых пределах "
                        "не связано со снижением уровня. Ползунки возвращены к нулю.")
            else:
                accusative = {"unemployment": "безработицу", "poverty": "долю бедных"}
                parts = [f"{accusative[r['код']]} на {abs(r['изменение показателя, %'])} %"
                         for _, r in used.iterrows()]
                st.success(f"Подобрано на {'1 год' if horizon == '1' else '3 года'}: снизить " + " и ".join(parts)
                           + ". Ползунки выставлены, результат показан ниже.")
                with st.expander("Подробности подбора"):
                    shown = table.drop(columns="код").copy()
                    shown["изменение показателя, %"] = shown["изменение показателя, %"].map(lambda v: num(v, 0))
                    shown["изменение уровня преступности, %"] = shown["изменение уровня преступности, %"].map(
                        lambda v: num(v, 1))
                    st.dataframe(shown, hide_index=True, width="stretch")
                    st.caption("Строки упорядочены от более сильной связи к более слабой; в каждой строке — эффект "
                               "изменения одного показателя. Если показатель в сценарий не вошёл (0 %), его "
                               "снижение в допустимых пределах не связано со снижением уровня.")
        with st.expander("Как устроен подбор"):
            st.markdown(SELECTION_NOTE + " Для каждого показателя перебираются снижения на 5, 10, 15 % и далее до "
                        "предела и выбирается то, с которым связано наибольшее снижение преступности. Результат "
                        "показывает связь в данных и не является прогнозом последствий мер.")


# ---------- Выводы для региона (правило перехода к рекомендациям, методика, шаг 7) ----------
MEASURES = {
    "unemployment": {
        "title": "Содействие занятости", "name": "Безработица",
        "what": "Профессиональное обучение безработных, общественные и временные работы, ярмарки вакансий.",
        "law": "закон № 414-ФЗ, ст. 44, ч. 1, п. 134"},
    "poverty": {
        "title": "Поддержка малоимущих", "name": "Доля бедных",
        "what": "Социальная поддержка малоимущих граждан, в том числе по социальному контракту.",
        "law": "закон № 414-ФЗ, ст. 44, ч. 1, п. 49 и 50"},
}
PREVENTION = {
    "title": "Профилактика правонарушений и анализ причин",
    "what": ("Правовое информирование, социальная адаптация, ресоциализация, помощь лицам, подверженным риску "
             "стать пострадавшими. Анализ причин, которые расчёт не учитывает: работа правоохранительных органов, "
             "особенности учёта, местные условия."),
    "law": "закон № 182-ФЗ, ст. 11 и 17"}


@st.cache_data
def expected_levels(features, method, params, target):
    """Ожидаемый уровень для каждой строки по данным других регионов: модель обучается без региона
    (деление на 5 групп регионов), поэтому оценка показывает, какой уровень типичен для таких условий."""
    model = make_model(method, dict(params))
    pred = cross_val_predict(model, data[list(features)], data[target], cv=GroupKFold(n_splits=5),
                             groups=data["region"])
    rel = (data[target] / pred - 1) * 100
    return pd.Series(pred, index=data.index), float(rel.abs().median())  # оценки и типичная ошибка, %


def region_conclusions(features, method, params, split_type, target, row, r2):
    """Расчёт выводов для региона по правилу методики. Возвращает словарь:
    1) проверка точности (при R² ниже 0,8 выводы не формулируются);
    2) сравнение фактического уровня с ожидаемым при таких условиях;
    3) управляемые показатели, снижение которых связано со снижением преступности на обоих сроках;
    4) приоритет меры — по положению региона относительно медианы."""
    if r2 < 0.8:
        return {"ok": False}
    pred, typical_err = expected_levels(features, method, params, target)
    fact, expected = row[target], pred.loc[row.name]
    gap = (fact / expected - 1) * 100
    same_year = data[data["year"] == row["year"]]
    sel = {h: select_scenario(features, method, params, split_type, target, h, row)[0].set_index("код")
           for h in ["1", "3"]}
    measures = []
    for f in MANAGED:
        e1, e3 = sel["1"].loc[f, "изменение уровня преступности, %"], sel["3"].loc[f, "изменение уровня преступности, %"]
        if e1 >= 0 or e3 >= 0:
            continue  # связь неустойчива: на одном из сроков снижение показателя не снижает оценку
        shift = int(sel["3"].loc[f, "изменение показателя, %"])
        measures.append({"код": f, "сдвиг": shift, "эффект": float(e3), "значение": row[f],
                         "после": row[f] * (1 + shift / 100), "доля": (same_year[f] < row[f]).mean() * 100,
                         "первоочередная": bool(row[f] > same_year[f].median())})
    measures.sort(key=lambda m: m["эффект"])
    joint = 0.0
    if measures:
        shifts = {m["код"]: m["сдвиг"] for m in measures}
        joint, _ = scenario_effects(features, method, params, split_type, target,
                                    tuple(shifts.get(g, 0) for g in features))
    verdict = "соответствует" if abs(gap) <= typical_err else ("выше" if gap > 0 else "ниже")
    return {"ok": True, "факт": fact, "ожидаемый": expected, "разрыв": gap, "точность": typical_err,
            "вердикт": verdict, "меры": measures, "совместно": joint, "профилактика": verdict == "выше",
            "недоучёт": verdict == "ниже" and bool(row.get("flag_low_registration", False))}


def verdict_text(c):
    """Фраза о том, соответствует ли уровень преступности условиям региона."""
    fact, expected, gap, err = num(c["факт"], 0), num(c["ожидаемый"], 0), num(abs(c["разрыв"]), 0), num(c["точность"], 0)
    if c["вердикт"] == "соответствует":
        return (f"**Уровень соответствует условиям региона.** В регионах с такими условиями обычно бывает около "
                f"{expected} преступлений на 100 тыс. жителей, фактически — {fact}. Разница {gap} % укладывается "
                f"в точность такой оценки (±{err} %).")
    if c["вердикт"] == "выше":
        return (f"**Уровень выше, чем обычно бывает при таких условиях:** {fact} против около {expected} на "
                f"100 тыс. жителей (на {gap} % больше при точности оценки ±{err} %). Часть преступности связана "
                "с причинами, которых нет среди показателей расчёта.")
    text = (f"**Уровень ниже, чем обычно бывает при таких условиях:** {fact} против около {expected} на 100 тыс. "
            f"жителей (на {gap} % меньше при точности оценки ±{err} %). Причины могут быть разными: действующие "
            "меры профилактики, местные условия или неполный учёт преступлений.")
    if c["недоучёт"]:
        text += (" Для этого региона при подготовке данных отмечен возможный недоучёт преступлений: регистрируемый "
                 "уровень устойчиво намного ниже, чем в регионах с похожими условиями. Сравнение следует толковать "
                 "осторожно.")
    return text


def measure_cards(c):
    """Карточки мер: название, что делать, положение региона, связанный эффект, полномочие мелкой строкой."""
    cards = []
    if c["профилактика"]:
        cards.append((PREVENTION, "первоочередное направление",
                      ["Уровень преступности выше, чем обычно бывает при таких условиях, и показателями расчёта "
                       "это превышение не объясняется."]))
    for m in c["меры"]:
        info = MEASURES[m["код"]]
        lines = [f"{info['name']} в регионе {value_text(m['код'], m['значение'])} — выше, чем у {m['доля']:.0f} % "
                 "регионов.",
                 f"Снижение на {abs(m['сдвиг'])} % за три года (до {value_text(m['код'], m['после'])}) связано со "
                 f"снижением уровня преступности примерно на **{num(abs(m['эффект']), 1)} %**."]
        cards.append((info, "первоочередное направление" if m["первоочередная"] else "дополнительное направление",
                      lines))
    if not cards:
        st.info("Снижение безработицы и бедности в допустимых пределах не связано устойчиво со снижением уровня "
                "этого вида преступлений, поэтому направления мер по этим показателям не предлагаются.")
        return
    for col, (info, priority, lines) in zip(st.columns(len(cards)), cards):
        with col.container(border=True):
            st.caption(priority.upper())
            st.markdown(f"**{info['title']}**")
            st.markdown(info["what"])
            for line in lines:
                st.markdown(line)
            st.caption(f"Полномочия региона: {info['law']}.")
    st.caption("Первоочередное направление — показатель в регионе хуже, чем у большинства регионов, или уровень "
               "преступности выше ожидаемого. Дополнительное — показатель уже не хуже типичного, резерв меньше. "
               "Меры предлагаются только по показателям, которые входят в полномочия региона и снижение которых "
               "связано со снижением преступности и за год, и за три года.")


def effects_chart(single, features, names=FEATURE_NAMES):
    """Столбцы: на сколько процентов меняется уровень преступности от каждого изменения по отдельности."""
    eff = pd.DataFrame({"фактор": [names[f] for f in features], "изменение, %": [single[f] for f in features]})
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
mode = st.radio("Кому показать расчёты", [SIMPLE, ADVANCED], horizontal=True, key="mode",
                captions=["обзор региона, выводы и сценарии обычными словами",
                          "методы, настройки и проверка качества моделей"])
regions = sorted(data["region"].unique())

if mode == SIMPLE:
    # Модель для простого режима зафиксирована: гибридная, все факторы, параметры из ноутбука 06
    s_features = tuple(BASE + CONTROL)
    s_method = "Гибридная модель (лес + kNN)"
    s_params = freeze(DEFAULTS[s_method])
    s_split = SPLITS[0]

    st.title("Преступность в регионах России и социально-экономические условия")
    st.markdown("Выберите регион: страница покажет уровень преступности, условия жизни в регионе и то, какие "
                "изменения условий связаны со снижением преступности. "
                f"Расчёты основаны на данных по 85 регионам за {data['year'].min()}–{data['year'].max()} гг.")

    col_r, col_c = st.columns(2)

    def reset_sliders():
        """Ползунки сценария возвращаются к нулю (кнопка сброса и смена региона: сценарий подбирается для региона)."""
        for f in s_features:
            st.session_state[f"p_{f}"] = 0

    s_region = col_r.selectbox("Регион", regions, index=regions.index(DEMO_REGION), key="s_region",
                               on_change=reset_sliders)
    category = col_c.selectbox("Вид преступлений", list(CATEGORIES), key="category", help=CATEGORY_HELP)
    target = CATEGORIES[category][1]
    reg = data[data["region"] == s_region].sort_values("year")
    last, first = reg.iloc[-1], reg.iloc[0]
    year = int(last["year"])
    same_year = data[data["year"] == year]
    level = last[target]
    median = same_year[target].median()
    rank = int((same_year[target] > level).sum()) + 1  # 1 — самый высокий уровень
    compare = "выше" if level > median else "ниже"

    # Модель и её точность на проверочной части данных
    model_s, tr_s, te_s = train(s_features, s_method, s_params, s_split, target)
    X_s, y_s = data[list(s_features)], data[target]
    pred_s = model_s.predict(X_s.iloc[te_s])
    r2_s, mae_s = r2_score(y_s.iloc[te_s], pred_s), mean_absolute_error(y_s.iloc[te_s], pred_s)
    mae_pct = mae_s / y_s.mean() * 100  # средняя ошибка в процентах среднего уровня
    concl = region_conclusions(s_features, s_method, s_params, s_split, target, last, r2_s)

    # --- Коротко: три главных вывода до подробностей ---
    with st.container(border=True):
        st.subheader(f"Коротко: {s_region}, {year} год")
        st.markdown(f"**Уровень.** {num(level, 0)} преступлений на 100 тыс. жителей — {rank}-е место из "
                    f"{len(same_year)} регионов (1-е место — самый высокий уровень). Это {compare} типичного "
                    f"для регионов России ({num(median, 0)}).")
        if not concl["ok"]:
            st.markdown(f"**Выводы.** Для этого вида преступлений расчёт недостаточно точен (ошибка в среднем "
                        f"{mae_pct:.0f} %), поэтому выводы и направления мер не формулируются. Данные ниже можно "
                        "использовать как справку.")
        else:
            expected = num(concl["ожидаемый"], 0)
            gap = num(abs(concl["разрыв"]), 0)
            if concl["вердикт"] == "соответствует":
                st.markdown(f"**Соответствие условиям.** Уровень близок к тому, что обычно бывает в регионах с "
                            f"такими условиями (около {expected}).")
            elif concl["вердикт"] == "выше":
                st.markdown(f"**Соответствие условиям.** Уровень на {gap} % выше, чем обычно бывает при таких "
                            f"условиях (около {expected}): часть причин лежит за пределами показателей расчёта.")
            else:
                st.markdown(f"**Соответствие условиям.** Уровень на {gap} % ниже, чем обычно бывает при таких "
                            f"условиях (около {expected})."
                            + (" Возможен неполный учёт преступлений." if concl["недоучёт"] else ""))
            genitive = {"unemployment": "безработицы", "poverty": "доли бедных"}
            if concl["меры"]:
                parts = [f"{genitive[m['код']]} на {abs(m['сдвиг'])} %" for m in concl["меры"]]
                text = ("Со снижением " + " и ".join(parts) + " за три года связано снижение уровня преступности "
                        f"примерно на {num(abs(concl['совместно']), 1)} %: с {num(level, 0)} до "
                        f"{num(level * (1 + concl['совместно'] / 100), 0)} на 100 тыс. жителей.")
                if concl["профилактика"]:
                    text += " Первое направление — профилактика правонарушений."
            elif concl["профилактика"]:
                text = "Первое направление — профилактика правонарушений и анализ причин, не учтённых расчётом."
            else:
                text = ("Снижение безработицы и бедности в выполнимых пределах с уровнем этого вида преступлений "
                        "устойчиво не связано.")
            st.markdown("**Что можно сделать.** " + text)
        st.caption("Подробности — в разделах ниже. Расчёт показывает связь в данных; результат мер он не гарантирует.")

    # --- 1. Ситуация ---
    st.header(f"1. Ситуация: уровень преступности в {year} году" + ("" if category == "Все преступления"
                                                                    else f" ({CATEGORY_PHRASE[category]})"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Преступлений на 100 тыс. жителей", num(level, 0),
              help="Число зарегистрированных преступлений в расчёте на 100 тысяч жителей региона. "
                   "Так можно сравнивать регионы с разной численностью населения.")
    c2.metric("Место среди регионов", f"{rank} из {len(same_year)}",
              help="1-е место — самый высокий уровень преступности, последнее — самый низкий.")
    c3.metric(f"Изменение с {int(first['year'])} года", f"{(level / first[target] - 1) * 100:+.0f} %".replace("-", "−"))

    # Все регионы по убыванию уровня, выбранный выделен
    order = same_year.sort_values(target, ascending=False)
    bars = pd.DataFrame({"регион": order["region"].tolist(), "преступлений на 100 тыс.": order[target].tolist(),
                         "ряд": [s_region if r == s_region else "остальные регионы" for r in order["region"]]})
    fig = px.bar(bars, x="регион", y="преступлений на 100 тыс.", color="ряд",
                 color_discrete_map={s_region: ACCENT, "остальные регионы": "#C5CAD3"},
                 category_orders={"регион": bars["регион"].tolist(), "ряд": [s_region, "остальные регионы"]},
                 title=f"Место региона среди {len(same_year)} регионов в {year} году")
    fig.add_hline(y=median, line_dash="dash", line_color=NEUTRAL, annotation_text="типичный уровень",
                  annotation_position="top right")
    fig.update_layout(xaxis_showticklabels=False, xaxis_title="регионы от самого высокого уровня к самому низкому",
                      legend_title="", height=340, bargap=0.15)
    show(fig)
    st.caption("Каждый столбец — один регион; название видно при наведении. Типичный уровень — медиана: у половины "
               "регионов уровень выше, у половины ниже.")

    med_by_year = data.groupby("year")[target].median()
    chart = pd.DataFrame({"год": reg["year"].values, s_region: reg[target].values,
                          "типичный уровень по регионам": med_by_year.loc[reg["year"]].values})
    fig = px.line(chart.melt("год", var_name="ряд", value_name="преступлений на 100 тыс."),
                  x="год", y="преступлений на 100 тыс.", color="ряд", markers=True,
                  title="Как менялся уровень преступности",
                  color_discrete_map={s_region: ACCENT, "типичный уровень по регионам": NEUTRAL})
    fig.update_layout(legend_title="", yaxis_rangemode="tozero", height=380)
    show(fig)

    # --- 2. Условия ---
    st.header("2. Условия в регионе")
    rows = []
    for f in s_features:
        share = (same_year[f] < last[f]).mean() * 100
        rows.append({"показатель": PLAIN[f], "в регионе": value_text(f, last[f]),
                     "типичное значение": value_text(f, same_year[f].median()),
                     "сравнение": compare_text(f, last[f], same_year[f].median()),
                     "место среди регионов": f"выше, чем у {share:.0f} % регионов"})
    marks = {"хуже типичного": f"color: {UP}; font-weight: 600", "лучше типичного": f"color: {DOWN}; font-weight: 600"}
    st.dataframe(pd.DataFrame(rows).style.map(lambda v: marks.get(v, ""), subset=["сравнение"]),
                 hide_index=True, width="stretch")
    st.caption(f"Данные за {year} год. Типичное значение — медиана по регионам. «Лучше» и «хуже» относятся к самим "
               "условиям жизни: ниже безработица, бедность и доля нераскрытых преступлений, выше зарплата и доля "
               "жителей с высшим образованием. Для остальных показателей указано только «выше» или «ниже». "
               "Как условия связаны с преступностью, показывает следующий раздел.")

    # --- 3. Что можно сделать ---
    st.header("3. Что можно сделать")
    if not concl["ok"]:
        st.warning("Для этого вида преступлений расчёт недостаточно точен, поэтому выводы и направления мер не "
                   "формулируются. Сценарий ниже следует считать грубой прикидкой. Причины — в режиме "
                   f"«{ADVANCED}», вкладка «Виды преступлений».")
    else:
        st.markdown(verdict_text(concl))
        measure_cards(concl)

    st.subheader("Проверить свой сценарий")
    st.markdown("Задайте, на сколько процентов изменится показатель. Расчёт покажет, как такое изменение в среднем "
                "связано с уровнем преступности в регионах России, и применит это к выбранному региону.")
    selection_block(s_features, s_method, s_params, s_split, target, last, "p_", "s_sel")

    def scenario_slider(col, f):
        """Ползунок изменения показателя и подпись: значение сейчас и значение по сценарию."""
        hint = ("Значение отрицательное: из региона уезжает больше людей, чем приезжает. Рост на 10 % "
                "означает, что отток усилится на 10 %." if f == "migration" and last[f] < 0 else None)
        st.session_state.setdefault(f"p_{f}", 0)  # начальное значение; дальше ползунком управляет его ключ
        change = col.slider(PLAIN[f], -30, 30, step=5, key=f"p_{f}", format="%d %%", help=hint)
        new = last[f] * (1 + change / 100)
        if f in PERCENT_FEATURES:
            new = min(new, 100)
        col.caption(f"сейчас {value_text(f, last[f])}"
                    + (f" → по сценарию **{value_text(f, new)}**" if change else ""))
        return change

    changes = {}
    st.markdown("**Показатели, на которые регион может влиять**")
    for col, f in zip(st.columns(len(MANAGED)), MANAGED):
        changes[f] = scenario_slider(col, f)
    others = [f for f in s_features if f not in MANAGED]
    with st.expander("Другие условия: регион не управляет ими напрямую"):
        st.caption("Эти ползунки показывают, как условия связаны с преступностью. В автоматический подбор они не "
                   "входят.")
        cols = st.columns(3)
        for i, f in enumerate(others):
            changes[f] = scenario_slider(cols[i % 3], f)

    if all(v == 0 for v in changes.values()):
        st.info("Сдвиньте ползунок или нажмите «Подобрать», чтобы увидеть результат.")
    else:
        st.button("Вернуть все ползунки к нулю", on_click=reset_sliders)
        total, single = scenario_effects(s_features, s_method, s_params, s_split, target,
                                         tuple(changes[f] for f in s_features))
        new_level = level * (1 + total / 100)
        m1, m2 = st.columns(2)
        m1.metric(f"Сейчас ({year})", num(level, 0))
        # в поле delta нужен обычный дефис: по нему Streamlit определяет знак и цвет стрелки
        delta = f"{num(new_level - level, 0)} ({num(total, 1)} %)".replace("−", "-")
        m2.metric("По сценарию", num(new_level, 0), delta=delta, delta_color="inverse")
        direction = "снизиться" if total < 0 else "вырасти"
        st.markdown(f"При заданных изменениях уровень преступности в регионе может **{direction} примерно на "
                    f"{num(abs(total), 1)} %**: с {num(level, 0)} до {num(new_level, 0)} преступлений на "
                    "100 тыс. жителей.")
        show(effects_chart(single, s_features, PLAIN))
        scenario_values = {f: last[f] * (1 + changes[f] / 100) for f in s_features}
        outside = [PLAIN[f] for f in s_features if not data[f].min() <= scenario_values[f] <= data[f].max()]
        if outside:
            st.warning("Значения выходят за пределы того, что встречалось в регионах России за 2011–2022 гг.: "
                       + ", ".join(outside) + ". Для таких значений расчёт ненадёжен.")

    # --- 4. Насколько этому верить ---
    st.header("4. Насколько этому верить")
    st.markdown(f"Расчёт проверен на данных, которые не использовались при его построении. Уровень преступности "
                f"он оценивает со средней ошибкой около **{mae_pct:.0f} %** ({num(mae_s, 0)} преступлений на "
                "100 тыс. жителей).")
    if r2_s < 0.8:
        st.warning(f"Для показателя «{CATEGORY_PHRASE[category]}» расчёт описывает различия между регионами хуже, "
                   "чем для всех преступлений. Результаты сценария здесь следует считать грубыми.")
    st.markdown(f"""
- Расчёт показывает **связь** условий жизни с уровнем преступности по данным всех регионов. Он не доказывает,
  что изменение показателя само по себе изменит уровень преступности: на преступность влияют и другие причины.
- Сценарий считается так: у регионов из проверочной части данных показатели сдвигаются на выбранные проценты,
  и сравнивается средний уровень преступности по расчёту до и после сдвига. Полученный процент применяется к
  выбранному региону, поэтому он показывает общую для регионов связь.
- Уровень преступности региона устойчив: назвать его на следующий год точнее всего позволяет прошлогоднее
  значение. Расчёт нужен для другого: показать, как уровень связан с условиями жизни в регионе.
- Оценка «какой уровень обычно бывает при таких условиях» грубее: она строится только по другим регионам,
  её типичная ошибка указана рядом с самой оценкой.
- {WAGE_NOTE}
""")
    st.caption(f"Для специалистов: R² на проверочных данных {num(r2_s, 2)}; подробная проверка моделей — в режиме "
               f"«{ADVANCED}». Данные: Генпрокуратура, МВД и Росстат (обработка «Если быть точным», ЕМИСС), "
               "переписи 2010 и 2020 гг. Данные о преступности по регионам доступны по 2022 год.")
    st.stop()

# ---------- Боковая панель ----------
for key, value in {"degree": DEFAULTS["Полиномиальная регрессия"]["degree"], "alpha": 10, "k": KNN_P["k"],
                   "trees": RF_P["trees"], "use_ridge": True, "knn_distance": KNN_P["weights"] == "distance",
                   "max_features": RF_P["max_features"]}.items():
    st.session_state.setdefault(key, value)

st.sidebar.header("Настройки модели")

category = st.sidebar.selectbox("Вид преступлений", list(CATEGORIES), key="category", help=CATEGORY_HELP)
target = CATEGORIES[category][1]



def on_method():
    """При смене метода сила регуляризации возвращается к значению, подобранному в ноутбуке 06:
    иначе цифры выбранного метода расходились бы с вкладкой «Сравнение методов»."""
    tuned = DEFAULTS[st.session_state.method]
    if "alpha" in tuned:
        st.session_state.alpha = tuned["alpha"]


method = st.sidebar.selectbox("Метод", list(METHODS), index=len(METHODS) - 1, key="method", on_change=on_method,
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
# Параметры метода убраны в раскрывающийся блок, чтобы боковая панель помещалась на экране
box = st.sidebar.expander("Параметры метода") if method != "Линейная регрессия" else st.sidebar
if method == "Полиномиальная регрессия":
    params["degree"] = box.slider(
        "Степень полинома", 1, 6, key="degree",
        help="1 — прямая линия; 2 — добавляются квадраты и произведения факторов; чем выше степень, "
             "тем гибче формула и тем выше риск переобучения.")
    params["use_ridge"] = box.checkbox(
        "Регуляризация (Ridge)", key="use_ridge",
        help="Штраф за слишком большие коэффициенты: не даёт формуле изгибаться под каждую точку.")
if method in ("Lasso", "Ridge", "ElasticNet") or (method == "Полиномиальная регрессия" and params["use_ridge"]):
    params["alpha"] = box.select_slider(
        "Сила регуляризации alpha", options=ALPHAS, key="alpha",
        format_func=lambda v: f"{v:g}".replace(".", ","),
        help="Насколько сильно сдерживать модель. Мало (0,01) — модель почти свободна и легко переобучается; "
             "много (1000) — модель осторожная и простая, может недоучиться.")
elif method == "Полиномиальная регрессия":
    params["alpha"] = None
if method in ("k ближайших соседей (kNN)", "Гибридная модель (лес + kNN)"):
    params["k"] = box.slider(
        "Число соседей k", 1, 30, key="k",
        help="Сколько самых похожих строк усреднять. Мало соседей — модель повторяет отдельные строки и "
             "переобучается; много — оценки сглаживаются и становятся грубее.")
if method in ("k ближайших соседей (kNN)", "Гибридная модель (лес + kNN)"):
    params["weights"] = "distance" if box.checkbox(
        "Ближние соседи весят больше", key="knn_distance",
        help="Если включено, вклад соседа в оценку тем больше, чем он ближе по факторам.") else "uniform"
if method in ("Случайный лес", "Гибридная модель (лес + kNN)"):
    params["trees"] = box.select_slider(
        "Число деревьев", options=[50, 100, 300, 500], key="trees",
        help="Больше деревьев — устойчивее результат, но дольше расчёт.")
    params["max_features"] = box.select_slider(
        "Доля факторов для каждого дерева", options=[0.33, 0.66, 1.0], key="max_features",
        format_func=lambda v: num(v, 2),
        help="Какая часть факторов доступна дереву при каждом разбиении. Меньше — деревья разнообразнее.")

if method == "ElasticNet":
    params["l1_ratio"] = FINAL["ElasticNet"]["l1_ratio"]  # доля штрафа Lasso: подобрана в ноутбуке 06, не настраивается
if method != "Линейная регрессия":
    labels = {"degree": "степень", "alpha": "alpha", "k": "соседей", "trees": "деревьев", "max_features": "доля факторов"}
    st.sidebar.caption("Сейчас: " + ", ".join(
        f"{labels[k]} {str(v).replace('.', ',')}" for k, v in params.items() if k in labels and v is not None))

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
st.title("Факторы преступности в регионах России: модели и их проверка")
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
c1.metric("R² на тесте", num(r2_te, 3),
          help="Главная цифра: какую долю различий в уровне преступности модель угадывает на спрятанных данных. "
               "1 — угадывает всё; 0 — не лучше, чем всем назвать среднее; меньше 0 — хуже среднего. "
               "Целевой уровень около 0,85, допустимый — от 0,8.")
c2.metric("R² на обучении", num(r2_tr, 3),
          help="То же на данных, на которых модель училась. Обычно выше, чем на тесте.")
c3.metric("Средняя ошибка на тесте", num(mae_te, 0),
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
        st.dataframe(ru_table(st.session_state.tune_table.head(10)), hide_index=True, width="stretch")

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
    show(fig)
    st.dataframe(ru_table(table.assign(**{"средняя ошибка": table["средняя ошибка"].round().astype(int)})),
                 hide_index=True, width="stretch")
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
                "обучающих данных). Для сравнения добавлены две простые оценки, в которых нет ни одного фактора: "
                "уровень того же региона в ближайшем году и средний уровень региона по обучающим данным.")
    st.markdown("- **случайно 70/30** — основной способ;\n"
                "- **вложенная проверка 5 × 5** — данные пять раз делятся на пять частей, каждая по очереди "
                "становится тестовой, а подбор параметров повторяется внутри каждого раунда; показано среднее и "
                "разброс;\n"
                "- **по годам** — обучение на 2011–2019 гг., проверка на 2020–2022 гг.;\n"
                "- **по регионам** — проверка на регионах, которых не было в обучении.")
    final = load_final(stamp(FINAL_PATH))
    # Простые оценки без факторов (ноутбук 09) добавляются к методам как точка отсчёта
    base_cmp, contrib = load_baseline(stamp(BASELINE_PATH, CONTRIB_PATH))
    simple = base_cmp[base_cmp["способ оценки"].str.startswith("Простая")]
    final = pd.concat([final, pd.DataFrame({
        "метод": simple["способ оценки"].values, "параметры": "факторы не используются",
        "R² тест 70/30": simple["R² 70/30"].values, "MAE тест": simple["MAE 70/30"].values,
        "R² вложенная, среднее": simple["R² 5 частей, среднее"].values,
        "R² вложенная, разброс": simple["R² 5 частей, разброс"].values,
        "R² по годам": simple["R² по годам"].values})], ignore_index=True)
    checks = {"R² тест 70/30": "случайно 70/30", "R² вложенная, среднее": "вложенная 5 × 5",
              "R² по годам": "по годам", "R² по регионам": "по регионам"}
    order = final.sort_values("R² вложенная, среднее")["метод"].tolist()  # лучший метод — вверху графика
    long = final.melt(id_vars=["метод"], value_vars=list(checks), var_name="проверка", value_name="R²")
    long = long.dropna(subset=["R²"])  # у простых оценок нет проверки по регионам: прошлых значений региона нет
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
    show(fig)
    worst = final.loc[final["R² по регионам"].idxmin()]
    st.caption("Точка — R² метода при одном способе проверки, чем правее, тем точнее. Горизонтальная черта у "
               "вложенной проверки — разброс между раундами. Методы упорядочены по результату вложенной "
               f"проверки. Значения ниже 0 (модель хуже простого среднего) показаны у левого края (например, {worst['метод'].lower()} по "
               f"регионам: {num(worst['R² по регионам'], 2)}); точные значения — при наведении и в таблице ниже.")
    hyb = final[final["метод"].str.startswith("Гибридная")].iloc[0]
    near = final[final["метод"].str.contains("ближайший год")].iloc[0]
    by_regions = contrib[contrib["проверка"].str.startswith("по регионам")].set_index("способ оценки")["R²"]
    within = contrib[contrib["проверка"].str.startswith("внутри региона")].set_index("способ оценки")["R²"]
    st.caption("У простых оценок вторая проверка — перекрёстная на 5 частей без подбора параметров; проверки по "
               "регионам у них нет: у региона, которого не было в обучении, нет прошлых значений.")
    st.markdown(f"**Вывод.** Среди методов, которые используют факторы, гибридная модель точнее остальных на "
                f"основном делении (R² = {num(hyb['R² тест 70/30'], 2)}), при вложенной проверке "
                f"({num(hyb['R² вложенная, среднее'], 2)} ± {num(hyb['R² вложенная, разброс'], 2)}) и при "
                f"проверке по годам ({num(hyb['R² по годам'], 2)}). Линейные методы объясняют меньше половины "
                f"различий: связь факторов с преступностью нелинейна.")
    st.markdown(f"**Точка отсчёта.** Простая оценка без единого фактора — уровень того же региона в ближайшем "
                f"году — при тех же проверках даёт {num(near['R² тест 70/30'], 2)}, "
                f"{num(near['R² вложенная, среднее'], 2)} и {num(near['R² по годам'], 2)}, то есть точнее модели. "
                f"Уровень преступности региона устойчив во времени, и модель узнаёт регион по показателям, "
                f"которые от года к году почти не меняются. Поэтому ориентир 0,85 сам по себе влияния факторов "
                f"не подтверждает.")
    st.markdown(f"**Вклад факторов.** Его показывают проверки, при которых узнать регион нельзя. Для регионов, "
                f"которых не было в обучении, гибридная модель объясняет {num(by_regions.iloc[0] * 100, 0)} % "
                f"различий (без факторов — около нуля). Изменения во времени внутри региона она описывает на "
                f"{num(within.iloc[0] * 100, 0)} % (оценка по ближайшему году — на {num(within.iloc[1] * 100, 0)} %). "
                f"Модель предназначена для анализа связи уровня преступности с условиями региона: ожидаемый "
                f"уровень при таких условиях, сравнение регионов, сценарии. Назвать уровень региона в следующем "
                f"году точнее позволяет его прошлое значение.")
    with st.expander("Таблица с точными значениями"):
        shown = final.replace({"участники с параметрами из шага 2": "лес и kNN с параметрами из строк ниже"})
        shown = shown.rename(columns={"параметры": "подобранные параметры", "MAE тест": "средняя ошибка 70/30"})
        shown["средняя ошибка 70/30"] = shown["средняя ошибка 70/30"].round().astype(int)
        st.dataframe(ru_table(shown), hide_index=True, width="stretch")
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
    show(fig)
    st.caption("Таблица: показатели региона по годам, на которых модель строит оценку.")
    st.dataframe(ru_table(reg[["year", target] + features].rename(
        columns={"year": "год", target: "преступлений на 100 тыс.", **FEATURE_NAMES}), 2),
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
        show(fig)
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
            show(fig)
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
        show(fig)
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
        show(fig)
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

    selection_block(tuple(features), method, freeze(params), split_type, target, row.iloc[0], "s_", "a_sel")
    st.caption("Ползунки: на сколько процентов изменить показатель относительно его реального значения "
               "в выбранном году.")
    scenario = base.copy()
    cols = st.columns(3)
    for i, f in enumerate(features):
        now = base[f].iloc[0]
        st.session_state.setdefault(f"s_{f}", 0)
        change = cols[i % 3].slider(FEATURE_NAMES[f], -30, 30, step=5, key=f"s_{f}", format="%d %%")
        scenario[f] = base[f] * (1 + change / 100)
        cols[i % 3].caption(f"сейчас {num(now, 2)}" + (f" → по сценарию **{num(scenario[f].iloc[0], 2)}**"
                                                       if change else ""))

    train_min, train_max = X.iloc[tr].min(), X.iloc[tr].max()
    outside = [FEATURE_NAMES[f] for f in features if not train_min[f] <= scenario[f].iloc[0] <= train_max[f]]
    if outside:
        st.warning("Значения вышли за пределы того, что встречалось в обучающих данных: " + ", ".join(outside)
                   + ". Модель таких регионов не видела, её оценка здесь ненадёжна.")

    changes = tuple(int(st.session_state[f"s_{f}"]) for f in features)
    fact = row[target].iloc[0]
    base_pred, new_pred = model.predict(base)[0], model.predict(scenario)[0]
    m1, m2 = st.columns(2)
    m1.metric("Факт", num(fact, 0),
              help="Реальный уровень преступности в регионе в выбранном году, на 100 тыс. жителей.")
    if any(changes):
        total, single = scenario_effects(tuple(features), method, freeze(params), split_type, target, changes)
        new_avg = fact * (1 + total / 100)
        # в поле delta нужен обычный дефис: по нему Streamlit определяет знак и цвет стрелки
        delta = f"{num(new_avg - fact, 0)} ({num(total, 1)} %)".replace("−", "-")
        m2.metric("Оценка по сценарию", num(new_avg, 0), delta=delta, delta_color="inverse",
                  help="Все строки тестовой выборки сдвигаются на выбранные проценты; относительное изменение "
                       "средней оценки модели применяется к факту региона.")
    else:
        m2.metric("Оценка по сценарию", num(fact, 0), help="Сдвиньте ползунки, чтобы получить сценарий.")
    with st.expander("Справочно: оценка модели в одной точке"):
        st.markdown(f"Оценка модели для одной строки с изменёнными показателями: **{num(new_pred, 0)}**; при "
                    f"реальных показателях — {num(base_pred, 0)} (разница {num(new_pred - base_pred, 0)}). "
                    "У леса и kNN отклик в одной точке ступенчатый: оценка берётся по похожим строкам обучающей "
                    "выборки, и при любом сдвиге похожими становятся другие регионы и годы. Поэтому основная "
                    "оценка сценария считается по всей тестовой выборке.")
    if any(changes):
        show(effects_chart(single, features))
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
    cat_cmp, cat_fac = load_categories(stamp(CAT_COMPARE_PATH, CAT_FACTORS_PATH))
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
    show(fig)
    st.caption("Значения ниже 0 показаны у левого края; точные значения — при наведении и в таблице ниже.")
    shown = cat_cmp[["вид", "R² тест 70/30", "средняя ошибка", "средняя ошибка, % среднего", "R² 5 частей",
                     "R² по годам", "R² по регионам", "R² линейная 70/30", "R² 70/30 без доли нераскрытых"]]
    st.dataframe(ru_table(shown), hide_index=True, width="stretch")
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
    show(fig)
    st.caption("Важность — насколько падает точность модели, если значения фактора перемешать. В каждом столбце "
               "сумма равна 100 %: так виды с разной точностью можно сравнивать.")
    fig = px.imshow(eff.round(1), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0, zmin=-6, zmax=6,
                    title="Изменение средней оценки модели при росте фактора на 10 %, %")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="", height=360,
                      margin=dict(t=135, b=10))
    fig.update_xaxes(side="top", tickangle=0)
    show(fig)
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
