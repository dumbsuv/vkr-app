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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
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
ALPHAS = [0.1, 1, 10, 100, 1000]
SPLITS = ["По годам (тест 2020–2022)", "Случайно 70/30", "По регионам (30 % регионов)"]

st.set_page_config(page_title="Факторы преступности в регионах РФ", layout="wide")


# ---------- Загрузка данных ----------
@st.cache_data
def load_data():
    """Читает таблицу «регион × год» и убирает строки с пропусками."""
    panel = pd.read_csv(DATA_PATH)
    return panel.dropna(subset=[TARGET] + list(FEATURE_NAMES)).reset_index(drop=True)


data = load_data()


# ---------- Вспомогательные функции ----------
def make_model(model_type, degree, use_ridge, alpha):
    """Собирает модель по настройкам."""
    if model_type == "Случайный лес":
        return RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=-1)
    final = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler(), final)


def split_rows(split_type):
    """Возвращает номера строк обучающей и тестовой выборок."""
    idx = np.arange(len(data))
    if split_type.startswith("По годам"):
        return idx[data["year"] <= 2019], idx[data["year"] >= 2020]
    if split_type.startswith("Случайно"):
        return train_test_split(idx, test_size=0.3, random_state=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    return next(gss.split(data, groups=data["region"]))


@st.cache_resource
def train(features, model_type, degree, use_ridge, alpha, split_type):
    """Обучает модель; результат запоминается, чтобы не переобучать при каждом клике."""
    tr, te = split_rows(split_type)
    model = make_model(model_type, degree, use_ridge, alpha)
    model.fit(data.loc[tr, list(features)], data.loc[tr, TARGET])
    return model, tr, te


@st.cache_data
def factor_effects(features, model_type, degree, use_ridge, alpha, split_type):
    """Два способа оценить роль факторов (для любой модели).

    1. Важность: насколько падает R² на тесте, если перемешать значения фактора
       (связь фактора с Y разрушается).
    2. Направление: насколько в среднем меняется оценка модели, если фактор вырастет на 10 %.
    """
    model, tr, te = train(features, model_type, degree, use_ridge, alpha, split_type)
    X_te, y_te = data.loc[te, list(features)], data.loc[te, TARGET]
    perm = permutation_importance(model, X_te, y_te, scoring="r2", n_repeats=5, random_state=1)
    base = model.predict(X_te)
    rows = []
    for i, f in enumerate(features):
        shifted = X_te.copy()
        shifted[f] = shifted[f] * 1.1
        rows.append({"фактор": FEATURE_NAMES[f],
                     "важность": perm.importances_mean[i],
                     "эффект +10 %": float(np.mean(model.predict(shifted) - base))})
    return pd.DataFrame(rows)


def tune(features, split_type, use_ridge):
    """Подбор степени и alpha перекрёстной проверкой только на обучающей выборке.

    Обучающая выборка делится на 5 частей: модель учится на четырёх и проверяется на пятой,
    по кругу. Тестовая выборка в подборе не участвует, иначе её оценка станет завышенной.
    """
    tr, _ = split_rows(split_type)
    X, y = data.loc[tr, list(features)], data.loc[tr, TARGET]
    if split_type.startswith("По регионам"):
        cv, groups = GroupKFold(n_splits=5), data.loc[tr, "region"]
    else:
        cv, groups = KFold(n_splits=5, shuffle=True, random_state=1), None
    rows = []
    for d in range(1, 5):
        for a in (ALPHAS if use_ridge else [None]):
            model = make_model("Полиномиальная регрессия", d, use_ridge, a)
            score = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2").mean()
            rows.append({"степень": d, "alpha": a if use_ridge else "—", "R² на проверке": score})
    return pd.DataFrame(rows).sort_values("R² на проверке", ascending=False).reset_index(drop=True)


def on_tune():
    """Запускается по кнопке: подбирает настройки и ставит лучшие в боковую панель."""
    table = tune(tuple(st.session_state.features), st.session_state.split_type, st.session_state.use_ridge)
    best = table.iloc[0]
    st.session_state.degree = int(best["степень"])
    if st.session_state.use_ridge:
        # приводим к значению из списка ALPHAS, иначе ползунок его не узнает
        st.session_state.alpha = min(ALPHAS, key=lambda a: abs(a - float(best["alpha"])))
    st.session_state.tune_table = table
    # запоминаем, для каких настроек сделан подбор
    st.session_state.tune_for = (tuple(st.session_state.features), st.session_state.split_type,
                                 st.session_state.use_ridge)


# ---------- Боковая панель: настройки модели ----------
# Начальные значения ползунков, которые может менять кнопка подбора
st.session_state.setdefault("degree", 2)
st.session_state.setdefault("alpha", 10)

st.sidebar.header("Настройки модели")

features = st.sidebar.multiselect(
    "Факторы", options=list(FEATURE_NAMES), default=list(FEATURE_NAMES),
    format_func=FEATURE_NAMES.get, key="features",
    help="Показатели региона, по которым модель оценивает уровень преступности.",
)
model_type = st.sidebar.radio(
    "Метод", ["Полиномиальная регрессия", "Случайный лес"], key="model_type",
    help="Полиномиальная регрессия — формула с квадратами и произведениями факторов (метод из рабочей тетради "
         "консультанта). Случайный лес — среднее множества деревьев решений. Остальные методы из ноутбука 02 "
         "будут добавлены после согласования с консультантом.",
)
is_poly = model_type == "Полиномиальная регрессия"
degree = st.sidebar.slider(
    "Степень полинома", 1, 6, key="degree", disabled=not is_poly,
    help="1 — прямая линия; 2 — добавляются квадраты и попарные произведения факторов; чем выше степень, "
         "тем гибче формула и тем выше риск переобучения.",
)
use_ridge = st.sidebar.checkbox(
    "Регуляризация (Ridge)", value=True, key="use_ridge", disabled=not is_poly,
    help="Штраф за слишком большие коэффициенты: не даёт формуле изгибаться под каждую точку.",
)
alpha = st.sidebar.select_slider(
    "Сила регуляризации alpha", options=ALPHAS, key="alpha", disabled=not (use_ridge and is_poly),
    help="Насколько сильно сдерживать модель. Мало (0,1) — модель почти свободна и легко переобучается; "
         "много (1000) — модель осторожная и простая, может недоучиться.",
)
split_type = st.sidebar.radio(
    "Способ проверки", SPLITS, key="split_type",
    help="Какие данные спрятать от модели при обучении, чтобы потом проверить, как она их угадывает. "
         "По годам — угадать 2020–2022 годы; случайно — угадать случайные 30 % строк; "
         "по регионам — угадать регионы, которых модель не видела.",
)
if is_poly:
    st.sidebar.button("Подобрать лучшие степень и alpha", on_click=on_tune, width="stretch",
                      help="Перебирает степени 1–4 и значения alpha и ставит лучшие. Проверка идёт только "
                           "на обучающих данных, тестовые в подборе не участвуют.")

if not features:
    st.warning("Выберите хотя бы один фактор.")
    st.stop()

model, tr, te = train(tuple(features), model_type, degree, use_ridge, alpha, split_type)
X, y = data[features], data[TARGET]
pred_tr, pred_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
r2_tr, r2_te = r2_score(y.iloc[tr], pred_tr), r2_score(y.iloc[te], pred_te)
mae_te = mean_absolute_error(y.iloc[te], pred_te)

# ---------- Заголовок и пояснение ----------
st.title("Влияние социально-экономических факторов на преступность в регионах РФ")
st.caption("Прототип. Данные: Генпрокуратура и Росстат (обработка «Если быть точным»), переписи 2010 и 2020 гг.; "
           "85 регионов, 2011–2022 гг.")

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
- «Регион» — реальный уровень преступности в регионе по годам и оценка модели;
- «Качество модели» — насколько точно модель угадывает спрятанные данные;
- «Влияние факторов» — какие показатели важнее для модели и в какую сторону они тянут оценку;
- «Что, если» — как изменится оценка модели, если изменить показатели региона.

Модель показывает **связь** показателей с преступностью; причинность она не доказывает.
""")

# ---------- Метрики ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² на тесте", f"{r2_te:.3f}",
          help="Главная цифра: какую долю различий в уровне преступности модель угадывает на спрятанных данных. "
               "1 — угадывает всё; 0 — не лучше, чем всем назвать среднее; меньше 0 — хуже среднего. "
               "Ориентир консультанта около 0,85.")
c2.metric("R² на обучении", f"{r2_tr:.3f}",
          help="То же на данных, на которых модель училась. Всегда выше, чем на тесте.")
c3.metric("Средняя ошибка на тесте", f"{mae_te:.0f}",
          help=f"На сколько преступлений на 100 тыс. жителей модель в среднем ошибается. "
               f"Для сравнения: средний уровень в данных {y.mean():.0f}.")
if is_poly:
    n_feat = comb(len(features) + degree, degree) - 1
    c4.metric("Признаков в модели", n_feat,
              help=f"Сколько комбинаций факторов (квадраты, произведения) строит полином. "
                   f"Строк в обучении: {len(tr)}. Если признаков больше, чем строк, модель может просто "
                   f"запомнить данные.")

if r2_te < 0:
    num = lambda v, d: f"{v:,.{d}f}".replace(",", " ").replace(".", ",")  # 12 345,67
    st.error(f"R² на тесте {num(r2_te, 2)}: модель на спрятанных данных хуже, чем простое среднее. Обычно это "
             f"значит, что формула слишком сложная для такого объёма данных и на незнакомых данных выдаёт "
             f"нереальные значения (самая низкая оценка на тесте: {num(pred_te.min(), 0)} преступлений на 100 тыс., "
             f"хотя меньше нуля их быть не может). Попробуйте меньшую степень, большую alpha или кнопку "
             f"«Подобрать лучшие степень и alpha».")
elif r2_tr - r2_te > 0.15:
    st.warning(f"Разрыв между обучением и тестом {r2_tr - r2_te:.2f}: признак переобучения. На знакомых данных "
               f"модель точнее, чем на новых, значит, часть данных она запомнила. Порог 0,15 взят из методических "
               f"правил работы.")

if is_poly and st.session_state.get("tune_for") == (tuple(features), split_type, use_ridge):
    with st.expander("Результат подбора степени и alpha", expanded=True):
        st.write(f"Подбор для способа проверки «{split_type}». Каждая строка — вариант настроек. «R² на проверке» "
                 "— среднее по пяти проверкам внутри обучающей выборки (тестовые данные в подборе не участвуют). "
                 "Лучший вариант (первая строка) выставлен слева; его итоговое качество — «R² на тесте» выше.")
        st.dataframe(st.session_state.tune_table.head(10).round(3), hide_index=True, width="stretch")

tab_region, tab_model, tab_factors, tab_whatif = st.tabs(
    ["Регион", "Качество модели", "Влияние факторов", "Что, если"])

# ---------- Вкладка 1: регион ----------
with tab_region:
    st.caption("Две линии по годам для выбранного региона: **факт** — реальный уровень преступности, "
               "**модель** — оценка модели по показателям региона в этом году. Чем ближе линии, тем лучше "
               "модель описывает регион. Годы, попавшие в тестовую выборку, модель при обучении не видела.")
    regions = sorted(data["region"].unique())
    region = st.selectbox("Регион", regions, index=regions.index("Москва"))
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

# ---------- Вкладка 2: качество модели ----------
with tab_model:
    left, right = st.columns(2)
    with left:
        fact_pred = pd.DataFrame({"факт": y.iloc[te].values, "оценка модели": pred_te,
                                  "регион": data["region"].iloc[te].values, "год": data["year"].iloc[te].values})
        fig = px.scatter(fact_pred, x="факт", y="оценка модели", hover_data=["регион", "год"],
                         title="Факт и оценка модели на тестовой выборке")
        lo, hi = y.min(), y.max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
        if fact_pred["оценка модели"].min() < 0 or fact_pred["оценка модели"].max() > 2 * hi:
            fig.update_yaxes(range=[0, 1.3 * hi])
            st.caption("Часть оценок вышла далеко за реальные значения и не видна на графике.")
        st.plotly_chart(fig, width="stretch")
        st.caption("Каждая точка — один регион в один год из **спрятанных** данных. По горизонтали — реальный "
                   "уровень преступности, по вертикали — оценка модели. Точка на пунктирной диагонали — модель "
                   "угадала точно; чем дальше от диагонали, тем больше ошибка. Нажмите на точку, чтобы увидеть "
                   "регион и год.")
    with right:
        if is_poly:
            rows = []
            for d in range(1, 7):
                m, _, _ = train(tuple(features), model_type, d, use_ridge, alpha, split_type)
                rows += [{"степень": d, "выборка": "обучение", "R²": r2_score(y.iloc[tr], m.predict(X.iloc[tr]))},
                         {"степень": d, "выборка": "тест", "R²": max(r2_score(y.iloc[te], m.predict(X.iloc[te])), -1)}]
            fig = px.line(pd.DataFrame(rows), x="степень", y="R²", color="выборка", markers=True,
                          title="R² при разных степенях полинома")
            fig.add_hline(y=0.85, line_dash="dash", line_color="grey", annotation_text="ориентир 0,85")
            fig.add_vline(x=degree, line_color="orange")
            fig.update_yaxes(range=[-1.05, 1.05])
            st.plotly_chart(fig, width="stretch")
            st.caption("Как меняется качество с ростом степени при текущих остальных настройках. Линия «обучение» "
                       "почти всегда растёт: формула всё точнее подгоняется под знакомые данные. Линия «тест» "
                       "сначала растёт, потом падает — это переобучение. Оранжевая вертикаль — выбранная степень, "
                       "пунктир — ориентир 0,85. Значения ниже −1 показаны как −1.")
        else:
            st.info("Для случайного леса степени нет. Важность факторов — на вкладке «Влияние факторов».")

# ---------- Вкладка 3: влияние факторов ----------
with tab_factors:
    effects = factor_effects(tuple(features), model_type, degree, use_ridge, alpha, split_type)
    left, right = st.columns(2)
    with left:
        fig = px.bar(effects.sort_values("важность"), x="важность", y="фактор", orientation="h",
                     title="Важность факторов для точности модели")
        fig.update_layout(yaxis_title="", xaxis_title="падение R² на тесте")
        st.plotly_chart(fig, width="stretch")
        st.caption("Мы по очереди перемешиваем значения одного фактора между строками (так его связь с "
                   "преступностью разрушается) и смотрим, насколько ухудшилась точность. Чем длиннее столбец, "
                   "тем сильнее модель опирается на этот фактор. Направление связи этот график не показывает.")
    with right:
        eff = effects.sort_values("эффект +10 %")
        eff["направление"] = np.where(eff["эффект +10 %"] > 0, "оценка растёт", "оценка снижается")
        fig = px.bar(eff, x="эффект +10 %", y="фактор", color="направление", orientation="h",
                     color_discrete_map={"оценка растёт": "#d62728", "оценка снижается": "#2ca02c"},
                     title="Что будет с оценкой, если фактор вырастет на 10 %")
        fig.update_layout(yaxis_title="", xaxis_title="изменение, преступлений на 100 тыс.")
        st.plotly_chart(fig, width="stretch")
        st.caption("Для каждой строки тестовой выборки увеличиваем один фактор на 10 % и смотрим, на сколько "
                   "в среднем изменилась оценка модели. Красный — оценка преступности растёт, зелёный — "
                   "снижается. Это связь при прочих равных внутри модели; причинность она не доказывает. "
                   "При низком R² на тесте выводы по этому графику ненадёжны.")

# ---------- Вкладка 4: сценарий «что, если» ----------
with tab_whatif:
    st.markdown("Выберите регион и год. Модель оценит уровень преступности в этом регионе **в этом же году**, "
                "если бы его показатели были другими. Это не прогноз на будущее: меняются условия, год остаётся "
                "тем же.")
    col_a, col_b = st.columns(2)
    w_region = col_a.selectbox("Регион", regions, key="w_region")
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

    # Предупреждение: показатель вышел за пределы того, что модель видела при обучении
    train_min, train_max = X.iloc[tr].min(), X.iloc[tr].max()
    outside = [FEATURE_NAMES[f] for f in features
               if not train_min[f] <= scenario[f].iloc[0] <= train_max[f]]
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
