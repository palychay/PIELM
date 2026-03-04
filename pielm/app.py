"""
app.py  —  GA-PIELM · Уравнение Дарси
Запуск:  streamlit run app.py
Зависит: pielm.py (в той же папке)
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pielm import PIELM, GeneticOptimizer, diffusion_2d_operator

# ─────────────────────────────────────────────────────────────────────────────
#  СТРАНИЦА
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GA-PIELM · Darcy",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  ФИЗИКА   ∇²P = 0  на  [0,1]²
#  P(0,y)=1,  P(1,y)=0,  ∂P/∂n=0 (top/bot)  →  P(x,y) = 1 − x
# ─────────────────────────────────────────────────────────────────────────────

def darcy_operator(W, b, X, act_name="tanh"):
    return diffusion_2d_operator(W, b, X, act_name=act_name)

def darcy_source(X):
    return np.zeros((len(X), 1))

def p_exact(x, y):
    return 1.0 - x

def rmse_model(model, X):
    return float(np.sqrt(np.mean(
        (model.predict(X).ravel() - p_exact(X[:, 0], X[:, 1])) ** 2)))

def pde_res(model, X):
    H = darcy_operator(model.W, model.b, X, act_name=model.act_name)
    return float(np.mean((H @ model.beta).ravel() ** 2))

# ─────────────────────────────────────────────────────────────────────────────
#  ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def make_colloc(n):
    s = int(np.sqrt(n))
    t = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(t, t)
    return np.column_stack([xx.ravel(), yy.ravel()])

@st.cache_data
def make_grid():
    t = np.linspace(0, 1, 60)
    xx, yy = np.meshgrid(t, t)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])

@st.cache_data
def make_bc(n_b=40):
    t      = np.linspace(0, 1, n_b)
    left   = np.column_stack([np.zeros(n_b), t])
    right  = np.column_stack([np.ones(n_b),  t])
    top    = np.column_stack([t, np.ones(n_b)])
    bottom = np.column_stack([t, np.zeros(n_b)])
    Xb = np.vstack([left, right, top, bottom])
    Yb = np.concatenate([
        np.ones(n_b), np.zeros(n_b),
        p_exact(top[:, 0],    top[:, 1]),
        p_exact(bottom[:, 0], bottom[:, 1]),
    ])
    return Xb, Yb

# ─────────────────────────────────────────────────────────────────────────────
#  ГА-ОБЁРТКА
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(Xf, Xb, Yb, n_pop, n_gen, seed, cb=None):
    ga = GeneticOptimizer(
        n_pop=n_pop, n_gen=n_gen,
        hidden_bounds=(50, 600),
        scale_bounds=(0.5, 10.0),
        lambda_pde_bounds=(0.01, 10.0),
        lambda_bc_bounds=(1.0, 100.0),
        seed=seed,
    )

    def _wrap(gen, loss, params):
        if cb:
            cb(gen + 1, n_gen, loss, params)

    best_p = ga.search(Xf, Xb, Yb, darcy_operator, darcy_source, callback=_wrap)

    model = PIELM(
        n_hidden   = best_p["n_hidden"],
        input_dim  = 2,
        scale      = best_p["scale"],
        act_name   = best_p["activation"],
        lambda_pde = best_p["lambda_pde"],
        lambda_bc  = best_p["lambda_bc"],
        seed       = seed,
    ).initialize()
    model.fit(Xf, Xb, Yb, darcy_operator, darcy_source)

    return model, best_p, ga.history

# ─────────────────────────────────────────────────────────────────────────────
#  ТЕМА PLOTLY  —  тёмная, все подписи и значения colorbar чётко видны
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#1e1e2e"    # фон области графика
_BG2  = "#16161e"    # фон бумаги (за графиком)
_GRID = "#313244"    # линии сетки
_TEXT = "#cdd6f4"    # основной текст, оси, тики
_SUB  = "#7f849c"    # второстепенный текст

# Базовый layout для всех фигур
_PL = dict(
    paper_bgcolor = _BG2,
    plot_bgcolor  = _BG,
    font          = dict(color=_TEXT, size=12, family="monospace"),
    margin        = dict(l=50, r=20, t=50, b=40),
    title_font    = dict(color=_TEXT, size=13),
)

# Стиль осей
_AX = dict(
    gridcolor  = _GRID,
    zeroline   = False,
    color      = _TEXT,           # цвет надписи оси и тиков
    tickfont   = dict(color=_TEXT, size=11),
    title_font = dict(color=_TEXT, size=11),
    linecolor  = _GRID,
    showline   = True,
)

# Базовый стиль colorbar — тёмный фон, светлый текст
_CB_BASE = dict(
    thickness    = 12,
    outlinecolor = _GRID,
    outlinewidth = 1,
    tickfont     = dict(color=_TEXT, size=10),
    title_font   = dict(color=_TEXT, size=10),
)

# ─────────────────────────────────────────────────────────────────────────────
#  ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────────────

def chart_pressure(model, xx, yy, Xg):
    """Три тепловые карты: PIELM, аналитика, |ошибка|."""
    Pp = model.predict(Xg).reshape(xx.shape)
    Pt = p_exact(xx, yy)
    Pe = np.abs(Pp - Pt)

    fig = make_subplots(
        1, 3,
        subplot_titles=["PIELM  P̂(x,y)", "Аналитика  1−x", "|Ошибка|"],
        horizontal_spacing=0.12,
    )
    # Цвет заголовков подграфиков
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=12)

    # Первые два — давление (одинаковая шкала), третий — ошибка в scientific
    for col, (z, cs, fmt) in enumerate(
        [(Pp, "RdYlBu_r", ".2f"),
         (Pt, "RdYlBu_r", ".2f"),
         (Pe, "Reds",     ".2e")],   # <-- scientific для ошибки
        start=1,
    ):
        cb_x = 0.305 * col - 0.085
        fig.add_trace(go.Heatmap(
            z=z, x=xx[0], y=yy[:, 0],
            colorscale=cs, showscale=True,
            colorbar=dict(x=cb_x, len=0.85, tickformat=fmt, **_CB_BASE),
        ), row=1, col=col)
        fig.update_xaxes(title_text="x", row=1, col=col, **_AX)
        fig.update_yaxes(
            title_text="y" if col == 1 else "",
            row=1, col=col, **_AX,
        )

    fig.update_layout(**_PL, height=310, title_text="Поле давления")
    return fig


def chart_velocity(model, xx, yy, Xg):
    """Тепловая карта скорости + стрелки направления."""
    P   = model.predict(Xg).reshape(xx.shape)
    ux  = -np.gradient(P, xx[0, 1] - xx[0, 0], axis=1)
    uy  = -np.gradient(P, yy[1, 0] - yy[0, 0], axis=0)
    spd = np.hypot(ux, uy)

    s  = 6
    xs = xx[::s, ::s].ravel()
    ys = yy[::s, ::s].ravel()
    us = ux[::s, ::s].ravel()
    vs = uy[::s, ::s].ravel()
    nrm = np.hypot(us, vs) + 1e-12

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=spd, x=xx[0], y=yy[:, 0],
        colorscale="Blues", showscale=True,
        # скорость ~1, показываем 3 знака после запятой
        colorbar=dict(title="│u│", len=0.85, tickformat=".3f", **_CB_BASE),
    ))
    for xi, yi, ui, vi in zip(xs[::2], ys[::2],
                               us[::2] / nrm[::2], vs[::2] / nrm[::2]):
        fig.add_annotation(
            x=xi + ui * 0.03, y=yi + vi * 0.03,
            ax=xi, ay=yi,
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=1.3,
            arrowcolor="#89b4fa",
        )

    fig.update_layout(
        **_PL, height=310,
        title_text="Скорость фильтрации   u = −(k/μ) ∇P",
    )
    fig.update_xaxes(title="x", **_AX)
    fig.update_yaxes(title="y", **_AX)
    return fig


def chart_profiles(model):
    """Профили P(x) при y = 0.25, 0.5, 0.75."""
    x      = np.linspace(0, 1, 300)
    # Цвета catppuccin — хорошо видны на тёмном фоне
    colors = ["#89b4fa", "#a6e3a1", "#f38ba8"]

    fig = go.Figure()
    for c, yv in zip(colors, [0.25, 0.5, 0.75]):
        xy = np.column_stack([x, np.full_like(x, yv)])
        # PIELM: сплошная с маркерами (видны даже при полном совпадении)
        fig.add_trace(go.Scatter(
            x=x, y=model.predict(xy).ravel(),
            mode="lines+markers",
            name=f"PIELM  y={yv}",
            line=dict(color=c, width=2),
            marker=dict(size=3, opacity=0.5),
        ))
        # Аналитика: пунктир того же цвета
        fig.add_trace(go.Scatter(
            x=x, y=p_exact(x, yv),
            mode="lines",
            name=f"Аналит. y={yv}",
            line=dict(color=c, width=1.5, dash="dot"),
        ))

    fig.update_layout(
        **_PL, height=310,
        title_text="Профили P(x) при фиксированных y",
        xaxis=dict(title="x", **_AX),
        yaxis=dict(title="P", **_AX),
        legend=dict(
            font=dict(color=_TEXT, size=10),
            bgcolor=_BG2,
            bordercolor=_GRID,
            borderwidth=1,
        ),
    )
    return fig


def chart_convergence(histories: dict):
    """Кривые сходимости ГА (log шкала)."""
    palette = px.colors.qualitative.D3
    fig = go.Figure()
    for i, (lbl, hist) in enumerate(histories.items()):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist) + 1)), y=hist,
            mode="lines+markers", name=lbl,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        **_PL, height=290,
        title_text="Сходимость ГА",
        xaxis=dict(title="Поколение", **_AX),
        yaxis=dict(title="RMSE", type="log", **_AX),
        legend=dict(
            font=dict(color=_TEXT, size=10),
            bgcolor=_BG2, bordercolor=_GRID, borderwidth=1,
        ),
    )
    return fig


def chart_heatmap(df: pd.DataFrame):
    """Тепловая карта RMSE: популяция × поколения."""
    pivot = df.pivot(index="Популяция", columns="Поколения", values="RMSE")
    zv    = np.log10(pivot.values.astype(float) + 1e-20)
    txt   = [[f"{v:.1e}" for v in row] for row in pivot.values]

    fig = go.Figure(go.Heatmap(
        z=zv,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="RdYlGn_r",
        text=txt,
        texttemplate="%{text}",
        textfont=dict(size=11, color=_TEXT),
        colorbar=dict(title="log₁₀(RMSE)", **_CB_BASE),
    ))
    fig.update_layout(
        **_PL, height=310,
        title_text="RMSE: популяция × поколения",
        xaxis=dict(title="Поколения", **_AX),
        yaxis=dict(title="Популяция", **_AX),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_INIT: dict = {
    "cfg_n_pop":     20,
    "cfg_n_gen":     10,
    "cfg_n_colloc":  400,       # значение по умолчанию для selectbox
    "cfg_seed":      42,
    "cfg_exp_pops":  [5, 10, 20],
    "cfg_exp_gens":  [3, 5, 10],
    "mode":          None,
    "best_model":    None,
    "best_params":   None,
    "ga_history":    [],
    "ga_elapsed":    0.0,
    "exp_df":        None,
    "exp_histories": {},
}

for _k, _v in _INIT.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def clear_results():
    """Сбросить результаты при изменении параметров."""
    for k in ("mode", "best_model", "best_params", "ga_history",
              "ga_elapsed", "exp_df", "exp_histories"):
        st.session_state[k] = _INIT[k]

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Параметры")

    # Слайдеры — значение из session_state, без value=
    st.slider(
        "Размер популяции", min_value=5, max_value=50, step=5,
        key="cfg_n_pop", on_change=clear_results,
    )
    st.slider(
        "Число поколений", min_value=3, max_value=30, step=1,
        key="cfg_n_gen", on_change=clear_results,
    )

    # ВАЖНО: selectbox с key= НЕ должен получать аргумент index=,
    # иначе Streamlit выдаёт предупреждение о конфликте default/session_state.
    # Streamlit сам читает текущее значение из session_state["cfg_n_colloc"].
    st.selectbox(
        "Коллокационных точек",
        options=[100, 225, 400, 625, 900],
        key="cfg_n_colloc",
        on_change=clear_results,
    )

    st.number_input(
        "Seed", min_value=0, step=1,
        key="cfg_seed", on_change=clear_results,
    )

    st.divider()
    st.markdown("**Эксперимент**")
    st.multiselect(
        "Популяции",  [5, 10, 15, 20, 30, 50],
        key="cfg_exp_pops", on_change=clear_results,
    )
    st.multiselect(
        "Поколения", [3, 5, 10, 15, 20],
        key="cfg_exp_gens", on_change=clear_results,
    )

    btn_single = st.button("▶ Запустить ГА",          type="primary")
    btn_exp    = st.button("🔬 Запустить эксперимент")

    # ── Параметры модели: st.metric корректен в любой теме ───────────────────
    st.divider()
    st.markdown("**Параметры модели**")
    sidebar_ph = st.empty()

    def render_sidebar(params=None, rmse=None, pde=None):
        with sidebar_ph.container():
            if params is None:
                st.caption("— нет данных —")
                return
            c1, c2 = st.columns(2)
            c1.metric("Нейронов",  params["n_hidden"])
            c2.metric("Активация", params["activation"])
            c3, c4 = st.columns(2)
            c3.metric("σ",      f"{params['scale']:.2f}")
            c4.metric("λ_pde",  f"{params['lambda_pde']:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("λ_bc",   f"{params['lambda_bc']:.2f}")
            c6.metric("",       "")
            if rmse is not None:
                st.metric("RMSE",        f"{rmse:.3e}")
            if pde is not None:
                st.metric("Невязка PDE", f"{pde:.3e}")

    # Начальный рендер при наличии сохранённой модели
    _xx0, _yy0, _Xg0 = make_grid()
    _Xf0 = make_colloc(st.session_state.cfg_n_colloc)
    render_sidebar(
        st.session_state.best_params,
        rmse=rmse_model(st.session_state.best_model, _Xg0)
             if st.session_state.best_model else None,
        pde=pde_res(st.session_state.best_model, _Xf0)
            if st.session_state.best_model else None,
    )

    st.divider()
    st.markdown("**Уравнение Дарси**")
    for _eq in ["∇·(k/μ ∇P) = 0", "u = −(k/μ) ∇P",
                "P(0,y)=1,  P(1,y)=0", "∂P/∂n = 0 (top/bot)",
                "Точное:  P = 1 − x"]:
        st.code(_eq, language=None)

# ─────────────────────────────────────────────────────────────────────────────
#  ШАПКА
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌊 GA-PIELM · Фильтрация Дарси")
st.caption(
    "Физически информированная ELM  ·  Генетический алгоритм  ·  "
    "2D стационарное уравнение Дарси"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────────────

Xf            = make_colloc(st.session_state.cfg_n_colloc)
Xb, Yb        = make_bc()
xx, yy, Xgrid = make_grid()

# ─────────────────────────────────────────────────────────────────────────────
#  ГЛАВНЫЙ КОНТЕЙНЕР  (единый, заменяется целиком при каждом запуске)
# ─────────────────────────────────────────────────────────────────────────────

main = st.empty()

# ─── Рендер одиночного запуска ────────────────────────────────────────────────

def show_single():
    model   = st.session_state.best_model
    params  = st.session_state.best_params
    history = st.session_state.ga_history
    elapsed = st.session_state.ga_elapsed
    if model is None:
        return

    rv = rmse_model(model, Xgrid)
    pv = pde_res(model, Xf)

    with main.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE",        f"{rv:.2e}")
        c2.metric("Невязка PDE", f"{pv:.2e}")
        c3.metric("Нейронов",    str(params["n_hidden"]))
        c4.metric("Активация",   params["activation"])
        c5.metric("Время",       f"{elapsed:.1f} с")

        with st.expander("Все гиперпараметры", expanded=False):
            st.dataframe(pd.DataFrame([{
                "N":       params["n_hidden"],
                "act":     params["activation"],
                "σ":       round(params["scale"],      3),
                "λ_pde":   round(params["lambda_pde"], 4),
                "λ_bc":    round(params["lambda_bc"],  3),
                "RMSE":    f"{rv:.3e}",
                "PDE res": f"{pv:.3e}",
            }]), hide_index=True)

        st.divider()
        st.plotly_chart(chart_pressure(model, xx, yy, Xgrid), width="stretch")

        col_v, col_p = st.columns(2)
        with col_v:
            st.plotly_chart(chart_velocity(model, xx, yy, Xgrid), width="stretch")
        with col_p:
            st.plotly_chart(chart_profiles(model), width="stretch")

        if history:
            st.plotly_chart(chart_convergence({"ГА": history}), width="stretch")


# ─── Рендер эксперимента ──────────────────────────────────────────────────────

def show_experiment():
    df    = st.session_state.exp_df
    hists = st.session_state.exp_histories
    if df is None:
        return

    bi = df["RMSE"].idxmin()
    with main.container():
        st.success(
            f"Лучший результат:  "
            f"pop = **{df.loc[bi,'Популяция']}**  ·  "
            f"gen = **{df.loc[bi,'Поколения']}**  ·  "
            f"RMSE = **{df.loc[bi,'RMSE']:.3e}**"
        )
        df_fmt = df.copy()
        df_fmt["RMSE"]        = df_fmt["RMSE"].map(lambda v: f"{v:.3e}")
        df_fmt["Невязка PDE"] = df_fmt["Невязка PDE"].map(lambda v: f"{v:.3e}")
        st.dataframe(df_fmt, hide_index=True)

        st.divider()
        col_h, col_c = st.columns(2)
        with col_h:
            st.plotly_chart(chart_heatmap(df), width="stretch")
        with col_c:
            if hists:
                st.plotly_chart(chart_convergence(hists), width="stretch")

        st.download_button(
            "⬇ Скачать CSV",
            data=df.to_csv(index=False).encode(),
            file_name="ga_pielm_experiment.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  ЗАПУСК — одиночный ГА
# ─────────────────────────────────────────────────────────────────────────────

if btn_single:
    clear_results()
    st.session_state.mode = "single"

    with main.container():
        st.markdown("#### Оптимизация гиперпараметров")
        prog   = st.progress(0)
        status = st.empty()
        log_ph = st.empty()
        log    = []

        def _cb(gen, total, loss, params):
            prog.progress(gen / total)
            status.markdown(
                f"Поколение **{gen}/{total}**  ·  "
                f"RMSE `{loss:.3e}`  ·  "
                f"N = {params['n_hidden']}  ·  "
                f"act = {params['activation']}"
            )
            log.append(
                f"[{gen:02d}/{total}]  RMSE={loss:.3e}  "
                f"N={params['n_hidden']:4d}  "
                f"act={params['activation']:8s}  "
                f"σ={params['scale']:.2f}  "
                f"λ_pde={params['lambda_pde']:.3f}  "
                f"λ_bc={params['lambda_bc']:.2f}"
            )
            log_ph.code("\n".join(log[-8:]))
            render_sidebar(params, rmse=loss)

        t0 = time.time()
        model, best_p, history = run_ga(
            Xf, Xb, Yb,
            st.session_state.cfg_n_pop,
            st.session_state.cfg_n_gen,
            seed=int(st.session_state.cfg_seed),
            cb=_cb,
        )
        elapsed = time.time() - t0
        prog.progress(1.0)
        status.success("✅ Оптимизация завершена")

    st.session_state.best_model  = model
    st.session_state.best_params = best_p
    st.session_state.ga_history  = history
    st.session_state.ga_elapsed  = elapsed

    render_sidebar(best_p,
                   rmse=rmse_model(model, Xgrid),
                   pde=pde_res(model, Xf))
    show_single()

# ─────────────────────────────────────────────────────────────────────────────
#  ЗАПУСК — эксперимент
# ─────────────────────────────────────────────────────────────────────────────

elif btn_exp and st.session_state.cfg_exp_pops and st.session_state.cfg_exp_gens:
    clear_results()
    st.session_state.mode = "experiment"

    combos = [
        (p, g)
        for p in sorted(st.session_state.cfg_exp_pops)
        for g in sorted(st.session_state.cfg_exp_gens)
    ]

    with main.container():
        st.markdown("#### Сравнительный эксперимент")
        prog = st.progress(0)
        info = st.empty()
        tbl  = st.empty()

        rows            = {}
        hists           = {}
        best_params_map = {}

        for i, (n_p, n_g) in enumerate(combos):
            label = f"pop={n_p} · gen={n_g}"
            info.markdown(f"⏳ **{label}**  ({i + 1}/{len(combos)})")

            t0 = time.time()
            m, p, h = run_ga(
                Xf, Xb, Yb, n_p, n_g,
                seed=int(st.session_state.cfg_seed),
            )
            elapsed = time.time() - t0

            rv = rmse_model(m, Xgrid)
            pv = pde_res(m, Xf)

            rows[(n_p, n_g)] = {
                "Популяция":    n_p,
                "Поколения":    n_g,
                "Оценок":       n_p * n_g,
                "RMSE":         rv,
                "Невязка PDE":  pv,
                "Нейронов":     p["n_hidden"],
                "Активация":    p["activation"],
                "λ_pde":        round(p["lambda_pde"], 4),
                "λ_bc":         round(p["lambda_bc"],  2),
                "Время, с":     round(elapsed, 2),
            }
            best_params_map[(n_p, n_g)] = p
            hists[label] = h
            prog.progress((i + 1) / len(combos))

            df_live = pd.DataFrame(list(rows.values())).copy()
            df_live["RMSE"]        = df_live["RMSE"].map(lambda v: f"{v:.3e}")
            df_live["Невязка PDE"] = df_live["Невязка PDE"].map(lambda v: f"{v:.3e}")
            tbl.dataframe(df_live, hide_index=True)

            best_key = min(rows, key=lambda k: rows[k]["RMSE"])
            render_sidebar(
                best_params_map[best_key],
                rmse=rows[best_key]["RMSE"],
                pde=rows[best_key]["Невязка PDE"],
            )

        info.success("✅ Эксперимент завершён")

    st.session_state.exp_df        = pd.DataFrame(list(rows.values()))
    st.session_state.exp_histories = hists
    show_experiment()

# ─────────────────────────────────────────────────────────────────────────────
#  ПОВТОРНЫЙ РЕНДЕР  (перезагрузка без нажатия кнопок)
# ─────────────────────────────────────────────────────────────────────────────

else:
    mode = st.session_state.mode
    if mode == "single" and st.session_state.best_model is not None:
        show_single()
    elif mode == "experiment" and st.session_state.exp_df is not None:
        show_experiment()
    else:
        with main.container():
            st.info(
                "Настройте параметры в боковой панели и нажмите  "
                "**▶ Запустить ГА**  или  **🔬 Запустить эксперимент**."
            )