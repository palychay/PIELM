"""
app.py — GA-PIELM / FGA-PIELM · Уравнение Дарси
Запуск: streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pielm import (
    PIELM, GeneticOptimizer, FuzzyGeneticOptimizer,
    diffusion_2d_operator,
    train_test_split_colloc,
)

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
#  ФИЗИКА   ∇²P = 0  →  P(x,y) = 1 − x
# ─────────────────────────────────────────────────────────────────────────────

def darcy_operator(W, b, X, act_name="tanh"):
    return diffusion_2d_operator(W, b, X, act_name=act_name)

def darcy_source(X):
    return np.zeros((len(X), 1))

def p_exact(x, y):
    return 1.0 - x

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
#  ЗАПУСК ОДНОГО АЛГОРИТМА (GA или FGA)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(Xf_all, Xb, Yb, n_pop, n_gen, seed, use_fuzzy, cb=None):
    """
    Общая функция запуска GA или FGA.
    Возвращает: model, best_p, history, X_train, X_test,
                rmse_tr, rmse_te, fuzzy_log (None для GA)
    """
    X_train, X_test, _, _ = train_test_split_colloc(Xf_all, test_size=0.25, seed=seed)

    OptimizerClass = FuzzyGeneticOptimizer if use_fuzzy else GeneticOptimizer
    opt = OptimizerClass(
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

    best_p = opt.search(
        X_train, Xb, Yb, darcy_operator, darcy_source, p_exact,
        callback=_wrap,
    )

    model = PIELM(
        n_hidden=best_p["n_hidden"], input_dim=2,
        scale=best_p["scale"], act_name=best_p["activation"],
        lambda_pde=best_p["lambda_pde"], lambda_bc=best_p["lambda_bc"],
        seed=seed,
    ).initialize()
    model.fit(X_train, Xb, Yb, darcy_operator, darcy_source)

    rmse_tr = model.rmse(X_train, p_exact(X_train[:, 0], X_train[:, 1]))
    rmse_te = model.rmse(X_test,  p_exact(X_test[:, 0],  X_test[:, 1]))

    fuzzy_log = getattr(opt, 'fuzzy_log', None)
    return model, best_p, opt.history, X_train, X_test, rmse_tr, rmse_te, fuzzy_log

# ─────────────────────────────────────────────────────────────────────────────
#  ТЕМА PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#1e1e2e"
_BG2  = "#16161e"
_GRID = "#313244"
_TEXT = "#cdd6f4"

_PL = dict(
    paper_bgcolor=_BG2, plot_bgcolor=_BG,
    font=dict(color=_TEXT, size=12, family="monospace"),
    margin=dict(l=50, r=20, t=50, b=40),
    title_font=dict(color=_TEXT, size=13),
)
_AX = dict(
    gridcolor=_GRID, zeroline=False, color=_TEXT,
    tickfont=dict(color=_TEXT, size=11),
    title_font=dict(color=_TEXT, size=11),
    linecolor=_GRID, showline=True,
)
_CB = dict(
    thickness=12, outlinecolor=_GRID, outlinewidth=1,
    tickfont=dict(color=_TEXT, size=10),
    title_font=dict(color=_TEXT, size=10),
)

# ─────────────────────────────────────────────────────────────────────────────
#  ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────────────

def chart_pressure(model, xx, yy, Xg):
    Pp = model.predict(Xg).reshape(xx.shape)
    Pt = p_exact(xx, yy)
    Pe = np.abs(Pp - Pt)
    fig = make_subplots(1, 3,
        subplot_titles=["PIELM  P̂(x,y)", "Аналитика  1−x", "|Ошибка|"],
        horizontal_spacing=0.12)
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=12)
    for col, (z, cs, fmt) in enumerate(
        [(Pp, "RdYlBu_r", ".2f"), (Pt, "RdYlBu_r", ".2f"), (Pe, "Reds", ".2e")],
        start=1):
        fig.add_trace(go.Heatmap(
            z=z, x=xx[0], y=yy[:, 0], colorscale=cs, showscale=True,
            colorbar=dict(x=0.305*col-0.085, len=0.85, tickformat=fmt, **_CB),
        ), row=1, col=col)
        fig.update_xaxes(title_text="x", row=1, col=col, **_AX)
        fig.update_yaxes(title_text="y" if col == 1 else "", row=1, col=col, **_AX)
    fig.update_layout(**_PL, height=310, title_text="Поле давления")
    return fig


def chart_velocity(model, xx, yy, Xg):
    P   = model.predict(Xg).reshape(xx.shape)
    ux  = -np.gradient(P, xx[0, 1]-xx[0, 0], axis=1)
    uy  = -np.gradient(P, yy[1, 0]-yy[0, 0], axis=0)
    spd = np.hypot(ux, uy)
    s   = 6
    xs, ys = xx[::s, ::s].ravel(), yy[::s, ::s].ravel()
    us, vs = ux[::s, ::s].ravel(), uy[::s, ::s].ravel()
    nrm = np.hypot(us, vs) + 1e-12
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=spd, x=xx[0], y=yy[:, 0], colorscale="Blues", showscale=True,
        colorbar=dict(title="│u│", len=0.85, tickformat=".3f", **_CB),
    ))
    for xi, yi, ui, vi in zip(xs[::2], ys[::2], us[::2]/nrm[::2], vs[::2]/nrm[::2]):
        fig.add_annotation(
            x=xi+ui*0.03, y=yi+vi*0.03, ax=xi, ay=yi,
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=1.3, arrowcolor="#89b4fa",
        )
    fig.update_layout(**_PL, height=310, title_text="Скорость фильтрации  u = −∇P")
    fig.update_xaxes(title="x", **_AX)
    fig.update_yaxes(title="y", **_AX)
    return fig


def chart_profiles(model):
    x      = np.linspace(0, 1, 300)
    colors = ["#89b4fa", "#a6e3a1", "#f38ba8"]
    fig    = go.Figure()
    for c, yv in zip(colors, [0.25, 0.5, 0.75]):
        xy = np.column_stack([x, np.full_like(x, yv)])
        fig.add_trace(go.Scatter(
            x=x, y=model.predict(xy).ravel(),
            mode="lines+markers", name=f"PIELM y={yv}",
            line=dict(color=c, width=2), marker=dict(size=3, opacity=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=p_exact(x, yv), mode="lines", name=f"Аналит. y={yv}",
            line=dict(color=c, width=1.5, dash="dot"),
        ))
    fig.update_layout(
        **_PL, height=310, title_text="Профили P(x)",
        xaxis=dict(title="x", **_AX), yaxis=dict(title="P", **_AX),
        legend=dict(font=dict(color=_TEXT, size=10),
                    bgcolor=_BG2, bordercolor=_GRID, borderwidth=1),
    )
    return fig


def chart_convergence(histories: dict):
    """Кривые сходимости — несколько серий на одном графике."""
    palette = px.colors.qualitative.D3
    fig = go.Figure()
    for i, (lbl, hist) in enumerate(histories.items()):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist)+1)), y=hist,
            mode="lines+markers", name=lbl,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        **_PL, height=290, title_text="Сходимость",
        xaxis=dict(title="Поколение", **_AX),
        yaxis=dict(title="RMSE", type="log", **_AX),
        legend=dict(font=dict(color=_TEXT, size=10),
                    bgcolor=_BG2, bordercolor=_GRID, borderwidth=1),
    )
    return fig


def chart_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="Популяция", columns="Поколения", values="RMSE test")
    zv    = np.log10(pivot.values.astype(float) + 1e-20)
    txt   = [[f"{v:.1e}" for v in row] for row in pivot.values]
    fig   = go.Figure(go.Heatmap(
        z=zv, x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="RdYlGn_r", text=txt,
        texttemplate="%{text}", textfont=dict(size=11, color=_TEXT),
        colorbar=dict(title="log₁₀(RMSE)", **_CB),
    ))
    fig.update_layout(
        **_PL, height=310, title_text="RMSE test: популяция × поколения",
        xaxis=dict(title="Поколения", **_AX),
        yaxis=dict(title="Популяция", **_AX),
    )
    return fig


def chart_test_scatter(X_train, X_test, model):
    err_tr = np.abs(model.predict(X_train).ravel() - p_exact(X_train[:, 0], X_train[:, 1]))
    err_te = np.abs(model.predict(X_test).ravel()  - p_exact(X_test[:, 0],  X_test[:, 1]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_train[:, 0], y=X_train[:, 1], mode="markers",
        marker=dict(size=5, color=err_tr, colorscale="Blues", showscale=True,
                    colorbar=dict(title="err train", x=0.45, **_CB)),
        name="Train (75%)",
    ))
    fig.add_trace(go.Scatter(
        x=X_test[:, 0], y=X_test[:, 1], mode="markers",
        marker=dict(size=7, symbol="diamond", color=err_te,
                    colorscale="Reds", showscale=True,
                    colorbar=dict(title="err test", x=1.0, **_CB)),
        name="Test (25%)",
    ))
    fig.update_layout(
        **_PL, height=340, title_text="Ошибка на train и test точках",
        xaxis=dict(title="x", **_AX),
        yaxis=dict(title="y", scaleanchor="x", **_AX),
        legend=dict(font=dict(color=_TEXT, size=11),
                    bgcolor=_BG2, bordercolor=_GRID, borderwidth=1),
    )
    return fig


def chart_fuzzy_log(fuzzy_log: list):
    """
    График нечёткого контроллера:
    e1, e2 и итоговая p_mut по поколениям.
    """
    gens  = [d['gen'] + 1 for d in fuzzy_log]
    e1s   = [d['e1']   for d in fuzzy_log]
    e2s   = [d['e2']   for d in fuzzy_log]
    pmuts = [d['p_mut'] for d in fuzzy_log]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Сигналы e₁ и e₂", "Вероятность мутации p_mut"],
        vertical_spacing=0.15,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=12)

    fig.add_trace(go.Scatter(
        x=gens, y=e1s, mode="lines+markers", name="e₁ (разнообразие)",
        line=dict(color="#89b4fa", width=2), marker=dict(size=5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gens, y=e2s, mode="lines+markers", name="e₂ (улучшение)",
        line=dict(color="#a6e3a1", width=2), marker=dict(size=5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gens, y=pmuts, mode="lines+markers", name="p_mut",
        line=dict(color="#f38ba8", width=2), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(243,139,168,0.15)",
    ), row=2, col=1)

    # Пороговые линии
    for y, dash, lbl in [(0.5, "dot", "граница МАЛО/ВЕЛИКО"),
                          (0.10, "dash", "LOW=0.10"),
                          (0.30, "dash", "MED=0.30"),
                          (0.60, "dash", "HIGH=0.60")]:
        row_ = 1 if lbl.startswith("г") else 2
        fig.add_hline(y=y, line_dash=dash, line_color="#585b70",
                      line_width=1, row=row_, col=1)

    fig.update_xaxes(title_text="Поколение", row=2, col=1, **_AX)
    fig.update_yaxes(title_text="e₁, e₂", row=1, col=1, range=[0, 1], **_AX)
    fig.update_yaxes(title_text="p_mut", row=2, col=1, range=[0, 0.75], **_AX)
    fig.update_layout(
        **_PL, height=420, title_text="Нечёткий контроллер мутации",
        legend=dict(font=dict(color=_TEXT, size=10),
                    bgcolor=_BG2, bordercolor=_GRID, borderwidth=1),
    )
    return fig


def chart_ga_fga_compare(hist_ga, hist_fga):
    """Сравнение кривых сходимости GA и FGA."""
    fig = go.Figure()
    for hist, lbl, clr in [
        (hist_ga,  "GA  (фиксированная мутация)", "#89b4fa"),
        (hist_fga, "FGA (нечёткая мутация)",       "#f38ba8"),
    ]:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist)+1)), y=hist,
            mode="lines+markers", name=lbl,
            line=dict(color=clr, width=2), marker=dict(size=5),
        ))
    fig.update_layout(
        **_PL, height=320, title_text="GA vs FGA: сходимость",
        xaxis=dict(title="Поколение", **_AX),
        yaxis=dict(title="RMSE (train)", type="log", **_AX),
        legend=dict(font=dict(color=_TEXT, size=11),
                    bgcolor=_BG2, bordercolor=_GRID, borderwidth=1),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_INIT = {
    "cfg_algo":      "GA",
    "cfg_n_pop":     20,
    "cfg_n_gen":     10,
    "cfg_n_colloc":  400,
    "cfg_seed":      42,
    "cfg_exp_pops":  [5, 10, 20],
    "cfg_exp_gens":  [3, 5, 10],
    "mode":          None,
    # одиночный запуск
    "best_model":    None,
    "best_params":   None,
    "best_split":    None,
    "best_rmse_tr":  None,
    "best_rmse_te":  None,
    "ga_history":    [],
    "ga_elapsed":    0.0,
    "fuzzy_log":     None,
    # сравнение GA vs FGA
    "cmp_done":      False,
    "cmp_ga":        None,   # (rmse_tr, rmse_te, history, elapsed)
    "cmp_fga":       None,
    # эксперимент
    "exp_df":        None,
    "exp_histories": {},
}
for _k, _v in _INIT.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def clear_results():
    for k in ("mode", "best_model", "best_params", "best_split",
              "best_rmse_tr", "best_rmse_te", "ga_history", "ga_elapsed",
              "fuzzy_log", "cmp_done", "cmp_ga", "cmp_fga",
              "exp_df", "exp_histories"):
        st.session_state[k] = _INIT[k]

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Параметры")

    st.radio("Алгоритм", ["GA", "FGA", "GA vs FGA"],
             key="cfg_algo", on_change=clear_results,
             help="GA — обычный, FGA — нечёткая мутация, GA vs FGA — сравнение")

    st.slider("Размер популяции", 5, 50, step=5,
              key="cfg_n_pop", on_change=clear_results)
    st.slider("Число поколений",  3, 30, step=1,
              key="cfg_n_gen", on_change=clear_results)
    st.selectbox("Коллокационных точек",
                 options=[100, 225, 400, 625, 900],
                 key="cfg_n_colloc", on_change=clear_results)
    st.number_input("Seed", min_value=0, step=1,
                    key="cfg_seed", on_change=clear_results)

    st.divider()
    st.markdown("**Эксперимент**")
    st.multiselect("Популяции",  [5, 10, 15, 20, 30, 50],
                   key="cfg_exp_pops", on_change=clear_results)
    st.multiselect("Поколения",  [3, 5, 10, 15, 20],
                   key="cfg_exp_gens", on_change=clear_results)

    btn_single = st.button("▶ Запустить",              type="primary")
    btn_exp    = st.button("🔬 Запустить эксперимент")

    st.divider()
    st.markdown("**Параметры модели**")
    sidebar_ph = st.empty()

    def render_sidebar(params=None, rmse_tr=None, rmse_te=None):
        with sidebar_ph.container():
            if params is None:
                st.caption("— нет данных —")
                return
            c1, c2 = st.columns(2)
            c1.metric("Нейронов",  params["n_hidden"])
            c2.metric("Активация", params["activation"])
            c3, c4 = st.columns(2)
            c3.metric("σ",     f"{params['scale']:.2f}")
            c4.metric("λ_pde", f"{params['lambda_pde']:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("λ_bc",  f"{params['lambda_bc']:.2f}")
            c6.metric("",      "")
            if rmse_tr is not None:
                st.metric("RMSE train", f"{rmse_tr:.3e}")
            if rmse_te is not None:
                st.metric("RMSE test",  f"{rmse_te:.3e}")

    render_sidebar(
        st.session_state.best_params,
        st.session_state.best_rmse_tr,
        st.session_state.best_rmse_te,
    )

    st.divider()
    st.markdown("**Уравнение Дарси**")
    for eq in ["∇²P = 0", "u = −(k/μ) ∇P",
               "P(0,y)=1,  P(1,y)=0",
               "∂P/∂n = 0  (top/bot)",
               "Точное:  P = 1 − x"]:
        st.code(eq, language=None)

# ─────────────────────────────────────────────────────────────────────────────
#  ШАПКА
# ─────────────────────────────────────────────────────────────────────────────

algo_label = st.session_state.cfg_algo
st.title(f"🌊 {algo_label}-PIELM · Фильтрация Дарси")
st.caption("Физически информированная ELM  ·  Разбивка 75/25  ·  "
           "Нечёткий ГА по: Herrera & Lozano (1995)")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────────────

Xf_all        = make_colloc(st.session_state.cfg_n_colloc)
Xb, Yb        = make_bc()
xx, yy, Xgrid = make_grid()
main          = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
#  РЕНДЕР: одиночный запуск (GA или FGA)
# ─────────────────────────────────────────────────────────────────────────────

def show_single():
    model      = st.session_state.best_model
    params     = st.session_state.best_params
    X_train, X_test = st.session_state.best_split
    rmse_tr    = st.session_state.best_rmse_tr
    rmse_te    = st.session_state.best_rmse_te
    history    = st.session_state.ga_history
    elapsed    = st.session_state.ga_elapsed
    fuzzy_log  = st.session_state.fuzzy_log
    is_fga     = (st.session_state.cfg_algo == "FGA")
    if model is None:
        return

    with main.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE train",  f"{rmse_tr:.2e}")
        c2.metric("RMSE test",   f"{rmse_te:.2e}")
        c3.metric("Нейронов",    str(params["n_hidden"]))
        c4.metric("Активация",   params["activation"])
        c5.metric("Время",       f"{elapsed:.1f} с")

        with st.expander("Все гиперпараметры", expanded=False):
            st.dataframe(pd.DataFrame([{
                "N":          params["n_hidden"],
                "act":        params["activation"],
                "σ":          round(params["scale"],      3),
                "λ_pde":      round(params["lambda_pde"], 4),
                "λ_bc":       round(params["lambda_bc"],  3),
                "RMSE train": f"{rmse_tr:.3e}",
                "RMSE test":  f"{rmse_te:.3e}",
            }]), hide_index=True)

        st.divider()

        # Вкладки зависят от режима
        tabs = ["📊 Графики", "🧪 Train / Test"]
        if is_fga and fuzzy_log:
            tabs.append("🔮 Нечёткий контроллер")
        tab_objs = st.tabs(tabs)

        with tab_objs[0]:
            st.plotly_chart(chart_pressure(model, xx, yy, Xgrid), width="stretch")
            col_v, col_p = st.columns(2)
            with col_v:
                st.plotly_chart(chart_velocity(model, xx, yy, Xgrid), width="stretch")
            with col_p:
                st.plotly_chart(chart_profiles(model), width="stretch")
            if history:
                lbl = "FGA" if is_fga else "GA"
                st.plotly_chart(chart_convergence({lbl: history}), width="stretch")

        with tab_objs[1]:
            st.markdown("### Оценка на тестовой выборке (25%)")
            st.caption("Тестовые точки не использовались в обучении и поиске.")
            ca, cb_, cc, cd = st.columns(4)
            ca.metric("RMSE train",  f"{rmse_tr:.3e}")
            cb_.metric("RMSE test",  f"{rmse_te:.3e}")
            cc.metric("Train точек", len(X_train))
            cd.metric("Test точек",  len(X_test))
            st.plotly_chart(chart_test_scatter(X_train, X_test, model), width="stretch")
            err_tr = np.abs(model.predict(X_train).ravel()
                            - p_exact(X_train[:, 0], X_train[:, 1]))
            err_te = np.abs(model.predict(X_test).ravel()
                            - p_exact(X_test[:, 0],  X_test[:, 1]))
            st.dataframe(pd.DataFrame({
                "Выборка":      ["Train (75%)", "Test (25%)"],
                "Точек":        [len(X_train),  len(X_test)],
                "RMSE":         [f"{rmse_tr:.3e}", f"{rmse_te:.3e}"],
                "Макс ошибка":  [f"{err_tr.max():.3e}", f"{err_te.max():.3e}"],
                "Средн ошибка": [f"{err_tr.mean():.3e}", f"{err_te.mean():.3e}"],
            }), hide_index=True)

        if is_fga and fuzzy_log and len(tab_objs) > 2:
            with tab_objs[2]:
                st.markdown("### Работа нечёткого контроллера мутации")
                st.caption(
                    "**e₁** — разнообразие популяции: (f_ave − f_min) / f_ave.  "
                    "**e₂** — скорость улучшения: (f_ave(t−1) − f_ave(t)) / f_ave(t−1).  "
                    "Оба нормированы в [0, 1]. Контроллер увеличивает мутацию, "
                    "когда популяция однородна или застряла."
                )
                st.plotly_chart(chart_fuzzy_log(fuzzy_log), width="stretch")
                st.dataframe(
                    pd.DataFrame(fuzzy_log).rename(columns={
                        "gen": "Поколение", "e1": "e₁", "e2": "e₂", "p_mut": "p_mut"
                    }).assign(**{"Поколение": lambda d: d["Поколение"] + 1})
                    .style.format({"e₁": "{:.3f}", "e₂": "{:.3f}", "p_mut": "{:.3f}"}),
                    hide_index=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
#  РЕНДЕР: сравнение GA vs FGA
# ─────────────────────────────────────────────────────────────────────────────

def show_compare():
    if not st.session_state.cmp_done:
        return
    ga  = st.session_state.cmp_ga   # (rmse_tr, rmse_te, history, elapsed, bp, model, Xtr, Xte)
    fga = st.session_state.cmp_fga
    fl  = st.session_state.fuzzy_log

    with main.container():
        st.markdown("### Сравнение GA и FGA")

        # Сводная таблица
        df_cmp = pd.DataFrame([
            {"Алгоритм": "GA",  "RMSE train": f"{ga[0]:.3e}",
             "RMSE test": f"{ga[1]:.3e}",  "Время, с": f"{ga[3]:.1f}",
             "Нейронов": ga[4]["n_hidden"], "Активация": ga[4]["activation"]},
            {"Алгоритм": "FGA", "RMSE train": f"{fga[0]:.3e}",
             "RMSE test": f"{fga[1]:.3e}", "Время, с": f"{fga[3]:.1f}",
             "Нейронов": fga[4]["n_hidden"], "Активация": fga[4]["activation"]},
        ])
        st.dataframe(df_cmp, hide_index=True)

        # Победитель
        if ga[1] <= fga[1]:
            winner_lbl, w_rte = "GA", ga[1]
        else:
            winner_lbl, w_rte = "FGA", fga[1]
        st.success(
            f"✅ Победитель по RMSE test: **{winner_lbl}**  ({w_rte:.3e})  "
            f"— модель сохранена, графики доступны ниже."
        )

        st.divider()

        # Графики сравнения
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(chart_ga_fga_compare(ga[2], fga[2]), width="stretch")
        with col_r:
            if fl:
                st.plotly_chart(chart_fuzzy_log(fl), width="stretch")

        # Графики победившей модели
        st.divider()
        st.markdown(f"#### Поле давления — модель **{winner_lbl}**")
        w_model = st.session_state.best_model
        if w_model is not None:
            st.plotly_chart(chart_pressure(w_model, xx, yy, Xgrid), width="stretch")
            col_v, col_p = st.columns(2)
            with col_v:
                st.plotly_chart(chart_velocity(w_model, xx, yy, Xgrid), width="stretch")
            with col_p:
                st.plotly_chart(chart_profiles(w_model), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
#  РЕНДЕР: эксперимент
# ─────────────────────────────────────────────────────────────────────────────

def show_experiment():
    df    = st.session_state.exp_df
    hists = st.session_state.exp_histories
    if df is None:
        return

    bi      = df["RMSE test"].idxmin()
    w_pop   = df.loc[bi, "Популяция"]
    w_gen   = df.loc[bi, "Поколения"]
    w_algo  = df.loc[bi, "Алгоритм"]
    w_rte   = df.loc[bi, "RMSE test"]
    w_n     = df.loc[bi, "Нейронов"]
    w_act   = df.loc[bi, "Активация"]

    with main.container():
        st.success(
            f"✅ Победитель: **{w_algo}**  ·  pop={w_pop}  gen={w_gen}  "
            f"·  RMSE test={w_rte:.3e}  ·  N={w_n}  act={w_act}"
        )

        df_fmt = df.copy()
        df_fmt["RMSE train"] = df_fmt["RMSE train"].map(lambda v: f"{v:.3e}")
        df_fmt["RMSE test"]  = df_fmt["RMSE test"].map(lambda v: f"{v:.3e}")
        st.dataframe(df_fmt, hide_index=True)

        st.divider()
        col_h, col_c = st.columns(2)
        with col_h:
            st.plotly_chart(chart_heatmap(df), width="stretch")
        with col_c:
            if hists:
                st.plotly_chart(chart_convergence(hists), width="stretch")

        # Графики победившей модели
        w_model = st.session_state.best_model
        if w_model is not None:
            st.divider()
            st.markdown(f"#### Поле давления — лучшая модель (pop={w_pop}, gen={w_gen})")
            st.plotly_chart(chart_pressure(w_model, xx, yy, Xgrid), width="stretch")
            col_v, col_p = st.columns(2)
            with col_v:
                st.plotly_chart(chart_velocity(w_model, xx, yy, Xgrid), width="stretch")
            with col_p:
                st.plotly_chart(chart_profiles(w_model), width="stretch")

        st.download_button(
            "⬇ Скачать CSV",
            data=df.to_csv(index=False).encode(),
            file_name="ga_pielm_results.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  ЗАПУСК
# ─────────────────────────────────────────────────────────────────────────────

if btn_single:
    clear_results()
    algo = st.session_state.cfg_algo

    # ── GA vs FGA: запускаем оба последовательно ─────────────────────────────
    if algo == "GA vs FGA":
        st.session_state.mode = "compare"
        results = {}
        flog    = None

        with main.container():
            for label, use_fz in [("GA", False), ("FGA", True)]:
                st.markdown(f"#### ⏳ {label} — оптимизация")
                prog   = st.progress(0)
                status = st.empty()

                def _cb(gen, total, loss, params, _lbl=label):
                    prog.progress(gen / total)
                    status.markdown(
                        f"**{_lbl}** · Поколение **{gen}/{total}**  ·  "
                        f"RMSE `{loss:.3e}`  ·  N={params['n_hidden']}"
                    )

                t0 = time.time()
                m, bp, hist, Xtr, Xte, rtr, rte, fl = run_optimizer(
                    Xf_all, Xb, Yb,
                    n_pop=st.session_state.cfg_n_pop,
                    n_gen=st.session_state.cfg_n_gen,
                    seed=int(st.session_state.cfg_seed),
                    use_fuzzy=use_fz, cb=_cb,
                )
                elapsed = time.time() - t0
                prog.progress(1.0)
                status.success(f"✅ {label} завершён")

                results[label] = (rtr, rte, hist, elapsed, bp, m, Xtr, Xte)
                if use_fz:
                    flog = fl

        st.session_state.cmp_ga    = results["GA"]
        st.session_state.cmp_fga   = results["FGA"]
        st.session_state.fuzzy_log = flog
        st.session_state.cmp_done  = True

        # Победитель по RMSE test → сохраняем как основную модель
        ga_rte  = results["GA"][1]
        fga_rte = results["FGA"][1]
        winner  = results["GA"] if ga_rte <= fga_rte else results["FGA"]
        w_rtr, w_rte, _, w_elapsed, w_bp, w_model, w_Xtr, w_Xte = winner

        st.session_state.best_model  = w_model
        st.session_state.best_params = w_bp
        st.session_state.best_split  = (w_Xtr, w_Xte)
        st.session_state.best_rmse_tr = w_rtr
        st.session_state.best_rmse_te = w_rte
        st.session_state.ga_elapsed   = w_elapsed
        render_sidebar(w_bp, w_rtr, w_rte)
        show_compare()

    # ── Одиночный GA или FGA ─────────────────────────────────────────────────
    else:
        st.session_state.mode = "single"
        use_fz = (algo == "FGA")

        with main.container():
            st.markdown(f"#### ⏳ {algo} — оптимизация гиперпараметров")
            prog   = st.progress(0)
            status = st.empty()
            log_ph = st.empty()
            log    = []

            def _cb(gen, total, loss, params):
                prog.progress(gen / total)
                status.markdown(
                    f"Поколение **{gen}/{total}**  ·  "
                    f"RMSE `{loss:.3e}`  ·  "
                    f"N={params['n_hidden']}  ·  act={params['activation']}"
                )
                log.append(
                    f"[{gen:02d}/{total}]  RMSE={loss:.3e}  "
                    f"N={params['n_hidden']:4d}  {params['activation']:8s}  "
                    f"σ={params['scale']:.2f}  "
                    f"λp={params['lambda_pde']:.3f}  λb={params['lambda_bc']:.2f}"
                )
                log_ph.code("\n".join(log[-8:]))
                render_sidebar(params, rmse_tr=loss)

            t0 = time.time()
            model, best_p, history, X_train, X_test, rmse_tr, rmse_te, fuzzy_log = \
                run_optimizer(
                    Xf_all, Xb, Yb,
                    n_pop=st.session_state.cfg_n_pop,
                    n_gen=st.session_state.cfg_n_gen,
                    seed=int(st.session_state.cfg_seed),
                    use_fuzzy=use_fz, cb=_cb,
                )
            elapsed = time.time() - t0
            prog.progress(1.0)
            status.success("✅ Оптимизация завершена")

        st.session_state.best_model  = model
        st.session_state.best_params = best_p
        st.session_state.best_split  = (X_train, X_test)
        st.session_state.best_rmse_tr= rmse_tr
        st.session_state.best_rmse_te= rmse_te
        st.session_state.ga_history  = history
        st.session_state.ga_elapsed  = elapsed
        st.session_state.fuzzy_log   = fuzzy_log

        render_sidebar(best_p, rmse_tr, rmse_te)
        show_single()

# ─────────────────────────────────────────────────────────────────────────────
#  ЗАПУСК — эксперимент
# ─────────────────────────────────────────────────────────────────────────────

elif btn_exp and st.session_state.cfg_exp_pops and st.session_state.cfg_exp_gens:
    clear_results()
    st.session_state.mode = "experiment"
    algo_lbl = st.session_state.cfg_algo
    is_compare = (algo_lbl == "GA vs FGA")
    use_fz = (algo_lbl == "FGA")

    combos = [(p, g)
              for p in sorted(st.session_state.cfg_exp_pops)
              for g in sorted(st.session_state.cfg_exp_gens)]

    with main.container():
        st.markdown(f"#### ⏳ {algo_lbl} — сравнительный эксперимент")
        prog = st.progress(0)
        info = st.empty()
        tbl  = st.empty()

        rows       = {}
        hists      = {}
        models_map = {}

        for i, (n_p, n_g) in enumerate(combos):
            label = f"pop={n_p} · gen={n_g}"
            info.markdown(f"⏳ **{label}**  ({i+1}/{len(combos)})")
            seed = int(st.session_state.cfg_seed)

            if is_compare:
                # Запускаем GA и FGA раздельно, замеряем время каждого
                results_pair = {}
                for lbl_pair, fz in [("GA", False), ("FGA", True)]:
                    t0_ = time.time()
                    m_, p_, h_, Xtr_, Xte_, rtr_, rte_, _ = run_optimizer(
                        Xf_all, Xb, Yb, n_p, n_g, seed=seed, use_fuzzy=fz)
                    results_pair[lbl_pair] = (m_, p_, h_, Xtr_, Xte_, rtr_, rte_,
                                              time.time() - t0_)

                ga_r  = results_pair["GA"]
                fga_r = results_pair["FGA"]
                if ga_r[6] <= fga_r[6]:
                    winner_lbl, m, p, h, Xtr, Xte, rtr, rte, elapsed = "GA",  *ga_r
                else:
                    winner_lbl, m, p, h, Xtr, Xte, rtr, rte, elapsed = "FGA", *fga_r
                hists[f"GA  pop={n_p} gen={n_g}"]  = ga_r[2]
                hists[f"FGA pop={n_p} gen={n_g}"]  = fga_r[2]
            else:
                t0 = time.time()
                m, p, h, Xtr, Xte, rtr, rte, _ = run_optimizer(
                    Xf_all, Xb, Yb, n_p, n_g, seed=seed, use_fuzzy=use_fz)
                elapsed  = time.time() - t0
                winner_lbl = algo_lbl
                hists[label] = h

            rows[(n_p, n_g)] = {
                "Алгоритм":   winner_lbl,
                "Популяция":  n_p,
                "Поколения":  n_g,
                "Оценок":     n_p * n_g,
                "RMSE train": rtr,
                "RMSE test":  rte,
                "Нейронов":   p["n_hidden"],
                "Активация":  p["activation"],
                "λ_pde":      round(p["lambda_pde"], 4),
                "λ_bc":       round(p["lambda_bc"],  2),
                "Время, с":   round(elapsed, 2),
            }
            models_map[(n_p, n_g)] = (m, Xtr, Xte, rtr, rte, p)
            prog.progress((i+1) / len(combos))

            df_live = pd.DataFrame(list(rows.values())).copy()
            df_live["RMSE train"] = df_live["RMSE train"].map(lambda v: f"{v:.3e}")
            df_live["RMSE test"]  = df_live["RMSE test"].map(lambda v: f"{v:.3e}")
            tbl.dataframe(df_live, hide_index=True)

            best_key = min(rows, key=lambda k: rows[k]["RMSE test"])
            bm = models_map[best_key]
            render_sidebar(bm[5], bm[3], bm[4])

        info.success("✅ Эксперимент завершён")

    # Победитель по RMSE test → сохраняем модель
    best_key = min(rows, key=lambda k: rows[k]["RMSE test"])
    bm = models_map[best_key]
    w_model, w_Xtr, w_Xte, w_rtr, w_rte, w_bp = bm
    st.session_state.best_model   = w_model
    st.session_state.best_params  = w_bp
    st.session_state.best_split   = (w_Xtr, w_Xte)
    st.session_state.best_rmse_tr = w_rtr
    st.session_state.best_rmse_te = w_rte
    st.session_state.exp_df        = pd.DataFrame(list(rows.values()))
    st.session_state.exp_histories = hists
    render_sidebar(w_bp, w_rtr, w_rte)
    show_experiment()

# ─────────────────────────────────────────────────────────────────────────────
#  ПОВТОРНЫЙ РЕНДЕР
# ─────────────────────────────────────────────────────────────────────────────

else:
    mode = st.session_state.mode
    if mode == "single" and st.session_state.best_model is not None:
        show_single()
    elif mode == "compare" and st.session_state.cmp_done:
        show_compare()
    elif mode == "experiment" and st.session_state.exp_df is not None:
        show_experiment()
    else:
        with main.container():
            st.info(
                "Выберите алгоритм (GA / FGA / GA vs FGA) в боковой панели  "
                "и нажмите **▶ Запустить**."
            )