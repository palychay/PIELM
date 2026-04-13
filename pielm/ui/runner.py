import time
import numpy as np
import streamlit as st

from equations.poisson import pde_operator_poisson, rhs_poisson, boundary_conditions_poisson
from equations.piezo   import pde_operator_piezo,   rhs_piezo,   boundary_conditions_piezo

from models.pielm   import PIELM
from models.genetic import GeneticOptimizer

from solvers.fdm import FDMPoisson, FDMPiezo

from utils.collocation import make_collocation_points, train_test_split

from ui.plots import plot_ga_progress


# ─────────────────────────────────────────────────────────────────────────────
# Построение замыканий оператора и источника
# ─────────────────────────────────────────────────────────────────────────────

def build_closures(cfg):
    """
    Возвращает operator(W,b,X,act_name,act_dict) и source_fn(X)
    в зависимости от выбранного уравнения.
    """
    if cfg['is_piezo']:
        kappa = cfg['kappa']
        def operator(W, b, X, act_name, act_dict):
            return pde_operator_piezo(W, b, X, kappa=kappa,
                                      act_name=act_name, act_dict=act_dict)
        def source_fn(X):
            return rhs_piezo(X, cfg['source_func'], cfg['source_params'])
    else:
        def operator(W, b, X, act_name, act_dict):
            return pde_operator_poisson(W, b, X,
                                        act_name=act_name, act_dict=act_dict)
        def source_fn(X):
            return rhs_poisson(X, cfg['source_func'], cfg['source_params'])

    return operator, source_fn


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка данных (точки коллокации и граница)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(cfg):
    """Генерирует коллокационные и граничные точки."""
    X_all = make_collocation_points(
        cfg['n_side'], dim=cfg['dim'],
        T=cfg['T_end'] if cfg['is_piezo'] else 1.0,
        n_t=cfg['n_t'],
    )
    X_train, X_test = train_test_split(X_all, test_size=0.25, seed=42)

    if cfg['is_piezo']:
        X_b, Y_b = boundary_conditions_piezo(
            n_bc=cfg['n_bc'], n_t=20, T=cfg['T_end']
        )
    else:
        X_b, Y_b = boundary_conditions_poisson(n_bc=cfg['n_bc'])

    return X_train, X_test, X_b, Y_b


# ─────────────────────────────────────────────────────────────────────────────
# Обучение PIELM с заданными гиперпараметрами
# ─────────────────────────────────────────────────────────────────────────────

def fit_model(hp, cfg, X_train, X_b, Y_b, operator, source_fn):
    """
    Создаёт и обучает PIELM с гиперпараметрами hp.
    Возвращает (model, t_fit).
    """
    model = PIELM(
        n_hidden   = hp['n_hidden'],
        input_dim  = cfg['dim'],
        scale      = hp['scale'],
        act_name   = hp['activation'],
        lambda_pde = hp['lambda_pde'],
        lambda_bc  = hp['lambda_bc'],
        seed       = 0,
    )
    t0 = time.time()
    model.fit(X_train, X_b, Y_b, operator, source_fn)
    t_fit = time.time() - t0
    return model, t_fit


# ─────────────────────────────────────────────────────────────────────────────
# Запуск GA
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(cfg, X_train, X_b, Y_b, operator, source_fn):
    """
    Запускает генетический алгоритм с live-обновлением в Streamlit.
    Возвращает (best_hp, ga, t_ga).
    """
    gp = cfg['ga_params_cfg']
    n_gen = gp['n_gen']

    st.markdown('<div class="section-header">🧬 Генетическая оптимизация</div>',
                unsafe_allow_html=True)
    progress_bar      = st.progress(0)
    status_text       = st.empty()
    chart_placeholder = st.empty()

    ga_gens  = []
    ga_best  = []
    ga_cur   = []

    def callback(gen, best_rmse, best_params):
        progress_bar.progress(int(gen / n_gen * 100))
        status_text.markdown(
            f'Поколение **{gen}/{n_gen}** — '
            f'RMSE невязки: **{best_rmse:.6f}** — '
            f'n_hidden: **{best_params["n_hidden"]}**, '
            f'activation: **{best_params["activation"]}**'
        )
        ga_gens.append(gen)
        ga_best.append(best_rmse)
        ga_cur.append(best_rmse)
        if gen % max(1, n_gen // 10) == 0:
            png = plot_ga_progress(ga_gens, ga_best, ga_cur)
            chart_placeholder.image(png, use_container_width=True)

    ga = GeneticOptimizer(
        n_pop         = gp['n_pop'],
        n_gen         = n_gen,
        hidden_bounds = gp['hidden_bounds'],
        scale_bounds  = gp['scale_bounds'],
        elite_frac    = gp['elite_frac'],
        mut_prob      = gp['mut_prob'],
    )

    t0 = time.time()
    best_hp = ga.search(
        X_train, X_b, Y_b,
        operator_func = operator,
        source_func   = source_fn,
        input_dim     = cfg['dim'],
        callback      = callback,
    )
    t_ga = time.time() - t0

    progress_bar.progress(100)
    status_text.success(f'✅ GA завершён за {t_ga:.1f} с')

    gens, best_h, gen_h = ga.get_history_arrays()
    png_ga = plot_ga_progress(gens, best_h, gen_h)
    chart_placeholder.image(png_ga, use_container_width=True)

    return best_hp, ga, t_ga, png_ga


# ─────────────────────────────────────────────────────────────────────────────
# МКР
# ─────────────────────────────────────────────────────────────────────────────

def run_fdm(cfg):
    """
    Запускает МКР.
    Возвращает (fdm_solver, t_fdm).
    """
    if cfg['is_piezo']:
        solver = FDMPiezo(
            n_grid = cfg['n_grid_fdm'],
            n_t    = cfg['n_t_fdm'],
            T      = cfg['T_end'],
            kappa  = cfg['kappa'],
        )
    else:
        solver = FDMPoisson(n_grid=cfg['n_grid_fdm'])

    t0 = time.time()
    solver.solve(cfg['source_func'], cfg['source_params'])
    t_fdm = time.time() - t0
    return solver, t_fdm


# ─────────────────────────────────────────────────────────────────────────────
# PDE-невязка для МКР (численное дифференцирование)
# ─────────────────────────────────────────────────────────────────────────────

def fdm_rmse_pde(fdm_solver, X_test, source_fn, cfg):
    """
    Вычисляет RMSE невязки PDE для решения МКР на точках X_test.

    Лапласиан оценивается численно через центральные разности
    по интерполированным значениям МКР:
      ∇²P ≈ (P(x+h)-2P(x)+P(x-h))/h² + (P(y+h)-2P(y)+P(y-h))/h²

    Для пьезопроводности добавляется производная по t.
    """
    h = 1e-4
    is_piezo = cfg['is_piezo']

    P0 = fdm_solver.predict(X_test)

    # ∂²P/∂x²
    Xp = X_test.copy(); Xp[:, 0] += h
    Xm = X_test.copy(); Xm[:, 0] -= h
    d2x = (fdm_solver.predict(Xp) - 2*P0 + fdm_solver.predict(Xm)) / h**2

    # ∂²P/∂y²
    Yp = X_test.copy(); Yp[:, 1] += h
    Ym = X_test.copy(); Ym[:, 1] -= h
    d2y = (fdm_solver.predict(Yp) - 2*P0 + fdm_solver.predict(Ym)) / h**2

    laplacian = d2x + d2y

    if is_piezo:
        kappa = cfg['kappa']
        Tp = X_test.copy(); Tp[:, 2] += h
        Tm = X_test.copy(); Tm[:, 2] -= h
        dPdt     = (fdm_solver.predict(Tp) - fdm_solver.predict(Tm)) / (2*h)
        residual = dPdt - kappa * laplacian - source_fn(X_test)
    else:
        residual = laplacian - source_fn(X_test)

    return float(np.sqrt(np.mean(residual**2)))


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция запуска
# ─────────────────────────────────────────────────────────────────────────────

def run_computation(cfg):
    """
    Полный цикл вычислений. Сохраняет результаты в st.session_state['results'].

    Порядок:
    1. Генерация коллокационных точек
    2. GA (если выбран) → best_hp
    3. Обучение лучшей модели (отдельный замер t_fit)
    4. Метрики PIELM: rmse_pde_train, rmse_pde_test
    5. МКР (если включён)
    6. Метрики МКР: rmse_pde через численное дифференцирование
       Обе метрики rmse_pde сравнимы: √mean((L[P]−f)²) на X_test
    """
    plots = {}

    # ── 1. Данные ──────────────────────────────────────────────────────────
    with st.spinner('Генерация коллокационных точек...'):
        X_train, X_test, X_b, Y_b = prepare_data(cfg)
        operator, source_fn       = build_closures(cfg)

    # ── 2. GA или ручной режим ─────────────────────────────────────────────
    ga_history = None
    t_ga       = None

    if cfg['use_ga']:
        best_hp, ga, t_ga, png_ga = run_ga(
            cfg, X_train, X_b, Y_b, operator, source_fn
        )
        ga_history = ga.history
        plots['ga_progress.png'] = png_ga
    else:
        best_hp = cfg['manual_params']

    # ── 3. Обучение лучшей модели — отдельный замер ────────────────────────
    # Важно: t_fit — это чистое время решения задачи,
    # именно его сравниваем с МКР, а не t_ga
    with st.spinner('Обучение лучшей модели...'):
        best_model, t_fit = fit_model(
            best_hp, cfg, X_train, X_b, Y_b, operator, source_fn
        )

    # ── 4. Метрики PIELM ───────────────────────────────────────────────────
    rmse_pde_train = best_model.rmse_pde(X_train, operator, source_fn)
    rmse_pde_test  = best_model.rmse_pde(X_test,  operator, source_fn)

    pielm_metrics = {
        'rmse_pde_train': rmse_pde_train,
        'rmse_pde_test':  rmse_pde_test,
        't_fit':          t_fit,          # чистое время обучения
        't_ga':           t_ga,           # время GA (None в ручном режиме)
    }

    # ── 5. МКР ─────────────────────────────────────────────────────────────
    fdm_metrics = None
    fdm_solver  = None

    if cfg['fdm_enabled']:
        with st.spinner('Вычисление МКР...'):
            fdm_solver, t_fdm = run_fdm(cfg)

        # ── 6. Метрики МКР ────────────────────────────────────────────────
        # rmse_pde — невязка ∇²P_fdm − f, считается численным дифференц-ем
        # Для сравнения с rmse_pde_test PIELM: обе метрики одного смысла
        fdm_rmse = fdm_rmse_pde(fdm_solver, X_test, source_fn, cfg)

        fdm_metrics = {
            'rmse_pde': fdm_rmse,   # невязка PDE на test
            't_fdm':    t_fdm,
            'n_grid':   cfg['n_grid_fdm'],
        }

    # ── Сохраняем в session_state ──────────────────────────────────────────
    st.session_state['results'] = {
        'model':         best_model,
        'fdm_solver':    fdm_solver,
        'X_train':       X_train,
        'X_test':        X_test,
        'X_b':           X_b,
        'best_hp':       best_hp,
        'pielm_metrics': pielm_metrics,
        'fdm_metrics':   fdm_metrics,
        **cfg,
    }
    st.session_state['ga_history'] = ga_history
    st.session_state['plots']      = plots
    st.session_state['run_complete'] = True