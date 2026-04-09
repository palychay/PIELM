import time
import numpy as np
import streamlit as st

from equations.piezo import (
    pde_operator_piezo, rhs_piezo,
    boundary_conditions_piezo,
    PIEZO_SOURCE_VARIANTS,
)


from equations.poisson import (
    pde_operator_poisson, rhs_poisson,
    boundary_conditions_poisson,
    POISSON_SOURCE_VARIANTS,
)

from models.pielm import PIELM, ACTIVATIONS
from models.genetic import GeneticOptimizer

from solvers.fdm import FDMPoisson, FDMPiezo

from utils.collocation import (
    make_collocation_points,
    train_test_split,
    get_grid_for_plot,
    count_points,
    SAMPLING_STRATEGIES_2D,
    SAMPLING_STRATEGIES_3D,
)

from ui.plots import (
    plot_ga_progress,
    plot_pielm_solution,
    plot_pielm_vs_fdm,
    plot_piezo_time_slice,
    plot_piezo_evolution,
    plot_comparison_bar,
    plot_collocation_points,
    plot_source_field,
)

from ui.export import (
    build_report_txt,
    report_to_bytes,
    build_plots_zip,
    make_filename,
)


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PIELM + GA — Решение уравнений фильтрации',
    page_icon='💧',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4a6fa5;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 0.3rem;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8faff;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Заголовок
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">💧 PIELM + Генетический алгоритм</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Физически информированное экстремальное машинное обучение '
    'для решения уравнений фильтрации'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Инициализация session_state
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    'results':      None,
    'ga_history':   None,
    'plots':        {},
    'run_complete': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('## ⚙️ Настройки')

    # ── Блок 1: Выбор уравнения ───────────────────────────────────────────
    st.markdown('### 📐 Уравнение')
    equation_name = st.selectbox(
        'Уравнение',
        options=[
            'Уравнение Пуассона (стационарное)',
            'Уравнение пьезопроводности (нестационарное)',
        ],
        help=(
            'Пуассон: ∇²P = f(x,y) — стационарная фильтрация\n\n'
            'Пьезопроводность: ∂P/∂t = κ·∇²P + q — нестационарная фильтрация'
        ),
    )
    is_piezo = 'пьезо' in equation_name.lower()
    dim      = 3 if is_piezo else 2

    st.markdown('---')

    # ── Блок 2: Параметры уравнения ───────────────────────────────────────
    st.markdown('### 📊 Параметры уравнения')

    source_variants = PIEZO_SOURCE_VARIANTS if is_piezo else POISSON_SOURCE_VARIANTS
    source_name     = st.selectbox('Источниковый член f(x,y)',
                                   options=list(source_variants.keys()))
    source_info     = source_variants[source_name]
    source_func     = source_info['func']
    source_params   = {}

    st.markdown('**Параметры источника:**')
    for param, default_val in source_info['params'].items():
        lo, hi = source_info['param_ranges'][param]
        source_params[param] = st.slider(
            param, min_value=float(lo), max_value=float(hi),
            value=float(default_val), step=(hi - lo) / 100,
        )

    if is_piezo:
        st.markdown('**Параметры пьезопроводности:**')
        kappa = st.number_input(
            'κ — коэффициент пьезопроводности',
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help='κ = k/(m·μ)·(1/Kρ + 1/Km)⁻¹  (Леонтьев §12, формула 12.2)',
        )
        T_end = st.number_input(
            'T — конечное время',
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        )
    else:
        kappa = None
        T_end = None

    st.markdown('---')

    # ── Блок 3: Коллокационные точки ─────────────────────────────────────
    st.markdown('### 🔢 Коллокационные точки')

    strategies = (SAMPLING_STRATEGIES_3D if is_piezo
                  else SAMPLING_STRATEGIES_2D)
    sampling   = st.selectbox('Стратегия размещения',
                              options=list(strategies.keys()))

    n_colloc = st.slider(
        'Число точек N (для сетки — N×N)',
        min_value=5, max_value=50, value=15,
        help='Для сетки: N² внутренних точек. Для случайной: N точек.',
    )
    n_actual = count_points(sampling, n_colloc, dim=dim)
    st.caption(f'Фактическое число точек: **{n_actual}**')

    n_bc = st.slider('Граничных точек на сторону',
                     min_value=10, max_value=60, value=30)

    st.markdown('---')

    # ── Блок 4: Гиперпараметры PIELM ─────────────────────────────────────
    st.markdown('### 🧠 Гиперпараметры PIELM')

    mode   = st.radio('Режим настройки',
                      options=['Ручной', 'Автоматический (GA)'],
                      horizontal=True)
    use_ga = mode == 'Автоматический (GA)'

    if use_ga:
        st.markdown('**Параметры генетического алгоритма:**')
        n_pop    = st.slider('Размер популяции',     10, 50, 20)
        n_gen    = st.slider('Число поколений',       5, 30, 10)
        elite_f  = st.slider('Доля элиты',          0.1, 0.4, 0.2, step=0.05)
        mut_prob = st.slider('Вероятность мутации', 0.1, 0.5, 0.3, step=0.05)

        st.markdown('**Границы поиска:**')
        h_min, h_max = st.slider('n_hidden (диапазон)', 50, 600, (50, 400))
        s_min, s_max = st.slider('scale (диапазон)',    0.5, 15.0, (0.5, 10.0))

        manual_params = None
        ga_params_cfg = {
            'n_pop':         n_pop,
            'n_gen':         n_gen,
            'elite_frac':    elite_f,
            'mut_prob':      mut_prob,
            'hidden_bounds': (h_min, h_max),
            'scale_bounds':  (s_min, s_max),
        }
    else:
        st.markdown('**Параметры PIELM:**')
        n_hidden   = st.slider('n_hidden',    50, 600, 200)
        scale      = st.slider('scale',       0.5, 15.0, 5.0, step=0.5)
        activation = st.selectbox('Функция активации',
                                  options=['tanh', 'sin', 'sigmoid'])
        lam_pde    = st.slider('λ_pde', 0.01, 10.0, 1.0, step=0.01)
        lam_bc     = st.slider('λ_bc',  1.0, 100.0, 10.0, step=1.0)

        manual_params = {
            'n_hidden':   n_hidden,
            'scale':      scale,
            'activation': activation,
            'lambda_pde': lam_pde,
            'lambda_bc':  lam_bc,
        }
        ga_params_cfg = None

    st.markdown('---')

    # ── Блок 5: МКР ───────────────────────────────────────────────────────
    st.markdown('### ⚖️ Метод конечных разностей')
    fdm_enabled = st.toggle('Включить МКР для сравнения', value=True)

    if fdm_enabled:
        n_grid_fdm = st.slider('Узлов сетки МКР (n_grid)',
                               10, 100, 50,
                               help='Полная сетка: (n_grid+2)²')
        if is_piezo:
            n_t_fdm = st.slider('Временных шагов МКР', 20, 200, 50)
        else:
            n_t_fdm = None

    st.markdown('---')

    # ── Кнопка запуска ────────────────────────────────────────────────────
    run_btn = st.button('▶ Запустить вычисление',
                        type='primary', use_container_width=True)

    st.markdown('---')
    st.caption('💾 Экспорт доступен после вычисления')


# ─────────────────────────────────────────────────────────────────────────────
# ОСНОВНАЯ ОБЛАСТЬ — ВЫЧИСЛЕНИЕ
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.session_state['run_complete'] = False
    st.session_state['plots']        = {}

    with st.spinner('Подготовка данных...'):

        # ── Генерация коллокационных точек ───────────────────────────────
        X_all = make_collocation_points(
            sampling, n_colloc, dim=dim,
            T=T_end if is_piezo else 1.0,
            seed=42,
        )
        X_train, X_test = train_test_split(X_all, test_size=0.25, seed=42)

        if is_piezo:
            X_b, Y_b = boundary_conditions_piezo(n_bc=n_bc, n_t=20, T=T_end)
        else:
            X_b, Y_b = boundary_conditions_poisson(n_bc=n_bc)

        # ── Замыкания оператора и источника ──────────────────────────────
        if is_piezo:
            def operator(W, b, X, act_name, act_dict):
                return pde_operator_piezo(
                    W, b, X, kappa=kappa,
                    act_name=act_name, act_dict=act_dict,
                )
            def source_fn(X):
                return rhs_piezo(X, source_func, source_params)
        else:
            def operator(W, b, X, act_name, act_dict):
                return pde_operator_poisson(
                    W, b, X,
                    act_name=act_name, act_dict=act_dict,
                )
            def source_fn(X):
                return rhs_poisson(X, source_func, source_params)

    # ── Режим GA ─────────────────────────────────────────────────────────
    if use_ga:
        st.markdown('<div class="section-header">🧬 Генетическая оптимизация</div>',
                    unsafe_allow_html=True)

        progress_bar      = st.progress(0)
        status_text       = st.empty()
        chart_placeholder = st.empty()

        ga_gen_list  = []
        ga_best_list = []
        ga_cur_list  = []

        def ga_callback(gen, best_rmse, best_params):
            progress_bar.progress(int(gen / n_gen * 100))
            status_text.markdown(
                f'Поколение **{gen}/{n_gen}** — '
                f'RMSE невязки: **{best_rmse:.6f}** — '
                f'n_hidden: **{best_params["n_hidden"]}**, '
                f'activation: **{best_params["activation"]}**'
            )
            ga_gen_list.append(gen)
            ga_best_list.append(best_rmse)
            ga_cur_list.append(best_rmse)
            if gen % max(1, n_gen // 10) == 0:
                png = plot_ga_progress(ga_gen_list, ga_best_list, ga_cur_list)
                chart_placeholder.image(png, use_container_width=True)

        ga = GeneticOptimizer(
            n_pop         = ga_params_cfg['n_pop'],
            n_gen         = ga_params_cfg['n_gen'],
            hidden_bounds = ga_params_cfg['hidden_bounds'],
            scale_bounds  = ga_params_cfg['scale_bounds'],
            elite_frac    = ga_params_cfg['elite_frac'],
            mut_prob      = ga_params_cfg['mut_prob'],
        )

        t0 = time.time()
        best_hp = ga.search(
            X_train, X_b, Y_b,
            operator_func = operator,
            source_func   = source_fn,
            input_dim     = dim,
            callback      = ga_callback,
        )
        t_ga = time.time() - t0

        progress_bar.progress(100)
        status_text.success(f'✅ GA завершён за {t_ga:.1f} с')

        gens, best_rmse_hist, gen_rmse_hist = ga.get_history_arrays()
        png_ga = plot_ga_progress(gens, best_rmse_hist, gen_rmse_hist)
        chart_placeholder.image(png_ga, use_container_width=True)
        st.session_state['plots']['ga_progress.png'] = png_ga
        st.session_state['ga_history'] = ga.history

        # Переобучаем лучшую модель отдельно — замеряем чистое время fit
        with st.spinner('Обучение лучшей модели...'):
            best_model = PIELM(
                n_hidden   = best_hp['n_hidden'],
                input_dim  = dim,
                scale      = best_hp['scale'],
                act_name   = best_hp['activation'],
                lambda_pde = best_hp['lambda_pde'],
                lambda_bc  = best_hp['lambda_bc'],
                seed       = 0,
            )
            t0 = time.time()
            best_model.fit(X_train, X_b, Y_b, operator, source_fn)
            t_fit = time.time() - t0
        t_pielm = t_ga + t_fit

    # ── Ручной режим ─────────────────────────────────────────────────────
    else:
        best_hp = manual_params
        st.session_state['ga_history'] = None

        with st.spinner('Обучение PIELM...'):
            best_model = PIELM(
                n_hidden   = manual_params['n_hidden'],
                input_dim  = dim,
                scale      = manual_params['scale'],
                act_name   = manual_params['activation'],
                lambda_pde = manual_params['lambda_pde'],
                lambda_bc  = manual_params['lambda_bc'],
                seed       = 0,
            )
            t0 = time.time()
            best_model.fit(X_train, X_b, Y_b, operator, source_fn)
            t_pielm = time.time() - t0
        t_ga  = None
        t_fit = t_pielm

    # ── Метрики PIELM ────────────────────────────────────────────────────
    # rmse_pde_train — невязка L[P]−f на точках построения МНК (X_train)
    # rmse_pde_test  — невязка L[P]−f на точках вне МНК (X_test)
    # Разница показывает равномерность аппроксимации по области Ω,
    # а не переобучение — в PIELM β находится из линейной задачи МНК,
    # классического переобучения здесь нет
    rmse_pde_train = best_model.rmse_pde(X_train, operator, source_fn)
    rmse_pde_test  = best_model.rmse_pde(X_test,  operator, source_fn)

    # t_ga   — время работы генетического алгоритма (только GA-режим)
    # t_fit  — время обучения лучшей модели (МНК, fit)
    # t_pielm — суммарное время (GA + fit) или просто fit в ручном режиме
    pielm_metrics = {
        'rmse_pde_train': rmse_pde_train,
        'rmse_pde_test':  rmse_pde_test,
        'time':           t_pielm,
        't_ga':           t_ga   if use_ga else None,
        't_fit':          t_fit  if use_ga else t_pielm,
    }

    # ── МКР ──────────────────────────────────────────────────────────────
    fdm_metrics = None
    fdm_solver  = None

    if fdm_enabled:
        with st.spinner('Вычисление МКР...'):
            if is_piezo:
                fdm_solver = FDMPiezo(
                    n_grid=n_grid_fdm, n_t=n_t_fdm,
                    T=T_end, kappa=kappa,
                )
                t0 = time.time()
                fdm_solver.solve(source_func, source_params)
                t_fdm = time.time() - t0
            else:
                fdm_solver = FDMPoisson(n_grid=n_grid_fdm)
                t0 = time.time()
                fdm_solver.solve(source_func, source_params)
                t_fdm = time.time() - t0

            # Расхождение решений PIELM и МКР на тестовых точках
            P_pielm_test      = best_model.predict(X_test)
            P_fdm_test        = fdm_solver.predict(X_test)
            rmse_pielm_vs_fdm = float(np.sqrt(
                np.mean((P_pielm_test - P_fdm_test) ** 2)
            ))
            fdm_metrics = {
                'rmse':   rmse_pielm_vs_fdm,
                'time':   t_fdm,
                'n_grid': n_grid_fdm,
            }

    # ── Сохраняем в session_state ─────────────────────────────────────────
    st.session_state['results'] = {
        'model':         best_model,
        'fdm_solver':    fdm_solver,
        'X_train':       X_train,
        'X_test':        X_test,
        'X_b':           X_b,
        'Y_b':           Y_b,
        'best_hp':       best_hp,
        'pielm_metrics': pielm_metrics,
        'fdm_metrics':   fdm_metrics,
        'fdm_enabled':   fdm_enabled,
        'is_piezo':      is_piezo,
        'dim':           dim,
        'kappa':         kappa,
        'T_end':         T_end,
        'source_func':   source_func,
        'source_params': source_params,
        'source_name':   source_name,
        'equation_name': equation_name,
        'sampling':      sampling,
        'n_colloc':      n_colloc,
        'n_bc':          n_bc,
    }
    st.session_state['run_complete'] = True


# ─────────────────────────────────────────────────────────────────────────────
# ВЫВОД РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state['run_complete'] and st.session_state['results'] is not None:
    res = st.session_state['results']

    model       = res['model']
    fdm_solver  = res['fdm_solver']
    is_piezo    = res['is_piezo']
    dim         = res['dim']
    kappa       = res['kappa']
    T_end       = res['T_end']
    source_func = res['source_func']
    src_params  = res['source_params']
    fdm_enabled = res['fdm_enabled']

    pm = res['pielm_metrics']
    fm = res['fdm_metrics']

    tabs = st.tabs([
        '📈 Прогресс GA',
        '🔵 Коллокационные точки',
        '🌊 Поле источника',
        '🗺️ Результаты PIELM',
        '⚖️ Сравнение методов',
        '💾 Экспорт',
    ])

    # ── Вкладка 1: Прогресс GA ────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">Прогресс генетической оптимизации</div>',
                    unsafe_allow_html=True)
        if st.session_state['ga_history'] is not None:
            hist = st.session_state['ga_history']
            gens = [h['generation'] for h in hist]
            best = [h['best_rmse']  for h in hist]
            cur  = [h['gen_rmse']   for h in hist]
            png  = plot_ga_progress(gens, best, cur)
            st.image(png, use_container_width=True)
            st.session_state['plots']['ga_progress.png'] = png

            import pandas as pd
            st.markdown('**История поколений:**')
            df = pd.DataFrame(hist)[['generation', 'best_rmse', 'gen_rmse']]
            df.columns = ['Поколение',
                          'Лучший RMSE невязки (глоб.)',
                          'RMSE невязки поколения']
            st.dataframe(df.style.format({
                'Лучший RMSE невязки (глоб.)': '{:.8f}',
                'RMSE невязки поколения':      '{:.8f}',
            }), use_container_width=True)
        else:
            st.info('GA не использовался. Переключитесь в режим '
                    '"Автоматический (GA)" для просмотра прогресса.')

    # ── Вкладка 2: Коллокационные точки ───────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header">Распределение коллокационных точек</div>',
                    unsafe_allow_html=True)
        png = plot_collocation_points(
            res['X_train'], res['X_test'], res['X_b'], dim=dim
        )
        st.image(png, use_container_width=False)
        st.session_state['plots']['collocation_points.png'] = png

        col1, col2, col3 = st.columns(3)
        col1.metric('Train',    len(res['X_train']))
        col2.metric('Test',     len(res['X_test']))
        col3.metric('Граница',  len(res['X_b']))

    # ── Вкладка 3: Поле источника ─────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">Поле источникового члена</div>',
                    unsafe_allow_html=True)

        if is_piezo:
            t_src = st.slider(
                'Момент времени t для f(x,y,t)',
                0.0, float(T_end), float(T_end) / 2,
                step=float(T_end) / 20,
                key='src_t_slider',
            )
            X_plot, xx, yy = get_grid_for_plot(n=80, dim=3,
                                               T=T_end, t_slice=t_src)
            F_vals = rhs_piezo(X_plot, source_func, src_params)
        else:
            X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
            F_vals = rhs_poisson(X_plot, source_func, src_params)

        png = plot_source_field(F_vals, xx, yy,
                                title=f'Источник: {res["source_name"]}')
        st.image(png, use_container_width=False)
        st.session_state['plots']['source_field.png'] = png

    # ── Вкладка 4: Результаты PIELM ───────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header">Решение PIELM</div>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric(
            'RMSE невязки PDE (train)',
            f'{pm["rmse_pde_train"]:.6f}',
            help=(
                'Невязка уравнения на точках построения МНК:\n'
                '√mean((L[P] − f)²) на X_train'
            ),
        )
        col2.metric(
            'RMSE невязки PDE (test)',
            f'{pm["rmse_pde_test"]:.6f}',
            help=(
                'Невязка уравнения на точках вне МНК:\n'
                '√mean((L[P] − f)²) на X_test.\n'
                'Показывает равномерность аппроксимации по области.'
            ),
        )
        col3.metric('Суммарное время', f'{pm["time"]:.3f} с',
                    help='GA + обучение лучшей модели (или только обучение в ручном режиме)')

        if use_ga and pm['t_ga'] is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric('Время GA (оптимизация)',      f'{pm["t_ga"]:.3f} с',
                      help='Время работы генетического алгоритма')
            c2.metric('Время обучения лучшей модели', f'{pm["t_fit"]:.3f} с',
                      help='Чистое время решения МНК для лучших гиперпараметров')
            c3.metric('Итого (GA + обучение)',         f'{pm["time"]:.3f} с')

        import pandas as pd
        st.markdown('**Лучшие гиперпараметры:**')
        st.dataframe(
            pd.DataFrame([res['best_hp']]).T.rename(columns={0: 'Значение'}),
            use_container_width=False,
        )

        if is_piezo:
            t_plot = st.slider(
                'Момент времени t для визуализации',
                0.0, float(T_end), float(T_end) / 2,
                step=float(T_end) / 20,
                key='result_t_slider',
            )
            X_plot, xx, yy = get_grid_for_plot(n=80, dim=3,
                                               T=T_end, t_slice=t_plot)
            P_pred = model.predict(X_plot)
            png = plot_pielm_solution(
                P_pred, xx, yy,
                title=f'PIELM: P(x,y,t={t_plot:.2f})',
            )
        else:
            X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
            P_pred = model.predict(X_plot)
            png = plot_pielm_solution(P_pred, xx, yy)

        st.image(png, use_container_width=False)
        st.session_state['plots']['pielm_solution.png'] = png

        if is_piezo:
            st.markdown('**Эволюция поля давления:**')
            t_snaps = np.linspace(0.0, T_end, 4)[1:]
            snaps   = []
            for ts in t_snaps:
                Xp, xx2, yy2 = get_grid_for_plot(n=60, dim=3,
                                                  T=T_end, t_slice=ts)
                snaps.append(model.predict(Xp))
            png_ev = plot_piezo_evolution(snaps, xx2, yy2, list(t_snaps))
            st.image(png_ev, use_container_width=True)
            st.session_state['plots']['piezo_evolution.png'] = png_ev
        else:
            st.info('Эволюция доступна только для уравнения пьезопроводности.')

    # ── Вкладка 5: Сравнение методов ──────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">Сравнение PIELM и МКР</div>',
                    unsafe_allow_html=True)

        if not fdm_enabled:
            st.info('МКР отключён. Включите в боковой панели.')
        else:
            col1, col2 = st.columns(2)
            col1.metric(
                'RMSE невязки PDE (PIELM, test)',
                f'{pm["rmse_pde_test"]:.6f}',
                help='√mean((L[P] − f)²) на тестовых точках',
            )
            col2.metric(
                'RMSE(PIELM vs МКР)',
                f'{fm["rmse"]:.6f}',
                help=(
                    'Расхождение решений на тестовых точках:\n'
                    '√mean((P_PIELM − P_МКР)²)\n'
                    'Показывает согласованность методов.'
                ),
            )

            # Временные метрики
            st.markdown('**Время вычисления:**')
            if use_ga and pm['t_ga'] is not None:
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric('GA (оптимизация)',    f'{pm["t_ga"]:.3f} с',
                           help='Время работы генетического алгоритма')
                tc2.metric('Обучение лучшей модели', f'{pm["t_fit"]:.3f} с',
                           help='Чистое время решения МНК')
                tc3.metric('PIELM итого',         f'{pm["time"]:.3f} с',
                           help='GA + обучение лучшей модели')
                tc4.metric('МКР',                 f'{fm["time"]:.3f} с',
                           help='Время решения методом конечных разностей')
            else:
                tc1, tc2 = st.columns(2)
                tc1.metric('Обучение PIELM', f'{pm["time"]:.3f} с')
                tc2.metric('МКР',            f'{fm["time"]:.3f} с')

            png_bar = plot_comparison_bar({
                'PIELM': {'rmse': pm['rmse_pde_test'], 'time': pm['time']},
                'МКР':   {'rmse': fm['rmse'],          'time': fm['time']},
            })
            st.image(png_bar, use_container_width=True)
            st.session_state['plots']['comparison_bar.png'] = png_bar

            st.markdown('**Поля давления: PIELM vs МКР**')
            if is_piezo:
                t_cmp = st.slider(
                    'Момент времени t для сравнения',
                    0.0, float(T_end), float(T_end) / 2,
                    step=float(T_end) / 20,
                    key='cmp_t_slider',
                )
                X_plot, xx, yy = get_grid_for_plot(n=80, dim=3,
                                                   T=T_end, t_slice=t_cmp)
                P_pielm_plot = model.predict(X_plot)
                P_fdm_plot   = fdm_solver.predict(X_plot)
                png_cmp = plot_piezo_time_slice(
                    P_pielm_plot, P_fdm_plot, xx, yy, t_cmp
                )
            else:
                X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
                P_pielm_plot = model.predict(X_plot)
                P_fdm_plot   = fdm_solver.predict(X_plot)
                png_cmp = plot_pielm_vs_fdm(P_pielm_plot, P_fdm_plot, xx, yy)

            st.image(png_cmp, use_container_width=True)
            st.session_state['plots']['comparison_fields.png'] = png_cmp

            import pandas as pd
            st.markdown('**Сводная таблица:**')

            # Формируем строки времени в зависимости от режима
            if use_ga and pm['t_ga'] is not None:
                pielm_time_str = (
                    f'GA: {pm["t_ga"]:.4f} с\n'
                    f'fit: {pm["t_fit"]:.4f} с\n'
                    f'итого: {pm["time"]:.4f} с'
                )
            else:
                pielm_time_str = f'{pm["time"]:.4f} с'

            table_data = {
                'Метод': ['PIELM + GA', 'МКР'],
                'RMSE невязки PDE (test)': [
                    f'{pm["rmse_pde_test"]:.8f}', '—',
                ],
                'RMSE(PIELM vs МКР)': [
                    f'{fm["rmse"]:.8f}', f'{fm["rmse"]:.8f}',
                ],
                'Время (с)': [
                    pielm_time_str, f'{fm["time"]:.4f} с',
                ],
                'Узлов / точек': [
                    f'{len(res["X_train"])} коллокац.',
                    f'{fm["n_grid"]}² = {fm["n_grid"] ** 2} узлов',
                ],
            }
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            st.info(
                '**Пояснение метрик:**\n\n'
                '— **RMSE невязки PDE** — √mean((L[P] − f)²) — '
                'насколько точно нейросеть удовлетворяет уравнению. '
                'Доступна только для PIELM.\n\n'
                '— **RMSE(PIELM vs МКР)** — √mean((P_PIELM − P_МКР)²) — '
                'расхождение двух решений на тестовых точках. '
                'Показывает согласованность методов между собой.'
            )

    # ── Вкладка 6: Экспорт ────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-header">💾 Экспорт результатов</div>',
                    unsafe_allow_html=True)

        eq_params = {}
        if is_piezo:
            eq_params['κ (пьезопроводность)'] = kappa
            eq_params['T (конечное время)']    = T_end

        txt_report = build_report_txt(
            equation_name      = res['equation_name'],
            equation_params    = eq_params,
            source_name        = res['source_name'],
            source_params      = src_params,
            collocation_params = {
                'Стратегия':                res['sampling'],
                'N (по оси)':               res['n_colloc'],
                'N граничных (на сторону)': res['n_bc'],
                'N train':                  len(res['X_train']),
                'N test':                   len(res['X_test']),
            },
            pielm_params       = {} if use_ga else manual_params,
            ga_params          = ga_params_cfg or {},
            ga_history         = st.session_state['ga_history'],
            best_hyperparams   = res['best_hp'],
            pielm_metrics      = {
                'rmse_pde':   pm['rmse_pde_train'],
                'rmse_train': pm['rmse_pde_train'],
                'rmse_test':  pm['rmse_pde_test'],
                'time':       pm['time'],
            },
            fdm_metrics        = fm,
            fdm_enabled        = fdm_enabled,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**📄 Текстовый отчёт**')
            st.download_button(
                label             = '⬇️ Скачать отчёт (.txt)',
                data              = report_to_bytes(txt_report),
                file_name         = make_filename('report', 'txt'),
                mime              = 'text/plain',
                use_container_width=True,
            )
            with st.expander('Предпросмотр отчёта'):
                st.text(txt_report)

        with col2:
            st.markdown('**🖼️ Графики (ZIP)**')
            if st.session_state['plots']:
                zip_bytes = build_plots_zip(st.session_state['plots'])
                st.download_button(
                    label             = '⬇️ Скачать все графики (.zip)',
                    data              = zip_bytes,
                    file_name         = make_filename('plots', 'zip'),
                    mime              = 'application/zip',
                    use_container_width=True,
                )
                st.caption(
                    f'В архиве: {len(st.session_state["plots"])} графиков — '
                    + ', '.join(st.session_state['plots'].keys())
                )
            else:
                st.info('Сначала просмотрите вкладки с графиками.')


# ─────────────────────────────────────────────────────────────────────────────
# Заглушка до первого запуска
# ─────────────────────────────────────────────────────────────────────────────

elif not st.session_state['run_complete']:
    st.info(
        '👈 Настройте параметры в боковой панели и нажмите '
        '**▶ Запустить вычисление**'
    )
    with st.expander('ℹ️ О приложении'):
        st.markdown("""
**PIELM** (Physics-Informed Extreme Learning Machine) — метод решения уравнений
математической физики с помощью нейронных сетей с фиксированными случайными весами.

**Поддерживаемые уравнения:**

| Уравнение | Вывод | Тип |
|-----------|-------|-----|
| Пуассона: ∇²P = f(x,y) | Закон Дарси + div u = 0 | Стационарное |
| Пьезопроводности: ∂P/∂t = κ·∇²P + q | Закон Дарси + сжимаемость | Нестационарное |

**Генетический алгоритм** автоматически подбирает гиперпараметры PIELM:
число нейронов, масштаб инициализации, функцию активации, веса λ_pde и λ_bc.

**МКР** (метод конечных разностей) используется как эталон для сравнения:
- Пуассон: центральные разности, прямое решение СЛАУ
- Пьезопроводность: схема Кранка–Николсона

**Метрики:**
- **RMSE невязки PDE (train/test)** — √mean((L[P] − f)²) — насколько
  точно нейросеть удовлетворяет уравнению в точках построения МНК
  и в остальной части области
- **RMSE(PIELM vs МКР)** — √mean((P_PIELM − P_МКР)²) — расхождение
  двух решений, показывает их согласованность
        """)