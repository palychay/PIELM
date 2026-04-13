import streamlit as st

from equations.poisson import POISSON_SOURCE_VARIANTS
from equations.piezo   import PIEZO_SOURCE_VARIANTS
from utils.collocation import count_points


def render_sidebar():
    """
    Отрисовывает боковую панель и возвращает словарь с настройками.

    Возвращает
    ----------
    cfg : dict — все параметры выбранные пользователем
    run : bool — нажата ли кнопка запуска
    """
    with st.sidebar:
        st.markdown('## ⚙️ Настройки')

        # ── Блок 1: Уравнение ─────────────────────────────────────────────
        st.markdown('### 📐 Уравнение')
        equation_name = st.selectbox(
            'Уравнение',
            options=[
                'Уравнение Пуассона (стационарное)',
                'Уравнение пьезопроводности (нестационарное)',
            ],
            help=(
                'Пуассон: ∇²P = f(x,y) — стационарная фильтрация\n\n'
                'Пьезопроводность: ∂P/∂t = κ·∇²P + q(x,y,t) — нестационарная'
            ),
        )
        is_piezo = 'пьезо' in equation_name.lower()
        dim      = 3 if is_piezo else 2

        st.markdown('---')

        # ── Блок 2: Источник и параметры уравнения ────────────────────────
        st.markdown('### 📊 Параметры уравнения')

        source_variants = PIEZO_SOURCE_VARIANTS if is_piezo else POISSON_SOURCE_VARIANTS
        source_name     = st.selectbox('Источниковый член',
                                       options=list(source_variants.keys()))
        source_info   = source_variants[source_name]
        source_func   = source_info['func']
        source_params = {}

        st.markdown('**Параметры источника:**')
        for param, default_val in source_info['params'].items():
            lo, hi = source_info['param_ranges'][param]
            source_params[param] = st.slider(
                param,
                min_value=float(lo), max_value=float(hi),
                value=float(default_val), step=(hi - lo) / 100,
            )

        kappa = None
        T_end = None
        if is_piezo:
            st.markdown('**Параметры пьезопроводности:**')
            kappa = st.number_input(
                'κ — коэффициент пьезопроводности',
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help='κ = k/(m·μ)·(1/Kρ + 1/Km)⁻¹  (Леонтьев §12)',
            )
            T_end = st.number_input(
                'T — конечное время',
                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            )

        st.markdown('---')

        # ── Блок 3: Коллокационные точки ──────────────────────────────────
        st.markdown('### 🔢 Коллокационные точки')
        st.caption('Используется равномерная сетка внутренних точек')

        n_side = st.slider(
            'N — точек по каждому направлению',
            min_value=5, max_value=30, value=10,
            help='Итого: N² точек (2D) или N²×N точек (3D)',
        )
        n_t = n_side if is_piezo else None
        n_actual = count_points(n_side, dim=dim, n_t=n_t)
        st.caption(f'Коллокационных точек: **{n_actual}**')

        n_bc = st.slider('Граничных точек на сторону',
                         min_value=10, max_value=60, value=20)

        st.markdown('---')

        # ── Блок 4: Гиперпараметры PIELM ──────────────────────────────────
        st.markdown('### 🧠 Гиперпараметры PIELM')

        mode   = st.radio('Режим настройки',
                          options=['Ручной', 'Автоматический (GA)'],
                          horizontal=True)
        use_ga = mode == 'Автоматический (GA)'

        manual_params = None
        ga_params_cfg = None

        if use_ga:
            st.markdown('**Параметры GA:**')
            n_pop    = st.slider('Размер популяции',     10, 50, 20)
            n_gen    = st.slider('Число поколений',       5, 30, 10)
            elite_f  = st.slider('Доля элиты',          0.1, 0.4, 0.2, step=0.05)
            mut_prob = st.slider('Вероятность мутации', 0.1, 0.5, 0.3, step=0.05)
            st.markdown('**Границы поиска:**')
            h_min, h_max = st.slider('n_hidden', 50, 600, (50, 400))
            s_min, s_max = st.slider('scale',    0.5, 15.0, (0.5, 10.0))
            ga_params_cfg = {
                'n_pop': n_pop, 'n_gen': n_gen,
                'elite_frac': elite_f, 'mut_prob': mut_prob,
                'hidden_bounds': (h_min, h_max),
                'scale_bounds':  (s_min, s_max),
            }
        else:
            st.markdown('**Параметры PIELM:**')
            n_hidden   = st.slider('n_hidden',    50, 600, 200)
            scale      = st.slider('scale',       0.5, 15.0, 5.0, step=0.5)
            activation = st.selectbox('Активация',
                                      options=['tanh', 'sin', 'sigmoid'])
            lam_pde    = st.slider('λ_pde', 0.01, 10.0, 1.0, step=0.01)
            lam_bc     = st.slider('λ_bc',  1.0, 100.0, 10.0, step=1.0)
            manual_params = {
                'n_hidden': n_hidden, 'scale': scale,
                'activation': activation,
                'lambda_pde': lam_pde, 'lambda_bc': lam_bc,
            }

        st.markdown('---')

        # ── Блок 5: МКР ───────────────────────────────────────────────────
        st.markdown('### ⚖️ Метод конечных разностей')
        fdm_enabled = st.toggle('Включить МКР для сравнения', value=True)

        n_grid_fdm = None
        n_t_fdm    = None
        if fdm_enabled:
            n_grid_fdm = st.slider('Узлов МКР (n_grid)', 10, 100, 50,
                                   help='Полная сетка: (n_grid+2)²')
            if is_piezo:
                n_t_fdm = st.slider('Временных шагов МКР', 20, 200, 50)

        st.markdown('---')

        run = st.button('▶ Запустить вычисление',
                        type='primary', use_container_width=True)
        st.caption('💾 Экспорт доступен после вычисления')

    cfg = dict(
        equation_name = equation_name,
        is_piezo      = is_piezo,
        dim           = dim,
        source_name   = source_name,
        source_func   = source_func,
        source_params = source_params,
        kappa         = kappa,
        T_end         = T_end,
        n_side        = n_side,
        n_t           = n_t,
        n_bc          = n_bc,
        use_ga        = use_ga,
        manual_params = manual_params,
        ga_params_cfg = ga_params_cfg,
        fdm_enabled   = fdm_enabled,
        n_grid_fdm    = n_grid_fdm,
        n_t_fdm       = n_t_fdm,
    )
    return cfg, run