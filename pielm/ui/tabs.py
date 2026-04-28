import numpy as np
import pandas as pd
import streamlit as st

from equations.poisson import rhs_poisson
from equations.piezo   import rhs_piezo

from utils.collocation import get_grid_for_plot

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
from models.model_io import (
    model_to_json_bytes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 1: Прогресс GA
# ─────────────────────────────────────────────────────────────────────────────

def tab_ga_progress():
    st.markdown('<div class="section-header">Прогресс генетической оптимизации</div>',
                unsafe_allow_html=True)
    hist = st.session_state.get('ga_history')

    if hist is None:
        st.info('GA не использовался. Выберите режим "Автоматический (GA)".')
        return

    gens = [h['generation'] for h in hist]
    best = [h['best_rmse']  for h in hist]
    cur  = [h['gen_rmse']   for h in hist]
    png  = plot_ga_progress(gens, best, cur)
    st.image(png, use_container_width=True)
    st.session_state['plots']['ga_progress.png'] = png

    df = pd.DataFrame(hist)[['generation', 'best_rmse', 'gen_rmse']]
    df.columns = ['Поколение', 'Лучший RMSE (глоб.)', 'RMSE поколения']
    st.dataframe(df.style.format({
        'Лучший RMSE (глоб.)': '{:.8f}',
        'RMSE поколения':      '{:.8f}',
    }), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 2: Коллокационные точки
# ─────────────────────────────────────────────────────────────────────────────

def tab_collocation(res):
    st.markdown('<div class="section-header">Распределение коллокационных точек</div>',
                unsafe_allow_html=True)
    png = plot_collocation_points(
        res['X_train'], res['X_test'], res['X_b'], dim=res['dim']
    )
    st.image(png, use_container_width=False)
    st.session_state['plots']['collocation_points.png'] = png

    c1, c2, c3 = st.columns(3)
    c1.metric('Train',   len(res['X_train']))
    c2.metric('Test',    len(res['X_test']))
    c3.metric('Граница', len(res['X_b']))


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 3: Поле источника
# ─────────────────────────────────────────────────────────────────────────────

def tab_source(res):
    st.markdown('<div class="section-header">Поле источникового члена</div>',
                unsafe_allow_html=True)

    is_piezo    = res['is_piezo']
    source_func = res['source_func']
    src_params  = res['source_params']

    if is_piezo:
        T_end = res['T_end']
        t_src = st.slider('Момент времени t', 0.0, float(T_end),
                          float(T_end) / 2, step=float(T_end) / 20,
                          key='src_t_slider')
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=3, T=T_end, t_slice=t_src)
        F_vals = rhs_piezo(X_plot, source_func, src_params)
    else:
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
        F_vals = rhs_poisson(X_plot, source_func, src_params)

    png = plot_source_field(F_vals, xx, yy,
                            title=f'Источник: {res["source_name"]}')
    st.image(png, use_container_width=False)
    st.session_state['plots']['source_field.png'] = png


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 4: Результаты PIELM
# ─────────────────────────────────────────────────────────────────────────────

def tab_pielm(res):
    st.markdown('<div class="section-header">Решение PIELM</div>',
                unsafe_allow_html=True)

    pm      = res['pielm_metrics']
    model   = res['model']
    is_piezo = res['is_piezo']

    # Метрики
    c1, c2, c3 = st.columns(3)
    c1.metric('RMSE невязки PDE (train)', f'{pm["rmse_pde_train"]:.6f}',
              help='√mean((L[P]−f)²) на X_train')
    c2.metric('RMSE невязки PDE (test)',  f'{pm["rmse_pde_test"]:.6f}',
              help='√mean((L[P]−f)²) на X_test')
    c3.metric('Время обучения (fit)',     f'{pm["t_fit"]:.4f} с',
              help='Чистое время решения МНК для лучших гиперпараметров')

    # Гиперпараметры
    st.markdown('**Гиперпараметры:**')
    st.dataframe(
        pd.DataFrame([res['best_hp']]).T.rename(columns={0: 'Значение'}),
        use_container_width=False,
    )

    # Тепловая карта
    if is_piezo:
        T_end  = res['T_end']
        t_plot = st.slider('Момент времени t', 0.0, float(T_end),
                           float(T_end) / 2, step=float(T_end) / 20,
                           key='result_t_slider')
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=3, T=T_end, t_slice=t_plot)
        png = plot_pielm_solution(model.predict(X_plot), xx, yy,
                                  title=f'PIELM: P(x,y,t={t_plot:.2f})')
    else:
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
        png = plot_pielm_solution(model.predict(X_plot), xx, yy)

    st.image(png, use_container_width=False)
    st.session_state['plots']['pielm_solution.png'] = png

    # Эволюция (только пьезо)
    if is_piezo:
        st.markdown('**Эволюция поля давления:**')
        T_end   = res['T_end']
        t_snaps = np.linspace(0.0, T_end, 4)[1:]
        snaps   = []
        for ts in t_snaps:
            Xp, xx2, yy2 = get_grid_for_plot(n=60, dim=3, T=T_end, t_slice=ts)
            snaps.append(model.predict(Xp))
        png_ev = plot_piezo_evolution(snaps, xx2, yy2, list(t_snaps))
        st.image(png_ev, use_container_width=True)
        st.session_state['plots']['piezo_evolution.png'] = png_ev


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 5: Сравнение методов
# ─────────────────────────────────────────────────────────────────────────────

def tab_comparison(res):
    st.markdown('<div class="section-header">Сравнение PIELM и МКР</div>',
                unsafe_allow_html=True)

    if not res['fdm_enabled']:
        st.info('МКР отключён. Включите в боковой панели.')
        return

    pm  = res['pielm_metrics']
    fm  = res['fdm_metrics']

    # ── Карточки метрик ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('RMSE PDE — PIELM', f'{pm["rmse_pde_test"]:.6f}',
              help='√mean((L[P]−f)²) на test, PIELM')
    c2.metric('RMSE PDE — МКР',   f'{fm["rmse_pde"]:.6f}',
              help='√mean((L[P]−f)²) на test, МКР (числ. дифф.)')
    c3.metric('Время PIELM (fit)', f'{pm["t_fit"]:.4f} с')
    c4.metric('Время МКР',         f'{fm["t_fdm"]:.4f} с')
    if pm['t_ga'] is not None:
        st.caption(f'Время GA: {pm["t_ga"]:.3f} с — overhead подбора гиперпараметров, в сравнение не входит')

    # ── Bar-график: RMSE PDE (оба метода) и время ───────────────────────
    png_bar = plot_comparison_bar(
        pielm_rmse = pm['rmse_pde_test'],
        fdm_rmse   = fm['rmse_pde'],
        pielm_time = pm['t_fit'],
        fdm_time   = fm['t_fdm'],
    )
    st.image(png_bar, use_container_width=True)
    st.session_state['plots']['comparison_bar.png'] = png_bar

    # ── Тепловые карты ───────────────────────────────────────────────────
    st.markdown('**Поля давления: PIELM vs МКР**')
    model      = res['model']
    fdm_solver = res['fdm_solver']
    is_piezo   = res['is_piezo']

    if is_piezo:
        T_end = res['T_end']
        t_cmp = st.slider('Момент времени t', 0.0, float(T_end),
                          float(T_end) / 2, step=float(T_end) / 20,
                          key='cmp_t_slider')
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=3, T=T_end, t_slice=t_cmp)
        png_cmp = plot_piezo_time_slice(
            model.predict(X_plot), fdm_solver.predict(X_plot), xx, yy, t_cmp
        )
    else:
        X_plot, xx, yy = get_grid_for_plot(n=80, dim=2)
        png_cmp = plot_pielm_vs_fdm(
            model.predict(X_plot), fdm_solver.predict(X_plot), xx, yy
        )

    st.image(png_cmp, use_container_width=True)
    st.session_state['plots']['comparison_fields.png'] = png_cmp

    # ── Сводная таблица ──────────────────────────────────────────────────
    st.markdown('**Сводная таблица:**')
    table = {
        'Метод':              ['PIELM', 'МКР'],
        'RMSE PDE (test)':    [f'{pm["rmse_pde_test"]:.8f}',
                               f'{fm["rmse_pde"]:.8f}'],
        'Время решения (с)':  [f'{pm["t_fit"]:.4f}', f'{fm["t_fdm"]:.4f}'],
        'Узлов / точек':      [
            f'{len(res["X_train"])} коллокац.',
            f'{fm["n_grid"]}² = {fm["n_grid"]**2} узлов',
        ],
    }
    st.dataframe(pd.DataFrame(table), use_container_width=True)

    st.info(
        '**Пояснение метрик:**\n\n'
        '— **RMSE PDE** = √mean((L[P]−f)²) — насколько точно каждый метод '
        'удовлетворяет уравнению на тестовых точках. '
        'Для PIELM вычисляется через оператор нейросети, '
        'для МКР — через численное дифференцирование интерполированного решения.\n\n'
        '— **Время** — t_fit (МНК PIELM) vs t_fdm (СЛАУ МКР). '
        'Время GA — overhead подбора гиперпараметров, в сравнение не входит.'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 6: Экспорт
# ─────────────────────────────────────────────────────────────────────────────

def tab_export(res, cfg):
    st.markdown('<div class="section-header">💾 Экспорт результатов</div>',
                unsafe_allow_html=True)

    pm = res['pielm_metrics']
    fm = res['fdm_metrics']

    eq_params = {}
    if res['is_piezo']:
        eq_params['κ'] = res['kappa']
        eq_params['T'] = res['T_end']

    txt = build_report_txt(
        equation_name      = res['equation_name'],
        equation_params    = eq_params,
        source_name        = res['source_name'],
        source_params      = res['source_params'],
        collocation_params = {
            'N (по оси)':   res['n_side'],
            'N train':      len(res['X_train']),
            'N test':       len(res['X_test']),
            'N граничных':  len(res['X_b']),
        },
        pielm_params       = {} if res['use_ga'] else res['manual_params'],
        ga_params          = res['ga_params_cfg'] or {},
        ga_history         = st.session_state.get('ga_history'),
        best_hyperparams   = res['best_hp'],
        pielm_metrics = {
            'rmse_pde':  pm['rmse_pde_train'],   # невязка PDE на train
            'rmse_test': pm['rmse_pde_test'],     # невязка PDE на test
            'time':      pm['t_fit'],             # время обучения
            't_ga':      pm['t_ga'],              # время GA (None если ручной)
        },
        fdm_metrics = {
            'rmse':   fm['rmse_pde'], # RMSE PDE для МКР
            'time':   fm['t_fdm'],    # время МКР
            'n_grid': fm['n_grid'],
        } if fm is not None else None,
        fdm_enabled = res['fdm_enabled'],
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**📄 Текстовый отчёт**')
        st.download_button(
            label    = '⬇️ Скачать отчёт (.txt)',
            data     = report_to_bytes(txt),
            file_name = make_filename('report', 'txt'),
            mime     = 'text/plain',
            use_container_width=True,
        )
        with st.expander('Предпросмотр'):
            st.text(txt)

    with c2:
        st.markdown('**🖼️ Графики (ZIP)**')
        plots = st.session_state.get('plots', {})
        if plots:
            st.download_button(
                label     = '⬇️ Скачать все графики (.zip)',
                data      = build_plots_zip(plots),
                file_name = make_filename('plots', 'zip'),
                mime      = 'application/zip',
                use_container_width=True,
            )
            st.caption(f'В архиве: {len(plots)} файлов — ' +
                       ', '.join(plots.keys()))
        else:
            st.info('Сначала откройте вкладки с графиками.')

    # ── Экспорт модели ────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('<div class="section-header">🧠 Экспорт модели PIELM</div>',
                unsafe_allow_html=True)

    st.markdown(
        'Сохранённая модель содержит все веса нейронной сети и может быть '
        'загружена на **любом языке программирования** (Python, JavaScript, '
        'C++, MATLAB) без ML-фреймворков. '
        'Формула предсказания: `P = activation(X @ W + b) @ beta`.'
    )

    model = res.get('model')
    if model is None or model.beta is None:
        st.warning('Модель не обучена. Запустите вычисление.')
        return

    # Подготовка конфигурации для метаданных
    model_cfg = {
        'equation_name': res.get('equation_name', 'unknown'),
        'is_piezo':      res.get('is_piezo', False),
        'source_name':   res.get('source_name', 'unknown'),
        'source_params': res.get('source_params', {}),
        'kappa':         res.get('kappa', None),
        'T':             res.get('T_end', None),
    }
    model_metrics = {
        'rmse_pde_train': pm['rmse_pde_train'],
        'rmse_pde_test':  pm['rmse_pde_test'],
        'time_fit':       pm['t_fit'],
    }

    json_bytes = model_to_json_bytes(
        model, cfg=model_cfg, metrics=model_metrics
    )
    st.download_button(
        label='⬇️ Скачать модель (.json)',
        data=json_bytes,
        file_name=make_filename('pielm_model', 'json'),
        mime='application/json',
        use_container_width=True,
    )
    st.caption(
        f'Размер: {len(json_bytes)/1024:.1f} КБ · '
        f'{model.n_hidden} нейронов · {model.act_name}'
    )

    # Пример кода для загрузки
    with st.expander('📋 Пример загрузки и предсказания (Python)'):
        st.code('''import json, numpy as np

# Загрузка модели
with open("pielm_model.json", "r") as f:
    m = json.load(f)

W    = np.array(m["weights"]["W"])
b    = np.array(m["weights"]["b"])
beta = np.array(m["weights"]["beta"])

# Выбор функции активации
activations = {"tanh": np.tanh, "sin": np.sin,
               "sigmoid": lambda z: 1/(1+np.exp(-z))}
act = activations[m["architecture"]["activation"]]

# Предсказание давления в точке (x=0.5, y=0.3)
X = np.array([[0.5, 0.3]])
P = (act(X @ W + b) @ beta).ravel()
print(f"P(0.5, 0.3) = {P[0]:.6f}")
''', language='python')

    with st.expander('📋 REST API (FastAPI)'):
        st.code('''from fastapi import FastAPI
import json, numpy as np

m = json.load(open("pielm_model.json"))
W, b, beta = np.array(m["weights"]["W"]), np.array(m["weights"]["b"]), np.array(m["weights"]["beta"])
act = np.tanh

app = FastAPI()

@app.post("/predict")
def predict(points: list[list[float]]):
    X = np.array(points)
    P = (act(X @ W + b) @ beta).ravel()
    return {"pressures": P.tolist()}

# Запуск: uvicorn server:app --reload
# Запрос: curl -X POST localhost:8000/predict -d '[[0.5,0.3]]'
''', language='python')