import streamlit as st

from ui.sidebar import render_sidebar
from ui.runner  import run_computation
from ui.tabs    import (
    tab_ga_progress,
    tab_collocation,
    tab_source,
    tab_pielm,
    tab_comparison,
    tab_export,
)


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='PIELM + GA — Уравнения фильтрации',
    page_icon='💧',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem; font-weight: 700;
        color: #1e3a5f; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #4a6fa5; margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 0.3rem;
        margin-top: 1rem; margin-bottom: 0.8rem;
    }
    div[data-testid="stSidebar"] { background-color: #f8faff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Инициализация session_state
# ─────────────────────────────────────────────────────────────────────────────

for key, val in [('results', None), ('ga_history', None),
                 ('plots', {}), ('run_complete', False)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Заголовок
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">💧 PIELM + Генетический алгоритм</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Физически информированное экстремальное '
    'машинное обучение для решения уравнений фильтрации</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Боковая панель
# ─────────────────────────────────────────────────────────────────────────────

cfg, run_btn = render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# Запуск вычислений
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.session_state['run_complete'] = False
    st.session_state['plots']        = {}
    run_computation(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Вывод результатов
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state['run_complete'] and st.session_state['results']:
    res = st.session_state['results']

    tabs = st.tabs([
        '📈 Прогресс GA',
        '🔵 Коллокационные точки',
        '🌊 Поле источника',
        '🗺️ Результаты PIELM',
        '⚖️ Сравнение методов',
        '💾 Экспорт',
    ])

    with tabs[0]: tab_ga_progress()
    with tabs[1]: tab_collocation(res)
    with tabs[2]: tab_source(res)
    with tabs[3]: tab_pielm(res)
    with tabs[4]: tab_comparison(res)
    with tabs[5]: tab_export(res, cfg)

else:
    st.info('👈 Настройте параметры и нажмите **▶ Запустить вычисление**')
    with st.expander('ℹ️ О приложении'):
        st.markdown("""
**PIELM** — решение уравнений фильтрации нейросетью с фиксированными весами.

| Уравнение | Тип |
|-----------|-----|
| Пуассона: ∇²P = f(x,y) | Стационарное |
| Пьезопроводности: ∂P/∂t = κ·∇²P + q | Нестационарное |

**Генетический алгоритм** подбирает: n_hidden, scale, activation, λ_pde, λ_bc.

**МКР** — эталон для сравнения:
- Пуассон: центральные разности + прямое решение СЛАУ
- Пьезопроводность: схема Кранка–Николсона

**Метрики:**
- RMSE невязки PDE = √mean((L[P]−f)²)
- RMSE(PIELM vs МКР) = √mean((P_PIELM−P_МКР)²)
- Время сравнивается: t_fit (PIELM) vs t_fdm (МКР)
        """)