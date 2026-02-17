import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pielm  # Файл с вашим классом PIELM

# --- Настройка страницы ---
st.set_page_config(page_title="PIELM Interactive Solver", layout="wide")

st.title("🚀 Physics Informed Extreme Learning Machine (PIELM)")
st.markdown("""
Интерактивная среда для решения дифференциальных уравнений с помощью нейросетей.
Выберите задачу, настройте параметры и найдите решение мгновенно.
""")

# --- БОКОВАЯ ПАНЕЛЬ: Настройки ---
st.sidebar.header("1. Выбор задачи")
test_case = st.sidebar.selectbox(
    "Test Case", 
    ["TC-1: Advection", "TC-2: Diffusion", "TC-3: Advection-Diffusion", 
     "TC-11: Sharp Gradient", "TC-12: High Frequency"]
)

st.sidebar.header("2. Гиперпараметры")
# Используем session_state, чтобы авто-подбор мог обновлять эти значения
if 'n_hidden' not in st.session_state: st.session_state.n_hidden = 200
if 'scale' not in st.session_state: st.session_state.scale = 2.0

n_hidden = st.sidebar.slider("Количество нейронов (N_hidden)", 10, 2000, st.session_state.n_hidden)
scale = st.sidebar.slider("Масштаб весов (Scale)", 0.1, 20.0, st.session_state.scale, step=0.1)
n_f = st.sidebar.slider("Точки коллокации (N_f)", 10, 500, 100)

# Дополнительные параметры для специфических тестов
nu_param = 0.1
k_freq = 4.0

if test_case == "TC-3: Advection-Diffusion":
    nu_param = st.sidebar.number_input("Вязкость (nu)", 0.001, 1.0, 0.1, format="%.3f")
elif test_case == "TC-11: Sharp Gradient":
    nu_param = st.sidebar.number_input("Вязкость (nu)", 0.001, 1.0, 0.05, format="%.3f")
elif test_case == "TC-12: High Frequency":
    k_freq = st.sidebar.number_input("Частота (k)", 1.0, 20.0, 8.0)
    nu_param = st.sidebar.number_input("Вязкость (nu)", 0.0001, 0.1, 0.0001, format="%.4f")

# --- ФУНКЦИЯ: Генерация данных и операторов ---
def get_problem_data(case, n_f, nu, k):
    # Данные для обучения
    x_f = np.random.uniform(0, 1, (n_f, 1))
    x_b = np.array([[0.0], [1.0]])
    
    # Данные для теста (графики)
    x_test = np.linspace(0, 1, 300).reshape(-1, 1)

    if "TC-1" in case:
        # u_x = R
        exact = lambda x: np.sin(2*np.pi*x) * np.cos(4*np.pi*x) + 1
        source = lambda x: 2*np.pi*np.cos(2*np.pi*x)*np.cos(4*np.pi*x) - 4*np.pi*np.sin(2*np.pi*x)*np.sin(4*np.pi*x)
        operator = pielm.advection_operator_tc1
        
    elif "TC-2" in case:
        # u_xx = R
        exact = lambda x: np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x) + 1
        # Точная вторая производная
        def source(x):
            a, b = np.pi * x / 2, 2 * np.pi * x
            f, g = np.sin(a), np.cos(b)
            df, dg = (np.pi/2)*np.cos(a), -2*np.pi*np.sin(b)
            ddf, ddg = -(np.pi/2)**2 * np.sin(a), -(2*np.pi)**2 * np.cos(b)
            return ddf*g + 2*df*dg + f*ddg
        operator = pielm.diffusion_operator_tc2
        
    elif "TC-3" in case:
        # u_x - nu*u_xx = 0
        exact = lambda x: np.expm1(x / nu) / np.expm1(1.0 / nu)
        source = lambda x: np.zeros_like(x)
        operator = lambda W, b, X: pielm.adv_diff_operator_tc3(W, b, X, nu=nu)
        
    elif "TC-11" in case:
        # Sharp Gradient: -nu*u_xx + u_x = 0
        exact = lambda x: np.expm1(x / nu) / np.expm1(1.0 / nu) # то же решение, что TC-3
        source = lambda x: np.zeros_like(x)
        operator = lambda W, b, X: pielm.adv_diff_operator_tc3(W, b, X, nu=nu)
        
    elif "TC-12" in case:
        # High Freq: -nu*u_xx + u_x = source
        exact = lambda x: np.sin(k * np.pi * x)
        def source(x):
            term1 = nu * (k * np.pi)**2 * np.sin(k * np.pi * x)
            term2 = k * np.pi * np.cos(k * np.pi * x)
            return term1 + term2
        operator = lambda W, b, X: pielm.adv_diff_operator_tc3(W, b, X, nu=nu)
        
    return x_f, x_b, x_test, exact, source, operator

# --- ЛОГИКА: Авто-подбор (Auto-ML) ---
st.sidebar.markdown("---")
if st.sidebar.button("🤖 AI Авто-подбор параметров"):
    with st.spinner("Генетический алгоритм ищет лучшее решение..."):
        # Получаем данные для оптимизации
        x_f_opt, x_b_opt, _, exact_opt, source_opt, op_opt = get_problem_data(test_case, 100, nu_param, k_freq)
        y_b_opt = exact_opt(x_b_opt)
        
        # Запускаем оптимизатор (проверьте, что GeneticOptimizer есть в pielm.py!)
        try:
            optimizer = pielm.GeneticOptimizer(n_pop=20, n_gen=15, 
                                               scale_bounds=(0.5, 15.0), 
                                               hidden_bounds=(50, 800))
            
            # Колбек для прогресс-бара
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            def ga_callback(gen, loss, params):
                progress_bar.progress((gen + 1) / 15)
                status_text.text(f"Gen {gen}: Loss {loss:.2e}")

            best_params = optimizer.search(x_f_opt, x_b_opt, y_b_opt, op_opt, source_opt, callback=ga_callback)
            
            # Сохраняем в session_state и перезагружаем
            st.session_state.n_hidden = best_params['n_hidden']
            st.session_state.scale = best_params['scale']
            st.success(f"Найдено! Scale: {best_params['scale']:.2f}, Neurons: {best_params['n_hidden']}")
            st.rerun()
            
        except AttributeError:
            st.error("Класс GeneticOptimizer не найден в pielm.py! Добавьте его, чтобы использовать эту кнопку.")

# --- ОСНОВНОЙ ПРОЦЕСС ---

# 1. Получаем данные
x_f, x_b, x_test, exact_u, source_r, operator = get_problem_data(test_case, n_f, nu_param, k_freq)
y_b = exact_u(x_b)

# 2. Создаем и обучаем модель
model = pielm.PIELM(n_hidden=n_hidden, scale=scale)

try:
    model.fit(x_f, x_b, y_b, operator, source_r)
    
    # 3. Предсказание
    u_pred = model.predict(x_test)
    u_true = exact_u(x_test)
    
    # 4. Расчет ошибки уравнения (Residual) на тестовых точках
    # Проверяем, насколько хорошо сеть выучила физику: L[u] - R ≈ 0
    H_test = operator(model.W, model.b, x_test)
    R_test = source_r(x_test)
    if R_test.ndim == 1: R_test = R_test.reshape(-1, 1)
    
    residual = (H_test @ model.beta) - R_test
    
    mse_u = np.mean((u_true - u_pred)**2)
    mse_res = np.mean(residual**2)

    # --- ВИЗУАЛИЗАЦИЯ ---
    
    # Метрики
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE (Решение)", f"{mse_u:.2e}")
    c2.metric("MSE (Уравнение/Физика)", f"{mse_res:.2e}")
    c3.metric("Число условий", f"{n_f} + 2")

    # Графики
    tab1, tab2 = st.tabs(["📉 Решение u(x)", "physics Ошибка уравнения (Residual)"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_test, u_true, 'k-', label='Точное решение', linewidth=2, alpha=0.6)
        ax.plot(x_test, u_pred, 'r--', label='PIELM Предсказание', linewidth=2)
        
        # Рисуем точки коллокации
        ax.scatter(x_f, exact_u(x_f), color='blue', alpha=0.3, s=20, label='Точки коллокации')
        ax.scatter(x_b, y_b, color='green', s=100, marker='x', label='Граничные условия', zorder=5)
        
        ax.set_title(f"Решение: {test_case}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(x_test, residual, 'g-', label='Residual (L[u] - R)')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title("Ошибка выполнения дифференциального уравнения")
        ax2.set_ylabel("Error")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        st.caption("Если эта линия близка к нулю, значит нейросеть соблюдает закон физики.")

except np.linalg.LinAlgError:
    st.error("Ошибка линейной алгебры! Вероятно, матрица вырождена. Попробуйте изменить Scale или количество нейронов.")
except Exception as e:
    st.error(f"Произошла ошибка: {e}")

# --- Доп. инфо ---
with st.expander("ℹ️ Справка по параметрам"):
    st.markdown("""
    * **Scale**: Отвечает за "резкость" базисных функций. Для плавных решений (TC-1, TC-2) подходит 1.0-3.0. Для резких (TC-11, TC-12) нужно 5.0-15.0.
    * **N_hidden**: Чем больше нейронов, тем точнее, но может возникнуть переобучение (шум).
    * **N_f**: Количество точек внутри области, где мы "учим" физику.
    """)