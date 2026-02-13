import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pielm  # Импортируем ваш существующий файл

st.set_page_config(page_title="PIELM Interactive Demo", layout="wide")

st.title("🚀 Physics Informed Extreme Learning Machine (PIELM)")
st.markdown("""
Визуализация первых трех тестов из статьи: 
* **TC-1**: Адвекция ($u_x = R$)
* **TC-2**: Диффузия ($u_{xx} = R$)
* **TC-3**: Адвекция-Диффузия ($u_x - \nu u_{xx} = 0$)
""")

# --- Боковая панель (Настройки) ---
st.sidebar.header("Общие параметры")
test_case = st.sidebar.selectbox("Выберите тест", ["TC-1", "TC-2", "TC-3"])
n_hidden = st.sidebar.slider("Количество нейронов", 10, 1000, 200)
n_f = st.sidebar.slider("Точки коллокации (N_f)", 5, 200, 50)
scale = st.sidebar.slider("Масштаб весов (Scale)", 0.1, 10.0, 2.0)

# --- Физика и точные решения ---
def get_tc_data(case, n_f):
    x_f = np.random.uniform(0, 1, (n_f, 1))
    x_b = np.array([[0.0], [1.0]])
    x_test = np.linspace(0, 1, 200).reshape(-1, 1)
    
    if case == "TC-1":
        exact_u = lambda x: np.sin(2*np.pi*x) * np.cos(4*np.pi*x) + 1
        source_r = lambda x: 2*np.pi*np.cos(2*np.pi*x)*np.cos(4*np.pi*x) - 4*np.pi*np.sin(2*np.pi*x)*np.sin(4*np.pi*x)
        operator = pielm.advection_operator_tc1
        return x_f, x_b, x_test, exact_u, source_r, operator
    
    elif case == "TC-2":
        exact_u = lambda x: np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x) + 1
        def source_r(x):
            a, b = np.pi * x / 2, 2 * np.pi * x
            f, g = np.sin(a), np.cos(b)
            df, dg = (np.pi/2) * np.cos(a), -2*np.pi * np.sin(b)
            ddf, ddg = -(np.pi/2)**2 * np.sin(a), -(2*np.pi)**2 * np.cos(b)
            # Формула u'' = f''g + 2f'g' + fg''
            return ddf * g + 2 * df * dg + f * ddg
            
        operator = pielm.diffusion_operator_tc2
        return x_f, x_b, x_test, exact_u, source_r, operator

    elif case == "TC-3":
        nu = st.sidebar.number_input("Вязкость (nu)", 0.01, 1.0, 0.1)
        exact_u = lambda x: np.expm1(x / nu) / np.expm1(1.0 / nu)
        source_r = lambda x: np.zeros_like(x)
        operator = lambda W, b, X: pielm.adv_diff_operator_tc3(W, b, X, nu=nu)
        return x_f, x_b, x_test, exact_u, source_r, operator

# --- Запуск модели ---
x_f, x_b, x_test, exact_u, source_r, operator = get_tc_data(test_case, n_f)

model = pielm.PIELM(n_hidden=n_hidden, scale=scale)
y_b = exact_u(x_b)

# Обучение
model.fit(x_f, x_b, y_b, operator, source_r)

# Предсказание
u_pred = model.predict(x_test)
u_true = exact_u(x_test)
mse = np.mean((u_true - u_pred)**2)

# --- Визуализация ---
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_test, u_true, 'b-', label='Точное решение', linewidth=2)
    ax.plot(x_test, u_pred, 'r--', label='PIELM', linewidth=2)
    ax.scatter(x_f, np.zeros_like(x_f), color='green', marker='|', s=100, label='Точки коллокации')
    ax.set_title(f"Результаты для {test_case}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.metric("MSE Error", f"{mse:.2e}")
    st.write(f"**Нейронов:** {n_hidden}")
    st.write(f"**Точек:** {n_f}")
    if mse > 0.1:
        st.error("Плохая точность. Попробуйте увеличить количество нейронов или изменить Scale.")
    else:
        st.success("Отличная аппроксимация!")

st.info("💡 Совет: В TC-3 при малых 'nu' (например, 0.05) решение становится очень резким. Увеличьте Scale до 5-8, чтобы модель могла его поймать.")