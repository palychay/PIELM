import numpy as np
import matplotlib.pyplot as plt
import pielm

# ==========================================
# ЧАСТЬ 2: 2D Задачи (TC-4, TC-5, TC-6)
# ==========================================

# --- Геометрия: Звездообразная область ---
def is_in_star_domain(x, y):
    """
    Фильтр точек внутри "Звезды" (как в статье).
    Уравнение в полярных координатах: r(theta) = R0 + A * cos(k * theta)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # 6 лучей, радиус меняется от 0.3 до 0.7 (примерно)
    boundary_r = 0.5 + 0.2 * np.cos(6 * theta) 
    return r <= boundary_r

def generate_2d_data(N_total, domain_func):
    """
    Генерирует случайные точки внутри области и точки на границе.
    """
    # 1. Внутренние точки (метод Rejection Sampling)
    points = []
    while len(points) < N_total:
        x_try = np.random.uniform(-1, 1, N_total*2)
        y_try = np.random.uniform(-1, 1, N_total*2)
        mask = domain_func(x_try, y_try)
        valid = np.column_stack((x_try[mask], y_try[mask]))
        points.extend(valid)
    
    X_f = np.array(points[:N_total])

    # 2. Граничные точки (параметрически)
    theta = np.linspace(0, 2*np.pi, 200) # 200 точек на границе
    r_b = 0.5 + 0.2 * np.cos(6 * theta)
    x_b = r_b * np.cos(theta)
    y_b = r_b * np.sin(theta)
    X_b = np.column_stack((x_b, y_b))
    
    return X_f, X_b

# --- ТОЧНЫЕ РЕШЕНИЯ И ИСТОЧНИКИ (Eq 40-42 в статье) ---

# TC-4: Стационарная адвекция. u = 0.5 * cos(pi*x)*sin(pi*y)
# Уравнение: u_x + u_y = R
def exact_u_tc4(x, y):
    return 0.5 * np.cos(np.pi * x) * np.sin(np.pi * y)

def source_R_tc4(x, y):
    # u_x = -0.5 * pi * sin(pi*x) * sin(pi*y)
    # u_y = 0.5 * pi * cos(pi*x) * cos(pi*y)
    u_x = -0.5 * np.pi * np.sin(np.pi*x) * np.sin(np.pi*y)
    u_y = 0.5 * np.pi * np.cos(np.pi*x) * np.cos(np.pi*y)
    return u_x + u_y

# TC-5: Стационарная диффузия. u = 0.5 + exp(-(2x^2 + 4y^2))
# Уравнение: u_xx + u_yy = R
def exact_u_tc5(x, y):
    return 0.5 + np.exp(-(2*x**2 + 4*y**2))

def source_R_tc5(x, y):
    # E = exp(-(2x^2 + 4y^2))
    # Производные вычисляются по правилу цепочки
    E = np.exp(-(2*x**2 + 4*y**2))
    u_xx = E * (16*x**2 - 4)
    u_yy = E * (64*y**2 - 8)
    return u_xx + u_yy

# TC-6: Диффузия (Сложный Гауссиан). u = 0.5 + exp(-((x-0.6)^2 + (y-0.6)^2))
# Уравнение: u_xx + u_yy = R
def exact_u_tc6(x, y):
    return 0.5 + np.exp(-((x - 0.6)**2 + (y - 0.6)**2))

def source_R_tc6(x, y):
    # Гауссиан смещенный. 
    E = np.exp(-((x - 0.6)**2 + (y - 0.6)**2))
    u_xx = E * (4*(x - 0.6)**2 - 2)
    u_yy = E * (4*(y - 0.6)**2 - 2)
    return u_xx + u_yy


# ==========================================
# ЗАПУСК ТЕСТОВ
# ==========================================

if __name__ == "__main__":
    
    # --- Запуск 2D задач ---
    print("\n=== Запуск 2D тестов (TC-4, TC-5, TC-6) ===")
    
    # 1. Параметры обучения 
    N_f_2d = 1000   # Точки внутри
    N_hidden_2d = 800 
    
    # 2. Генерируем данные для области "Звезда"
    X_f, X_b = generate_2d_data(N_f_2d, is_in_star_domain)
    
    # 3. Генерируем тестовую сетку для валидации и графиков
    x_lin = np.linspace(-1, 1, 60)
    y_lin = np.linspace(-1, 1, 60)
    xx, yy = np.meshgrid(x_lin, y_lin)
    flat_x, flat_y = xx.flatten(), yy.flatten()
    # Фильтруем тестовые точки, чтобы проверять только внутри звезды
    mask_test = is_in_star_domain(flat_x, flat_y)
    X_test_2d = np.column_stack((flat_x[mask_test], flat_y[mask_test]))

    # --- TC-4: 2D Адвекция ---
    print("\nОбучение TC-4 (Адвекция)...")
    model_tc4 = pielm.PIELM(n_hidden=N_hidden_2d, input_dim=2, scale=2.0)
    
    # Граничные значения берем из точного решения
    Y_b_tc4 = exact_u_tc4(X_b[:,0], X_b[:,1]).reshape(-1, 1)
    
    model_tc4.fit(X_f, X_b, Y_b_tc4,
                  lambda W,b,X: pielm.advection_2d_operator(W,b,X, a=1.0, b_coef=1.0),
                  source_R_tc4)
    
    u_pred_4 = model_tc4.predict(X_test_2d)
    u_true_4 = exact_u_tc4(X_test_2d[:,0], X_test_2d[:,1]).reshape(-1, 1)
    pielm.print_info(u_pred_4, u_true_4, N_hidden_2d, N_f_2d, "TC-4 (2D Адвекция)")
    pielm.draw_graphics_2d(X_test_2d, u_true_4, u_pred_4, title="TC-4: 2D Advection")

    # --- TC-5: 2D Диффузия ---
    print("Обучение TC-5 (Диффузия)...")
    model_tc5 = pielm.PIELM(n_hidden=N_hidden_2d, input_dim=2, scale=3.0)
    
    Y_b_tc5 = exact_u_tc5(X_b[:,0], X_b[:,1]).reshape(-1, 1)
    
    model_tc5.fit(X_f, X_b, Y_b_tc5,
                  pielm.diffusion_2d_operator,
                  source_R_tc5)
    
    u_pred_5 = model_tc5.predict(X_test_2d)
    u_true_5 = exact_u_tc5(X_test_2d[:,0], X_test_2d[:,1]).reshape(-1, 1)
    pielm.print_info(u_pred_5, u_true_5, N_hidden_2d, N_f_2d, "TC-5 (2D Диффузия)")
    pielm.draw_graphics_2d(X_test_2d, u_true_5, u_pred_5, title="TC-5: 2D Diffusion")

    # --- TC-6: 2D Диффузия (Сложный пик) ---
    print("Обучение TC-6 (Сложный Гауссиан)...")
    # Для острого пика нужно больше нейронов или другой scale
    model_tc6 = pielm.PIELM(n_hidden=1500, input_dim=2, scale=4.0) 
    
    Y_b_tc6 = exact_u_tc6(X_b[:,0], X_b[:,1]).reshape(-1, 1)
    
    model_tc6.fit(X_f, X_b, Y_b_tc6,
                  pielm.diffusion_2d_operator,
                  source_R_tc6)
                  
    u_pred_6 = model_tc6.predict(X_test_2d)
    u_true_6 = exact_u_tc6(X_test_2d[:,0], X_test_2d[:,1]).reshape(-1, 1)
    pielm.print_info(u_pred_6, u_true_6, 1500, N_f_2d, "TC-6 (2D Sharp Gaussian)")
    pielm.draw_graphics_2d(X_test_2d, u_true_6, u_pred_6, title="TC-6: Complex Domain Diffusion")
