import numpy as np
import matplotlib.pyplot as plt
import pielm

# ==========================================
# ЧАСТЬ 2: 2D Задачи (TC-4, TC-5, TC-6)
# ==========================================

def generate_square_data(N_f, N_b_per_side=50):
    """
    Генерирует данные для квадратной области [-1, 1] x [-1, 1].
    
    Аргументы:
    N_f -- количество точек коллокации внутри.
    N_b_per_side -- количество точек на каждой из 4-х сторон квадрата.
    """
    # 1. Точки внутри (Collocation points)
    # Просто равномерное распределение от -1 до 1
    X_f = np.random.uniform(-1, 1, (N_f, 2))

    # 2. Граничные точки (Boundary points)
    # Генерируем точки для 4 сторон квадрата
    
    # Нижняя сторона (y = -1)
    bottom = np.column_stack((np.linspace(-1, 1, N_b_per_side), np.full(N_b_per_side, -1.0)))
    # Верхняя сторона (y = 1)
    top = np.column_stack((np.linspace(-1, 1, N_b_per_side), np.full(N_b_per_side, 1.0)))
    # Левая сторона (x = -1)
    left = np.column_stack((np.full(N_b_per_side, -1.0), np.linspace(-1, 1, N_b_per_side)))
    # Правая сторона (x = 1)
    right = np.column_stack((np.full(N_b_per_side, 1.0), np.linspace(-1, 1, N_b_per_side)))
    
    # Объединяем все границы
    X_b = np.vstack((bottom, top, left, right))
    
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
    
# 1. Параметры
    N_f_2d = 1000       # Точки внутри
    N_b_side = 50       # Точек на одну сторону (всего будет 200)
    N_hidden_2d = 800   # Нейроны
    
    # 2. Генерируем данные (Квадрат [-1, 1])
    X_f, X_b = generate_square_data(N_f_2d, N_b_side)
    
    # 3. Генерируем тестовую сетку для валидации и графиков
    x_lin = np.linspace(-1, 1, 50)
    y_lin = np.linspace(-1, 1, 50)
    xx, yy = np.meshgrid(x_lin, y_lin)

    # Превращаем сетку в список точек (N, 2)
    X_test_2d = np.column_stack((xx.flatten(), yy.flatten()))

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
