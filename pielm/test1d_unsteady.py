import numpy as np
import matplotlib.pyplot as plt
import pielm

# ==========================================
# ЧАСТЬ 3: 1D Нестационарные задачи (Space-Time)
# Область: x: [-1, 1], t: [0, 0.5] (из Eq 46)
# ==========================================

def generate_spacetime_data(N_f, N_b_side=50):
    """
    Генерирует данные для прямоугольника x=[-1,1], t=[0, 0.5]
    """
    # 1. Внутри области (Collocation)
    x_f = np.random.uniform(-1, 1, N_f)
    t_f = np.random.uniform(0, 0.5, N_f)
    X_f = np.column_stack((x_f, t_f))

    # 2. Границы + Начальное условие
    # Initial Condition (t=0)
    x_init = np.linspace(-1, 1, N_b_side)
    t_init = np.zeros(N_b_side)
    bc_init = np.column_stack((x_init, t_init))

    t_side = np.linspace(0, 0.5, N_b_side)
    x_left = np.full(N_b_side, -1.0)
    bc_left = np.column_stack((x_left, t_side))
    
    x_right = np.full(N_b_side, 1.0)
    bc_right = np.column_stack((x_right, t_side))

    X_b = np.vstack((bc_init, bc_left, bc_right))
    return X_f, X_b

# --- ТОЧНЫЕ РЕШЕНИЯ (Eq 48, 49) ---

def exact_F(x):
    """ Начальное условие F(x) = sin(pi * x) из текста под Eq 47 """
    return np.sin(np.pi * x)

# TC-7: Constant coefficient a=1
# u(x,t) = F(x - t)
def exact_u_tc7(x, t):
    return exact_F(x - t)

# TC-8: Variable coefficient a(x) = 1 + x
# u(x,t) = F((1+x)*exp(-t) - 1)
def exact_u_tc8(x, t):
    arg = (1 + x) * np.exp(-t) - 1
    return exact_F(arg)


if __name__ == "__main__":
    
    # Настройки как в статье (Section 4.3)
    # Они используют около 420 точек, мы возьмем чуть больше для гарантии
    N_f = 1000 
    N_b_side = 100
    N_hidden = 500
    
    X_f, X_b = generate_spacetime_data(N_f, N_b_side)
    
    # Сетка для отрисовки
    x_grid = np.linspace(-1, 1, 100)
    t_grid = np.linspace(0, 0.5, 100)
    xx, tt = np.meshgrid(x_grid, t_grid)
    X_test = np.column_stack((xx.flatten(), tt.flatten()))

    # ==========================
    # TC-7: Constant Advection
    # ==========================
    print("\n--- TC-7: 1D Unsteady Advection (Constant a=1) ---")
    model_7 = pielm.PIELM(n_hidden=N_hidden, input_dim=2, scale=3.0)
    
    # Точные значения на границах
    Y_b_7 = exact_u_tc7(X_b[:,0], X_b[:,1]).reshape(-1, 1)
    
    # Оператор: u_t + 1.0 * u_x = 0
    # Правая часть R = 0
    model_7.fit(X_f, X_b, Y_b_7,
                lambda W,b,X: pielm.advection_unsteady_1d_operator(W,b,X, a_coeff=1.0),
                lambda x,t: np.zeros_like(x)) # R=0
                
    u_pred_7 = model_7.predict(X_test)
    u_true_7 = exact_u_tc7(X_test[:,0], X_test[:,1]).reshape(-1, 1)
    
    pielm.print_info(u_pred_7, u_true_7, N_hidden, N_f, "TC-7 (a=1)")
    # Рисуем карту Space-Time (ось Y - время)
    pielm.draw_graphics_2d(X_test, u_true_7, u_pred_7, title="TC-7: Space-Time Solution")

    # ==========================
    # TC-8: Variable Advection
    # ==========================
    print("\n--- TC-8: 1D Unsteady Advection (Variable a = 1+x) ---")
    model_8 = pielm.PIELM(n_hidden=N_hidden, input_dim=2, scale=3.0)
    
    # Точные значения на границах
    Y_b_8 = exact_u_tc8(X_b[:,0], X_b[:,1]).reshape(-1, 1)
    
    # Коэффициент a(x) зависит от x. Вычисляем его для точек коллокации X_f
    # X_f[:, 0] - это координаты x
    a_variable = 1.0 + X_f[:, 0] 
    
    # Оператор: u_t + (1+x) * u_x = 0
    model_8.fit(X_f, X_b, Y_b_8,
                lambda W,b,X: pielm.advection_unsteady_1d_operator(W,b,X, a_coeff=a_variable),
                lambda x,t: np.zeros_like(x)) # R=0
    
    u_pred_8 = model_8.predict(X_test)
    u_true_8 = exact_u_tc8(X_test[:,0], X_test[:,1]).reshape(-1, 1)
    
    pielm.print_info(u_pred_8, u_true_8, N_hidden, N_f, "TC-8 (a = 1+x)")
    pielm.draw_graphics_2d(X_test, u_true_8, u_pred_8, title="TC-8: Space-Time (Variable Coeff)")