import numpy as np
import matplotlib.pyplot as plt
import pielm

# ==========================================
# ЧАСТЬ 4: Unsteady Advection-Diffusion (TC-9, TC-10)
# ==========================================

# --- Генераторы данных ---

def generate_data_tc9(N_f, N_b_side=50):
    # Область TC-9: x in [0, 1], t in [0, 0.5] (Eq 54)
    # Внутренние точки
    X_f = np.random.uniform(0, 1, (N_f, 2))
    X_f[:, 1] *= 0.5 # t масштабируем до 0.5

    # Границы
    # t=0 (IC)
    ic = np.column_stack((np.linspace(0, 1, N_b_side), np.zeros(N_b_side)))
    # x=0 (BC left)
    bc_l = np.column_stack((np.zeros(N_b_side), np.linspace(0, 0.5, N_b_side)))
    # x=1 (BC right)
    bc_r = np.column_stack((np.ones(N_b_side), np.linspace(0, 0.5, N_b_side)))
    
    X_b = np.vstack((ic, bc_l, bc_r))
    return X_f, X_b

def generate_data_tc10(N_f, N_b_side=20):
    # Область TC-10: x,y in [0, 1], t in [0, 0.2] (Eq 59)
    # Внутренние точки (N, 3) -> x, y, t
    X_f = np.random.uniform(0, 1, (N_f, 3))
    X_f[:, 2] *= 0.2 # t до 0.2

    # Границы (6 граней куба, упростим до основных)
    # 1. t=0 (IC) - плоскость xy
    xy_grid = np.random.uniform(0, 1, (N_b_side**2, 2))
    ic = np.column_stack((xy_grid, np.zeros(len(xy_grid))))
    
    # 2. Стенки (x=0, x=1, y=0, y=1) для всех t
    t_vals = np.linspace(0, 0.2, N_b_side)
    pos_vals = np.linspace(0, 1, N_b_side)
    # Для простоты генерируем случайные точки на гранях
    # В реальной задаче лучше делать grid mesh
    bc_walls = []
    for _ in range(4 * N_b_side): # Просто накидаем точек на границы
        t = np.random.uniform(0, 0.2)
        p = np.random.uniform(0, 1)
        bc_walls.append([0, p, t]) # x=0
        bc_walls.append([1, p, t]) # x=1
        bc_walls.append([p, 0, t]) # y=0
        bc_walls.append([p, 1, t]) # y=1
    
    X_b = np.vstack((ic, np.array(bc_walls)))
    return X_f, X_b

# --- Точные решения ---

# TC-9 Exact (Eq 55)
def exact_u_tc9(x, t, a=1.0, nu=0.005):
    # Gaussian moving and decaying
    term1 = 1.0 / np.sqrt(4*t + 1)
    numerator = (x - 0.2 - a*t)**2
    denominator = nu * (4*t + 1)
    return term1 * np.exp(-numerator / denominator)

# TC-10 Exact (Eq 62)
def exact_u_tc10(x, y, t, a, b, nu):
    # 2D Gaussian moving
    term1 = 1.0 / (4*t + 1)
    num = (x - a*t)**2 + (y - b*t)**2
    den = nu * (4*t + 1)
    return term1 * np.exp(-num / den)


if __name__ == "__main__":
    
    # --- TC-9: 1D Unsteady Adv-Diff ---
    print("\n--- TC-9: 1D Unsteady Advection-Diffusion ---")
    # Параметры из статьи: nu=0.005, a=1.0 (обычно)
    NU_9 = 0.005
    A_9 = 1.0
    
    X_f_9, X_b_9 = generate_data_tc9(N_f=1500, N_b_side=100)
    
    # Решение на границе
    Y_b_9 = exact_u_tc9(X_b_9[:,0], X_b_9[:,1], a=A_9, nu=NU_9).reshape(-1, 1)
    
    model_9 = pielm.PIELM(n_hidden=800, input_dim=2, scale=3.0)
    model_9.fit(X_f_9, X_b_9, Y_b_9,
                lambda W,b,X: pielm.adv_diff_1d_unsteady_operator(W,b,X, a=A_9, nu=NU_9),
                lambda x,t: np.zeros_like(x)) # R=0
    
    # Проверка на карте x-t
    x_lin = np.linspace(0, 1, 60)
    t_lin = np.linspace(0, 0.5, 60)
    xx, tt = np.meshgrid(x_lin, t_lin)
    X_test_9 = np.column_stack((xx.flatten(), tt.flatten()))
    
    u_pred_9 = model_9.predict(X_test_9)
    u_true_9 = exact_u_tc9(X_test_9[:,0], X_test_9[:,1], a=A_9, nu=NU_9).reshape(-1, 1)
    
    pielm.print_info(u_pred_9, u_true_9, 800, 1500, "TC-9 (x-t Heatmap)")
    pielm.draw_graphics_2d(X_test_9, u_true_9, u_pred_9, title="TC-9 Solution")


    # --- TC-10: 2D Unsteady Adv-Diff ---
    print("\n--- TC-10: 2D Unsteady Advection-Diffusion ---")
    # Параметры 
    angle = 22.5 * np.pi / 180
    A_10 = np.cos(angle)
    B_10 = np.sin(angle)
    NU_10 = 0.005
    
    # Входных измерений теперь 3 (x, y, t)
    X_f_10, X_b_10 = generate_data_tc10(N_f=2000, N_b_side=30)
    
    Y_b_10 = exact_u_tc10(X_b_10[:,0], X_b_10[:,1], X_b_10[:,2], A_10, B_10, NU_10).reshape(-1, 1)
    
    model_10 = pielm.PIELM(n_hidden=1200, input_dim=3, scale=3.0)
    model_10.fit(X_f_10, X_b_10, Y_b_10,
                 lambda W,b,X: pielm.adv_diff_2d_unsteady_operator(W,b,X, a=A_10, b_coef=B_10, nu=NU_10),
                 lambda x,y,t: np.zeros_like(x)) # R=0
                 
    # Визуализация: Делаем СРЕЗ по времени t = 0.1
    t_slice = 0.1
    print(f"Visualizing slice at t = {t_slice}...")
    
    x_lin = np.linspace(0, 1, 50)
    y_lin = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_lin, y_lin)
    
    # Создаем тестовые данные (x, y, t_fixed)
    X_flat = xx.flatten()
    Y_flat = yy.flatten()
    T_flat = np.full_like(X_flat, t_slice)
    
    X_test_10 = np.column_stack((X_flat, Y_flat, T_flat))
    
    u_pred_10 = model_10.predict(X_test_10)
    u_true_10 = exact_u_tc10(X_flat, Y_flat, T_flat, A_10, B_10, NU_10).reshape(-1, 1)
    
    # Для отрисовки передаем только (x, y) в draw_graphics_2d, так как t константа
    X_plot = np.column_stack((X_flat, Y_flat))
    
    pielm.print_info(u_pred_10, u_true_10, 1200, 2000, f"TC-10 (Slice t={t_slice})")
    pielm.draw_graphics_2d(X_plot, u_true_10, u_pred_10, title=f"TC-10: Slice at t={t_slice}")