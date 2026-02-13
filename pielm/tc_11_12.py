import numpy as np
import matplotlib.pyplot as plt
import pielm

# ==========================================
# ЧАСТЬ 5: Стресс-тесты (TC-11, TC-12)
# ==========================================

# --- Генератор данных 1D (стандартный) ---
def generate_1d_data(N_f):
    X_f = np.random.uniform(0, 1, (N_f, 1))
    # Границы: x=0 и x=1
    X_b = np.array([[0.0], [1.0]])
    return X_f, X_b

# --- TC-11: High Gradient ---
# Eq 63: -nu * u_xx + u_x = 0
# Exact: (exp(x/nu) - 1) / (exp(1/nu) - 1)
# BC: u(0)=0, u(1)=1 (или наоборот, в статье u(0)=0, u(1)=1)
def exact_u_tc11(x, nu):
    # Используем expm1 чтобы не было переполнения
    try:
        return np.expm1(x / nu) / np.expm1(1.0 / nu)
    except OverflowError:
        # Если nu очень мало, экспонента огромная
        return np.exp((x - 1) / nu)

# --- TC-12: High Frequency ---
# Eq 66: -nu * u_xx + u_x = f(x)
# Exact: u = sin(k * pi * x)
def exact_u_tc12(x, k):
    return np.sin(k * np.pi * x)

def source_R_tc12(x, k, nu):
    # Подставляем u = sin(k*pi*x) в уравнение -nu*u_xx + u_x
    # u_x = k*pi * cos(k*pi*x)
    # u_xx = -(k*pi)^2 * sin(k*pi*x)
    # R = -nu * (-(k*pi)^2 * sin) + (k*pi * cos)
    term1 = nu * (k * np.pi)**2 * np.sin(k * np.pi * x)
    term2 = k * np.pi * np.cos(k * np.pi * x)
    return term1 + term2


if __name__ == "__main__":
    
    # Сетка для графиков
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)

    # ==========================
    # TC-11: Sharp Gradient
    # ==========================
    print("\n--- TC-11: High Gradient Problem ---")
    # Пробуем nu = 0.01 (умеренно сложно) и nu = 0.001 (очень сложно)
    NU_11 = 0.05 
    N_hidden = 1000
    N_f = 500
    
    X_f, X_b = generate_1d_data(N_f)
    Y_b = exact_u_tc11(X_b, NU_11)
    
    # Важный момент: для резких функций scale должен быть выше!
    model_11 = pielm.PIELM(n_hidden=N_hidden, input_dim=1, scale=10.0)
    
    # Оператор: u_x - nu*u_xx = 0 (Это наш adv_diff_operator_tc3)
    # Обратите внимание: в статье "-nu*u_xx + u_x", это то же самое.
    model_11.fit(X_f, X_b, Y_b,
                 lambda W,b,X: pielm.adv_diff_operator_tc3(W,b,X, nu=NU_11),
                 lambda x: np.zeros_like(x)) # R=0
    
    u_pred_11 = model_11.predict(X_test)
    u_true_11 = exact_u_tc11(X_test, NU_11)
    
    pielm.print_info(u_pred_11, u_true_11, N_hidden, N_f, f"TC-11 (nu={NU_11})")
    pielm.draw_graphics(X_test, u_true_11, u_pred_11, X_f, f"TC-11: Gradient (nu={NU_11})")


    # ==========================
    # TC-12: High Frequency
    # ==========================
    print("\n--- TC-12: High Frequency Problem ---")
    k_freq = 8.0 # Частота колебаний (в статье тестировали k=4, k=8, k=16)
    NU_12 = 0.0001 # Очень малая вязкость
    
    X_f, X_b = generate_1d_data(1000) # Нужно больше точек, чтобы поймать волны
    Y_b = exact_u_tc12(X_b, k_freq)
    
    # Scale еще выше, чтобы нейроны были "резкими"
    model_12 = pielm.PIELM(n_hidden=1500, input_dim=1, scale=15.0)
    
    model_12.fit(X_f, X_b, Y_b,
                 lambda W,b,X: pielm.adv_diff_operator_tc3(W,b,X, nu=NU_12),
                 lambda x: source_R_tc12(x, k=k_freq, nu=NU_12))
    
    u_pred_12 = model_12.predict(X_test)
    u_true_12 = exact_u_tc12(X_test, k_freq)
    
    pielm.print_info(u_pred_12, u_true_12, 1500, 1000, f"TC-12 (Freq k={k_freq})")
    pielm.draw_graphics(X_test, u_true_12, u_pred_12, X_f, f"TC-12: Frequency k={k_freq}")