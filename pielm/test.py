import numpy as np
import pielm

# --- 1. Физика задачи  ---
def exact_u_tc1(x):
    # Из TC-1: sin(2*pi*x)*cos(4*pi*x) + 1
    return np.sin(2*np.pi*x) * np.cos(4*np.pi*x) + 1

def exact_u_tc2(x):
    # u = sin(pi*x/2) * cos(2*pi*x) + 1
    return np.sin(np.pi * x / 2) * np.cos(2 * np.pi * x) + 1

def exact_u_tc3(x, nu):
    # u(x) = (exp(x/nu) - 1) / (exp(1/nu) - 1)
    # Используем expm1 для точности при малых числах
    numerator = np.expm1(x / nu)
    denominator = np.expm1(1.0 / nu)
    return numerator / denominator


def source_R_tc1(x):
    # Производная du/dx = R(x)
    # d/dx [sin(2wx)cos(4wx) + 1] = 2w*cos(2wx)cos(4wx) - 4w*sin(2wx)sin(4wx)
    # где w = pi
    term1 = 2*np.pi * np.cos(2*np.pi*x) * np.cos(4*np.pi*x)
    term2 = -4*np.pi * np.sin(2*np.pi*x) * np.sin(4*np.pi*x)
    return term1 + term2

def source_R_tc2(x):
    # Нам нужно R(x) = u''(x).
    # Пусть u = f(x) * g(x) + 1
    # u' = f'g + fg'
    # u'' = f''g + 2f'g' + fg''
    
    # f = sin(pi*x/2), g = cos(2*pi*x)
    # Аргументы для краткости
    a = np.pi * x / 2
    b = 2 * np.pi * x
    
    # Сами функции
    f = np.sin(a)
    g = np.cos(b)
    
    # Первые производные
    # f' = (pi/2) * cos(a)
    df = (np.pi / 2) * np.cos(a)
    # g' = -2pi * sin(b)
    dg = -2 * np.pi * np.sin(b)
    
    # Вторые производные
    # f'' = -(pi/2)^2 * sin(a)
    ddf = -(np.pi / 2)**2 * np.sin(a)
    # g'' = -(2pi)^2 * cos(b)
    ddg = -(2 * np.pi)**2 * np.cos(b)
    
    # Сборка по формуле Лейбница: f''g + 2f'g' + fg''
    term1 = ddf * g
    term2 = 2 * df * dg
    term3 = f * ddg
    
    return term1 + term2 + term3



# 2. Настройки
N_f = 40       # Точки коллокации
N_bc = 2       # Граничные точки
N_hidden = 42  # Количество нейронов (N_f + N_bc)

x_f = np.linspace(0, 1, N_f).reshape(-1, 1)
x_b = np.array([[0.0], [1.0]])

# --- ТЕСТ TC-1 (Адвекция/Первая производная) ---
model_tc1 = pielm.PIELM(n_hidden=N_hidden)
# Граничное условие из точного решения u(0)
y_b_tc1 = exact_u_tc1(x_b) 
model_tc1.fit(x_f, x_b, y_b_tc1, 
              operator_func=pielm.advection_operator_tc1, 
              source_func=source_R_tc1)

# Проверка на тестовой сетке
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
u_pred = model_tc1.predict(X_test)
u_true = exact_u_tc1(X_test)

pielm.print_info(u_pred ,u_true, N_hidden, N_f)
pielm.draw_graphics(X_test, u_true, u_pred, x_f, 'TC-1: 1D Advection Equation')


# --- ТЕСТ TC-2 (Диффузия/Вторая производная) ---
model_tc2 = pielm.PIELM(n_hidden=N_hidden)
y_b_tc2 = exact_u_tc2(x_b)
model_tc2.fit(x_f, x_b, y_b_tc2, 
              operator_func=pielm.diffusion_operator_tc2, 
              source_func=source_R_tc2)

u_pred_tc2 = model_tc2.predict(X_test)
u_true_tc2 = exact_u_tc2(X_test)

pielm.print_info(u_pred_tc2 ,u_true_tc2, N_hidden, N_f)
pielm.draw_graphics(X_test, u_true_tc2, u_pred_tc2, x_f, 'TC-2: Diffusion Equation')


# --- ТЕСТ TC-3 (Адвекция-Диффузия с параметром nu) ---
nu_val = 0.2
N_f_tc3 = 20        # Всего 20 точек внутри
N_bc_tc3 = 2
N_hidden_tc3 = 22

X_f_tc3 = np.linspace(0, 1, N_f_tc3).reshape(-1, 1)
X_b_tc3 = np.array([[0.0], [1.0]])
Y_b_tc3 = np.array([[0.0], [1.0]])

model_tc3 = pielm.PIELM(n_hidden=N_hidden_tc3, scale=5.0)
# Используем lambda, чтобы "пробросить" nu в оператор
model_tc3.fit(X_f_tc3, X_b_tc3, Y_b_tc3, 
              operator_func=lambda W, b, X: pielm.adv_diff_operator_tc3(W, b, X, nu_val), 
              source_func=lambda x: np.zeros_like(x)) # В TC-3 правая часть 0

u_pred_tc3 = model_tc3.predict(X_test)
u_true_tc3 = exact_u_tc3(X_test, nu_val)

pielm.print_info(u_pred_tc3 ,u_true_tc3, N_hidden_tc3, N_f_tc3)
pielm.draw_graphics(X_test, u_true_tc3, u_pred_tc3, X_f_tc3, 'TC-3: Advection-Diffusion')
