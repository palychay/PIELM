import numpy as np
import matplotlib.pyplot as plt

# --- 1. Базовые функции ---
def phi(z):
    return np.tanh(z)

def d_phi(z):
    return 1 - np.tanh(z)**2

def dd_phi(z):
    t = np.tanh(z)
    return -2 * t * (1 - t**2)


# --- 2. Физика задачи  ---
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


def advection_operator_tc1(W, b, X):
    """
    Оператор L = d/dx
    u = sum beta * phi(wx+b)
    L[u] = sum beta * w * phi'(wx+b)
    """
    Z = X @ W + b
    # Производная внутренней функции: d(wx+b)/dx = w
    # H = phi'(Z) * W
    H_val = d_phi(Z) * W 
    return H_val


def diffusion_operator_tc2(W, b, X):
    """
    Оператор: d^2/dx^2
    L[u] = sum beta * w^2 * phi''(wx+b)
    """
    Z = X @ W + b
    H_val = dd_phi(Z) * (W**2)
    return H_val


def adv_diff_operator_tc3(W, b, X, nu):
    """
    Оператор: u_x - nu * u_xx
    """
    Z = X @ W + b
    
    # Первая производная: w * phi'
    d_val = d_phi(Z) * W
    
    # Вторая производная: w^2 * phi''
    dd_val = dd_phi(Z) * (W**2)
    
    # Комбинируем: u_x - nu*u_xx
    H_val = d_val - nu * dd_val
    return H_val

#------3 Class PIELM------------------------------------------
class PIELM:
    def __init__(self, n_hidden, scale=5.0):
        self.n_hidden = n_hidden
        self.scale = scale
        self.W = None
        self.b = None
        self.beta = None
        
    def fit(self, X_f, X_b, Y_b, operator_func, source_func):
        # 1. Инициализация весов
        self.W = np.random.normal(0, self.scale, (1, self.n_hidden))
        self.b = np.random.normal(0, self.scale, (1, self.n_hidden))
        
        # 2. Матрица для физики (внутренние точки)
        # operator_func должен возвращать H_f на основе W и b
        H_f = operator_func(self.W, self.b, X_f)
        Y_f = source_func(X_f)
        
        # 3. Матрица для граничных условий
        Z_b = X_b @ self.W + self.b
        H_b = phi(Z_b)
        # Y_b уже содержит значения на границах
        
        # 4. Сборка глобальной системы H * beta = Y
        H = np.vstack((H_f, H_b))
        Y = np.vstack((Y_f, Y_b))
        
        # 5. Решение методом наименьших квадратов
        self.beta = np.linalg.pinv(H, rcond=1e-15) @ Y
        
    def predict(self, X):
        Z = X @ self.W + self.b
        return phi(Z) @ self.beta
 #--------------------------------------------------------------   


def print_info(u_pred, u_true, N_hidden, N_f):
    mse = np.mean((u_pred - u_true)**2)
    print(f"Test Case 1 (Advection Equation)")
    print(f"Neurons: {N_hidden}, Collocation Points: {N_f}")
    print(f"MSE Error: {mse:.2e}")

def draw_graphics(X_test, u_true, u_pred, x_f, name):
    plt.figure(figsize=(10, 5))
    plt.plot(X_test, u_true, 'b-', label='Exact Solution', linewidth=2)
    plt.plot(X_test, u_pred, 'r--', label='PIELM Prediction', linewidth=2, dashes=(4, 4))
    plt.scatter(x_f, np.zeros_like(x_f), color='green', marker='|', s=50, label='Collocation Pts')
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()




# 4. Настройки
N_f = 40       # Точки коллокации
N_bc = 2       # Граничные точки
N_hidden = 42  # Количество нейронов (N_f + N_bc)

x_f = np.linspace(0, 1, N_f).reshape(-1, 1)
x_b = np.array([[0.0], [1.0]])

# --- ТЕСТ TC-1 (Адвекция/Первая производная) ---
model_tc1 = PIELM(n_hidden=N_hidden)
# Граничное условие из точного решения u(0)
y_b_tc1 = exact_u_tc1(x_b) 
model_tc1.fit(x_f, x_b, y_b_tc1, 
              operator_func=advection_operator_tc1, 
              source_func=source_R_tc1)

# Проверка на тестовой сетке
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
u_pred = model_tc1.predict(X_test)
u_true = exact_u_tc1(X_test)

print_info(u_pred ,u_true, N_hidden, N_f)
draw_graphics(X_test, u_true, u_pred, x_f, 'TC-1: 1D Advection Equation')


# --- ТЕСТ TC-2 (Диффузия/Вторая производная) ---
model_tc2 = PIELM(n_hidden=N_hidden)
y_b_tc2 = exact_u_tc2(x_b)
model_tc2.fit(x_f, x_b, y_b_tc2, 
              operator_func=diffusion_operator_tc2, 
              source_func=source_R_tc2)

u_pred_tc2 = model_tc2.predict(X_test)
u_true_tc2 = exact_u_tc2(X_test)

print_info(u_pred_tc2 ,u_true_tc2, N_hidden, N_f)
draw_graphics(X_test, u_true_tc2, u_pred_tc2, x_f, 'TC-2: Diffusion Equation')


# --- ТЕСТ TC-3 (Адвекция-Диффузия с параметром nu) ---
nu_val = 0.2
N_f_tc3 = 20        # Всего 20 точек внутри
N_bc_tc3 = 2
N_hidden_tc3 = 22

X_f_tc3 = np.linspace(0, 1, N_f_tc3).reshape(-1, 1)
X_b_tc3 = np.array([[0.0], [1.0]])
Y_b_tc3 = np.array([[0.0], [1.0]])

model_tc3 = PIELM(n_hidden=N_hidden_tc3, scale=5.0)
# Используем lambda, чтобы "пробросить" nu в оператор
model_tc3.fit(X_f_tc3, X_b_tc3, Y_b_tc3, 
              operator_func=lambda W, b, X: adv_diff_operator_tc3(W, b, X, nu_val), 
              source_func=lambda x: np.zeros_like(x)) # В TC-3 правая часть 0

u_pred_tc3 = model_tc3.predict(X_test)
u_true_tc3 = exact_u_tc3(X_test, nu_val)

print_info(u_pred_tc3 ,u_true_tc3, N_hidden_tc3, N_f_tc3)
draw_graphics(X_test, u_true_tc3, u_pred_tc3, X_f_tc3, 'TC-3: Advection-Diffusion')
