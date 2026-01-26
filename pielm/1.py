import numpy as np
import matplotlib.pyplot as plt

# --- 1. Активация и производные ---
def phi(z):
    return np.tanh(z)

def d_phi(z):
    # d/dz tanh(z) = 1 - tanh^2(z)
    return 1 - np.tanh(z)**2

# --- 2. Определение точного решения и операторов ---

def exact_u(x):
    return 5 * x**2

def source_R(x):
    return 10 * x

def advection_operator(W, b, X):
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

# --- 3. Класс PIELM (с псевдообращением) ---

class PIELM_TC1:
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        self.W = None
        self.b = None
        self.beta = None
        
    def fit(self, X_f, Y_f, X_b, Y_b):
        # Инициализация весов
        # Для осциллирующих функций (sin/cos) нужен масштаб побольше, 
        # чтобы нейроны могли "поймать" частоту.
        scale = 10.0  
        self.W = np.random.normal(0, scale, (1, self.n_hidden))
        self.b = np.random.normal(0, scale, (1, self.n_hidden))
        
        # Матрица физики (в точках коллокации)
        H_f = advection_operator(self.W, self.b, X_f)
        
        # Матрица границ (в граничных точках)
        Z_b = X_b @ self.W + self.b
        H_b = phi(Z_b)
        
        # Сборка системы
        H = np.vstack((H_f, H_b))
        Y = np.vstack((Y_f, Y_b))
        
        # Решение через псевдообращение (pinv)
        # Это точный аналог формулы beta = H_dagger * T из статьи
        self.beta = np.linalg.pinv(H) @ Y
        
    def predict(self, X):
        Z = X @ self.W + self.b
        return phi(Z) @ self.beta

# --- 4. Запуск эксперимента (как в статье) ---

# Параметры из таблицы 2 для TC-1
N_f = 40       # Точки коллокации
N_bc = 2       # Граничные точки
N_hidden = 42  # Количество нейронов (N_f + N_bc)

# Генерация данных
X_f = np.linspace(0, 1, N_f).reshape(-1, 1) # Интервал [0, 1]
Y_f = source_R(X_f)

X_b = np.array([[0.0], [1.0]])
Y_b = exact_u(X_b) 

# Обучение
model = PIELM_TC1(n_hidden=N_hidden)
model.fit(X_f, Y_f, X_b, Y_b)

# Проверка на тестовой сетке
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
u_pred = model.predict(X_test)
u_true = exact_u(X_test)

# Вывод ошибки
mse = np.mean((u_pred - u_true)**2)
print(f"Test Case 1 (Advection Equation)")
print(f"Neurons: {N_hidden}, Collocation Points: {N_f}")
print(f"MSE Error: {mse:.2e}")

# Визуализация
plt.figure(figsize=(10, 5))
plt.plot(X_test, u_true, 'b-', label='Exact', linewidth=2)
plt.plot(X_test, u_pred, 'r--', label='PIELM (pinv)', linewidth=2)
plt.scatter(X_f, np.zeros_like(X_f), color='gray', marker='|', alpha=0.5, label='Collocation pts')
plt.title('TC-1: 1D Advection Equation')
plt.legend()
plt.grid(True)
plt.show()