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

#------2 Class PIELM------------------------------------------
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