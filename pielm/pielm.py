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


# ---2. 2D Операторы (Новые для TC-4, 5, 6) ---
def advection_2d_operator(W, b, X, a=1.0, b_coef=1.0):
    """
    Уравнение (37): a * u_x + b * u_y = R
    W имеет форму (2, n_hidden). 
    W[0, :] - веса для x (w_x)
    W[1, :] - веса для y (w_y)
    """
    Z = X @ W + b
    
    w_x = W[0, :]
    w_y = W[1, :]
    
    # L[u] = phi'(Z) * (a * w_x + b * w_y)
    H_val = d_phi(Z) * (a * w_x + b_coef * w_y)
    return H_val

def diffusion_2d_operator(W, b, X):
    """
    Уравнение: u_xx + u_yy = R
    """
    Z = X @ W + b
    
    w_x = W[0, :]
    w_y = W[1, :]
    
    # Вторая производная сложной функции:
    # u_xx = phi''(Z) * w_x^2
    # u_yy = phi''(Z) * w_y^2
    # L[u] = phi''(Z) * (w_x^2 + w_y^2)
    H_val = dd_phi(Z) * (w_x**2 + w_y**2)
    return H_val



def advection_unsteady_1d_operator(W, b, X, a_coeff=1.0):
    """
    Решает уравнение: u_t + a(x) * u_x = 0
    X: (N, 2), где X[:,0] - это x, X[:,1] - это t
    a_coeff: может быть числом (для TC-7) или массивом той же длины, что и X (для TC-8)
    """
    Z = X @ W + b
    
    # W имеет форму (2, N_hidden)
    # W[0, :] -> веса для x
    # W[1, :] -> веса для t
    w_x = W[0, :]
    w_t = W[1, :]
    
    # Производная по времени u_t
    dt_val = d_phi(Z) * w_t
    
    # Производная по пространству u_x
    dx_val = d_phi(Z) * w_x
    
    # Если коэффициент переменный (массив), нужно сделать reshape для broadcasting
    if isinstance(a_coeff, np.ndarray):
        a_coeff = a_coeff.reshape(-1, 1)
        
    # L[u] = u_t + a(x)*u_x
    return dt_val + a_coeff * dx_val


# Оператор для TC-9 (1D Unsteady Advection-Diffusion)
# Уравнение: u_t + a*u_x = nu * u_xx
def adv_diff_1d_unsteady_operator(W, b, X, a=1.0, nu=0.01):
    Z = X @ W + b
    w_x = W[0, :]
    w_t = W[1, :]
    
    dt_val = d_phi(Z) * w_t           # u_t
    dx_val = d_phi(Z) * w_x           # u_x
    ddx_val = dd_phi(Z) * (w_x**2)    # u_xx
    
    # u_t + a*u_x - nu*u_xx
    return dt_val + a * dx_val - nu * ddx_val

# Оператор для TC-10 (2D Unsteady Advection-Diffusion)
# Уравнение: u_t + a*u_x + b*u_y = nu * (u_xx + u_yy)
def adv_diff_2d_unsteady_operator(W, b, X, a=1.0, b_coef=1.0, nu=0.01):
    Z = X @ W + b
    w_x = W[0, :]
    w_y = W[1, :]
    w_t = W[2, :]
    
    dt_val = d_phi(Z) * w_t
    
    d_adv = d_phi(Z) * (a * w_x + b_coef * w_y)      # a*u_x + b*u_y
    d_diff = dd_phi(Z) * (w_x**2 + w_y**2)           # u_xx + u_yy
    
    # u_t + Adv - nu*Diff
    return dt_val + d_adv - nu * d_diff



#------3 Class PIELM------------------------------------------
class PIELM:
    def __init__(self, n_hidden, input_dim=1, scale=5.0):
        self.n_hidden = n_hidden
        self.input_dim = input_dim # 1 для 1D, 2 для 2D
        self.scale = scale
        self.W = None
        self.b = None
        self.beta = None
        
    def fit(self, X_f, X_b, Y_b, operator_func, source_func):
        """
        X_f: Точки коллокации (N_f, input_dim)
        X_b: Граничные точки (N_b, input_dim)
        Y_b: Значения на границе (N_b, 1)
        """
        # 1. Инициализация весов
        self.W = np.random.normal(0, self.scale, (self.input_dim, self.n_hidden))
        self.b = np.random.normal(0, self.scale, (1, self.n_hidden))
        
        # 2. Матрица для физики (внутренние точки)
        # operator_func возвращает H_f на основе W и b
        H_f = operator_func(self.W, self.b, X_f)

        # Вычисляем правую часть R(x, y)
        if self.input_dim == 1:
            Y_f = source_func(X_f)
        elif self.input_dim == 2:
            # Для 2D передаем x и y отдельно
            Y_f = source_func(X_f[:, 0], X_f[:, 1]).reshape(-1, 1)
        elif self.input_dim == 3:
            # Для TC-10: x, y, t
            Y_f = source_func(X_f[:, 0], X_f[:, 1], X_f[:, 2]).reshape(-1, 1)
        
        # 3. Матрица для граничных условий
        Z_b = X_b @ self.W + self.b
        H_b = phi(Z_b)
        
        # 4. Сборка глобальной системы H * beta = Y
        H = np.vstack((H_f, H_b))
        Y = np.vstack((Y_f, Y_b))
        
        # 5. Решение методом наименьших квадратов
        self.beta = np.linalg.pinv(H, rcond=1e-15) @ Y
        
    def predict(self, X):
        Z = X @ self.W + self.b
        return phi(Z) @ self.beta
 #--------------------------------------------------------------   


def print_info(u_pred, u_true, N_hidden, N_f, title="Test Case"):
    mse = np.mean((u_pred - u_true)**2)
    print(f"--- {title} ---")
    print(f"Neurons: {N_hidden}, Collocation Points: {N_f}")
    print(f"MSE Error: {mse:.2e}")
    print("-" * 30)

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

def draw_graphics_2d(X_test, u_true, u_pred, title="PIELM 2D Result"):
    """
    Универсальная отрисовка для 2D задач.
    Сравнивает точное решение, результат модели и показывает карту ошибок.
    """
    error = np.abs(u_true - u_pred)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Список для итерации: (данные, заголовок, цветовая карта)
    plots = [
        (u_true, "Exact Solution", "jet"),
        (u_pred, "PIELM Prediction", "jet"),
        (error, "Absolute Error", "viridis")
    ]
    
    for i, (data, t, cmap) in enumerate(plots):
        # Используем scatter, так как X_test может быть произвольной формой (как звезда)
        sc = ax[i].scatter(X_test[:, 0], X_test[:, 1], c=data.flatten(), cmap=cmap, s=10)
        ax[i].set_title(t)
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("y")
        ax[i].axis("equal")
        plt.colorbar(sc, ax=ax[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()




# -----------------------------------------------------------
#  GENETIC ALGORITHM FOR PIELM
# -----------------------------------------------------------
class GeneticOptimizer:
    def __init__(self, n_pop=20, n_gen=10, 
                 scale_bounds=(0.1, 15.0), 
                 hidden_bounds=(10, 500)):
        """
        n_pop: размер популяции (сколько моделей проверяем за раз)
        n_gen: количество поколений
        scale_bounds: (min, max) для scale
        hidden_bounds: (min, max) для количества нейронов
        """
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.scale_min, self.scale_max = scale_bounds
        self.hidden_min, self.hidden_max = hidden_bounds
        self.best_params = None
        self.best_loss = float('inf')
        self.history = []

    def _init_population(self):
        # Популяция: список словарей {'scale': float, 'n_hidden': int}
        pop = []
        for _ in range(self.n_pop):
            ind = {
                'scale': np.random.uniform(self.scale_min, self.scale_max),
                'n_hidden': int(np.random.randint(self.hidden_min, self.hidden_max))
            }
            pop.append(ind)
        return pop

    def _fitness(self, individual, X_f, X_b, Y_b, operator_func, source_func):
        # 1. Создаем модель с генами индивида
        model = PIELM(n_hidden=individual['n_hidden'], scale=individual['scale'])
        
        # 2. Обучаем
        try:
            model.fit(X_f, X_b, Y_b, operator_func, source_func)
            
            # 3. Считаем ошибку (Loss)
            # L[u] - R(x) должно быть равно 0.
            
            # Предсказываем веса и производные через оператор
            # H_f * beta ≈ Y_f
            H_f = operator_func(model.W, model.b, X_f)
            residual_pred = H_f @ model.beta
            residual_true = source_func(X_f)
            if residual_true.ndim == 1: residual_true = residual_true.reshape(-1, 1)
                
            mse_res = np.mean((residual_pred - residual_true)**2)
            return mse_res
            
        except np.linalg.LinAlgError:
            return float('inf') # Убиваем особи с ошибками

    def _mutate(self, individual):
        # Мутация scale (сдвиг на случайную величину)
        if np.random.rand() < 0.3:
            individual['scale'] += np.random.normal(0, 1.0)
            individual['scale'] = np.clip(individual['scale'], self.scale_min, self.scale_max)
        
        # Мутация n_hidden
        if np.random.rand() < 0.3:
            individual['n_hidden'] += int(np.random.randint(-50, 50))
            individual['n_hidden'] = np.clip(individual['n_hidden'], self.hidden_min, self.hidden_max)
        
        return individual

    def _crossover(self, p1, p2):
        # Скрещивание: берем среднее или случайный выбор
        child = {
            'scale': (p1['scale'] + p2['scale']) / 2,
            'n_hidden': int((p1['n_hidden'] + p2['n_hidden']) / 2)
        }
        return child

    def search(self, X_f, X_b, Y_b, operator_func, source_func, callback=None):
        population = self._init_population()
        
        for gen in range(self.n_gen):
            # Оценка
            scores = []
            for ind in population:
                loss = self._fitness(ind, X_f, X_b, Y_b, operator_func, source_func)
                scores.append((loss, ind))
            
            # Сортировка (меньше ошибка -> лучше)
            scores.sort(key=lambda x: x[0])
            
            # Лучший в поколении
            current_best_loss, current_best_params = scores[0]
            if current_best_loss < self.best_loss:
                self.best_loss = current_best_loss
                self.best_params = current_best_params
            
            self.history.append(current_best_loss)
            
            if callback:
                callback(gen, current_best_loss, current_best_params)
            
            # Селекция (Элитизм: берем топ 20%)
            n_elites = int(self.n_pop * 0.2)
            elites = [x[1] for x in scores[:n_elites]]
            
            # Создание нового поколения
            new_pop = elites[:]
            while len(new_pop) < self.n_pop:
                # Случайные родители из элиты
                p1 = elites[np.random.randint(0, len(elites))]
                p2 = elites[np.random.randint(0, len(elites))]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
        return self.best_params