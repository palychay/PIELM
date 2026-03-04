import numpy as np
import matplotlib.pyplot as plt

# --- 1. Словари функций активации и их производных ---
ACTIVATIONS = {
    'tanh': {
        'f':   lambda z: np.tanh(z),
        'df':  lambda z: 1 - np.tanh(z)**2,
        'ddf': lambda z: -2 * np.tanh(z) * (1 - np.tanh(z)**2)
    },
    'sin': {
        'f':   lambda z: np.sin(z),
        'df':  lambda z: np.cos(z),
        'ddf': lambda z: -np.sin(z)
    },
    'sigmoid': {
        'f':   lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),  # ИСПРАВЛЕНО: clip против overflow
        'df':  lambda z: (lambda s: s * (1 - s))(1 / (1 + np.exp(-np.clip(z, -500, 500)))),
        'ddf': lambda z: (lambda s: s * (1 - s) * (1 - 2*s))(1 / (1 + np.exp(-np.clip(z, -500, 500))))
    }
}

# --- 2. Операторы ---

def advection_operator_tc1(W, b, X, act_name='tanh'):
    Z = X @ W + b
    df = ACTIVATIONS[act_name]['df']
    return df(Z) * W

def diffusion_operator_tc2(W, b, X, act_name='tanh'):
    Z = X @ W + b
    ddf = ACTIVATIONS[act_name]['ddf']
    return ddf(Z) * (W**2)

def adv_diff_operator_tc3(W, b, X, nu, act_name='tanh'):
    Z = X @ W + b
    df  = ACTIVATIONS[act_name]['df']
    ddf = ACTIVATIONS[act_name]['ddf']
    return df(Z) * W - nu * ddf(Z) * (W**2)

def advection_2d_operator(W, b, X, a=1.0, b_coef=1.0, act_name='tanh'):
    Z = X @ W + b
    w_x, w_y = W[0, :], W[1, :]
    df = ACTIVATIONS[act_name]['df']
    return df(Z) * (a * w_x + b_coef * w_y)

def diffusion_2d_operator(W, b, X, act_name='tanh'):
    Z = X @ W + b
    w_x, w_y = W[0, :], W[1, :]
    ddf = ACTIVATIONS[act_name]['ddf']
    return ddf(Z) * (w_x**2 + w_y**2)

def advection_unsteady_1d_operator(W, b, X, a_coeff=1.0, act_name='tanh'):
    Z = X @ W + b
    w_x, w_t = W[0, :], W[1, :]
    df = ACTIVATIONS[act_name]['df']
    if isinstance(a_coeff, np.ndarray): a_coeff = a_coeff.reshape(-1, 1)
    return df(Z) * w_t + a_coeff * (df(Z) * w_x)

def adv_diff_1d_unsteady_operator(W, b, X, a=1.0, nu=0.01, act_name='tanh'):
    Z = X @ W + b
    w_x, w_t = W[0, :], W[1, :]
    df  = ACTIVATIONS[act_name]['df']
    ddf = ACTIVATIONS[act_name]['ddf']
    return df(Z) * w_t + a * (df(Z) * w_x) - nu * (ddf(Z) * (w_x**2))

def adv_diff_2d_unsteady_operator(W, b, X, a=1.0, b_coef=1.0, nu=0.01, act_name='tanh'):
    Z = X @ W + b
    w_x, w_y, w_t = W[0, :], W[1, :], W[2, :]
    df  = ACTIVATIONS[act_name]['df']
    ddf = ACTIVATIONS[act_name]['ddf']
    return df(Z) * w_t + df(Z) * (a * w_x + b_coef * w_y) - nu * ddf(Z) * (w_x**2 + w_y**2)

# --- 3. Class PIELM ---
class PIELM:
    def __init__(self, n_hidden, input_dim=1, scale=5.0, act_name='tanh',
                 lambda_pde=1.0, lambda_bc=10.0, seed=None):  # ДОБАВЛЕНО: lambda_pde, lambda_bc
        self.n_hidden   = n_hidden
        self.input_dim  = input_dim
        self.scale      = scale
        self.act_name   = act_name
        self.lambda_pde = lambda_pde   # ДОБАВЛЕНО: вес невязки PDE
        self.lambda_bc  = lambda_bc    # ДОБАВЛЕНО: вес граничных условий
        self.phi = ACTIVATIONS[act_name]['f']
        self.W    = None
        self.b    = None
        self.beta = None

        # ИСПРАВЛЕНО: изолированный RNG вместо глобального np.random.seed
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        # ДОБАВЛЕНО: инициализация весов отдельно от fit()
        # Это позволяет зафиксировать W, b до обучения и переиспользовать их.
        # Важно для ГА: фитнес оценивает гиперпараметры, а не случайность.
        self.W = self.rng.normal(0, self.scale, (self.input_dim, self.n_hidden))
        self.b = self.rng.normal(0, self.scale, (1, self.n_hidden))
        return self

    def fit(self, X_f, X_b, Y_b, operator_func, source_func):
        if self.W is None:          # ИСПРАВЛЕНО: инициализируем только если ещё не сделано
            self.initialize()

        H_f = operator_func(self.W, self.b, X_f, act_name=self.act_name)

        # ИСПРАВЛЕНО: убран хрупкий if/elif по input_dim — source_func принимает весь X
        Y_f = np.asarray(source_func(X_f)).reshape(-1, 1)

        Z_b = X_b @ self.W + self.b
        H_b = self.phi(Z_b)

        if Y_f.ndim == 1: Y_f = Y_f.reshape(-1, 1)
        if Y_b.ndim == 1: Y_b = Y_b.reshape(-1, 1)

        # ИСПРАВЛЕНО: взвешенная сборка через sqrt(lambda).
        # Математически корректно: МНК минимизирует сумму квадратов,
        # поэтому sqrt(λ) на строках эквивалентен λ в функции потерь.
        sp = np.sqrt(self.lambda_pde)
        sb = np.sqrt(self.lambda_bc)

        H = np.vstack((sp * H_f, sb * H_b))
        Y = np.vstack((sp * Y_f, sb * Y_b))

        # ИСПРАВЛЕНО: lstsq эффективнее pinv (не строит полную псевдообратную матрицу)
        self.beta, _, _, _ = np.linalg.lstsq(H, Y, rcond=1e-15)

    def predict(self, X):
        Z = X @ self.W + self.b
        return self.phi(Z) @ self.beta


# --- 4. Визуализация и инфо ---
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
    plt.title(name); plt.xlabel('x'); plt.ylabel('u(x)')
    plt.legend(); plt.grid(True); plt.show()

def draw_graphics_2d(X_test, u_true, u_pred, title="PIELM 2D Result"):
    error = np.abs(u_true - u_pred)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plots = [(u_true, "Exact Solution", "jet"),
             (u_pred, "PIELM Prediction", "jet"),
             (error,  "Absolute Error",   "viridis")]
    for i, (data, t, cmap) in enumerate(plots):
        sc = ax[i].scatter(X_test[:, 0], X_test[:, 1], c=data.flatten(), cmap=cmap, s=10)
        ax[i].set_title(t); ax[i].set_xlabel("x"); ax[i].set_ylabel("y"); ax[i].axis("equal")
        plt.colorbar(sc, ax=ax[i])
    plt.suptitle(title, fontsize=16); plt.tight_layout(); plt.show()


# --- 5. GENETIC ALGORITHM FOR PIELM ---
class GeneticOptimizer:
    def __init__(self, n_pop=20, n_gen=10,
                 scale_bounds=(0.1, 15.0),
                 hidden_bounds=(10, 500),
                 lambda_pde_bounds=(0.01, 100.0),   # ДОБАВЛЕНО: диапазон lambda_pde
                 lambda_bc_bounds=(0.1, 100.0),      # ДОБАВЛЕНО: диапазон lambda_bc
                 seed=42):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.scale_min,      self.scale_max      = scale_bounds
        self.hidden_min,     self.hidden_max     = hidden_bounds
        self.lambda_pde_min, self.lambda_pde_max = lambda_pde_bounds  # ДОБАВЛЕНО
        self.lambda_bc_min,  self.lambda_bc_max  = lambda_bc_bounds   # ДОБАВЛЕНО
        self.seed        = seed
        self.best_params = None
        self.best_loss   = float('inf')
        self.history     = []

        # ИСПРАВЛЕНО: изолированный RNG
        self.rng = np.random.default_rng(seed)

    def _init_population(self):
        acts = ['tanh', 'sin', 'sigmoid']
        pop  = []
        for _ in range(self.n_pop):
            ind = {
                'scale':      self.rng.uniform(self.scale_min, self.scale_max),
                'n_hidden':   int(self.rng.integers(self.hidden_min, self.hidden_max)),
                'activation': self.rng.choice(acts),
                # ДОБАВЛЕНО: lambda-параметры как часть хромосомы
                'lambda_pde': self.rng.uniform(self.lambda_pde_min, self.lambda_pde_max),
                'lambda_bc':  self.rng.uniform(self.lambda_bc_min,  self.lambda_bc_max),
            }
            pop.append(ind)
        return pop

    def _fitness(self, individual, X_f, X_b, Y_b, operator_func, source_func):
        model = PIELM(
            n_hidden   = individual['n_hidden'],
            scale      = individual['scale'],
            act_name   = individual['activation'],
            lambda_pde = individual['lambda_pde'],   # ДОБАВЛЕНО
            lambda_bc  = individual['lambda_bc'],    # ДОБАВЛЕНО
            input_dim  = X_f.shape[1],
        )
        try:
            model.fit(X_f, X_b, Y_b, operator_func, source_func)
            H_f = operator_func(model.W, model.b, X_f, act_name=individual['activation'])
            residual_pred = H_f @ model.beta

            # ИСПРАВЛЕНО: единый способ вызова source_func — принимает весь X
            residual_true = np.asarray(source_func(X_f)).reshape(-1, 1)

            return float(np.mean((residual_pred - residual_true)**2))
        except np.linalg.LinAlgError:
            return float('inf')

    def _mutate(self, individual):
        ind = individual.copy()
        if self.rng.random() < 0.3:
            ind['scale'] = float(np.clip(
                ind['scale'] + self.rng.normal(0, 1.0),
                self.scale_min, self.scale_max))
        if self.rng.random() < 0.3:
            ind['n_hidden'] = int(np.clip(
                ind['n_hidden'] + int(self.rng.integers(-50, 50)),
                self.hidden_min, self.hidden_max))
        if self.rng.random() < 0.2:
            ind['activation'] = self.rng.choice(['tanh', 'sin', 'sigmoid'])
        # ДОБАВЛЕНО: мутация lambda-параметров
        if self.rng.random() < 0.3:
            ind['lambda_pde'] = float(np.clip(
                ind['lambda_pde'] * self.rng.uniform(0.5, 2.0),
                self.lambda_pde_min, self.lambda_pde_max))
        if self.rng.random() < 0.3:
            ind['lambda_bc'] = float(np.clip(
                ind['lambda_bc'] * self.rng.uniform(0.5, 2.0),
                self.lambda_bc_min, self.lambda_bc_max))
        return ind

    def _crossover(self, p1, p2):
        return {
            'scale':      (p1['scale'] + p2['scale']) / 2,
            'n_hidden':   int((p1['n_hidden'] + p2['n_hidden']) / 2),
            'activation': p1['activation'] if self.rng.random() < 0.5 else p2['activation'],
            # ДОБАВЛЕНО: скрещивание lambda-параметров (среднее)
            'lambda_pde': (p1['lambda_pde'] + p2['lambda_pde']) / 2,
            'lambda_bc':  (p1['lambda_bc']  + p2['lambda_bc'])  / 2,
        }

    def search(self, X_f, X_b, Y_b, operator_func, source_func, callback=None):
        # ИСПРАВЛЕНО: убран глобальный np.random.seed — используем self.rng (уже инициализирован)
        population = self._init_population()

        for gen in range(self.n_gen):
            scores = [
                (self._fitness(ind, X_f, X_b, Y_b, operator_func, source_func), ind)
                for ind in population
            ]
            scores.sort(key=lambda x: x[0])

            if scores[0][0] < self.best_loss:
                self.best_loss, self.best_params = scores[0]

            self.history.append(self.best_loss)
            if callback: callback(gen, self.best_loss, self.best_params)

            elites  = [x[1] for x in scores[:max(1, int(self.n_pop * 0.2))]]
            new_pop = elites[:]
            while len(new_pop) < self.n_pop:
                p1 = elites[self.rng.integers(len(elites))]   # ИСПРАВЛЕНО: rng вместо np.random.choice
                p2 = elites[self.rng.integers(len(elites))]
                new_pop.append(self._mutate(self._crossover(p1, p2)))
            population = new_pop

        return self.best_params