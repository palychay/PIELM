import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Функции активации
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATIONS = {
    'tanh': {
        'f':   lambda z: np.tanh(z),
        'df':  lambda z: 1.0 - np.tanh(z)**2,
        'ddf': lambda z: -2.0 * np.tanh(z) * (1.0 - np.tanh(z)**2),
    },
    'sin': {
        'f':   lambda z: np.sin(z),
        'df':  lambda z: np.cos(z),
        'ddf': lambda z: -np.sin(z),
    },
    'sigmoid': {
        'f':   lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
        'df':  lambda z: (
            lambda s: s * (1.0 - s)
        )(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))),
        'ddf': lambda z: (
            lambda s: s * (1.0 - s) * (1.0 - 2.0 * s)
        )(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Разбивка train / test
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_colloc(X_f, test_size=0.25, seed=0):
    """
    Случайное разбиение коллокационных точек 75% / 25%.

    Параметры
    ----------
    X_f       : ndarray (N, d)   — все коллокационные точки
    test_size : float            — доля тестовой выборки
    seed      : int              — зерно генератора

    Возвращает
    ----------
    X_train, X_test, train_idx, test_idx
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(X_f))
    n_test = max(1, int(len(X_f) * test_size))
    return (
        X_f[idx[n_test:]],
        X_f[idx[:n_test]],
        idx[n_test:],
        idx[:n_test],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Класс PIELM
# ─────────────────────────────────────────────────────────────────────────────

class PIELM:
    """
    Physics-Informed Extreme Learning Machine.

    Поддерживает два режима:
      - 2D (x, y)      — стационарное уравнение Дарси
      - 3D (x, y, t)   — нестационарное уравнение фильтрации

    Параметры
    ----------
    n_hidden    : int    — число нейронов скрытого слоя
    input_dim   : int    — размерность входа (2 или 3)
    scale       : float  — масштаб инициализации весов
    act_name    : str    — имя функции активации ('tanh', 'sin', 'sigmoid')
    lambda_pde  : float  — вес уравнения PDE в МНК
    lambda_bc   : float  — вес граничных условий в МНК
    seed        : int    — зерно генератора
    """

    def __init__(self, n_hidden=200, input_dim=2, scale=5.0,
                 act_name='tanh', lambda_pde=1.0, lambda_bc=10.0,
                 seed=None):
        self.n_hidden   = n_hidden
        self.input_dim  = input_dim
        self.scale      = scale
        self.act_name   = act_name
        self.lambda_pde = lambda_pde
        self.lambda_bc  = lambda_bc
        self.phi        = ACTIVATIONS[act_name]['f']
        self.W          = None
        self.b          = None
        self.beta       = None
        self.rng        = np.random.default_rng(seed)

    # ── инициализация случайных весов ──────────────────────────────────────

    def initialize(self):
        """Случайная инициализация фиксированных весов W и b."""
        self.W = self.rng.normal(
            0, self.scale, (self.input_dim, self.n_hidden)
        )
        self.b = self.rng.normal(
            0, self.scale, (1, self.n_hidden)
        )
        return self

    # ── обучение МНК ───────────────────────────────────────────────────────

    def fit(self, X_train, X_b, Y_b, operator_func, source_func):
        """
        Обучение методом наименьших квадратов на тренировочных точках.

        Параметры
        ----------
        X_train       : ndarray (N_train, d)  — внутренние коллокационные точки
        X_b           : ndarray (N_b, d)      — граничные точки
        Y_b           : ndarray (N_b,)        — значения P на границе
        operator_func : callable              — L[φ](X_train) → (N_train, n_hidden)
                        Сигнатура: operator_func(W, b, X, act_name, act_dict)
        source_func   : callable              — q(X_train) → (N_train,)

        Возвращает
        ----------
        self
        """
        if self.W is None:
            self.initialize()

        # Матрица коллокационных уравнений (PDE)
        H_f = operator_func(
            self.W, self.b, X_train,
            act_name=self.act_name,
            act_dict=ACTIVATIONS,
        )                                           # (N_train, n_hidden)

        Y_f = np.asarray(source_func(X_train)).reshape(-1, 1)  # (N_train, 1)

        # Матрица граничных условий
        H_b = self.phi(X_b @ self.W + self.b)      # (N_b, n_hidden)
        Y_b = np.asarray(Y_b).reshape(-1, 1)       # (N_b, 1)

        # Взвешенная система МНК
        sp = np.sqrt(self.lambda_pde)
        sb = np.sqrt(self.lambda_bc)

        H = np.vstack([sp * H_f, sb * H_b])        # (N_train+N_b, n_hidden)
        Y = np.vstack([sp * Y_f, sb * Y_b])        # (N_train+N_b, 1)

        self.beta, _, _, _ = np.linalg.lstsq(H, Y, rcond=1e-15)
        return self

    # ── предсказание ───────────────────────────────────────────────────────

    def predict(self, X):
        """
        Предсказание P в точках X.

        Параметры
        ----------
        X : ndarray (N, d)

        Возвращает
        ----------
        P_pred : ndarray (N,)
        """
        return (self.phi(X @ self.W + self.b) @ self.beta).ravel()

    # ── метрики ────────────────────────────────────────────────────────────

    def rmse(self, X, y_true):
        """
        RMSE между предсказанием и эталонным решением.

        Параметры
        ----------
        X      : ndarray (N, d)
        y_true : ndarray (N,) или callable

        Возвращает
        ----------
        rmse : float
        """
        y_ref = y_true(X) if callable(y_true) else np.asarray(y_true).ravel()
        return float(np.sqrt(np.mean((self.predict(X) - y_ref) ** 2)))

    def rmse_pde(self, X, operator_func, source_func):
        """
        RMSE невязки PDE в точках X (residual loss).

        Параметры
        ----------
        X             : ndarray (N, d)
        operator_func : callable
        source_func   : callable

        Возвращает
        ----------
        rmse_res : float
        """
        H_f   = operator_func(
            self.W, self.b, X,
            act_name=self.act_name,
            act_dict=ACTIVATIONS,
        )
        L_phi = (H_f @ self.beta).ravel()          # L[P] предсказанное
        q     = np.asarray(source_func(X)).ravel() # q(x,y) или q(x,y,t)
        return float(np.sqrt(np.mean((L_phi - q) ** 2)))

    # ── информация о модели ────────────────────────────────────────────────

    def summary(self):
        """
        Возвращает словарь с параметрами модели.
        """
        return {
            'n_hidden':   self.n_hidden,
            'input_dim':  self.input_dim,
            'scale':      self.scale,
            'activation': self.act_name,
            'lambda_pde': self.lambda_pde,
            'lambda_bc':  self.lambda_bc,
        }