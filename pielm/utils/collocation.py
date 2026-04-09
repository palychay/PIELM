import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Генерация коллокационных точек
#
# Поддерживаются три стратегии размещения точек:
#   1. Равномерная сетка   (grid)
#   2. Случайная выборка   (random)
#   3. Квазислучайная      (sobol — последовательность Соболя)
#
# Для уравнения Пуассона:   X ∈ [0,1]², shape (N, 2)
# Для пьезопроводности:     X ∈ [0,1]²×[0,T], shape (N, 3)
# ─────────────────────────────────────────────────────────────────────────────


# ── 2D: уравнение Пуассона ────────────────────────────────────────────────────

def collocation_grid_2d(n_side):
    """
    Равномерная сетка внутренних точек [0,1]²
    (без граничных точек — они задаются отдельно).

    Параметры
    ----------
    n_side : int — число точек по каждому направлению

    Возвращает
    ----------
    X : ndarray (n_side², 2)
    """
    h = 1.0 / (n_side + 1)
    coords = np.linspace(h, 1.0 - h, n_side)
    xx, yy = np.meshgrid(coords, coords)
    X = np.column_stack([xx.ravel(), yy.ravel()])
    return X


def collocation_random_2d(n_points, seed=0):
    """
    Случайные внутренние точки в (0,1)².

    Параметры
    ----------
    n_points : int — число точек
    seed     : int — зерно генератора

    Возвращает
    ----------
    X : ndarray (n_points, 2)
    """
    rng = np.random.default_rng(seed)
    X   = rng.uniform(0.0, 1.0, (n_points, 2))
    # Убираем точки слишком близко к границе (< 1e-3)
    mask = np.all((X > 1e-3) & (X < 1.0 - 1e-3), axis=1)
    X    = X[mask]
    # Если после фильтрации точек меньше — добираем
    while len(X) < n_points:
        extra = rng.uniform(1e-3, 1.0 - 1e-3, (n_points - len(X), 2))
        X = np.vstack([X, extra])
    return X[:n_points]


def collocation_sobol_2d(n_points, seed=0):
    """
    Квазислучайные точки (последовательность Соболя) в (0,1)².
    Обеспечивают более равномерное покрытие, чем случайные.

    Параметры
    ----------
    n_points : int — число точек
    seed     : int — зерно

    Возвращает
    ----------
    X : ndarray (n_points, 2)
    """
    try:
        import warnings
        from scipy.stats.qmc import Sobol
        # Округляем до ближайшей степени 2 для оптимальных свойств Соболя
        n_sobol = int(2 ** np.ceil(np.log2(n_points)))
        sampler = Sobol(d=2, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = sampler.random(n_sobol)[:n_points]
        # Масштабируем в (eps, 1-eps) — избегаем граничных точек
        eps = 1e-3
        X   = eps + (1.0 - 2 * eps) * X
        return X
    except ImportError:
        # Fallback на случайные точки если scipy.stats.qmc недоступен
        return collocation_random_2d(n_points, seed)


# ── 3D: уравнение пьезопроводности ────────────────────────────────────────────

def collocation_grid_3d(n_side, n_t, T=1.0):
    """
    Равномерная сетка внутренних точек [0,1]²×(0,T].

    Параметры
    ----------
    n_side : int   — число точек по x и y
    n_t    : int   — число точек по t (без t=0, это начальное условие)
    T      : float — конечное время

    Возвращает
    ----------
    X : ndarray (n_side²·n_t, 3)
    """
    h      = 1.0 / (n_side + 1)
    coords = np.linspace(h, 1.0 - h, n_side)
    # Временные точки: не включаем t=0 (начальное условие задаётся отдельно)
    dt     = T / (n_t + 1)
    t_vals = np.linspace(dt, T - dt, n_t)

    xx, yy, tt = np.meshgrid(coords, coords, t_vals)
    X = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
    return X


def collocation_random_3d(n_points, T=1.0, seed=0):
    """
    Случайные внутренние точки в (0,1)²×(0,T).

    Параметры
    ----------
    n_points : int   — число точек
    T        : float — конечное время
    seed     : int   — зерно генератора

    Возвращает
    ----------
    X : ndarray (n_points, 3)
    """
    rng = np.random.default_rng(seed)
    xy  = rng.uniform(1e-3, 1.0 - 1e-3, (n_points, 2))
    t   = rng.uniform(1e-3, T - 1e-3,   (n_points, 1))
    X   = np.hstack([xy, t])
    return X


def collocation_sobol_3d(n_points, T=1.0, seed=0):
    """
    Квазислучайные точки (последовательность Соболя) в (0,1)²×(0,T).

    Параметры
    ----------
    n_points : int   — число точек
    T        : float — конечное время
    seed     : int   — зерно

    Возвращает
    ----------
    X : ndarray (n_points, 3)
    """
    try:
        import warnings
        from scipy.stats.qmc import Sobol
        n_sobol = int(2 ** np.ceil(np.log2(n_points)))
        sampler = Sobol(d=3, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = sampler.random(n_sobol)[:n_points]
        # Масштабируем: x,y → (eps, 1-eps), t → (eps, T-eps)
        eps      = 1e-3
        X[:, :2] = eps + (1.0 - 2 * eps) * X[:, :2]
        X[:, 2]  = eps + (T   - 2 * eps) * X[:, 2]
        return X
    except ImportError:
        return collocation_random_3d(n_points, T, seed)


# ── Словари стратегий (для GUI) ────────────────────────────────────────────────

SAMPLING_STRATEGIES_2D = {
    "Равномерная сетка":          collocation_grid_2d,
    "Случайная выборка":          collocation_random_2d,
    "Квазислучайная (Соболь)":    collocation_sobol_2d,
}

SAMPLING_STRATEGIES_3D = {
    "Равномерная сетка":          collocation_grid_3d,
    "Случайная выборка":          collocation_random_3d,
    "Квазислучайная (Соболь)":    collocation_sobol_3d,
}


# ── Разбивка train / test ──────────────────────────────────────────────────────

def train_test_split(X, test_size=0.25, seed=0):
    """
    Случайное разбиение коллокационных точек на train и test.

    Параметры
    ----------
    X         : ndarray (N, d)
    test_size : float   — доля тестовой выборки (по умолчанию 25%)
    seed      : int

    Возвращает
    ----------
    X_train : ndarray (N_train, d)
    X_test  : ndarray (N_test,  d)
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(X))
    n_test = max(1, int(len(X) * test_size))
    return X[idx[n_test:]], X[idx[:n_test]]


# ── Вспомогательные функции ───────────────────────────────────────────────────

def make_collocation_points(strategy, n_points, dim=2, T=1.0, seed=0):
    """
    Единая точка входа для генерации коллокационных точек.

    Параметры
    ----------
    strategy  : str   — ключ из SAMPLING_STRATEGIES_2D / 3D
    n_points  : int   — число точек (для сетки — число точек по одной стороне)
    dim       : int   — 2 (Пуассон) или 3 (пьезопроводность)
    T         : float — конечное время (только для dim=3)
    seed      : int

    Возвращает
    ----------
    X : ndarray (N, dim)
    """
    if dim == 2:
        strategies = SAMPLING_STRATEGIES_2D
        if strategy == "Равномерная сетка":
            return strategies[strategy](n_points)
        else:
            return strategies[strategy](n_points, seed=seed)
    else:
        strategies = SAMPLING_STRATEGIES_3D
        if strategy == "Равномерная сетка":
            return strategies[strategy](n_points, n_points, T=T)
        else:
            return strategies[strategy](n_points, T=T, seed=seed)


def count_points(strategy, n_points, dim=2):
    """
    Возвращает фактическое число точек после генерации.
    Нужно для отображения в GUI (сетка даёт n² или n²·n_t точек).

    Параметры
    ----------
    strategy : str
    n_points : int
    dim      : int

    Возвращает
    ----------
    n_actual : int
    """
    if strategy == "Равномерная сетка":
        if dim == 2:
            return n_points ** 2
        else:
            return n_points ** 2 * n_points   # n_side² × n_t
    return n_points


def get_grid_for_plot(n=100, dim=2, T=1.0, t_slice=0.5):
    """
    Генерирует равномерную сетку для визуализации результатов.

    Параметры
    ----------
    n       : int   — число точек по каждому направлению
    dim     : int   — 2 или 3
    T       : float — конечное время
    t_slice : float — момент времени для среза (только dim=3)

    Возвращает
    ----------
    X_plot  : ndarray (n², dim) — точки для предсказания
    xx, yy  : ndarray (n, n)    — координатные сетки для plt/imshow
    """
    coords = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords)

    if dim == 2:
        X_plot = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        t_col  = np.full(n * n, t_slice)
        X_plot = np.column_stack([xx.ravel(), yy.ravel(), t_col])

    return X_plot, xx, yy