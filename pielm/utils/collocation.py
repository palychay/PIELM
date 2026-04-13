import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Генерация коллокационных точек — равномерная сетка
#
# Для уравнения Пуассона:   X ∈ (0,1)²,      shape (n_side², 2)
# Для пьезопроводности:     X ∈ (0,1)²×(0,T), shape (n_side²·n_t, 3)
#
# Граничные точки не включаются — они задаются отдельно через
# boundary_conditions_poisson / boundary_conditions_piezo
# ─────────────────────────────────────────────────────────────────────────────


def collocation_grid_2d(n_side):
    """
    Равномерная сетка внутренних точек (0,1)².

    Параметры
    ----------
    n_side : int — число точек по каждому направлению

    Возвращает
    ----------
    X : ndarray (n_side², 2)
    """
    h      = 1.0 / (n_side + 1)
    coords = np.linspace(h, 1.0 - h, n_side)
    xx, yy = np.meshgrid(coords, coords)
    return np.column_stack([xx.ravel(), yy.ravel()])


def collocation_grid_3d(n_side, n_t, T=1.0):
    """
    Равномерная сетка внутренних точек (0,1)²×(0,T).

    Начальный момент t=0 не включается — он задаётся как
    начальное условие в boundary_conditions_piezo.

    Параметры
    ----------
    n_side : int   — число точек по x и y
    n_t    : int   — число временных слоёв
    T      : float — конечное время

    Возвращает
    ----------
    X : ndarray (n_side²·n_t, 3)
    """
    h      = 1.0 / (n_side + 1)
    coords = np.linspace(h, 1.0 - h, n_side)
    dt     = T / (n_t + 1)
    t_vals = np.linspace(dt, T - dt, n_t)

    xx, yy, tt = np.meshgrid(coords, coords, t_vals)
    return np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])


# ─────────────────────────────────────────────────────────────────────────────
# Разбивка train / test (75% / 25%)
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split(X, test_size=0.25, seed=0):
    """
    Случайное разбиение коллокационных точек на train и test.

    Параметры
    ----------
    X         : ndarray (N, d)
    test_size : float   — доля тестовой выборки
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


# ─────────────────────────────────────────────────────────────────────────────
# Единая точка входа для генерации точек
# ─────────────────────────────────────────────────────────────────────────────

def make_collocation_points(n_side, dim=2, T=1.0, n_t=None):
    """
    Генерирует коллокационные точки равномерной сеткой.

    Параметры
    ----------
    n_side : int   — число точек по каждому пространственному направлению
    dim    : int   — 2 (Пуассон) или 3 (пьезопроводность)
    T      : float — конечное время (только для dim=3)
    n_t    : int   — число временных слоёв (только для dim=3);
                     если None — берётся равным n_side

    Возвращает
    ----------
    X : ndarray (N, dim)
        dim=2: N = n_side²
        dim=3: N = n_side² · n_t
    """
    if dim == 2:
        return collocation_grid_2d(n_side)
    else:
        nt = n_t if n_t is not None else n_side
        return collocation_grid_3d(n_side, nt, T=T)


def count_points(n_side, dim=2, n_t=None):
    """
    Возвращает фактическое число точек без генерации.
    Используется для отображения в GUI.

    Параметры
    ----------
    n_side : int
    dim    : int
    n_t    : int или None

    Возвращает
    ----------
    n : int
    """
    if dim == 2:
        return n_side ** 2
    else:
        nt = n_t if n_t is not None else n_side
        return n_side ** 2 * nt


# ─────────────────────────────────────────────────────────────────────────────
# Сетка для визуализации результатов
# ─────────────────────────────────────────────────────────────────────────────

def get_grid_for_plot(n=80, dim=2, T=1.0, t_slice=0.5):
    """
    Равномерная сетка n×n для построения тепловых карт.

    Параметры
    ----------
    n       : int   — число точек по каждому направлению
    dim     : int   — 2 или 3
    T       : float — конечное время (только для dim=3)
    t_slice : float — момент времени для среза (только для dim=3)

    Возвращает
    ----------
    X_plot : ndarray (n², dim)
    xx     : ndarray (n, n)
    yy     : ndarray (n, n)
    """
    coords = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords)

    if dim == 2:
        X_plot = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        t_col  = np.full(n * n, t_slice)
        X_plot = np.column_stack([xx.ravel(), yy.ravel(), t_col])

    return X_plot, xx, yy