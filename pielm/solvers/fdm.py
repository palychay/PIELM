import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ─────────────────────────────────────────────────────────────────────────────
# Метод конечных разностей (МКР)
#
# Уравнение Пуассона (стационарное):
#   ∇²P = f(x,y)  в Ω = [0,1]²
#   P = 0 на ∂Ω
#
#   Дискретизация центральными разностями:
#   (P[i-1,j] - 2P[i,j] + P[i+1,j]) / h² +
#   (P[i,j-1] - 2P[i,j] + P[i,j+1]) / h² = f[i,j]
#
# Уравнение пьезопроводности (нестационарное):
#   ∂P/∂t = κ·∇²P + q(x,y,t)  в Ω × [0,T]
#   P = 0 на ∂Ω,  P(x,y,0) = 0
#
#   Дискретизация:
#   - по пространству: центральные разности (второй порядок)
#   - по времени: неявная схема Кранка–Николсона (второй порядок,
#                 безусловно устойчивая)
# ─────────────────────────────────────────────────────────────────────────────


class FDMPoisson:
    """
    МКР для уравнения Пуассона:
        ∇²P = f(x,y),  P = 0 на ∂Ω = [0,1]²

    Параметры
    ----------
    n_grid : int — число внутренних узлов по каждому направлению
                   (полная сетка: (n_grid+2) × (n_grid+2))
    """

    def __init__(self, n_grid=50):
        self.n_grid  = n_grid
        self.h       = 1.0 / (n_grid + 1)        # шаг сетки
        self.n_inner = n_grid                     # число внутренних узлов
        self.P_grid  = None                       # решение на сетке
        self.x_inner = None                       # координаты внутренних узлов
        self.y_inner = None

    # ── Построение матрицы системы ─────────────────────────────────────────

    def _build_matrix(self):
        """
        Строит разреженную матрицу A системы линейных уравнений
        для внутренних узлов сетки.

        Нумерация: узел (i, j) → глобальный индекс k = i*n + j
        где i, j ∈ {0, ..., n-1} — индексы внутренних узлов.

        Возвращает
        ----------
        A : sparse matrix (n², n²)
        """
        n   = self.n_inner
        n2  = n * n
        h2  = self.h ** 2
        A   = lil_matrix((n2, n2))

        for i in range(n):
            for j in range(n):
                k = i * n + j          # глобальный индекс

                # Диагональный элемент: -4/h²
                A[k, k] = -4.0 / h2

                # Левый сосед: (i, j-1)
                if j > 0:
                    A[k, k - 1] = 1.0 / h2

                # Правый сосед: (i, j+1)
                if j < n - 1:
                    A[k, k + 1] = 1.0 / h2

                # Нижний сосед: (i-1, j)
                if i > 0:
                    A[k, k - n] = 1.0 / h2

                # Верхний сосед: (i+1, j)
                if i < n - 1:
                    A[k, k + n] = 1.0 / h2

        return A.tocsr()

    # ── Решение ───────────────────────────────────────────────────────────

    def solve(self, source_func, source_params):
        """
        Решает уравнение Пуассона методом конечных разностей.

        Параметры
        ----------
        source_func   : callable — f(x, y, **source_params)
        source_params : dict

        Возвращает
        ----------
        P_grid : ndarray ((n_grid+2), (n_grid+2)) — решение на полной сетке
                 (включая граничные узлы P = 0)
        """
        n    = self.n_inner
        h    = self.h

        # Координаты внутренних узлов
        coords  = np.linspace(h, 1.0 - h, n)
        self.x_inner = coords
        self.y_inner = coords
        xx, yy  = np.meshgrid(coords, coords)      # (n, n)

        # Правая часть
        F = source_func(xx.ravel(), yy.ravel(), **source_params)  # (n²,)

        # Матрица системы
        A = self._build_matrix()

        # Решение СЛАУ
        P_inner = spsolve(A, F)                    # (n²,)

        # Сборка полного поля (с нулевыми границами)
        n_full    = n + 2
        P_full    = np.zeros((n_full, n_full))
        P_inner2d = P_inner.reshape(n, n)
        P_full[1:-1, 1:-1] = P_inner2d

        self.P_grid = P_full
        return P_full

    # ── Интерполяция в произвольные точки ─────────────────────────────────

    def predict(self, X):
        """
        Билинейная интерполяция решения в точки X.

        Параметры
        ----------
        X : ndarray (N, 2) — точки (x, y) ∈ [0,1]²

        Возвращает
        ----------
        P : ndarray (N,)
        """
        if self.P_grid is None:
            raise RuntimeError("Сначала вызовите solve().")

        n_full = self.n_grid + 2
        h      = self.h
        coords = np.linspace(0.0, 1.0, n_full)    # включая границу

        x = X[:, 0]
        y = X[:, 1]

        # Индексы нижнего левого угла ячейки
        ix = np.clip((x / h).astype(int), 0, n_full - 2)
        iy = np.clip((y / h).astype(int), 0, n_full - 2)

        # Веса интерполяции
        tx = (x - coords[ix]) / h
        ty = (y - coords[iy]) / h

        # Билинейная интерполяция
        P = (
            (1 - tx) * (1 - ty) * self.P_grid[iy,     ix    ] +
                 tx  * (1 - ty) * self.P_grid[iy,     ix + 1] +
            (1 - tx) *      ty  * self.P_grid[iy + 1, ix    ] +
                 tx  *      ty  * self.P_grid[iy + 1, ix + 1]
        )
        return P

    def rmse(self, X, y_ref):
        """RMSE между интерполированным решением и эталоном."""
        y_ref = y_ref(X) if callable(y_ref) else np.asarray(y_ref).ravel()
        return float(np.sqrt(np.mean((self.predict(X) - y_ref) ** 2)))

    def get_grid_coords(self):
        """
        Возвращает координатные сетки для визуализации.

        Возвращает
        ----------
        X_grid, Y_grid : ndarray ((n_grid+2), (n_grid+2))
        """
        n_full = self.n_grid + 2
        coords = np.linspace(0.0, 1.0, n_full)
        return np.meshgrid(coords, coords)

    def summary(self):
        return {
            'method':  'МКР (метод конечных разностей)',
            'equation': 'Пуассон: ∇²P = f(x,y)',
            'n_grid':  self.n_grid,
            'h':       self.h,
            'n_dof':   self.n_inner ** 2,
        }


# ─────────────────────────────────────────────────────────────────────────────

class FDMPiezo:
    """
    МКР для уравнения пьезопроводности:
        ∂P/∂t = κ·∇²P + q(x,y,t)
        P = 0 на ∂Ω,   P(x,y,0) = 0

    Схема Кранка–Николсона (неявная, второй порядок по времени и пространству,
    безусловно устойчивая).

    Параметры
    ----------
    n_grid : int   — число внутренних узлов по каждому направлению
    n_t    : int   — число временных шагов
    T      : float — конечное время
    kappa  : float — коэффициент пьезопроводности κ
    """

    def __init__(self, n_grid=30, n_t=50, T=1.0, kappa=1.0):
        self.n_grid  = n_grid
        self.n_t     = n_t
        self.T       = T
        self.kappa   = kappa
        self.h       = 1.0 / (n_grid + 1)
        self.dt      = T / n_t
        self.n_inner = n_grid
        self.P_all   = None    # решение на всех временных слоях (n_t+1, n², )
        self.t_vals  = None

    # ── Построение матриц схемы Кранка–Николсона ──────────────────────────

    def _build_laplacian(self):
        """
        Строит матрицу дискретного Лапласиана L для внутренних узлов.
        ∇²P ≈ L·P_vec / h²
        """
        n   = self.n_inner
        n2  = n * n
        h2  = self.h ** 2
        L   = lil_matrix((n2, n2))

        for i in range(n):
            for j in range(n):
                k = i * n + j
                L[k, k] = -4.0 / h2
                if j > 0:
                    L[k, k - 1] = 1.0 / h2
                if j < n - 1:
                    L[k, k + 1] = 1.0 / h2
                if i > 0:
                    L[k, k - n] = 1.0 / h2
                if i < n - 1:
                    L[k, k + n] = 1.0 / h2

        return L.tocsr()

    # ── Решение ───────────────────────────────────────────────────────────

    def solve(self, source_func, source_params):
        """
        Решает уравнение пьезопроводности схемой Кранка–Николсона.

        На каждом шаге по времени решается СЛАУ:
        (I - κ·dt/2·L)·P^{n+1} = (I + κ·dt/2·L)·P^n
                                  + dt/2·(q^n + q^{n+1})

        Параметры
        ----------
        source_func   : callable — q(x, y, t, **source_params)
        source_params : dict

        Возвращает
        ----------
        P_all  : ndarray (n_t+1, n_inner, n_inner) — решение на всех слоях
        t_vals : ndarray (n_t+1,) — временные метки
        """
        n     = self.n_inner
        n2    = n * n
        h     = self.h
        dt    = self.dt
        kappa = self.kappa

        # Координаты внутренних узлов
        coords = np.linspace(h, 1.0 - h, n)
        xx, yy = np.meshgrid(coords, coords)      # (n, n)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        # Временные метки
        self.t_vals = np.linspace(0.0, self.T, self.n_t + 1)

        # Дискретный Лапласиан
        L = self._build_laplacian()

        # Матрицы схемы Кранка–Николсона
        from scipy.sparse import eye as speye
        I    = speye(n2, format='csr')
        A_lhs = I - (kappa * dt / 2.0) * L       # левая часть
        A_rhs = I + (kappa * dt / 2.0) * L       # правая часть (матрица)

        # Начальное условие P = 0
        P_vec = np.zeros(n2)

        # Хранение решений
        P_all = np.zeros((self.n_t + 1, n, n))
        P_all[0] = P_vec.reshape(n, n)

        # Временной цикл
        for step in range(self.n_t):
            t_curr = self.t_vals[step]
            t_next = self.t_vals[step + 1]

            # Источник на текущем и следующем шаге
            q_curr = source_func(x_flat, y_flat, t_curr, **source_params)
            q_next = source_func(x_flat, y_flat, t_next, **source_params)

            # Правая часть СЛАУ
            rhs = A_rhs @ P_vec + (dt / 2.0) * (q_curr + q_next)

            # Решение СЛАУ
            P_vec = spsolve(A_lhs, rhs)
            P_all[step + 1] = P_vec.reshape(n, n)

        self.P_all = P_all
        return P_all, self.t_vals

    # ── Получение полного поля на временном слое ──────────────────────────

    def get_full_field(self, step):
        """
        Возвращает полное поле давления (с нулевыми границами)
        на временном шаге step.

        Параметры
        ----------
        step : int — индекс временного шага (0 ... n_t)

        Возвращает
        ----------
        P_full : ndarray ((n_grid+2), (n_grid+2))
        """
        if self.P_all is None:
            raise RuntimeError("Сначала вызовите solve().")

        n_full = self.n_grid + 2
        P_full = np.zeros((n_full, n_full))
        P_full[1:-1, 1:-1] = self.P_all[step]
        return P_full

    # ── Интерполяция в произвольные точки на шаге step ────────────────────

    def predict_at_step(self, X, step):
        """
        Билинейная интерполяция решения в точки X на шаге step.

        Параметры
        ----------
        X    : ndarray (N, 2) — точки (x, y) ∈ [0,1]²
        step : int

        Возвращает
        ----------
        P : ndarray (N,)
        """
        P_full = self.get_full_field(step)
        n_full = self.n_grid + 2
        h      = self.h
        coords = np.linspace(0.0, 1.0, n_full)

        x = X[:, 0]
        y = X[:, 1]

        ix = np.clip((x / h).astype(int), 0, n_full - 2)
        iy = np.clip((y / h).astype(int), 0, n_full - 2)

        tx = (x - coords[ix]) / h
        ty = (y - coords[iy]) / h

        P = (
            (1 - tx) * (1 - ty) * P_full[iy,     ix    ] +
                 tx  * (1 - ty) * P_full[iy,     ix + 1] +
            (1 - tx) *      ty  * P_full[iy + 1, ix    ] +
                 tx  *      ty  * P_full[iy + 1, ix + 1]
        )
        return P

    def predict(self, X):
        """
        Интерполяция в точки X = (x, y, t).
        Находит ближайший временной шаг к t и интерполирует.

        Параметры
        ----------
        X : ndarray (N, 3) — точки (x, y, t)

        Возвращает
        ----------
        P : ndarray (N,)
        """
        if self.P_all is None:
            raise RuntimeError("Сначала вызовите solve().")

        t_vals = self.t_vals
        P_out  = np.zeros(len(X))

        # Группируем точки по ближайшему временному шагу
        t_points = X[:, 2]
        steps    = np.argmin(
            np.abs(t_points[:, None] - t_vals[None, :]), axis=1
        )

        for step in np.unique(steps):
            mask      = steps == step
            P_out[mask] = self.predict_at_step(X[mask, :2], step)

        return P_out

    def rmse(self, X, y_ref):
        """RMSE между интерполированным решением и эталоном."""
        y_ref = y_ref(X) if callable(y_ref) else np.asarray(y_ref).ravel()
        return float(np.sqrt(np.mean((self.predict(X) - y_ref) ** 2)))

    def get_grid_coords(self):
        """
        Возвращает координатные сетки для визуализации.

        Возвращает
        ----------
        X_grid, Y_grid : ndarray ((n_grid+2), (n_grid+2))
        """
        n_full = self.n_grid + 2
        coords = np.linspace(0.0, 1.0, n_full)
        return np.meshgrid(coords, coords)

    def summary(self):
        return {
            'method':   'МКР (схема Кранка–Николсона)',
            'equation': 'Пьезопроводность: ∂P/∂t = κ·∇²P + q',
            'n_grid':   self.n_grid,
            'n_t':      self.n_t,
            'h':        self.h,
            'dt':       self.dt,
            'kappa':    self.kappa,
            'n_dof':    self.n_inner ** 2,
        }