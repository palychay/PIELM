import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение пьезопроводности (нестационарная фильтрация)
#
# Вывод (Леонтьев §12, формула 12.2):
#   ∂(mρ)/∂t + div(ρu) = 0
#   u = -(k/μ) · grad P
#   ρ(p) = ρ0·(1 + (p-p0)/Kρ),  m(p) = m0·(1 + (p-p0)/Km)
#   → ∂P/∂t = κ·∇²P + q(x,y,t)
#
#   κ = k/(m0·μ) · (1/Kρ + 1/Km)⁻¹  — коэффициент пьезопроводности
#
# Область: Ω = [0,1]², t ∈ [0, T]
# НУ: P(x,y,0) = 0
# ГУ: P = 0 на ∂Ω
#
# Аналитического решения нет при сложном источнике q(x,y,t).
# ─────────────────────────────────────────────────────────────────────────────


# ── Варианты источникового члена q(x,y,t) ────────────────────────────────────

def source_pulsing(x, y, t, A=1.0, sigma=0.15, T=1.0):
    """
    Пульсирующий гауссов источник в центре:
        q(x,y,t) = A · sin(πt/T) · exp(-((x-0.5)²+(y-0.5)²) / σ²)

    Параметры
    ----------
    A     : float — амплитуда
    sigma : float — ширина гауссианы
    T     : float — период пульсации
    """
    spatial  = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / sigma**2)
    temporal = np.sin(np.pi * t / T)
    return A * temporal * spatial


def source_harmonic(x, y, t, A=1.0, B=0.5, T=1.0):
    """
    Гармонический источник с пространственной модуляцией:
        q(x,y,t) = A·sin(2πx)·sin(2πy)·cos(2πt/T)
                 + B·sin(πx)·sin(πy)·sin(4πt/T)

    Параметры
    ----------
    A, B : float — амплитуды
    T    : float — временной масштаб
    """
    mode1 = np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y) \
            * np.cos(2.0 * np.pi * t / T)
    mode2 = np.sin(np.pi * x) * np.sin(np.pi * y) \
            * np.sin(4.0 * np.pi * t / T)
    return A * mode1 + B * mode2


def source_moving(x, y, t, A=1.0, sigma=0.1, T=1.0):
    """
    Движущийся источник по диагонали области:
        центр: (0.1 + 0.8·t/T, 0.1 + 0.8·t/T)
        q(x,y,t) = A · exp(-((x-cx)²+(y-cy)²) / σ²)

    Параметры
    ----------
    A     : float — амплитуда
    sigma : float — ширина гауссианы
    T     : float — время полного прохода
    """
    cx = 0.1 + 0.8 * t / T
    cy = 0.1 + 0.8 * t / T
    return A * np.exp(-((x - cx)**2 + (y - cy)**2) / sigma**2)


# ── Словарь вариантов источника (для GUI) ─────────────────────────────────────

PIEZO_SOURCE_VARIANTS = {
    "Пульсирующий: A·sin(πt/T)·exp(-r²/σ²)": {
        "func":   source_pulsing,
        "params": {"A": 1.0, "sigma": 0.15, "T": 1.0},
        "param_ranges": {
            "A":     (0.1, 5.0),
            "sigma": (0.05, 0.4),
            "T":     (0.5, 5.0),
        },
    },
    "Гармонический: A·sin(2πx)·sin(2πy)·cos(2πt/T) + ...": {
        "func":   source_harmonic,
        "params": {"A": 1.0, "B": 0.5, "T": 1.0},
        "param_ranges": {
            "A": (0.1, 5.0),
            "B": (0.1, 5.0),
            "T": (0.5, 5.0),
        },
    },
    "Движущийся источник по диагонали": {
        "func":   source_moving,
        "params": {"A": 1.0, "sigma": 0.1, "T": 1.0},
        "param_ranges": {
            "A":     (0.1, 5.0),
            "sigma": (0.05, 0.3),
            "T":     (0.5, 5.0),
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PDE-оператор для PIELM
#
# ∂P/∂t = κ·∇²P + q(x,y,t)
#
# Переносим всё влево:
# ∂P/∂t - κ·∇²P = q(x,y,t)
#
# Для нейрона φ_j(x,y,t) = σ(W[0,j]·x + W[1,j]·y + W[2,j]·t + b[0,j]):
#   ∂φ_j/∂t   = σ'(z_j) · W[2,j]
#   ∂²φ_j/∂x² = σ''(z_j) · W[0,j]²
#   ∂²φ_j/∂y² = σ''(z_j) · W[1,j]²
#
# Оператор: L[φ_j] = σ'(z_j)·W[2,j] - κ·σ''(z_j)·(W[0,j]² + W[1,j]²)
#
# Входной слой: input_dim = 3, X = (x, y, t)
# Возвращает матрицу H_f размера (N_train, n_hidden)
# ─────────────────────────────────────────────────────────────────────────────

def pde_operator_piezo(W, b, X, kappa=1.0, act_name='tanh', act_dict=None):
    """
    Вычисляет матрицу коллокационных уравнений для уравнения пьезопроводности.

    Параметры
    ----------
    W        : ndarray (3, n_hidden)  — веса входного слоя (x, y, t)
    b        : ndarray (1, n_hidden)  — смещения
    X        : ndarray (N, 3)         — коллокационные точки (x, y, t)
    kappa    : float                  — коэффициент пьезопроводности κ
    act_name : str                    — имя функции активации
    act_dict : dict                   — словарь активаций {name: {f, df, ddf}}

    Возвращает
    ----------
    H_f : ndarray (N, n_hidden)
    """
    Z   = X @ W + b                              # (N, n_hidden)

    df  = act_dict[act_name]['df']
    ddf = act_dict[act_name]['ddf']

    sigma_p  = df(Z)                             # σ'(z),  (N, n_hidden)
    sigma_pp = ddf(Z)                            # σ''(z), (N, n_hidden)

    # ∂φ/∂t = σ'(z)·W[2,j]
    dphi_dt  = sigma_p  * W[2, :]               # (N, n_hidden)

    # κ·(∂²φ/∂x² + ∂²φ/∂y²) = κ·σ''(z)·(W[0,j]² + W[1,j]²)
    laplacian = sigma_pp * (W[0, :]**2 + W[1, :]**2)  # (N, n_hidden)

    # L[φ_j] = ∂φ/∂t - κ·∇²φ
    H_f = dphi_dt - kappa * laplacian           # (N, n_hidden)

    return H_f


# ─────────────────────────────────────────────────────────────────────────────
# Правая часть уравнения в коллокационных точках
# ─────────────────────────────────────────────────────────────────────────────

def rhs_piezo(X, source_func, source_params):
    """
    Вычисляет q(x,y,t) в точках X.

    Параметры
    ----------
    X             : ndarray (N, 3)       — точки (x, y, t)
    source_func   : callable             — q(x, y, t, **source_params)
    source_params : dict

    Возвращает
    ----------
    rhs : ndarray (N,)
    """
    x = X[:, 0]
    y = X[:, 1]
    t = X[:, 2]
    return source_func(x, y, t, **source_params)


# ─────────────────────────────────────────────────────────────────────────────
# Граничные и начальные условия
# ─────────────────────────────────────────────────────────────────────────────

def boundary_conditions_piezo(n_bc=20, n_t=20, T=1.0):
    """
    Генерирует граничные и начальные точки для уравнения пьезопроводности.

    Граничные условия (ГУ): P = 0 на ∂Ω для всех t ∈ [0, T]
    Начальное условие  (НУ): P = 0 при t = 0 для всех (x,y) ∈ Ω

    Параметры
    ----------
    n_bc : int   — число точек на каждой стороне квадрата
    n_t  : int   — число временных слоёв для ГУ
    T    : float — конечное время

    Возвращает
    ----------
    X_b : ndarray (N_b, 3)  — граничные и начальные точки (x, y, t)
    Y_b : ndarray (N_b,)    — P = 0
    """
    t_vals = np.linspace(0.0, T, n_t)
    s_vals = np.linspace(0.0, 1.0, n_bc)

    bc_points = []

    # ── Пространственные границы для каждого t ──
    for t in t_vals:
        # Нижняя: y = 0
        bc_points.append(np.column_stack([
            s_vals, np.zeros(n_bc), np.full(n_bc, t)
        ]))
        # Верхняя: y = 1
        bc_points.append(np.column_stack([
            s_vals, np.ones(n_bc), np.full(n_bc, t)
        ]))
        # Левая: x = 0
        bc_points.append(np.column_stack([
            np.zeros(n_bc), s_vals, np.full(n_bc, t)
        ]))
        # Правая: x = 1
        bc_points.append(np.column_stack([
            np.ones(n_bc), s_vals, np.full(n_bc, t)
        ]))

    # ── Начальное условие: t = 0 ──
    # Сетка точек (x, y) при t = 0
    xx, yy = np.meshgrid(s_vals, s_vals)
    ic_points = np.column_stack([
        xx.ravel(), yy.ravel(), np.zeros(n_bc * n_bc)
    ])
    bc_points.append(ic_points)

    X_b = np.vstack(bc_points)              # (N_b, 3)
    Y_b = np.zeros(len(X_b))               # P = 0
    return X_b, Y_b


# ─────────────────────────────────────────────────────────────────────────────
# Вычисление коэффициента пьезопроводности κ из физических параметров
# (Леонтьев §12, формула 12.2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kappa(k0, m0, mu, K_rho, K_m):
    """
    κ = k0 / (m0·μ) · (1/Kρ + 1/Km)⁻¹

    Параметры
    ----------
    k0    : float — проницаемость [м²]
    m0    : float — пористость [-]
    mu    : float — динамическая вязкость [Па·с]
    K_rho : float — модуль сжимаемости жидкости [Па]
    K_m   : float — модуль сжимаемости скелета [Па]

    Возвращает
    ----------
    kappa : float — коэффициент пьезопроводности [м²/с]
    """
    return (k0 / (m0 * mu)) / (1.0 / K_rho + 1.0 / K_m)