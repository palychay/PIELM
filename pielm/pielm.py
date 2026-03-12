"""
pielm.py — Physics-Informed Extreme Learning Machine (PIELM) + Genetic Algorithm

Задача: стационарная фильтрация (закон Дарси) на [0,1]²
    ∇²P = 0,  P(0,y)=1,  P(1,y)=0,  ∂P/∂n=0
    Точное решение: P(x,y) = 1 − x

Разбивка коллокационных точек:
    train_test_split_colloc() — 75% обучение / 25% тест
    GA оптимизирует гиперпараметры только на train.
    Тест — только для итоговой проверки.
"""

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
        'df':  lambda z: (lambda s: s*(1.0-s))(
                    1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))),
        'ddf': lambda z: (lambda s: s*(1.0-s)*(1.0-2.0*s))(
                    1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Оператор Дарси ∇²P
# ─────────────────────────────────────────────────────────────────────────────

def diffusion_2d_operator(W, b, X, act_name='tanh'):
    """∇²u = ∂²u/∂x² + ∂²u/∂y²  — возвращает матрицу (N, n_hidden)."""
    Z   = X @ W + b
    ddf = ACTIVATIONS[act_name]['ddf']
    return ddf(Z) * (W[0, :]**2 + W[1, :]**2)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Разбивка 75% train / 25% test
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_colloc(X_f, test_size=0.25, seed=0):
    """
    Случайное разбиение коллокационных точек.
    Возвращает: X_train, X_test, train_idx, test_idx
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(X_f))
    n_test = max(1, int(len(X_f) * test_size))
    return X_f[idx[n_test:]], X_f[idx[:n_test]], idx[n_test:], idx[:n_test]

# ─────────────────────────────────────────────────────────────────────────────
# 4. PIELM
# ─────────────────────────────────────────────────────────────────────────────

class PIELM:
    def __init__(self, n_hidden, input_dim=2, scale=5.0, act_name='tanh',
                 lambda_pde=1.0, lambda_bc=10.0, seed=None):
        self.n_hidden   = n_hidden
        self.input_dim  = input_dim
        self.scale      = scale
        self.act_name   = act_name
        self.lambda_pde = lambda_pde
        self.lambda_bc  = lambda_bc
        self.phi        = ACTIVATIONS[act_name]['f']
        self.W = self.b = self.beta = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        self.W = self.rng.normal(0, self.scale, (self.input_dim, self.n_hidden))
        self.b = self.rng.normal(0, self.scale, (1, self.n_hidden))
        return self

    def fit(self, X_train, X_b, Y_b, operator_func, source_func):
        """Обучение МНК на X_train (75% коллокационных точек)."""
        if self.W is None:
            self.initialize()
        H_f = operator_func(self.W, self.b, X_train, act_name=self.act_name)
        Y_f = np.asarray(source_func(X_train)).reshape(-1, 1)
        H_b = self.phi(X_b @ self.W + self.b)
        Y_b = np.asarray(Y_b).reshape(-1, 1)
        sp, sb = np.sqrt(self.lambda_pde), np.sqrt(self.lambda_bc)
        H = np.vstack([sp * H_f, sb * H_b])
        Y = np.vstack([sp * Y_f, sb * Y_b])
        self.beta, _, _, _ = np.linalg.lstsq(H, Y, rcond=1e-15)
        return self

    def predict(self, X):
        return self.phi(X @ self.W + self.b) @ self.beta

    def rmse(self, X, y_true):
        """RMSE относительно точного решения."""
        return float(np.sqrt(np.mean(
            (self.predict(X).ravel() - np.asarray(y_true).ravel()) ** 2)))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Генетический оптимизатор
# ─────────────────────────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Оптимизация гиперпараметров PIELM.
    Хромосома: {n_hidden, scale, activation, lambda_pde, lambda_bc}
    Фитнес: RMSE на X_train. Тестовая выборка в GA не участвует.
    """

    def __init__(self, n_pop=20, n_gen=10,
                 hidden_bounds=(50, 600),
                 scale_bounds=(0.5, 10.0),
                 lambda_pde_bounds=(0.01, 10.0),
                 lambda_bc_bounds=(1.0, 100.0),
                 seed=42):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.hidden_min,     self.hidden_max     = hidden_bounds
        self.scale_min,      self.scale_max      = scale_bounds
        self.lambda_pde_min, self.lambda_pde_max = lambda_pde_bounds
        self.lambda_bc_min,  self.lambda_bc_max  = lambda_bc_bounds
        self.best_params = None
        self.best_loss   = float('inf')
        self.history     = []
        self.rng         = np.random.default_rng(seed)

    def _init_population(self):
        return [{
            'n_hidden':   int(self.rng.integers(self.hidden_min, self.hidden_max)),
            'scale':      float(self.rng.uniform(self.scale_min, self.scale_max)),
            'activation': str(self.rng.choice(['tanh', 'sin', 'sigmoid'])),
            'lambda_pde': float(self.rng.uniform(self.lambda_pde_min, self.lambda_pde_max)),
            'lambda_bc':  float(self.rng.uniform(self.lambda_bc_min,  self.lambda_bc_max)),
        } for _ in range(self.n_pop)]

    def _fitness(self, ind, X_train, X_b, Y_b, operator_func, source_func, p_exact_func):
        model = PIELM(
            n_hidden=ind['n_hidden'], input_dim=X_train.shape[1],
            scale=ind['scale'], act_name=ind['activation'],
            lambda_pde=ind['lambda_pde'], lambda_bc=ind['lambda_bc'],
        )
        try:
            model.fit(X_train, X_b, Y_b, operator_func, source_func)
            y_true = p_exact_func(X_train[:, 0], X_train[:, 1])
            return model.rmse(X_train, y_true)
        except (np.linalg.LinAlgError, FloatingPointError):
            return float('inf')

    def _crossover(self, p1, p2):
        return {
            'n_hidden':   int((p1['n_hidden']   + p2['n_hidden'])   / 2),
            'scale':      (p1['scale']      + p2['scale'])      / 2.0,
            'activation': p1['activation'] if self.rng.random() < 0.5 else p2['activation'],
            'lambda_pde': (p1['lambda_pde'] + p2['lambda_pde']) / 2.0,
            'lambda_bc':  (p1['lambda_bc']  + p2['lambda_bc'])  / 2.0,
        }

    def _mutate(self, ind):
        c = ind.copy()
        if self.rng.random() < 0.3:
            c['n_hidden'] = int(np.clip(
                c['n_hidden'] + int(self.rng.integers(-50, 51)),
                self.hidden_min, self.hidden_max))
        if self.rng.random() < 0.3:
            c['scale'] = float(np.clip(
                c['scale'] + self.rng.normal(0, 1.0),
                self.scale_min, self.scale_max))
        if self.rng.random() < 0.2:
            c['activation'] = str(self.rng.choice(['tanh', 'sin', 'sigmoid']))
        if self.rng.random() < 0.3:
            c['lambda_pde'] = float(np.clip(
                c['lambda_pde'] * self.rng.uniform(0.5, 2.0),
                self.lambda_pde_min, self.lambda_pde_max))
        if self.rng.random() < 0.3:
            c['lambda_bc'] = float(np.clip(
                c['lambda_bc'] * self.rng.uniform(0.5, 2.0),
                self.lambda_bc_min, self.lambda_bc_max))
        return c

    def search(self, X_train, X_b, Y_b, operator_func, source_func,
               p_exact_func, callback=None):
        """
        Поиск лучших гиперпараметров.
        callback(gen, best_rmse, best_params) — вызывается после каждого поколения.
        """
        pop = self._init_population()
        for gen in range(self.n_gen):
            scored = sorted(
                [(self._fitness(ind, X_train, X_b, Y_b,
                                operator_func, source_func, p_exact_func), ind)
                 for ind in pop],
                key=lambda x: x[0],
            )
            if scored[0][0] < self.best_loss:
                self.best_loss   = scored[0][0]
                self.best_params = scored[0][1]
            self.history.append(self.best_loss)
            if callback:
                callback(gen, self.best_loss, self.best_params)

            n_elite = max(1, int(self.n_pop * 0.2))
            elites  = [x[1] for x in scored[:n_elite]]
            new_pop = elites[:]
            while len(new_pop) < self.n_pop:
                p1 = elites[self.rng.integers(len(elites))]
                p2 = elites[self.rng.integers(len(elites))]
                new_pop.append(self._mutate(self._crossover(p1, p2)))
            pop = new_pop
        return self.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 6. Нечёткий генетический оптимизатор (FGA)
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyGeneticOptimizer(GeneticOptimizer):
    """
    Нечёткий генетический алгоритм (Fuzzy Genetic Algorithm, FGA).

    Расширяет GeneticOptimizer: после каждого поколения нечёткий контроллер
    автоматически корректирует вероятности мутации, опираясь на два сигнала
    (по псевдокоду из лекции Эррера–Лозано):

        e1(t) = (f_max(t) − f_ave(t)) / f_max(t)
            — разнообразие популяции (насколько лучший отличается от среднего)

        e2(t) = (f_ave(t) − f_ave(t−1)) / f_max(t)
            — скорость улучшения популяции между поколениями

    Нечёткие правила (Mamdani, min-max):
        Если e1 МАЛО  И e2 МАЛО  → мутация ВЫСОКАЯ   (популяция застряла)
        Если e1 МАЛО  И e2 ВЕЛИКО → мутация СРЕДНЯЯ   (прогресс есть, но однородны)
        Если e1 ВЕЛИКО И e2 МАЛО  → мутация СРЕДНЯЯ   (разнообразны, но не прогрессируют)
        Если e1 ВЕЛИКО И e2 ВЕЛИКО → мутация НИЗКАЯ   (всё хорошо, не мешаем)

    Функции принадлежности — треугольные (трапециевидные на краях):
        МАЛО:   1 при x=0,  0 при x≥0.5
        ВЕЛИКО: 0 при x≤0.5, 1 при x=1

    Дефаззификация: центр тяжести (CoG) трёх синглтонов:
        НИЗКАЯ=0.1,  СРЕДНЯЯ=0.3,  ВЫСОКАЯ=0.6

    История сигналов хранится в self.fuzzy_log — удобно для графиков в дипломе.
    """

    # ── Функции принадлежности ────────────────────────────────────────────────

    @staticmethod
    def _mu_low(x: float) -> float:
        """μ_МАЛО(x): линейно падает от 1 (x=0) до 0 (x=0.5), затем 0."""
        return float(np.clip(1.0 - x / 0.5, 0.0, 1.0))

    @staticmethod
    def _mu_high(x: float) -> float:
        """μ_ВЕЛИКО(x): линейно растёт от 0 (x=0.5) до 1 (x=1.0)."""
        return float(np.clip((x - 0.5) / 0.5, 0.0, 1.0))

    # ── Нечёткий контроллер ───────────────────────────────────────────────────

    def _fuzzy_mutation_prob(self, e1: float, e2: float) -> float:
        """
        Возвращает вероятность мутации p_mut ∈ [0.05, 0.70].

        Метод Мамдани: min для «И», max для активации правил.
        Дефаззификация: центр тяжести по трём синглтонам.

        Синглтоны выходной переменной:
            LOW    = 0.10  (низкая мутация)
            MEDIUM = 0.30  (средняя)
            HIGH   = 0.60  (высокая)
        """
        LOW, MEDIUM, HIGH = 0.10, 0.30, 0.60

        ml_e1 = self._mu_low(e1);   mh_e1 = self._mu_high(e1)
        ml_e2 = self._mu_low(e2);   mh_e2 = self._mu_high(e2)

        # Активация четырёх правил
        r_high   = min(ml_e1, ml_e2)   # оба малы   → HIGH мутация
        r_med_1  = min(ml_e1, mh_e2)   # e1 мал,  e2 велик → MEDIUM
        r_med_2  = min(mh_e1, ml_e2)   # e1 велик, e2 мал  → MEDIUM
        r_low    = min(mh_e1, mh_e2)   # оба велики → LOW мутация

        # Агрегация (max по одинаковым консеквентам)
        w_high   = r_high
        w_medium = max(r_med_1, r_med_2)
        w_low    = r_low

        total = w_high + w_medium + w_low
        if total < 1e-9:
            return 0.30   # нет активации → средняя вероятность по умолчанию

        # Центр тяжести
        p_mut = (w_high * HIGH + w_medium * MEDIUM + w_low * LOW) / total
        return float(np.clip(p_mut, 0.05, 0.70))

    # ── Мутация с адаптивной вероятностью ────────────────────────────────────

    def _mutate_fuzzy(self, ind: dict, p_mut: float) -> dict:
        """
        Мутация с вероятностью p_mut для каждого гена.
        Для активации используем чуть меньшую вероятность (смена активации — редко).
        """
        c = ind.copy()
        if self.rng.random() < p_mut:
            c['n_hidden'] = int(np.clip(
                c['n_hidden'] + int(self.rng.integers(-50, 51)),
                self.hidden_min, self.hidden_max))
        if self.rng.random() < p_mut:
            c['scale'] = float(np.clip(
                c['scale'] + self.rng.normal(0, 1.0),
                self.scale_min, self.scale_max))
        if self.rng.random() < p_mut * 0.5:   # активацию меняем реже
            c['activation'] = str(self.rng.choice(['tanh', 'sin', 'sigmoid']))
        if self.rng.random() < p_mut:
            c['lambda_pde'] = float(np.clip(
                c['lambda_pde'] * self.rng.uniform(0.5, 2.0),
                self.lambda_pde_min, self.lambda_pde_max))
        if self.rng.random() < p_mut:
            c['lambda_bc'] = float(np.clip(
                c['lambda_bc'] * self.rng.uniform(0.5, 2.0),
                self.lambda_bc_min, self.lambda_bc_max))
        return c

    # ── Основной цикл ─────────────────────────────────────────────────────────

    def search(self, X_train, X_b, Y_b, operator_func, source_func,
               p_exact_func, callback=None):
        """
        FGA-поиск с нечётким адаптивным управлением мутацией.

        Дополнительно к базовому callback возвращает self.fuzzy_log —
        список словарей {gen, e1, e2, p_mut} для каждого поколения.
        """
        self.fuzzy_log: list[dict] = []

        pop      = self._init_population()
        f_ave_prev = None   # f_ave(t-1) — нужен для e2

        for gen in range(self.n_gen):
            # ── Оценка фитнеса всей популяции ────────────────────────────────
            fitnesses = [
                self._fitness(ind, X_train, X_b, Y_b,
                              operator_func, source_func, p_exact_func)
                for ind in pop
            ]
            # Отфильтруем inf для статистики
            finite = [f for f in fitnesses if np.isfinite(f)]
            f_max  = min(finite) if finite else 1.0   # у нас минимизация: "лучший" = min
            f_ave  = float(np.mean(finite)) if finite else 1.0

            # ── Сигналы нечёткого контроллера ────────────────────────────────
            # e1: нормированный разброс (насколько средний хуже лучшего)
            e1 = float(np.clip((f_ave - f_max) / (f_ave + 1e-20), 0.0, 1.0))

            # e2: нормированное улучшение среднего за последнее поколение
            if f_ave_prev is None:
                e2 = 0.0
            else:
                e2 = float(np.clip((f_ave_prev - f_ave) / (f_ave_prev + 1e-20),
                                   0.0, 1.0))
            f_ave_prev = f_ave

            p_mut = self._fuzzy_mutation_prob(e1, e2)
            self.fuzzy_log.append({'gen': gen, 'e1': e1, 'e2': e2, 'p_mut': p_mut})

            # ── Сортировка и обновление лучшего ──────────────────────────────
            scored = sorted(zip(fitnesses, pop), key=lambda x: x[0])
            if scored[0][0] < self.best_loss:
                self.best_loss   = scored[0][0]
                self.best_params = scored[0][1]
            self.history.append(self.best_loss)
            if callback:
                callback(gen, self.best_loss, self.best_params)

            # ── Элитизм + скрещивание + нечёткая мутация ─────────────────────
            n_elite = max(1, int(self.n_pop * 0.2))
            elites  = [x[1] for x in scored[:n_elite]]
            new_pop = elites[:]
            while len(new_pop) < self.n_pop:
                p1 = elites[self.rng.integers(len(elites))]
                p2 = elites[self.rng.integers(len(elites))]
                new_pop.append(self._mutate_fuzzy(self._crossover(p1, p2), p_mut))
            pop = new_pop

        return self.best_params