import numpy as np
from .pielm import PIELM, ACTIVATIONS


# ─────────────────────────────────────────────────────────────────────────────
# Генетический оптимизатор гиперпараметров PIELM
#
# Хромосома: {n_hidden, scale, activation, lambda_pde, lambda_bc}
# Фитнес   : RMSE PDE-невязки на X_train
#             (тестовая выборка в GA не участвует — только в финальной оценке)
# ─────────────────────────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Генетический алгоритм для оптимизации гиперпараметров PIELM.

    Параметры
    ----------
    n_pop              : int    — размер популяции
    n_gen              : int    — число поколений
    hidden_bounds      : tuple  — (min, max) для n_hidden
    scale_bounds       : tuple  — (min, max) для scale
    lambda_pde_bounds  : tuple  — (min, max) для lambda_pde
    lambda_bc_bounds   : tuple  — (min, max) для lambda_bc
    elite_frac         : float  — доля элитных особей
    mut_prob           : float  — вероятность мутации каждого гена
    seed               : int    — зерно генератора
    """

    def __init__(self,
                 n_pop=20,
                 n_gen=10,
                 hidden_bounds=(50, 600),
                 scale_bounds=(0.5, 10.0),
                 lambda_pde_bounds=(0.01, 10.0),
                 lambda_bc_bounds=(1.0, 100.0),
                 elite_frac=0.2,
                 mut_prob=0.3,
                 seed=42):

        self.n_pop   = n_pop
        self.n_gen   = n_gen

        self.hidden_min,     self.hidden_max     = hidden_bounds
        self.scale_min,      self.scale_max      = scale_bounds
        self.lambda_pde_min, self.lambda_pde_max = lambda_pde_bounds
        self.lambda_bc_min,  self.lambda_bc_max  = lambda_bc_bounds

        self.elite_frac = elite_frac
        self.mut_prob   = mut_prob

        self.best_params  = None
        self.best_loss    = float('inf')
        self.best_model   = None

        # История: список словарей по каждому поколению
        self.history = []

        self.rng = np.random.default_rng(seed)

    # ── инициализация популяции ────────────────────────────────────────────

    def _init_population(self):
        return [{
            'n_hidden':   int(self.rng.integers(self.hidden_min, self.hidden_max)),
            'scale':      float(self.rng.uniform(self.scale_min, self.scale_max)),
            'activation': str(self.rng.choice(['tanh', 'sin', 'sigmoid'])),
            'lambda_pde': float(self.rng.uniform(self.lambda_pde_min, self.lambda_pde_max)),
            'lambda_bc':  float(self.rng.uniform(self.lambda_bc_min,  self.lambda_bc_max)),
        } for _ in range(self.n_pop)]

    # ── фитнес-функция ─────────────────────────────────────────────────────

    def _fitness(self, ind, X_train, X_b, Y_b,
                 operator_func, source_func, input_dim):
        """
        Обучает PIELM с гиперпараметрами ind и возвращает RMSE невязки PDE.
        При ошибке возвращает inf.
        """
        model = PIELM(
            n_hidden   = ind['n_hidden'],
            input_dim  = input_dim,
            scale      = ind['scale'],
            act_name   = ind['activation'],
            lambda_pde = ind['lambda_pde'],
            lambda_bc  = ind['lambda_bc'],
        )
        try:
            model.fit(X_train, X_b, Y_b, operator_func, source_func)
            loss = model.rmse_pde(X_train, operator_func, source_func)
            return loss, model
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return float('inf'), None

    # ── скрещивание ────────────────────────────────────────────────────────

    def _crossover(self, p1, p2):
        return {
            'n_hidden':   int((p1['n_hidden']   + p2['n_hidden'])   / 2),
            'scale':      (p1['scale']      + p2['scale'])      / 2.0,
            'activation':  p1['activation'] if self.rng.random() < 0.5
                           else p2['activation'],
            'lambda_pde': (p1['lambda_pde'] + p2['lambda_pde']) / 2.0,
            'lambda_bc':  (p1['lambda_bc']  + p2['lambda_bc'])  / 2.0,
        }

    # ── мутация ────────────────────────────────────────────────────────────

    def _mutate(self, ind):
        c = ind.copy()

        if self.rng.random() < self.mut_prob:
            c['n_hidden'] = int(np.clip(
                c['n_hidden'] + int(self.rng.integers(-50, 51)),
                self.hidden_min, self.hidden_max,
            ))

        if self.rng.random() < self.mut_prob:
            c['scale'] = float(np.clip(
                c['scale'] + self.rng.normal(0, 1.0),
                self.scale_min, self.scale_max,
            ))

        if self.rng.random() < 0.2:
            c['activation'] = str(self.rng.choice(['tanh', 'sin', 'sigmoid']))

        if self.rng.random() < self.mut_prob:
            c['lambda_pde'] = float(np.clip(
                c['lambda_pde'] * self.rng.uniform(0.5, 2.0),
                self.lambda_pde_min, self.lambda_pde_max,
            ))

        if self.rng.random() < self.mut_prob:
            c['lambda_bc'] = float(np.clip(
                c['lambda_bc'] * self.rng.uniform(0.5, 2.0),
                self.lambda_bc_min, self.lambda_bc_max,
            ))

        return c

    # ── основной цикл поиска ───────────────────────────────────────────────

    def search(self, X_train, X_b, Y_b,
               operator_func, source_func,
               input_dim=2,
               callback=None):
        """
        Поиск лучших гиперпараметров.

        Параметры
        ----------
        X_train       : ndarray (N_train, d)
        X_b           : ndarray (N_b, d)
        Y_b           : ndarray (N_b,)
        operator_func : callable  — PDE-оператор
        source_func   : callable  — правая часть уравнения
        input_dim     : int       — 2 (Дарси) или 3 (фильтрация)
        callback      : callable или None
                        Сигнатура: callback(gen, best_rmse, best_params)
                        Вызывается после каждого поколения.
                        Используется для обновления прогресс-бара в Streamlit.

        Возвращает
        ----------
        best_params : dict
        """
        self.history     = []
        self.best_params = None
        self.best_loss   = float('inf')
        self.best_model  = None

        pop = self._init_population()

        for gen in range(self.n_gen):

            # Оценка популяции
            scored = []
            for ind in pop:
                loss, model = self._fitness(
                    ind, X_train, X_b, Y_b,
                    operator_func, source_func, input_dim,
                )
                scored.append((loss, ind, model))

            scored.sort(key=lambda x: x[0])

            # Обновление глобального лучшего
            gen_best_loss, gen_best_params, gen_best_model = scored[0]
            if gen_best_loss < self.best_loss:
                self.best_loss   = gen_best_loss
                self.best_params = gen_best_params
                self.best_model  = gen_best_model

            # Запись истории поколения
            self.history.append({
                'generation':  gen + 1,
                'best_rmse':   self.best_loss,
                'gen_rmse':    gen_best_loss,
                'best_params': self.best_params.copy(),
            })

            # Callback для GUI (прогресс-бар, live-обновление графика)
            if callback is not None:
                callback(gen + 1, self.best_loss, self.best_params)

            # Отбор элиты
            n_elite = max(1, int(self.n_pop * self.elite_frac))
            elites  = [x[1] for x in scored[:n_elite]]

            # Формирование нового поколения
            new_pop = elites[:]
            while len(new_pop) < self.n_pop:
                p1 = elites[self.rng.integers(len(elites))]
                p2 = elites[self.rng.integers(len(elites))]
                new_pop.append(self._mutate(self._crossover(p1, p2)))

            pop = new_pop

        return self.best_params

    # ── вспомогательные методы ─────────────────────────────────────────────

    def get_history_arrays(self):
        """
        Возвращает массивы для построения графиков прогресса.

        Возвращает
        ----------
        generations : list[int]
        best_rmse   : list[float]   — глобальный минимум к поколению
        gen_rmse    : list[float]   — лучший в поколении
        """
        generations = [h['generation'] for h in self.history]
        best_rmse   = [h['best_rmse']  for h in self.history]
        gen_rmse    = [h['gen_rmse']   for h in self.history]
        return generations, best_rmse, gen_rmse

    def summary(self):
        """
        Возвращает словарь с результатами оптимизации.
        """
        return {
            'best_params': self.best_params,
            'best_loss':   self.best_loss,
            'n_gen':       self.n_gen,
            'n_pop':       self.n_pop,
        }