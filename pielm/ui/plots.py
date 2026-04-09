import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import io


# ─────────────────────────────────────────────────────────────────────────────
# Единая цветовая схема и стиль
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    'cmap_solution':  'RdYlBu_r',   # тепловая карта давления
    'cmap_error':     'hot_r',       # тепловая карта ошибки
    'cmap_diff':      'RdBu_r',      # карта разности (знакопеременная)
    'color_pielm':    '#2563EB',     # синий — PIELM
    'color_fdm':      '#DC2626',     # красный — МКР
    'color_best':     '#16A34A',     # зелёный — лучший результат
    'color_gen':      '#9333EA',     # фиолетовый — поколение
    'figsize_single': (6, 5),
    'figsize_double': (12, 5),
    'figsize_triple': (15, 5),
    'figsize_tall':   (8, 6),
    'dpi':            120,
    'fontsize_title': 12,
    'fontsize_label': 10,
    'fontsize_tick':  9,
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})


def _fig_to_bytes(fig):
    """Конвертирует Figure в байты PNG для Streamlit st.image и скачивания."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=STYLE['dpi'],
                bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1. График прогресса генетического алгоритма
# ─────────────────────────────────────────────────────────────────────────────

def plot_ga_progress(generations, best_rmse, gen_rmse):
    """
    Две кривые RMSE по поколениям:
      — глобальный минимум (накопленный лучший)
      — лучший в поколении

    Параметры
    ----------
    generations : list[int]
    best_rmse   : list[float] — глобальный минимум к поколению
    gen_rmse    : list[float] — лучший в текущем поколении

    Возвращает
    ----------
    bytes PNG
    """
    fig, ax = plt.subplots(figsize=STYLE['figsize_tall'], dpi=STYLE['dpi'])

    ax.plot(generations, gen_rmse,
            color=STYLE['color_gen'], lw=1.5, ls='--',
            marker='o', ms=4, alpha=0.7, label='Лучший в поколении')

    ax.plot(generations, best_rmse,
            color=STYLE['color_best'], lw=2.5,
            marker='s', ms=5, label='Глобальный минимум')

    # Отмечаем финальное значение
    ax.annotate(
        f'  {best_rmse[-1]:.5f}',
        xy=(generations[-1], best_rmse[-1]),
        fontsize=STYLE['fontsize_label'],
        color=STYLE['color_best'],
        fontweight='bold',
    )

    ax.set_xlabel('Поколение', fontsize=STYLE['fontsize_label'])
    ax.set_ylabel('RMSE невязки PDE', fontsize=STYLE['fontsize_label'])
    ax.set_title('Прогресс генетической оптимизации',
                 fontsize=STYLE['fontsize_title'], fontweight='bold')
    ax.legend(fontsize=STYLE['fontsize_label'])
    ax.tick_params(labelsize=STYLE['fontsize_tick'])
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Тепловые карты решения PIELM
# ─────────────────────────────────────────────────────────────────────────────

def plot_pielm_solution(P_pred, xx, yy, title='Решение PIELM: P(x,y)'):
    """
    Одна тепловая карта предсказанного поля давления.

    Параметры
    ----------
    P_pred : ndarray (N,)  — предсказанные значения на сетке
    xx, yy : ndarray (n,n) — координатные сетки
    title  : str

    Возвращает
    ----------
    bytes PNG
    """
    n   = xx.shape[0]
    P2d = P_pred.reshape(n, n)

    fig, ax = plt.subplots(figsize=STYLE['figsize_single'], dpi=STYLE['dpi'])

    im = ax.pcolormesh(xx, yy, P2d,
                       cmap=STYLE['cmap_solution'], shading='auto')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=STYLE['fontsize_tick'])
    cb.set_label('P', fontsize=STYLE['fontsize_label'])

    ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
    ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
    ax.set_title(title, fontsize=STYLE['fontsize_title'], fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=STYLE['fontsize_tick'])
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_pielm_vs_fdm(P_pielm, P_fdm, xx, yy):
    """
    Три тепловые карты рядом:
      1. PIELM — предсказание
      2. МКР   — предсказание
      3. |PIELM - МКР| — разность

    Параметры
    ----------
    P_pielm : ndarray (N,)
    P_fdm   : ndarray (N,)
    xx, yy  : ndarray (n,n)

    Возвращает
    ----------
    bytes PNG
    """
    n      = xx.shape[0]
    P1     = P_pielm.reshape(n, n)
    P2     = P_fdm.reshape(n, n)
    P_diff = np.abs(P1 - P2)

    fig, axes = plt.subplots(1, 3, figsize=STYLE['figsize_triple'],
                             dpi=STYLE['dpi'])

    # Общий диапазон для первых двух карт
    vmin = min(P1.min(), P2.min())
    vmax = max(P1.max(), P2.max())

    titles = ['PIELM', 'МКР (конечные разности)', '|PIELM − МКР|']
    data   = [P1, P2, P_diff]
    cmaps  = [STYLE['cmap_solution'], STYLE['cmap_solution'], STYLE['cmap_error']]

    for ax, d, t, cm in zip(axes, data, titles, cmaps):
        if t.startswith('|'):
            im = ax.pcolormesh(xx, yy, d, cmap=cm, shading='auto')
        else:
            im = ax.pcolormesh(xx, yy, d, cmap=cm, shading='auto',
                               vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=STYLE['fontsize_tick'])
        ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
        ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
        ax.set_title(t, fontsize=STYLE['fontsize_title'], fontweight='bold')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=STYLE['fontsize_tick'])

    fig.suptitle('Сравнение методов: поля давления',
                 fontsize=STYLE['fontsize_title'] + 1,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Тепловые карты для нестационарного случая (срез по времени)
# ─────────────────────────────────────────────────────────────────────────────

def plot_piezo_time_slice(P_pielm, P_fdm, xx, yy, t_val):
    """
    Три тепловые карты для фиксированного момента времени t:
      1. PIELM
      2. МКР
      3. |PIELM - МКР|

    Параметры
    ----------
    P_pielm : ndarray (N,)
    P_fdm   : ndarray (N,)
    xx, yy  : ndarray (n,n)
    t_val   : float — момент времени для заголовка

    Возвращает
    ----------
    bytes PNG
    """
    n      = xx.shape[0]
    P1     = P_pielm.reshape(n, n)
    P2     = P_fdm.reshape(n, n)
    P_diff = np.abs(P1 - P2)

    fig, axes = plt.subplots(1, 3, figsize=STYLE['figsize_triple'],
                             dpi=STYLE['dpi'])

    vmin = min(P1.min(), P2.min())
    vmax = max(P1.max(), P2.max())

    titles = [f'PIELM, t={t_val:.2f}',
              f'МКР, t={t_val:.2f}',
              f'|PIELM − МКР|, t={t_val:.2f}']
    data   = [P1, P2, P_diff]
    cmaps  = [STYLE['cmap_solution'], STYLE['cmap_solution'], STYLE['cmap_error']]

    for ax, d, t, cm in zip(axes, data, titles, cmaps):
        if '|' in t:
            im = ax.pcolormesh(xx, yy, d, cmap=cm, shading='auto')
        else:
            im = ax.pcolormesh(xx, yy, d, cmap=cm, shading='auto',
                               vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=STYLE['fontsize_tick'])
        ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
        ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
        ax.set_title(t, fontsize=STYLE['fontsize_title'], fontweight='bold')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=STYLE['fontsize_tick'])

    fig.suptitle('Уравнение пьезопроводности: сравнение методов',
                 fontsize=STYLE['fontsize_title'] + 1,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_piezo_evolution(P_snapshots, xx, yy, t_vals):
    """
    Несколько снимков поля давления PIELM в разные моменты времени.

    Параметры
    ----------
    P_snapshots : list[ndarray (N,)] — предсказания в моменты t_vals
    xx, yy      : ndarray (n,n)
    t_vals      : list[float]

    Возвращает
    ----------
    bytes PNG
    """
    n_snap = len(P_snapshots)
    n      = xx.shape[0]

    fig, axes = plt.subplots(1, n_snap,
                             figsize=(5 * n_snap, 5),
                             dpi=STYLE['dpi'])
    if n_snap == 1:
        axes = [axes]

    # Общий диапазон для всех снимков
    all_vals = np.concatenate([p.ravel() for p in P_snapshots])
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, P, t in zip(axes, P_snapshots, t_vals):
        P2d = P.reshape(n, n)
        im  = ax.pcolormesh(xx, yy, P2d,
                            cmap=STYLE['cmap_solution'], shading='auto',
                            vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, pad=0.02).ax.tick_params(
            labelsize=STYLE['fontsize_tick']
        )
        ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
        ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
        ax.set_title(f't = {t:.2f}', fontsize=STYLE['fontsize_title'],
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=STYLE['fontsize_tick'])

    fig.suptitle('PIELM: эволюция поля давления',
                 fontsize=STYLE['fontsize_title'] + 1,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Сравнительный график метрик (RMSE, время)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(metrics):
    """
    Два bar-графика рядом: RMSE и время вычисления.

    Параметры
    ----------
    metrics : dict
        {
          'PIELM': {'rmse': float, 'time': float},
          'МКР':   {'rmse': float, 'time': float},
        }

    Возвращает
    ----------
    bytes PNG
    """
    methods = list(metrics.keys())
    rmse    = [metrics[m]['rmse'] for m in methods]
    times   = [metrics[m]['time'] for m in methods]
    colors  = [STYLE['color_pielm'], STYLE['color_fdm']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=STYLE['figsize_double'],
                                   dpi=STYLE['dpi'])

    # ── RMSE ──
    bars1 = ax1.bar(methods, rmse, color=colors, width=0.5,
                    edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars1, rmse):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(rmse) * 0.01,
                 f'{val:.5f}',
                 ha='center', va='bottom',
                 fontsize=STYLE['fontsize_label'], fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=STYLE['fontsize_label'])
    ax1.set_title('Точность решения', fontsize=STYLE['fontsize_title'],
                  fontweight='bold')
    ax1.tick_params(labelsize=STYLE['fontsize_tick'])
    ax1.set_ylim(0, max(rmse) * 1.2)

    # ── Время ──
    bars2 = ax2.bar(methods, times, color=colors, width=0.5,
                    edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(times) * 0.01,
                 f'{val:.3f} с',
                 ha='center', va='bottom',
                 fontsize=STYLE['fontsize_label'], fontweight='bold')
    ax2.set_ylabel('Время вычисления, с', fontsize=STYLE['fontsize_label'])
    ax2.set_title('Скорость решения', fontsize=STYLE['fontsize_title'],
                  fontweight='bold')
    ax2.tick_params(labelsize=STYLE['fontsize_tick'])
    ax2.set_ylim(0, max(times) * 1.2)

    fig.suptitle('Сравнение PIELM и МКР',
                 fontsize=STYLE['fontsize_title'] + 1,
                 fontweight='bold')
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Распределение коллокационных точек
# ─────────────────────────────────────────────────────────────────────────────

def plot_collocation_points(X_train, X_test, X_b, dim=2):
    """
    Визуализация распределения коллокационных точек.

    Параметры
    ----------
    X_train : ndarray (N_train, d)
    X_test  : ndarray (N_test,  d)
    X_b     : ndarray (N_b,     d)
    dim     : int — 2 или 3 (для 3D показывается проекция x-y)

    Возвращает
    ----------
    bytes PNG
    """
    fig, ax = plt.subplots(figsize=STYLE['figsize_single'], dpi=STYLE['dpi'])

    ax.scatter(X_train[:, 0], X_train[:, 1],
               c=STYLE['color_pielm'], s=8, alpha=0.5, label=f'Train ({len(X_train)})')
    ax.scatter(X_test[:, 0],  X_test[:, 1],
               c=STYLE['color_fdm'], s=8, alpha=0.5, label=f'Test ({len(X_test)})')
    ax.scatter(X_b[:, 0],     X_b[:, 1],
               c=STYLE['color_best'], s=15, alpha=0.8,
               marker='x', label=f'Граница ({len(X_b)})')

    ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
    ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
    suffix = ' (проекция x-y)' if dim == 3 else ''
    ax.set_title(f'Коллокационные точки{suffix}',
                 fontsize=STYLE['fontsize_title'], fontweight='bold')
    ax.legend(fontsize=STYLE['fontsize_label'] - 1, markerscale=2)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=STYLE['fontsize_tick'])
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Поле источника f(x,y) или q(x,y,t)
# ─────────────────────────────────────────────────────────────────────────────

def plot_source_field(F_vals, xx, yy, title='Поле источника f(x,y)'):
    """
    Тепловая карта источникового члена.

    Параметры
    ----------
    F_vals : ndarray (N,)  — значения источника на сетке xx, yy
    xx, yy : ndarray (n,n)
    title  : str

    Возвращает
    ----------
    bytes PNG
    """
    n   = xx.shape[0]
    F2d = F_vals.reshape(n, n)

    # Знакопеременный источник — центрированная цветовая шкала
    abs_max = np.abs(F2d).max()
    if abs_max > 0:
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        cmap = STYLE['cmap_diff']
    else:
        norm = None
        cmap = STYLE['cmap_solution']

    fig, ax = plt.subplots(figsize=STYLE['figsize_single'], dpi=STYLE['dpi'])
    im = ax.pcolormesh(xx, yy, F2d, cmap=cmap, norm=norm, shading='auto')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=STYLE['fontsize_tick'])
    cb.set_label('f(x,y)', fontsize=STYLE['fontsize_label'])

    ax.set_xlabel('x', fontsize=STYLE['fontsize_label'])
    ax.set_ylabel('y', fontsize=STYLE['fontsize_label'])
    ax.set_title(title, fontsize=STYLE['fontsize_title'], fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=STYLE['fontsize_tick'])
    fig.tight_layout()
    return _fig_to_bytes(fig)