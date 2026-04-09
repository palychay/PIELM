import io
import json
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Экспорт результатов в текстовый файл
# ─────────────────────────────────────────────────────────────────────────────

def build_report_txt(
    equation_name,
    equation_params,
    source_name,
    source_params,
    collocation_params,
    pielm_params,
    ga_params,
    ga_history,
    best_hyperparams,
    pielm_metrics,
    fdm_metrics,
    fdm_enabled=True,
):
    """
    Формирует текстовый отчёт о запуске.

    Параметры
    ----------
    equation_name      : str   — название уравнения
    equation_params    : dict  — параметры уравнения (kappa и т.д.)
    source_name        : str   — название источникового члена
    source_params      : dict  — параметры источника
    collocation_params : dict  — стратегия и число точек
    pielm_params       : dict  — гиперпараметры PIELM (если ручной режим)
    ga_params          : dict  — параметры GA (n_pop, n_gen и т.д.)
    ga_history         : list[dict] — история поколений (или None)
    best_hyperparams   : dict  — лучшие гиперпараметры
    pielm_metrics      : dict  — {'rmse_pde': ..., 'rmse_train': ...,
                                   'rmse_test': ..., 'time': ...}
    fdm_metrics        : dict  — {'rmse': ..., 'time': ..., 'n_grid': ...}
    fdm_enabled        : bool

    Возвращает
    ----------
    txt : str
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    def section(title):
        lines.append('')
        lines.append('=' * 60)
        lines.append(f'  {title}')
        lines.append('=' * 60)

    def row(key, val, indent=2):
        lines.append(f'{" " * indent}{key:<35} {val}')

    # ── Заголовок ────────────────────────────────────────────────────────────
    lines.append('PIELM + Генетический алгоритм — Отчёт о вычислении')
    lines.append(f'Дата и время: {ts}')

    # ── Уравнение ────────────────────────────────────────────────────────────
    section('1. УРАВНЕНИЕ')
    row('Уравнение:', equation_name)
    for k, v in equation_params.items():
        row(f'  {k}:', v)
    row('Источниковый член:', source_name)
    for k, v in source_params.items():
        row(f'  {k}:', v)

    # ── Коллокационные точки ─────────────────────────────────────────────────
    section('2. КОЛЛОКАЦИОННЫЕ ТОЧКИ')
    for k, v in collocation_params.items():
        row(k + ':', v)

    # ── Гиперпараметры PIELM ─────────────────────────────────────────────────
    section('3. ГИПЕРПАРАМЕТРЫ PIELM (ЛУЧШИЕ)')
    for k, v in best_hyperparams.items():
        row(k + ':', v)

    # ── Параметры GA (если применялся) ───────────────────────────────────────
    if ga_history is not None:
        section('4. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ')
        for k, v in ga_params.items():
            row(k + ':', v)
        lines.append('')
        lines.append('  История поколений:')
        lines.append(f'  {"Поколение":<12} {"Лучший RMSE":<20} {"RMSE поколения":<20}')
        lines.append('  ' + '-' * 52)
        for h in ga_history:
            lines.append(
                f'  {h["generation"]:<12} '
                f'{h["best_rmse"]:<20.8f} '
                f'{h["gen_rmse"]:<20.8f}'
            )
    else:
        section('4. РЕЖИМ')
        row('Режим:', 'Ручной (без генетической оптимизации)')
        for k, v in pielm_params.items():
            row(k + ':', v)

    # ── Метрики PIELM ────────────────────────────────────────────────────────
    section('5. МЕТРИКИ PIELM')
    row('RMSE невязки PDE (train):',  f'{pielm_metrics["rmse_pde"]:.8f}')
    row('RMSE на train-точках:',       f'{pielm_metrics["rmse_train"]:.8f}')
    row('RMSE на test-точках:',        f'{pielm_metrics["rmse_test"]:.8f}')
    row('Время вычисления:',           f'{pielm_metrics["time"]:.4f} с')

    # ── Метрики МКР ──────────────────────────────────────────────────────────
    if fdm_enabled and fdm_metrics is not None:
        section('6. МЕТРИКИ МКР')
        row('Число внутренних узлов (n_grid):',
            str(fdm_metrics.get('n_grid', '—')))
        row('RMSE МКР (vs PIELM на test):',
            f'{fdm_metrics["rmse"]:.8f}')
        row('Время вычисления МКР:',
            f'{fdm_metrics["time"]:.4f} с')

        section('7. СРАВНЕНИЕ')
        rmse_ratio = (fdm_metrics['rmse'] / pielm_metrics['rmse_test']
                      if pielm_metrics['rmse_test'] > 0 else float('inf'))
        time_ratio = (fdm_metrics['time'] / pielm_metrics['time']
                      if pielm_metrics['time'] > 0 else float('inf'))
        row('RMSE МКР / RMSE PIELM:', f'{rmse_ratio:.4f}')
        row('Время МКР / Время PIELM:', f'{time_ratio:.4f}')

    lines.append('')
    lines.append('=' * 60)
    lines.append('Конец отчёта')

    return '\n'.join(lines)


def report_to_bytes(txt):
    """Конвертирует строку отчёта в байты UTF-8 для st.download_button."""
    return txt.encode('utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Экспорт графиков — объединение нескольких PNG в один ZIP
# ─────────────────────────────────────────────────────────────────────────────

def build_plots_zip(plots_dict):
    """
    Упаковывает словарь {имя_файла: bytes_png} в ZIP-архив.

    Параметры
    ----------
    plots_dict : dict[str, bytes]
        Например: {
            'ga_progress.png':    <bytes>,
            'pielm_solution.png': <bytes>,
            'comparison.png':     <bytes>,
        }

    Возвращает
    ----------
    zip_bytes : bytes
    """
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in plots_dict.items():
            zf.writestr(filename, data)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Формирование имён файлов с временной меткой
# ─────────────────────────────────────────────────────────────────────────────

def make_filename(prefix, ext):
    """
    Формирует имя файла с временной меткой.

    Например: 'report_20250101_120000.txt'

    Параметры
    ----------
    prefix : str — префикс ('report', 'plots', и т.д.)
    ext    : str — расширение без точки ('txt', 'zip', 'png')

    Возвращает
    ----------
    filename : str
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{ts}.{ext}'