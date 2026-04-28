"""
model_io.py — Сохранение и загрузка моделей PIELM.

Форматы:
  - JSON  (.json)  — универсальный, читаемый из любого языка
  - NumPy (.npz)   — компактный бинарный для Python-экосистемы

Использование
-------------
    from models.model_io import save_model, load_model, predict_from_dict

    # Сохранение после обучения
    save_model(model, "model_poisson", cfg=cfg, metrics=metrics)

    # Загрузка и предсказание
    model = load_model("model_poisson.json")
    P = model.predict(X)

    # Или предсказание без класса PIELM (для интеграции)
    data = load_model_dict("model_poisson.json")
    P = predict_from_dict(data, X)
"""

import json
import io
import numpy as np
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Сохранение модели
# ─────────────────────────────────────────────────────────────────────────────

def save_model_json(model, filepath, *, cfg=None, metrics=None):
    """
    Сохраняет обученную модель PIELM в формате JSON.

    Параметры
    ----------
    model    : PIELM          — обученная модель
    filepath : str или Path   — путь к файлу (расширение .json добавляется)
    cfg      : dict, optional — конфигурация задачи (уравнение, источник, ...)
    metrics  : dict, optional — метрики качества (rmse_pde, rmse_test, ...)

    Возвращает
    ----------
    filepath : str — путь к сохранённому файлу
    """
    filepath = Path(filepath)
    if filepath.suffix != '.json':
        filepath = filepath.with_suffix('.json')

    data = {
        # ── Метаинформация ──
        "model_type": "PIELM",
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),

        # ── Архитектура ──
        "architecture": {
            "n_hidden": model.n_hidden,
            "input_dim": model.input_dim,
            "activation": model.act_name,
            "scale": model.scale,
            "lambda_pde": model.lambda_pde,
            "lambda_bc": model.lambda_bc,
        },

        # ── Формула предсказания ──
        # P(X) = activation(X @ W + b) @ beta
        # Это всё, что нужно для инференса в любом языке.
        "inference": {
            "formula": "P = activation(X @ W + b) @ beta",
            "activation_options": {
                "tanh": "tanh(z)",
                "sin": "sin(z)",
                "sigmoid": "1 / (1 + exp(-z))",
            },
        },

        # ── Веса (как вложенные списки для JSON-совместимости) ──
        "weights": {
            "W": model.W.tolist(),         # (input_dim, n_hidden)
            "b": model.b.tolist(),         # (1, n_hidden)
            "beta": model.beta.tolist(),   # (n_hidden, 1)
        },
    }

    # ── Конфигурация задачи (опционально) ──
    if cfg is not None:
        task = {
            "equation": cfg.get("equation_name", "unknown"),
            "is_piezo": cfg.get("is_piezo", False),
            "source_name": cfg.get("source_name", "unknown"),
        }
        # Параметры области
        domain = {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
        if cfg.get("is_piezo"):
            domain["t_range"] = [0.0, cfg.get("T", 1.0)]
            task["kappa"] = cfg.get("kappa", None)
        task["domain"] = domain

        # Параметры источника (только числовые)
        src_params = {}
        for k, v in cfg.get("source_params", {}).items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                src_params[k] = float(v)
        task["source_params"] = src_params
        data["task"] = task

    # ── Метрики (опционально) ──
    if metrics is not None:
        data["metrics"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def save_model_npz(model, filepath, *, cfg=None, metrics=None):
    """
    Сохраняет обученную модель PIELM в формате NumPy .npz.

    Параметры
    ----------
    model    : PIELM          — обученная модель
    filepath : str или Path   — путь к файлу
    cfg      : dict, optional — конфигурация задачи
    metrics  : dict, optional — метрики качества

    Возвращает
    ----------
    filepath : str — путь к сохранённому файлу
    """
    filepath = Path(filepath)
    if filepath.suffix != '.npz':
        filepath = filepath.with_suffix('.npz')

    save_dict = {
        "W": model.W,
        "b": model.b,
        "beta": model.beta,
    }

    # Метаданные как JSON-строка (хранится как 0-мерный массив)
    meta = {
        "model_type": "PIELM",
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "n_hidden": model.n_hidden,
        "input_dim": model.input_dim,
        "activation": model.act_name,
        "scale": model.scale,
        "lambda_pde": model.lambda_pde,
        "lambda_bc": model.lambda_bc,
    }
    if cfg:
        meta["equation"] = cfg.get("equation_name", "unknown")
        meta["is_piezo"] = cfg.get("is_piezo", False)
        meta["kappa"] = cfg.get("kappa", None)
    if metrics:
        meta["metrics"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
    save_dict["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))

    np.savez_compressed(filepath, **save_dict)
    return str(filepath)


def save_model(model, filepath, *, cfg=None, metrics=None, formats=("json", "npz")):
    """
    Сохраняет модель в указанных форматах (по умолчанию — оба).

    Параметры
    ----------
    model    : PIELM
    filepath : str или Path  — базовое имя файла (без расширения)
    cfg      : dict, optional
    metrics  : dict, optional
    formats  : tuple of str  — ('json', 'npz') или подмножество

    Возвращает
    ----------
    paths : dict[str, str] — {формат: путь}
    """
    base = Path(filepath).with_suffix('')
    paths = {}
    if "json" in formats:
        paths["json"] = save_model_json(model, base, cfg=cfg, metrics=metrics)
    if "npz" in formats:
        paths["npz"] = save_model_npz(model, base, cfg=cfg, metrics=metrics)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 2. Сохранение в байты (для Streamlit st.download_button)
# ─────────────────────────────────────────────────────────────────────────────

def model_to_json_bytes(model, *, cfg=None, metrics=None):
    """
    Возвращает JSON-представление модели как bytes (для скачивания).
    """
    # Сохраняем во временный буфер
    data = _build_json_dict(model, cfg=cfg, metrics=metrics)
    return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')


def model_to_npz_bytes(model, *, cfg=None, metrics=None):
    """
    Возвращает NPZ-представление модели как bytes (для скачивания).
    """
    buf = io.BytesIO()
    save_dict = {"W": model.W, "b": model.b, "beta": model.beta}
    meta = {
        "model_type": "PIELM", "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "n_hidden": model.n_hidden, "input_dim": model.input_dim,
        "activation": model.act_name, "scale": model.scale,
        "lambda_pde": model.lambda_pde, "lambda_bc": model.lambda_bc,
    }
    if cfg:
        meta["equation"] = cfg.get("equation_name", "unknown")
        meta["is_piezo"] = cfg.get("is_piezo", False)
        meta["kappa"] = cfg.get("kappa", None)
    if metrics:
        meta["metrics"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
    save_dict["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(buf, **save_dict)
    buf.seek(0)
    return buf.getvalue()


def _build_json_dict(model, *, cfg=None, metrics=None):
    """Внутренняя функция: собирает dict для JSON."""
    data = {
        "model_type": "PIELM", "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "architecture": {
            "n_hidden": model.n_hidden, "input_dim": model.input_dim,
            "activation": model.act_name, "scale": model.scale,
            "lambda_pde": model.lambda_pde, "lambda_bc": model.lambda_bc,
        },
        "inference": {
            "formula": "P = activation(X @ W + b) @ beta",
        },
        "weights": {
            "W": model.W.tolist(),
            "b": model.b.tolist(),
            "beta": model.beta.tolist(),
        },
    }
    if cfg:
        data["task"] = {
            "equation": cfg.get("equation_name", "unknown"),
            "is_piezo": cfg.get("is_piezo", False),
            "kappa": cfg.get("kappa", None),
            "domain": {"x_range": [0, 1], "y_range": [0, 1]},
        }
        if cfg.get("is_piezo"):
            data["task"]["domain"]["t_range"] = [0, cfg.get("T", 1.0)]
    if metrics:
        data["metrics"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. Загрузка модели
# ─────────────────────────────────────────────────────────────────────────────

def load_model_dict(filepath):
    """
    Загружает модель из файла (JSON или NPZ) в виде словаря.

    Параметры
    ----------
    filepath : str или Path

    Возвращает
    ----------
    data : dict  с ключами:
        - W     : ndarray (input_dim, n_hidden)
        - b     : ndarray (1, n_hidden)
        - beta  : ndarray (n_hidden, 1)
        - activation : str ('tanh', 'sin', 'sigmoid')
        - n_hidden   : int
        - input_dim  : int
        - meta       : dict  — вся метаинформация
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return {
            "W": np.array(raw["weights"]["W"]),
            "b": np.array(raw["weights"]["b"]),
            "beta": np.array(raw["weights"]["beta"]),
            "activation": raw["architecture"]["activation"],
            "n_hidden": raw["architecture"]["n_hidden"],
            "input_dim": raw["architecture"]["input_dim"],
            "meta": raw,
        }

    elif filepath.suffix == '.npz':
        npz = np.load(filepath, allow_pickle=True)
        meta = json.loads(str(npz["meta_json"]))
        return {
            "W": npz["W"],
            "b": npz["b"],
            "beta": npz["beta"],
            "activation": meta["activation"],
            "n_hidden": meta["n_hidden"],
            "input_dim": meta["input_dim"],
            "meta": meta,
        }

    else:
        raise ValueError(f"Неизвестный формат: {filepath.suffix}. Ожидается .json или .npz")


def load_model(filepath):
    """
    Загружает модель из файла и возвращает готовый экземпляр PIELM.

    Параметры
    ----------
    filepath : str или Path

    Возвращает
    ----------
    model : PIELM — обученная модель, готовая к predict()
    """
    from models.pielm import PIELM

    data = load_model_dict(filepath)

    model = PIELM(
        n_hidden=data["n_hidden"],
        input_dim=data["input_dim"],
        act_name=data["activation"],
    )
    model.W = data["W"]
    model.b = data["b"]
    model.beta = data["beta"]

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Предсказание без класса PIELM (для интеграции в другие системы)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVATIONS = {
    'tanh':    np.tanh,
    'sin':     np.sin,
    'sigmoid': lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
}


def predict_from_dict(model_data, X):
    """
    Вычисляет P(X) по загруженному словарю модели — без класса PIELM.

    Это минимальная функция инференса, которую легко
    перенести на любой язык программирования.

    Параметры
    ----------
    model_data : dict — результат load_model_dict()
    X          : ndarray (N, input_dim)

    Возвращает
    ----------
    P : ndarray (N,)

    Пример
    ------
    >>> data = load_model_dict("model.json")
    >>> X = np.array([[0.5, 0.3], [0.2, 0.8]])
    >>> P = predict_from_dict(data, X)
    """
    W = model_data["W"]           # (input_dim, n_hidden)
    b = model_data["b"]           # (1, n_hidden)
    beta = model_data["beta"]     # (n_hidden, 1)
    act = _ACTIVATIONS[model_data["activation"]]

    Z = X @ W + b                 # (N, n_hidden)
    H = act(Z)                    # (N, n_hidden)
    P = (H @ beta).ravel()        # (N,)
    return P


def predict_from_json(json_path, X):
    """
    Загружает JSON и предсказывает P(X) — одна функция для всего.

    Параметры
    ----------
    json_path : str — путь к JSON-файлу модели
    X         : ndarray (N, input_dim)

    Возвращает
    ----------
    P : ndarray (N,)
    """
    return predict_from_dict(load_model_dict(json_path), X)