import os, time, yaml
from pathlib import Path
from typing import Iterable, List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------- Config / FS ----------
def load_cfg(path: str = "app/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths: Iterable[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 3)

def time_np_load(path: str):
    t0 = time.perf_counter()
    arr = np.load(path)
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return arr, ms

def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dirs(Path(path).parent)
    np.save(path, arr)

# ---------- Math / Bench ----------
def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)

def to_dtype_and_norm(x: np.ndarray, dtype) -> np.ndarray:
    return l2norm(x.astype(dtype, copy=False))

def percentile(arr: Iterable[float], p: float) -> float:
    return float(np.percentile(arr, p))

def warmup_dot(M: np.ndarray, Q: np.ndarray, k: int = 10, n_warm: int = 2) -> None:
    n_warm = min(n_warm, len(Q))
    for i in range(n_warm):
        sims = Q[i] @ M.T
        _ = np.argpartition(sims, -k)[-k:]

def time_query_dot(q: np.ndarray, M: np.ndarray, k: int = 10) -> float:
    import time as _t
    t0 = _t.perf_counter()
    sims = q @ M.T
    _ = np.argpartition(sims, -k)[-k:]
    return (_t.perf_counter() - t0) * 1000.0

def bench_latency(M: np.ndarray, Q: np.ndarray, k: int = 10) -> tuple[float, float]:
    warmup_dot(M, Q, k)
    lat = [time_query_dot(Q[i], M, k) for i in range(len(Q))]
    return round(percentile(lat, 50), 2), round(percentile(lat, 95), 2)

# ---------- Embedding ----------
def load_model(model_id: str):
    return SentenceTransformer(model_id)

def embed_texts(model, texts: List[str], dtype=np.float32) -> np.ndarray:
    arr = model.encode(texts, convert_to_numpy=True).astype(np.float32, copy=False)
    return arr.astype(dtype) if dtype is not None and dtype != np.float32 else arr

# ---------- Quality ----------
def recall_at_k_from_labels(
    topk_idx_per_query: List[np.ndarray],
    q_labels: List[str],
    corpus_labels: List[str],
    k: int = 10,
) -> float:
    hits = 0
    for i, idxs in enumerate(topk_idx_per_query):
        labs = {corpus_labels[j] for j in idxs[:k]}
        hits += (q_labels[i] in labs)
    return hits / len(topk_idx_per_query)

# ---------- Cost ----------
def add_money_columns(df: pd.DataFrame, cost_per_gb_month: float) -> pd.DataFrame:
    df = df.copy()
    df["monthly_cost_usd"] = ((df["size_mb"] / 1024.0) * cost_per_gb_month).round(4)
    df["annual_cost_usd"] = (df["monthly_cost_usd"] * 12.0).round(4)
    return df

def add_savings_vs_fp32(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["savings_vs_fp32_usd_month"] = 0.0
    for n in df["n_docs"].unique():
        base = df[(df["n_docs"] == n) & (df["storage_mode"] == "fp32_npy")]["monthly_cost_usd"].iloc[0]
        mask = df["n_docs"] == n
        df.loc[mask, "savings_vs_fp32_usd_month"] = (base - df.loc[mask, "monthly_cost_usd"]).round(4)
    return df
