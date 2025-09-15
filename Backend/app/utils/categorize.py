# -*- coding: utf-8 -*-
"""
Classificazione colonne: numeriche (continue/discrete), categoriche, boolean,
datetime (già gestite a monte), testo libero e long text (euristiche).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd

__all__ = [
    "categorize_columns",
]

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def _text_stats(ser: pd.Series) -> Dict[str, float]:
    """
    Statistiche di base per distinguere testo breve vs long text.
    """
    s = ser.dropna().astype(str)
    if s.empty:
        return dict(avg_chars=0.0, avg_words=0.0, unique_token_ratio=0.0, pct_long_lines=0.0)
    lens = s.str.len()
    words = s.str.findall(_WORD_RE).apply(len)

    # campione per calcolare vocabulary ratio senza esplodere in RAM
    sample = s.sample(min(len(s), 500), random_state=0)
    tokens = _WORD_RE.findall(" ".join(sample.tolist()).lower())
    vocab = set(tokens)
    utr = len(vocab) / max(len(tokens), 1)

    return dict(
        avg_chars=float(lens.mean()),
        avg_words=float(words.mean()),
        unique_token_ratio=float(utr),
        pct_long_lines=float((lens >= 120).mean()),
    )

def _looks_like_long_text(
    stats: Dict[str, float],
    min_avg_chars: int = 120,
    min_avg_words: int = 20,
    min_utr: float = 0.2
) -> bool:
    """
    Heuristica long text: oppure avg_chars o avg_words sopra soglia
    + una minima varietà lessicale.
    """
    return (
        (stats["avg_chars"] >= min_avg_chars or stats["avg_words"] >= min_avg_words)
        and (stats["unique_token_ratio"] >= min_utr)
    )

def _categorical_values_preview(
    ser: pd.Series,
    max_cat_values: int = 10
) -> Tuple[Optional[List[str]], int, bool]:
    """
    Ritorna (values_preview, n_unique, truncated).
    Se n_unique > max_cat_values => values_preview=None, truncated=True.
    """
    nuniq = ser.nunique(dropna=True)
    if nuniq == 0:
        return [], 0, False
    if nuniq > max_cat_values:
        return None, int(nuniq), True
    vals = sorted(ser.dropna().astype(str).unique().tolist(), key=lambda x: x.lower())
    return vals, int(nuniq), False

def categorize_columns(
    df: pd.DataFrame,
    threshold_cat: int = 20,
    max_cat_values: int = 10,
    nlp_avg_chars: int = 120,
    nlp_avg_words: int = 20,
    nlp_min_unique_token_ratio: float = 0.2,
    auto_parse_dates: bool = False,  # la coercizione date è gestita a monte
) -> Dict[str, object]:
    """
    Classifica le colonne del DataFrame in:
    - numeriche_continue / numeriche_discrete
    - categoriche (con anteprima e flag truncated)
    - boolean
    - datetime
    - testo_libero
    - long_text_nlp (con stats)
    Ritorna un dict serializzabile.
    """
    cats: Dict[str, object] = {
        "numeriche_continue": [],
        "numeriche_discrete": [],
        "categoriche": {},      # {col: {n_unique, values/None, truncated}}
        "boolean": [],
        "datetime": [],
        "testo_libero": [],
        "long_text_nlp": {},    # {col: stats}
    }

    for col in df.columns:
        ser = df[col]

        # boolean: dtype bool o interi {0,1}
        if pd.api.types.is_bool_dtype(ser) or (
            pd.api.types.is_integer_dtype(ser) and ser.dropna().isin([0, 1]).all()
        ):
            cats["boolean"].append(col)
            continue

        # datetime
        if pd.api.types.is_datetime64_any_dtype(ser):
            cats["datetime"].append(col)
            continue

        # numeriche
        if pd.api.types.is_numeric_dtype(ser):
            if ser.nunique(dropna=True) <= threshold_cat:
                cats["numeriche_discrete"].append(col)
            else:
                cats["numeriche_continue"].append(col)
            continue

        # object / category → testo o categorica
        if pd.api.types.is_object_dtype(ser) or pd.api.types.is_categorical_dtype(ser):
            stats = _text_stats(ser)
            if _looks_like_long_text(stats, nlp_avg_chars, nlp_avg_words, nlp_min_unique_token_ratio):
                cats["long_text_nlp"][col] = stats
            else:
                nuniq = ser.nunique(dropna=True)
                if nuniq <= threshold_cat:
                    values, n_unique, truncated = _categorical_values_preview(ser, max_cat_values=max_cat_values)
                    cats["categoriche"][col] = {
                        "n_unique": int(n_unique),
                        "values": values,
                        "truncated": bool(truncated),
                    }
                else:
                    cats["testo_libero"].append(col)
            continue

        # fallback: tratta come categorica con anteprima
        values, n_unique, truncated = _categorical_values_preview(ser, max_cat_values=max_cat_values)
        cats["categoriche"][col] = {
            "n_unique": int(n_unique),
            "values": values,
            "truncated": bool(truncated),
        }

    # normalizza tipi delle liste (per sicurezza)
    for k in ["numeriche_continue", "numeriche_discrete", "boolean", "datetime", "testo_libero"]:
        cats[k] = list(cats[k])  # type: ignore[assignment]

    return cats
