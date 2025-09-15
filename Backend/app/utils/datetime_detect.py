# -*- coding: utf-8 -*-
"""
Riconoscimento e coercizione colonne datetime con euristiche EU/IT.
Compatibile con pandas ≥ 2.x (usa format="mixed" quando possibile).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import re
import pandas as pd

__all__ = [
    "COMMON_DATE_FORMATS",
    "try_parse_datetime",
    "detect_datetime_columns",
    "coerce_datetime_inplace",
]

# Formati espliciti di fallback
COMMON_DATE_FORMATS: List[str] = [
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y",
]

# Pattern grossolano per date con separatori o mesi testuali (es: 12/10/2024, 12 Oct 2024)
_DATEY_LOOK = re.compile(r"^[\d]{1,4}([\-\/\.\s])[\dA-Za-z]{1,3}\1[\d]{2,4}$")

def _infer_dayfirst(series: pd.Series) -> bool:
    """
    Euristica: se la prima componente numerica è > 12 con una certa frequenza,
    assumo giorno-prima (EU). Default EU se la serie è vuota.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return True  # default europeo
    sample = s.sample(min(len(s), 400), random_state=0)
    tokens = sample.str.replace(r"[^\d\/\-\.\s]", "", regex=True).str.split(r"[\/\-\.\s]+", regex=True)
    def _first_num(tok):
        return int(tok[0]) if tok and tok[0].isdigit() else None
    first_nums = tokens.apply(_first_num)
    ratio_day_gt_12 = (first_nums.dropna() > 12).mean() if first_nums.notna().any() else 0.0
    return bool(ratio_day_gt_12 >= 0.2)  # soglia morbida

def try_parse_datetime(
    series: pd.Series,
    min_parse_rate: float = 0.8,
    dayfirst_default: Optional[bool] = None
) -> Optional[pd.Series]:
    """
    Prova a fare il parsing della serie come datetime.
    - Usa prima pd.to_datetime(format="mixed", dayfirst=euristico).
    - Se il tasso di parse è basso, prova i formati in COMMON_DATE_FORMATS.
    - Ritorna la serie parsata se il tasso ≥ min_parse_rate, altrimenti None.
    """
    s_nonnull = series.dropna()
    if len(s_nonnull) == 0:
        return None

    # pre-check veloce: sembra una data?
    sample = s_nonnull.astype(str).sample(min(len(s_nonnull), 200), random_state=0)
    looks_like_date = (
        sample.str.len().between(6, 32) &
        (sample.str.match(_DATEY_LOOK) | sample.str.contains(r"[A-Za-z]{3}", regex=True))
    )
    if looks_like_date.mean() < 0.3:
        return None

    # inferisci dayfirst se non specificato
    dayfirst = _infer_dayfirst(series) if dayfirst_default is None else bool(dayfirst_default)

    # 1) parser misto
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")
    rate = parsed.notna().mean()

    # 2) fallback su formati espliciti
    if rate < min_parse_rate:
        best, best_rate = parsed, rate
        for fmt in COMMON_DATE_FORMATS:
            p = pd.to_datetime(series, errors="coerce", format=fmt)
            r = p.notna().mean()
            if r > best_rate:
                best, best_rate = p, r
            if best_rate >= min_parse_rate:
                break
        parsed, rate = best, best_rate

    return parsed if rate >= min_parse_rate else None

def detect_datetime_columns(df: pd.DataFrame, min_parse_rate: float = 0.8) -> Dict[str, float]:
    """
    Scansiona il DataFrame e restituisce {colonna: tasso_parse} per le colonne
    che risultano coerenti con datetime (≥ min_parse_rate).
    Nota: se una colonna è già dtype datetime64, riporta 1.0.
    """
    results: Dict[str, float] = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            results[col] = 1.0
        elif pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            parsed = try_parse_datetime(ser, min_parse_rate=min_parse_rate, dayfirst_default=None)
            if parsed is not None:
                results[col] = float(parsed.notna().mean())
    return results

def coerce_datetime_inplace(
    df: pd.DataFrame,
    min_parse_rate: float = 0.8,
    dayfirst_default: Optional[bool] = None
) -> List[str]:
    """
    Converte in-place le colonne che superano la soglia di parse come datetime.
    Ritorna la lista delle colonne convertite.
    """
    converted: List[str] = []
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            continue
        if pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            parsed = try_parse_datetime(ser, min_parse_rate=min_parse_rate, dayfirst_default=dayfirst_default)
            if parsed is not None:
                df[col] = parsed
                converted.append(col)
    return converted
