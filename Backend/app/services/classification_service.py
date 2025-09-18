# app/services/classification_service.py
from typing import Dict, Any, List
import pandas as pd
from app.utils.io import read_dataframe
from app.utils.datetime_detect import coerce_datetime_inplace, detect_datetime_columns
from app.utils.categorize import categorize_columns
from app.storage.memory_store import save_dataset

# --- in cima al file ---
CATEGORY_PALETTE = {
    "numeriche_continue": "#4C78A8",
    "numeriche_discrete": "#9EC9E6",
    "categoriche":        "#F58518",
    "boolean":            "#54A24B",
    "datetime":           "#B279A2",
    "testo_libero":       "#E45756",
    "long_text_nlp":      "#72B7B2",
    "unknown":            "#BAB0AC",
}

def _dtype_name(s) -> str:
    try:
        return str(s.dtype)
    except Exception:
        return "unknown"

def _sample_values(series, n=6):
    try:
        vals = series.dropna().astype(str).unique()[:n]
        return [str(v) for v in vals.tolist()]
    except Exception:
        return None

def schema_to_columns_list(schema: dict, df: pd.DataFrame) -> list[dict]:
    """
    Converte lo schema (dict) in una lista di colonne:
    [{name, category, dtype, sample_values}, ...]
    """
    cols = []
    # map col -> category
    cat_map = {}

    for c in schema.get("numeriche_continue", []): cat_map[c] = "numeriche_continue"
    for c in schema.get("numeriche_discrete", []): cat_map[c] = "numeriche_discrete"
    for c in schema.get("boolean", []):            cat_map[c] = "boolean"
    for c in schema.get("datetime", []):           cat_map[c] = "datetime"
    for c in schema.get("testo_libero", []):       cat_map[c] = "testo_libero"
    for c in schema.get("long_text_nlp", {}).keys(): cat_map[c] = "long_text_nlp"
    for c in schema.get("categoriche", {}).keys():   cat_map[c] = "categoriche"

    for name in df.columns:
        category = cat_map.get(name, "unknown")
        dtype = _dtype_name(df[name])
        if category == "categoriche":
            info = schema["categoriche"].get(name, {})
            sample = info.get("values")  # già pronto se non troncato
            if sample is None:  # se troncato, prendo qualche valore dal df
                sample = _sample_values(df[name], n=6)
        elif category == "long_text_nlp":
            sample = None
        else:
            sample = _sample_values(df[name], n=6)

        cols.append({
            "name": name,
            "category": category,
            "dtype": dtype,
            "sample_values": sample
        })
    return cols


def classify_file(filename: str, content: bytes,
                  delimiter=None, decimal=None, encoding=None,
                  dayfirst: bool | None = None,
                  threshold_cat: int = 20,
                  max_cat_values: int = 10,
                  nlp_avg_chars: int = 120,
                  nlp_avg_words: int = 20,
                  nlp_min_unique_token_ratio: float = 0.2,
                  auto_parse_dates: bool = True) -> Dict[str, Any]:

    df, meta = read_dataframe(filename, content, delimiter=delimiter, decimal=decimal, encoding=encoding)
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        raise ValueError("File senza dati")

    # Date parsing (prima/dopo per tassi)
    before_rates = detect_datetime_columns(df, min_parse_rate=0.0)
    converted = coerce_datetime_inplace(df, min_parse_rate=0.8, dayfirst_default=dayfirst)
    after_rates = detect_datetime_columns(df, min_parse_rate=0.0)

    schema = categorize_columns(
        df=df,
        threshold_cat=threshold_cat,
        max_cat_values=max_cat_values,
        nlp_avg_chars=nlp_avg_chars,
        nlp_avg_words=nlp_avg_words,
        nlp_min_unique_token_ratio=nlp_min_unique_token_ratio,
        auto_parse_dates=auto_parse_dates
    )

    columns = schema_to_columns_list(schema, df)   # <-- NEW


    warnings: List[str] = []
    for col, cfg in schema.get("categoriche", {}).items():
        if bool(cfg.get("truncated")):
            warnings.append(f"Valori categoria di '{col}' troncati a top-{max_cat_values} (truncated=true).")

    dataset_id = save_dataset(df)

    return {
    "dataset_id": dataset_id,
    "schema": schema,
    "columns": columns,                  # <-- NEW
    "palette": CATEGORY_PALETTE,         # <-- NEW
    "meta": {
        "rows": rows,
        "cols": cols,
        "format": meta["format"],
        "encoding_used": meta["encoding_used"],
        "converted_datetime_cols": converted,
        "datetime_parse_rate": after_rates
    },
    "warnings": warnings
}
