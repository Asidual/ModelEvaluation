# app/services/classification_service.py
from typing import Dict, Any, List
import pandas as pd
from app.utils.io import read_dataframe
from app.utils.datetime_detect import coerce_datetime_inplace, detect_datetime_columns
from app.utils.categorize import categorize_columns
from app.storage.memory_store import save_dataset

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

    warnings: List[str] = []
    for col, cfg in schema.get("categoriche", {}).items():
        if bool(cfg.get("truncated")):
            warnings.append(f"Valori categoria di '{col}' troncati a top-{max_cat_values} (truncated=true).")

    dataset_id = save_dataset(df)

    return {
        "dataset_id": dataset_id,
        "schema": schema,
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
