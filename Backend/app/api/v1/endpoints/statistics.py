# app/api/v1/endpoints/statistics.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Literal, Optional, Tuple
from sqlalchemy.orm import Session
import io
import pandas as pd
import numpy as np

from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.dataset_service import load_dataset_bytes
from app.schemas.statistics import ReadOptions, ColumnStatsRequest, CorrelationSpec, Filter, OverallStatsRequest, TextStatsSpec, DatetimeStatsSpec

router = APIRouter(prefix="/v1", tags=["Statistics"])



# ----------------------------
# Helpers
# ----------------------------

def _to_py(o: Any) -> Any:
    """Converte tipi numpy/pandas in tipi Python serializzabili."""
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        # attento ai NaN
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_, )):
        return bool(o)
    if isinstance(o, (pd.Timestamp, )):
        return o.isoformat()
    if pd.isna(o):
        return None
    return o

def _serialize_mapping(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): _to_py(v) for k, v in d.items()}

def _read_df_from_bytes(filename: str, content: bytes, opts: ReadOptions) -> pd.DataFrame:
    name_lower = (filename or "").lower()
    if name_lower.endswith(".parquet") or name_lower.endswith(".pq"):
        buf = io.BytesIO(content)
        return pd.read_parquet(buf, engine="pyarrow")
    # fallback CSV
    buf = io.BytesIO(content)
    read_csv_kwargs = {}
    if opts.delimiter is not None: read_csv_kwargs["sep"] = opts.delimiter
    if opts.decimal is not None: read_csv_kwargs["decimal"] = opts.decimal
    if opts.encoding is not None: read_csv_kwargs["encoding"] = opts.encoding
    if opts.dayfirst is not None: read_csv_kwargs["dayfirst"] = opts.dayfirst
    if opts.header is not None: read_csv_kwargs["header"] = opts.header
    if opts.quotechar is not None: read_csv_kwargs["quotechar"] = opts.quotechar
    if opts.thousands is not None: read_csv_kwargs["thousands"] = opts.thousands
    if opts.dtype is not None: read_csv_kwargs["dtype"] = opts.dtype
    if opts.nrows is not None: read_csv_kwargs["nrows"] = opts.nrows
    return pd.read_csv(buf, **read_csv_kwargs)

def _apply_filters(df: pd.DataFrame, filters: List[Filter]) -> pd.DataFrame:
    out = df
    for f in filters:
        col = f.column
        if col not in out.columns:
            # ignora filtri su colonne inesistenti (oppure alza eccezione, a scelta)
            continue
        s = out[col]
        op = f.op
        val = f.value

        if op in ("isnull", "notnull"):
            mask = s.isna() if op == "isnull" else s.notna()
        elif op in ("eq", "ne", "lt", "lte", "gt", "gte"):
            if isinstance(s.dtype, pd.StringDtype) or s.dtype == object:
                if not f.case_sensitive and isinstance(val, str):
                    mask = (s.str.lower() == val.lower()) if op == "eq" else (s.str.lower() != val.lower())
                    if op not in ("eq", "ne"):
                        # prova a convertire a numerico per confronti di ordine
                        s_num = pd.to_numeric(s, errors="coerce")
                        val_num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                        cmp_map = {
                            "lt": s_num < val_num,
                            "lte": s_num <= val_num,
                            "gt": s_num > val_num,
                            "gte": s_num >= val_num,
                        }
                        mask = cmp_map[op]
                else:
                    cmp_map = {
                        "eq": s == val,
                        "ne": s != val,
                        "lt": s < val,
                        "lte": s <= val,
                        "gt": s > val,
                        "gte": s >= val,
                    }
                    mask = cmp_map[op]
            else:
                cmp_map = {
                    "eq": s == val,
                    "ne": s != val,
                    "lt": s < val,
                    "lte": s <= val,
                    "gt": s > val,
                    "gte": s >= val,
                }
                mask = cmp_map[op]
        elif op in ("in", "nin"):
            vals = set(val or [])
            mask = s.isin(vals)
            if op == "nin":
                mask = ~mask
        elif op in ("contains", "startswith", "endswith"):
            if not isinstance(val, str):
                mask = pd.Series(False, index=s.index)
            else:
                series = s.astype(str)
                if not f.case_sensitive:
                    series = series.str.lower()
                    val_cmp = val.lower()
                else:
                    val_cmp = val
                if op == "contains":
                    mask = series.str.contains(val_cmp, na=False)
                elif op == "startswith":
                    mask = series.str.startswith(val_cmp, na=False)
                else:
                    mask = series.str.endswith(val_cmp, na=False)
        elif op == "between":
            low, high = (val or [None, None])[:2]
            s_num = pd.to_numeric(s, errors="coerce")
            low = -np.inf if low is None else float(low)
            high = np.inf if high is None else float(high)
            mask = (s_num >= low) & (s_num <= high)
        else:
            mask = pd.Series(True, index=s.index)

        out = out[mask]
    return out

def _pick_numeric(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _pick_categorical(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object","category","string"]).columns.tolist()

def _pick_datetime(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

# ----------------------------
# Endpoints
# ----------------------------

@router.post("/overallStatistics", summary="Statistiche generali dell'intero dataset (post-filtri)")
def overall_statistics(req: OverallStatsRequest, db: Session = Depends(get_db)):
    # 1) ownership + load
    ds = db.get(Dataset, req.dataset_id)
    if not ds or ds.user_id != req.user_id:
        raise HTTPException(status_code=404, detail="Dataset non trovato.")
    
    try:
        content, filename = load_dataset_bytes(db, req.dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        df = _read_df_from_bytes(filename, content, req.read_opts)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Errore lettura dataset: {e}")

    # 2) filtri
    df = _apply_filters(df, req.filters)

    # 3) sampling (se richiesto)
    if req.sample_rows is not None and req.sample_rows > 0 and len(df) > req.sample_rows:
        df = df.sample(req.sample_rows, random_state=42)

    # 4) core stats
    n_rows, n_cols = df.shape
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns} if req.include_missing_by_column else None

    # type splits
    numeric_cols = _pick_numeric(df)
    categorical_cols = _pick_categorical(df)
    # prova a inferire datetime per colonne object plausibili
    dt_inferred = {}
    for c in df.columns:
        if c not in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=req.read_opts.dayfirst)
                if parsed.notna().sum() > max(5, int(0.5 * len(df))):  # euristica
                    dt_inferred[c] = True
            except Exception:
                pass

    # correlation (solo su numeric)
    corr = None
    if req.include_correlations.enabled and numeric_cols:
        corr_df = df[numeric_cols].corr(method=req.include_correlations.method)
        corr = corr_df.replace({np.nan: None}).to_dict(orient="index")

    payload = {
        "dataset_id": req.dataset_id,
        "filtered_rows": n_rows,
        "columns_count": n_cols,
        "columns": list(df.columns),
        "dtypes": dtypes if req.include_dtypes else None,
        "missing_by_column": missing_by_col,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "likely_datetime_columns": list(dt_inferred.keys()),
        "correlations": corr,
    }
    return JSONResponse(payload)


@router.post("/{column}/statistics", summary="Statistiche dettagliate per una colonna (post-filtri)")
def column_statistics(
    column: str,
    req: ColumnStatsRequest,
    db: Session = Depends(get_db),
):
    # 1) ownership + load
    ds = db.get(Dataset, req.dataset_id)
    if not ds or ds.user_id != req.user_id:
        raise HTTPException(status_code=404, detail="Dataset non trovato.")

    try:
        content, filename = load_dataset_bytes(db, req.dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        df = _read_df_from_bytes(filename, content, req.read_opts)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Errore lettura dataset: {e}")

    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Colonna '{column}' non trovata.")

    # 2) filtri
    df = _apply_filters(df, req.filters)
    s = df[column]

    # 3) base stats comuni
    non_null = s.dropna()
    base = {
        "count": int(s.shape[0]),
        "non_null": int(non_null.shape[0]),
        "nulls": int(s.isna().sum()),
        "dtype": str(s.dtype),
        "unique": int(non_null.nunique()),
        "sample_head": [_to_py(v) for v in non_null.head(5).tolist()],
    }

    # 4) branch per tipo
    is_numeric = pd.api.types.is_numeric_dtype(s)
    is_datetime = pd.api.types.is_datetime64_any_dtype(s)
    is_textlike = (pd.api.types.is_string_dtype(s) or s.dtype == object) and not is_datetime and not is_numeric

    details: Dict[str, Any] = {}

    # ---- Numeric
    if is_numeric:
        desc = non_null.describe(percentiles=req.percentiles.values, include="all")
        # desc può contenere np types
        stats_map = {k: _to_py(v) for k, v in desc.to_dict().items()}
        # extra
        stats_map.update({
            "skew": _to_py(non_null.skew()),
            "kurtosis": _to_py(non_null.kurtosis()),
            "var": _to_py(non_null.var()),
            "mad": _to_py(non_null.mad()),
        })

        # histogram
        if req.histogram.enabled and non_null.shape[0] > 0:
            if req.histogram.strategy == "auto":
                bin_edges = np.histogram_bin_edges(non_null.dropna().to_numpy(), bins="auto")
            else:
                bin_edges = np.histogram_bin_edges(non_null.dropna().to_numpy(), bins=req.histogram.bins)
            counts, edges = np.histogram(non_null.dropna().to_numpy(), bins=bin_edges)
            details["histogram"] = {
                "bin_edges": [float(x) for x in edges.tolist()],
                "counts": [int(x) for x in counts.tolist()],
            }

        details["numeric"] = stats_map

    # ---- Datetime
    # se non già datetime, prova parse soft (non invasivo)
    if not is_datetime:
        try:
            s_parsed = pd.to_datetime(s, errors="coerce", dayfirst=req.read_opts.dayfirst)
            if s_parsed.notna().sum() > max(5, int(0.5 * len(s))):
                s_dt = s_parsed
                is_datetime = True
            else:
                s_dt = None
        except Exception:
            s_dt = None
    else:
        s_dt = s

    if is_datetime and req.datetime_stats.enabled and s_dt is not None:
        s_dt = s_dt.dropna()
        if req.datetime_stats.tz_localize:
            # prova a localizzare se naive
            if s_dt.dt.tz is None:
                try:
                    s_dt = s_dt.dt.tz_localize(req.datetime_stats.tz_localize)
                except Exception:
                    pass

        dt_stats = {}
        if req.datetime_stats.infer_minmax and len(s_dt) > 0:
            dt_stats["min"] = _to_py(s_dt.min())
            dt_stats["max"] = _to_py(s_dt.max())
            dt_stats["range_days"] = _to_py((s_dt.max() - s_dt.min()).days)

        # granularità
        gran = {}
        if len(s_dt) > 0:
            # NB: .dt is safe after ensuring datetime dtype
            if "year" in req.datetime_stats.granularity_counts:
                gran["by_year"] = _serialize_mapping(s_dt.dt.year.value_counts().to_dict())
            if "quarter" in req.datetime_stats.granularity_counts:
                gran["by_quarter"] = _serialize_mapping(s_dt.dt.quarter.value_counts().to_dict())
            if "month" in req.datetime_stats.granularity_counts:
                gran["by_month"] = _serialize_mapping(s_dt.dt.month.value_counts().to_dict())
            if "week" in req.datetime_stats.granularity_counts:
                # ISO week
                gran["by_iso_week"] = _serialize_mapping(s_dt.dt.isocalendar().week.value_counts().to_dict())
            if "weekday" in req.datetime_stats.granularity_counts:
                gran["by_weekday"] = _serialize_mapping(s_dt.dt.weekday.value_counts().to_dict())
            if "day" in req.datetime_stats.granularity_counts:
                gran["by_day"] = _serialize_mapping(s_dt.dt.day.value_counts().to_dict())
            if "hour" in req.datetime_stats.granularity_counts:
                gran["by_hour"] = _serialize_mapping(s_dt.dt.hour.value_counts().to_dict())

        details["datetime"] = {"basic": dt_stats, "granularity": gran}

    # ---- Categorical / Text-like
    if (is_textlike or not is_numeric) and req.value_counts.enabled:
        vc = non_null.astype(str) if is_textlike else non_null
        vc_series = vc.value_counts(dropna=req.value_counts.dropna, sort=True)
        total_unique = int(vc_series.shape[0])

        # paginazione semplice
        start = max(0, req.value_counts.offset)
        end = start + max(1, req.value_counts.limit)
        vc_slice = vc_series.iloc[start:end]

        if req.value_counts.normalize:
            vc_slice = vc_slice / (non_null.shape[0] if non_null.shape[0] > 0 else 1)

        details["value_counts"] = {
            "total_unique": total_unique,
            "offset": start,
            "limit": req.value_counts.limit,
            "items": [
                {"value": str(idx), "count": _to_py(val)}
                for idx, val in vc_slice.items()
            ],
        }

    # ---- Extra text stats
    if is_textlike and req.text_stats.enabled:
        s_text = non_null.astype(str)
        lens = s_text.str.len()
        words = s_text.str.split().map(lambda x: len(x) if isinstance(x, list) else np.nan)

        details["text"] = {
            "avg_len": _to_py(lens.mean()),
            "median_len": _to_py(lens.median()),
            "min_len": _to_py(lens.min()),
            "max_len": _to_py(lens.max()),
            "avg_words": _to_py(words.mean()),
            "median_words": _to_py(words.median()),
            "unique_ratio": _to_py(non_null.nunique() / non_null.shape[0] if non_null.shape[0] else None),
            "examples": s_text.head(req.text_stats.sample_examples).tolist(),
        }

    # payload finale
    payload = {
        "dataset_id": req.dataset_id,
        "column": column,
        "base": base,
        "details": details,
    }
    return JSONResponse(payload)
