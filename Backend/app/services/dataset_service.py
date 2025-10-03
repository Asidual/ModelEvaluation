# app/services/dataset_service.py
from __future__ import annotations

import os
import re
import stat
import time
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings, DATASETS_BASE_PATH
from app.models.dataset import Dataset


# ---------- Utils ----------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-.]", "", s)   # keep letters/digits/_/-/. and spaces
    s = re.sub(r"\s+", "-", s)
    return s or "unknown"


def _on_rm_error(func, path, exc_info):
    # Windows: rimuovi read-only e ritenta
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def _safe_rmtree(path: str, attempts: int = 3) -> bool:
    if not path or not os.path.exists(path):
        return True
    for i in range(attempts):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return True
        except Exception:
            time.sleep(0.2 * (i + 1))
    return False


def _inside_base_dir(path: str, base_abs: str) -> bool:
    try:
        ap = os.path.abspath(path)
        ab = os.path.abspath(base_abs)
        return os.path.commonpath([ap, ab]) == ab
    except Exception:
        return False


# Path base assoluto per lo storage
BASE_DIR_ABS = str(DATASETS_BASE_PATH)


# ---------- Low-level helpers ----------
def _read_anytable(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = BytesIO(file_bytes)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(bio)  # richiede pyarrow/fastparquet
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(bio)
    # default CSV
    return pd.read_csv(bio)


def _infer_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    schema: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        schema.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "missing_ratio": float(s.isna().mean()),
        })
    return schema


# ---------- Public services ----------
def update_columns_schema(db: Session, dataset_id: int, columns_schema: List[dict]) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise FileNotFoundError("Dataset non trovato.")
    ds.columns_schema = columns_schema
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def load_dataset_bytes(db: Session, dataset_id: int) -> Tuple[bytes, str]:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise FileNotFoundError("Dataset non trovato.")
    p = Path(ds.file_path)
    if not p.exists():
        raise FileNotFoundError("File del dataset non presente su disco.")
    return p.read_bytes(), p.name


def save_dataset_file_and_meta(
    db: Session,
    user_id: str,                 # per ora string (DEFAULT_USER)
    dataset_name: str,
    description: str | None,
    file_bytes: bytes,
    filename: str,
    version: str = "v1",
    user_name: str | None = None,  # es. "Pippo"
) -> Dataset:
    user_slug = slugify(user_name or "default")
    ds_slug = slugify(dataset_name)

    user_dir = os.path.join(BASE_DIR_ABS, user_slug)
    ds_dir = os.path.join(user_dir, ds_slug)
    os.makedirs(ds_dir, exist_ok=True)

    ext = os.path.splitext(filename or "")[1] or ".csv"
    safe_name = f"{ds_slug}{ext}"
    file_path = os.path.join(ds_dir, safe_name)

    # salva file
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # opzionale: calcolo n_rows/n_cols leggendo al volo (robusto su file piccoli/medi)
    n_rows = None
    n_cols = None
    try:
        df_preview = _read_anytable(file_bytes, filename or safe_name)
        n_rows, n_cols = df_preview.shape
    except Exception:
        pass

    ds = Dataset(
        user_id=user_id,
        dataset_name=dataset_name,
        dataset_slug=ds_slug,
        version=version,
        description=description,
        file_path=file_path,
        folder_path=ds_dir,
        n_rows=n_rows,
        n_cols=n_cols,
        # columns_schema: sarà popolato dopo la classificazione
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def delete_dataset(db: Session, dataset_id: int, user_id: str, remove_storage: bool = True) -> bool:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise FileNotFoundError("Dataset non trovato.")
    if getattr(ds, "user_id", None) not in (None, user_id):
        raise PermissionError("Non autorizzato a eliminare questo dataset.")

    folder_path = getattr(ds, "folder_path", None)
    file_path = getattr(ds, "file_path", None)

    # 1) elimina record dal DB
    db.delete(ds)
    db.commit()

    # 2) rimuovi storage su disco (cartella dataset o singolo file)
    if not remove_storage:
        return True

    user_dir = None
    # preferisci rimuovere l'intera cartella del dataset
    if folder_path and _inside_base_dir(folder_path, BASE_DIR_ABS):
        user_dir = os.path.dirname(folder_path)  # .../datasets/<user_slug>
        _safe_rmtree(folder_path)
    elif file_path and _inside_base_dir(file_path, BASE_DIR_ABS):
        # fallback: rimuovi file e prova a rimuovere la cartella dataset se vuota
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass
        ds_dir = os.path.dirname(file_path)
        try:
            if _inside_base_dir(ds_dir, BASE_DIR_ABS) and not os.listdir(ds_dir):
                os.rmdir(ds_dir)
        except Exception:
            pass
        user_dir = os.path.dirname(ds_dir)

    # 3) se la cartella utente è vuota, rimuovila (…/datasets/<user_slug>/)
    try:
        if user_dir and _inside_base_dir(user_dir, BASE_DIR_ABS) and not os.listdir(user_dir):
            os.rmdir(user_dir)
    except Exception:
        pass

    return True
