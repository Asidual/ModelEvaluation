from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sqlalchemy.orm import Session
from app.core.config import DATASETS_DIR
from app.utils.io import sha256_bytes, slugify, ensure_dir
from app.models.dataset import Dataset
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.dataset import Dataset

def update_columns_schema(db: Session, dataset_id: int, columns_schema: list[dict]):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise FileNotFoundError("Dataset non trovato.")
    ds.columns_schema = columns_schema
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def load_dataset_bytes(db: Session, dataset_id: int) -> tuple[bytes, str]:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise FileNotFoundError("Dataset non trovato.")
    p = Path(ds.file_path)
    if not p.exists():
        raise FileNotFoundError("File del dataset non presente su disco.")
    return p.read_bytes(), p.name


def _infer_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    schema = []
    for col in df.columns:
        s = df[col]
        schema.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "missing_ratio": float(s.isna().mean()),
        })
    return schema

def _read_anytable(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(pd.io.common.BytesIO(file_bytes))
    # default CSV
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))

def save_dataset_file_and_meta(
    db: Session,
    user_id: str,
    dataset_name: str,
    description: str | None,
    file_bytes: bytes,
    filename: str,
    version: str = "v1",
) -> Dataset:
    file_hash = sha256_bytes(file_bytes)

    df = _read_anytable(file_bytes, filename)
    n_rows, n_cols = df.shape
    columns_schema = _infer_schema(df)

    dataset_slug = slugify(dataset_name)
    base_dir = DATASETS_DIR / user_id / dataset_slug
    ensure_dir(base_dir)
    file_path = base_dir / f"{version}.parquet"

    # Salvo il file sempre nello stesso path
    df.to_parquet(file_path, index=False)

    # Controllo se esiste già il dataset
    ds = db.query(Dataset).filter(
        Dataset.user_id == user_id,
        Dataset.dataset_name == dataset_name
    ).first()

    if ds:
        # Aggiorno record esistente
        ds.description = description
        ds.file_path = str(file_path)
        ds.file_hash = file_hash
        ds.n_rows = n_rows
        ds.n_cols = n_cols
        ds.columns_schema = columns_schema
        ds.version = version
    else:
        # Creo nuovo record
        ds = Dataset(
            user_id=user_id,
            dataset_name=dataset_name,
            dataset_slug=dataset_slug,
            version=version,
            description=description,
            file_path=str(file_path),
            file_hash=file_hash,
            n_rows=n_rows,
            n_cols=n_cols,
            columns_schema=columns_schema,
        )
        db.add(ds)

    db.commit()
    db.refresh(ds)
    return ds

