# app/schemas/datasets.py
from __future__ import annotations

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ---------- Schema di classificazione ----------
class CategoricalPreview(BaseModel):
    n_unique: int
    values: Optional[List[str]] = None
    truncated: bool


class SchemaModel(BaseModel):
    numeriche_continue: List[str] = Field(default_factory=list)
    numeriche_discrete: List[str] = Field(default_factory=list)
    categoriche: Dict[str, CategoricalPreview] = Field(default_factory=dict)
    boolean: List[str] = Field(default_factory=list)
    datetime: List[str] = Field(default_factory=list)
    testo_libero: List[str] = Field(default_factory=list)
    long_text_nlp: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class MetaModel(BaseModel):
    rows: int
    cols: int
    format: str
    encoding_used: str
    converted_datetime_cols: List[str] = Field(default_factory=list)
    datetime_parse_rate: Dict[str, float] = Field(default_factory=dict)


class ColumnInfo(BaseModel):
    """Colonna normalizzata che il BE salva in Dataset.columns_schema e invia al FE."""
    name: str
    category: str
    dtype: str
    sample_values: Optional[List[str]] = None


class ClassifyResponse(BaseModel):
    """
    Risposta della classificazione file.
    Espone la chiave JSON 'schema' usando un alias per evitare warning Pydantic.
    """
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str | int
    data_schema: SchemaModel = Field(..., alias="schema")   # JSON → "schema"
    meta: MetaModel
    warnings: List[str] = Field(default_factory=list)
    columns: Optional[List[ColumnInfo]] = None
    palette: Optional[Dict[str, str]] = None


# ---------- Dataset CRUD ----------
class DatasetCreate(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    description: Optional[str] = None


class DatasetOut(BaseModel):
    id: int
    user_id: str

    dataset_name: str
    dataset_slug: str
    version: str
    description: Optional[str] = None

    folder_path: Optional[str] = None
    file_path: str
    file_hash: Optional[str] = None

    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    columns_schema: Optional[List[ColumnInfo]] = None

    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
