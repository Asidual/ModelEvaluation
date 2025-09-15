from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class CategoricalPreview(BaseModel):
    n_unique: int
    values: Optional[List[str]] = None
    truncated: bool

class SchemaModel(BaseModel):
    numeriche_continue: List[str]
    numeriche_discrete: List[str]
    categoriche: Dict[str, CategoricalPreview]
    boolean: List[str]
    datetime: List[str]
    testo_libero: List[str]
    long_text_nlp: Dict[str, Dict[str, float]]

class MetaModel(BaseModel):
    rows: int
    cols: int
    format: str
    encoding_used: str
    converted_datetime_cols: List[str] = []
    datetime_parse_rate: Dict[str, float] = {}

class ClassifyResponse(BaseModel):
    dataset_id: str
    schema: SchemaModel
    meta: MetaModel
    warnings: List[str] = []
