from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated

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


############################


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    missing_ratio: float = Field(ge=0.0, le=1.0)

ColumnsList = Annotated[List[ColumnSchema], Field(min_length=1)]

class DatasetCreate(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    description: Optional[str] = None

class DatasetOut(BaseModel):
    id: int
    user_id: str
    dataset_name: str
    dataset_slug: str
    version: str
    description: Optional[str]
    file_path: str
    n_rows: int
    n_cols: int
    columns_schema: ColumnsList
    model_config = ConfigDict(from_attributes=True)
    columns_schema: list[dict] | None = None
    


