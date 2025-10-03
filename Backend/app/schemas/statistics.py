from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Any, Optional, Dict
from app.core.config import DEFAULT_USER
# ----------------------------
# Request/Response Schemas
# ----------------------------

class Filter(BaseModel):
    column: str
    op: Literal[
        "eq", "ne", "lt", "lte", "gt", "gte",
        "in", "nin",
        "contains", "startswith", "endswith",
        "between",
        "isnull", "notnull"
    ]
    value: Any | None = None  # for between -> [low, high]; for in/nin -> list
    case_sensitive: bool = False

class ReadOptions(BaseModel):
    # Consente override di parsing (CSV/Parquet)
    delimiter: Optional[str] = None
    decimal: Optional[str] = None
    encoding: Optional[str] = None
    dayfirst: Optional[bool] = None
    # Per CSV edge-cases
    header: Optional[int] = 0
    quotechar: Optional[str] = None
    thousands: Optional[str] = None
    # Force dtype? (usa pandas dtype mapping, es: {"col":"Int64"})
    dtype: Optional[Dict[str, str]] = None
    # Limit row read (per preview/velocità)
    nrows: Optional[int] = None

class HistogramSpec(BaseModel):
    enabled: bool = True
    bins: int = 20
    # strategy: "auto" usa numpy.histogram_bin_edges(..., bins="auto")
    strategy: Literal["fixed", "auto"] = "fixed"
    dropna: bool = True

class ValueCountsSpec(BaseModel):
    enabled: bool = True
    top_k: int = 20
    normalize: bool = False
    dropna: bool = True
    offset: int = 0  # paginazione lato FE
    limit: int = 20

class TextStatsSpec(BaseModel):
    enabled: bool = True
    sample_examples: int = 3  # esempi brevi di valori (head filtrata non null)

class DatetimeStatsSpec(BaseModel):
    enabled: bool = True
    tz_localize: Optional[str] = None  # es. "Europe/Rome"
    granularity_counts: List[Literal["year","quarter","month","week","weekday","day","hour"]] = Field(default_factory=lambda: ["year","month","weekday"])
    infer_minmax: bool = True

class PercentilesSpec(BaseModel):
    # es: [0.01, 0.05, 0.95, 0.99]
    values: List[float] = Field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])

    @field_validator("values")
    @classmethod
    def check_bounds(cls, v: List[float]):
        for p in v:
            if p < 0 or p > 1:
                raise ValueError("Percentili devono essere tra 0 e 1.")
        return sorted(set(v))

class CorrelationSpec(BaseModel):
    enabled: bool = False
    method: Literal["pearson", "spearman", "kendall"] = "pearson"

class OverallStatsRequest(BaseModel):
    dataset_id: int
    user_id: str = Field(default=DEFAULT_USER)
    read_opts: ReadOptions = Field(default_factory=ReadOptions)
    filters: List[Filter] = Field(default_factory=list)
    sample_rows: Optional[int] = Field(default=None, description="Se impostato, usa un campione casuale di righe.")
    include_dtypes: bool = True
    include_missing_by_column: bool = True
    include_correlations: CorrelationSpec = Field(default_factory=CorrelationSpec)

class ColumnStatsRequest(BaseModel):
    dataset_id: int
    user_id: str = Field(default=DEFAULT_USER)
    read_opts: ReadOptions = Field(default_factory=ReadOptions)
    filters: List[Filter] = Field(default_factory=list)
    percentiles: PercentilesSpec = Field(default_factory=PercentilesSpec)
    histogram: HistogramSpec = Field(default_factory=HistogramSpec)
    value_counts: ValueCountsSpec = Field(default_factory=ValueCountsSpec)
    text_stats: TextStatsSpec = Field(default_factory=TextStatsSpec)
    datetime_stats: DatetimeStatsSpec = Field(default_factory=DatetimeStatsSpec)