# app/api/v1/explainability.py
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist, field_validator
from typing import Any, Dict, List, Optional
from openai import OpenAI
import os

router = APIRouter(tags=["datasets"], prefix="/v1")

# Config modello (override con env var MODEL_NAME)
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")
client = OpenAI()

# ----------------------------
# Pydantic Schemas
# ----------------------------
class ColumnStats(BaseModel):
    count: Optional[int] = None
    unique: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None

class ColumnMeta(BaseModel):
    name: str = Field(..., description="Nome della colonna")
    dtype: Optional[str] = Field(None, description="Tipo logico (es. int, float, categorical, datetime)")
    missing_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quota di valori mancanti [0,1]")
    sample_values: Optional[List[Any]] = Field(None, description="Esempi di valori")
    stats: Optional[ColumnStats] = None
    cardinality: Optional[int] = None

class DatasetContext(BaseModel):
    description: str = Field(..., description="Descrizione in linguaggio naturale del dataset e del caso d’uso")
    columns: conlist(ColumnMeta, min_items=1) = Field(..., description="Metadati delle colonne")

    @field_validator("description")
    def description_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("La descrizione del dataset non può essere vuota.")
        return v

class ColumnExplainRequest(BaseModel):
    description: str = Field(..., description="Descrizione del dataset/caso d’uso")
    columns: conlist(ColumnMeta, min_items=1) = Field(..., description="Metadati delle colonne")
    objective: Optional[str] = Field(None, description="Obiettivo analitico (es. previsione, classificazione, profiling)")
    selected_filters: Optional[Dict[str, Any]] = Field(None, description="Eventuali filtri utente correnti")
    target: Optional[str] = Field(None, description="Nome della colonna target, se presente")

# ----------------------------
# System Prompt
# ----------------------------
EXPLAIN_SYSTEM_PROMPT = """Sei un assistente di data explainability.
Fornisci spiegazioni chiare, sintetiche e azionabili per un data analyst o product owner.
Stile: bullet brevi, niente gergo inutile, esempi concreti.

Requisiti di output JSON (chiavi fisse):
- "summary": 3-6 bullet su insight principali
- "data_quality": elenco problemi/avvisi (missing, cardinalità, outlier, tipi)
- "relationships": possibili relazioni tra colonne (ipotesi, non certezze)
- "suggested_plots": lista di grafici suggeriti (nome + col/asse + perché)
- "assumptions": 2-5 ipotesi dichiarate
- "next_steps": 3-6 azioni successive consigliate

Mantieni riferimenti ai nomi delle colonne esattamente come forniti.
Se mancano info, dichiara l’incertezza in 'assumptions' invece di inventare.
Rispondi in italiano.
"""

# ----------------------------
# Helpers
# ----------------------------
def build_general_messages(ctx: DatasetContext) -> List[Dict[str, str]]:
    user_payload = {
        "description": ctx.description,
        "columns": [c.dict() for c in ctx.columns],
    }
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Genera la spiegazione generale del dataset seguendo lo schema JSON richiesto.\n\n"
                f"CONTESTO:\n{user_payload}"
            ),
        },
    ]

def build_column_messages(column_name: str, req: ColumnExplainRequest) -> List[Dict[str, str]]:
    user_payload = {
        "description": req.description,
        "objective": req.objective,
        "target": req.target,
        "selected_filters": req.selected_filters,
        "columns": [c.dict() for c in req.columns],
        "focus_column": column_name,
    }
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Fornisci spiegazione SPECIFICA per la colonna indicata, in relazione al dataset, "
                "all’obiettivo e (se presente) al target. Segui lo schema JSON richiesto.\n\n"
                f"CONTESTO:\n{user_payload}"
            ),
        },
    ]

def call_openai(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        # Puoi usare response_format={"type": "json_object"} se il tuo piano/modello lo supporta
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        # Il modello restituisce già JSON; FastAPI risponderà come JSONResponse
        return {"explanation": content}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

# ----------------------------
# Endpoints
# ----------------------------
@router.post("/explainability", response_class=JSONResponse, summary="Spiegazione generale del dataset")
async def explain_general(ctx: DatasetContext):
    """
    Ritorna una spiegazione generale del dataset (insight, qualità dati, relazioni ipotizzate,
    grafici consigliati, assunzioni e next steps) basata sulla descrizione e sui metadati delle colonne.
    """
    messages = build_general_messages(ctx)
    result = call_openai(messages)
    return JSONResponse(result)

@router.post(
    "/explainability/{column_name}",
    response_class=JSONResponse,
    summary="Spiegazione specifica di una colonna"
)
async def explain_column(
    column_name: str = Path(..., description="Nome esatto della colonna su cui focalizzarsi"),
    req: ColumnExplainRequest = ...
):
    """
    Ritorna una spiegazione specifica per la colonna selezionata, includendo possibili relazioni col target
    e grafici/analisi consigliate in base al contesto e all’obiettivo.
    """
    # Verifica che la colonna esista nel payload
    col_names = {c.name for c in req.columns}
    if column_name not in col_names:
        raise HTTPException(status_code=400, detail=f"La colonna '{column_name}' non è presente nel payload.")
    messages = build_column_messages(column_name, req)
    result = call_openai(messages)
    return JSONResponse(result)
