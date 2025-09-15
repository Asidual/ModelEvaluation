# app/api/v1/endpoints/datasets.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.services.classification_service import classify_file

router = APIRouter(tags=["datasets"], prefix="/v1")

@router.post("/datasets")
async def upload_and_classify(
    file: UploadFile = File(...),
    delimiter: str | None = Form(None),
    encoding: str | None = Form(None),
    decimal: str | None = Form(None),
    dayfirst: bool | None = Form(None),
    threshold_cat: int = Form(20),
    max_cat_values: int = Form(10),
    nlp_avg_chars: int = Form(120),
    nlp_avg_words: int = Form(20),
    nlp_min_unique_token_ratio: float = Form(0.2),
    auto_parse_dates: bool = Form(True),
):
    try:
        content = await file.read()
        result = classify_file(
            filename=file.filename or "uploaded",
            content=content,
            delimiter=delimiter,
            decimal=decimal,
            encoding=encoding,
            dayfirst=dayfirst,
            threshold_cat=threshold_cat,
            max_cat_values=max_cat_values,
            nlp_avg_chars=nlp_avg_chars,
            nlp_avg_words=nlp_avg_words,
            nlp_min_unique_token_ratio=nlp_min_unique_token_ratio,
            auto_parse_dates=auto_parse_dates
        )
        return JSONResponse(result)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno")
