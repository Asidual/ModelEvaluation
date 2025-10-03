# app/api/v1/endpoints/datasets.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import DEFAULT_USER
from app.schemas.datasets import DatasetOut
from app.models.dataset import Dataset
from app.services.classification_service import classify_file
from app.services.dataset_service import (
    save_dataset_file_and_meta,
    update_columns_schema,
    load_dataset_bytes,
    delete_dataset as svc_delete_dataset,
)

router = APIRouter(tags=["datasets"], prefix="/v1")


@router.post("/datasets/upload", response_model=DatasetOut, summary="Upload dataset (CSV/Parquet)")
async def upload_dataset(
    dataset_name: str = Form(...),
    description: str | None = Form(None),
    user_name: str | None = Form(None),       # nome utente lato FE; se assente → DEFAULT_USER
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File vuoto.")
    try:
        # user_id effettivo con cui salviamo e poi filtreremo i dataset
        owner = (user_name or DEFAULT_USER).strip() or DEFAULT_USER

        ds = save_dataset_file_and_meta(
            db=db,
            user_id=owner,                             # <-- salva il dataset per questo utente
            dataset_name=dataset_name,
            description=description,
            file_bytes=content,
            filename=file.filename or "upload.csv",
            version="v1",
            user_name=owner,                           # <-- usato anche per creare la cartella utente
        )
        return ds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


@router.get("/datasets", response_model=list[DatasetOut], summary="Lista dataset utente")
def list_datasets(
    user_id: str = Query(DEFAULT_USER, description="Identificativo utente (es. 'Pippo')"),
    db: Session = Depends(get_db),
):
    return (
        db.query(Dataset)
        .filter(Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc())
        .all()
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetOut, summary="Dettaglio dataset")
def get_dataset(
    dataset_id: int,
    user_id: str = Query(DEFAULT_USER, description="Identificativo utente proprietario del dataset"),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.user_id != user_id:
        # 404 per non rivelare l'esistenza di dataset di altri utenti
        raise HTTPException(status_code=404, detail="Dataset non trovato.")
    return ds


@router.get("/datasets/{dataset_id}/columns", summary="Estrae nomi colonne e categoria dal dataset salvato")
def get_columns_and_categories(
    dataset_id: int,
    user_id: str = Query(DEFAULT_USER, description="Identificativo utente proprietario del dataset"),
    db: Session = Depends(get_db),
):
    # 0) verifica ownership
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.user_id != user_id:
        raise HTTPException(status_code=404, detail="Dataset non trovato.")

    # 1) carico i bytes del file
    try:
        content, filename = load_dataset_bytes(db, dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2) classifico
    try:
        result = classify_file(
            filename=filename or "saved",
            content=content,
            delimiter=None,
            decimal=None,
            encoding=None,
            dayfirst=None,
            threshold_cat=20,
            max_cat_values=10,
            nlp_avg_chars=120,
            nlp_avg_words=20,
            nlp_min_unique_token_ratio=0.2,
            auto_parse_dates=True,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno in classificazione")

    columns = result.get("columns", [])
    palette = result.get("palette", {})

    # 3) salvo nel DB per riuso (Dataset.columns_schema)
    try:
        update_columns_schema(db, dataset_id=dataset_id, columns_schema=columns)
    except Exception:
        # non bloccare la risposta: logga e prosegui
        pass

    return JSONResponse(
        {
            "dataset_id": dataset_id,
            "columns": columns,
            "category_palette": palette,
        }
    )


@router.delete(
    "/datasets/{dataset_id}",
    status_code=204,
    summary="Elimina dataset dal DB e dallo storage su disco",
)
def delete_dataset_endpoint(
    dataset_id: int,
    user_id: str = Query(DEFAULT_USER, description="Identificativo utente proprietario del dataset"),
    hard_delete_storage: bool = Query(True, description="Se True rimuove anche la cartella/file su disco"),
    db: Session = Depends(get_db),
):
    # controllo ownership prima di eliminare
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.user_id != user_id:
        raise HTTPException(status_code=404, detail="Dataset non trovato.")
    try:
        svc_delete_dataset(
            db,
            dataset_id=dataset_id,
            user_id=user_id,
            remove_storage=hard_delete_storage,
        )
        return Response(status_code=204)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset non trovato.")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Non autorizzato a eliminare questo dataset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")


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
    """Classificazione ad-hoc di un file senza salvarlo a DB."""
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
            auto_parse_dates=auto_parse_dates,
        )
        return JSONResponse(result)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno")
