from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path
from typing import List

# percorso root del progetto (Backend/)
BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_USER = "default"

class Settings(BaseSettings):
    APP_NAME: str = "DF Classifier API"
    APP_ENV: str = "local"
    DEBUG: bool = True
    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx", "parquet"]
    MAX_UPLOAD_MB: int = 200
    CORS_ORIGINS: List[str] = ["*"]

    # 👇 ROOT dei dataset (NON includere l'utente qui!)
    DATASETS_BASE_DIR: str = "app/storage/datasets"

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _split_ext(cls, v):
        if isinstance(v, str):
            return [x.strip().lower() for x in v.split(",") if x.strip()]
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

settings = Settings()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_BASE_PATH = (BASE_DIR / settings.DATASETS_BASE_DIR).resolve()
DATASETS_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Alias legacy (così non esplode più chi importa DATASETS_DIR)
DATASETS_DIR = DATASETS_BASE_PATH
