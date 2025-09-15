from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "DF Classifier API"
    APP_ENV: str = "local"
    DEBUG: bool = True
    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx", "parquet"]
    MAX_UPLOAD_MB: int = 200
    CORS_ORIGINS: List[str] = ["*"]

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

settings = Settings()  # carica da .env
