# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.database import Base, engine
from app.api.v1.endpoints import datasets as datasets_v1
from app.api.v1.endpoints import explenability as explenability_v1
from app.api.v1.endpoints import statistics as statistics_v1
# IMPORTA i modelli PRIMA di create_all, così le tabelle esistono in metadata
from app.models import dataset  # noqa: F401

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    Base.metadata.create_all(bind=engine, checkfirst=True)
    # (facoltativo) log tabelle create/viste
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        print("[DB] Tables:", [r[0] for r in rows])

    yield  # <--- l'app è in esecuzione

    # --- SHUTDOWN ---
    # (facoltativo) chiudi pool/risorse
    engine.dispose()

app = FastAPI(title="ModelEvaluation", lifespan=lifespan)

# routers
app.include_router(datasets_v1.router)
app.include_router(explenability_v1.router)
app.include_router(statistics_v1.router)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
