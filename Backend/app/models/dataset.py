from sqlalchemy import Column, Integer, String, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base
# app/models/dataset.py




class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    # utente (per ora string “default”/“pippo”)
    user_id = Column(String, index=True, default="default", nullable=False)

    # identificazione dataset
    dataset_name = Column(String, index=True, nullable=False)
    dataset_slug = Column(String, index=True, nullable=False)
    version = Column(String, default="v1", nullable=False)
    description = Column(String, nullable=True)

    # storage
    folder_path = Column(String, nullable=True)   # <— cartella del dataset (usata in delete)
    file_path   = Column(String, nullable=False)  # <— path file caricato (csv/xlsx/parquet)
    file_hash   = Column(String, index=True, nullable=True)

    # meta
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    columns_schema = Column(JSON, nullable=True)  # <— lista [{name, category, dtype, ...}]

    # audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "dataset_slug", "version", name="uq_user_datasetslug_version"),
    )
