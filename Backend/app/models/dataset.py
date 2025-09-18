from sqlalchemy import Column, Integer, String, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base
# app/models/dataset.py




class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, default="default")

    dataset_name = Column(String, index=True)
    dataset_slug = Column(String, index=True)
    version = Column(String, default="v1")
    description = Column(String, nullable=True)

    file_path = Column(String, nullable=False)
    file_hash = Column(String, index=True)

    n_rows = Column(Integer)
    n_cols = Column(Integer)
    columns_schema = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    columns_schema = Column(JSON, nullable=True)  # <— per memorizzare la lista [{name,category,dtype,...}]

    __table_args__ = (UniqueConstraint("user_id", "dataset_name", name="uq_user_datasetname"),)
