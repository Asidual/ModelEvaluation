from typing import Dict, Any
import uuid

_DATASETS: Dict[str, Any] = {}

def save_dataset(df) -> str:
    dataset_id = str(uuid.uuid4())
    _DATASETS[dataset_id] = df
    return dataset_id

def get_dataset(dataset_id: str):
    return _DATASETS.get(dataset_id)

def drop_dataset(dataset_id: str) -> bool:
    return _DATASETS.pop(dataset_id, None) is not None
