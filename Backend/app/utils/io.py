import io, pandas as pd, chardet


def sniff_encoding(content: bytes) -> str:
    try:
        return chardet.detect(content)["encoding"] or "utf-8"
    except Exception:
        return "utf-8"

def read_dataframe(filename: str, content: bytes, delimiter=None, decimal=None, encoding=None):
    name = (filename or "").lower()
    encoding = encoding or sniff_encoding(content)
    bio = io.BytesIO(content)

    if name.endswith(".csv"):
        df = pd.read_csv(bio, encoding=encoding, sep=delimiter or None, decimal=decimal or ".")
        fmt = "csv"
    elif name.endswith(".xlsx"):
        df = pd.read_excel(bio)
        fmt = "xlsx"
    elif name.endswith(".parquet"):
        df = pd.read_parquet(bio)
        fmt = "parquet"
    else:
        raise ValueError("Formato non supportato")
    return df, {"encoding_used": encoding, "format": fmt}


###############################
# DB Creation
###############################

import hashlib, re
from pathlib import Path

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

_slug_re = re.compile(r"[^a-z0-9\-]+")
def slugify(text: str) -> str:
    s = text.strip().lower().replace(" ", "-")
    s = _slug_re.sub("-", s)
    return s.strip("-")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

