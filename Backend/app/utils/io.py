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
