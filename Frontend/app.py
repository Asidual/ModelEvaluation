import os, io, json, requests
import streamlit as st
import pandas as pd

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------
# HTTP helpers
# ---------------------------
def post_upload_dataset(dataset_name: str, description: str | None, file_bytes: bytes, filename: str):
    url = f"{API_BASE_URL}/v1/datasets/upload"
    files = {"file": (filename, io.BytesIO(file_bytes))}
    data = {"dataset_name": dataset_name}
    if description: data["description"] = description
    r = requests.post(url, data=data, files=files, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", f"Upload failed ({r.status_code})"))
    return r.json()

def list_datasets():
    url = f"{API_BASE_URL}/v1/datasets"
    r = requests.get(url, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", f"List failed ({r.status_code})"))
    return r.json()

def get_dataset_detail(dataset_id: int):
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}"
    r = requests.get(url, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", f"Detail failed ({r.status_code})"))
    return r.json()

def get_classification(dataset_id: int):
    """Chiama l’endpoint di classificazione. Fallback a /columns se /classify non c’è (ancora)."""
    # Tentativo 1: endpoint suggerito
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}/classify"
    r = requests.get(url, timeout=180)
    if r.status_code == 200:
        return r.json()
    # Fallback: endpoint /columns (ritorna columns + palette; schema assente)
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}/columns"
    r = requests.get(url, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", f"Classify failed ({r.status_code})"))
    payload = r.json()
    return {
        "dataset_id": dataset_id,
        "schema": None,        # non disponibile su /columns
        "meta": None,          # non disponibile su /columns
        "warnings": [],
        "columns": payload.get("columns", []),
        "palette": payload.get("category_palette", {}),
    }

# ---------------------------
# UI helpers
# ---------------------------
DEFAULT_PALETTE = {
    "numeriche_continue": "#4C78A8",
    "numeriche_discrete": "#9EC9E6",
    "categoriche":        "#F58518",
    "boolean":            "#54A24B",
    "datetime":           "#B279A2",
    "testo_libero":       "#E45756",
    "long_text_nlp":      "#72B7B2",
    "unknown":            "#BAB0AC",
}

def _badge(label: str, color: str) -> str:
    fg = "#000" if color.lower() in {"#ffffcc", "#fff", "#ffffff"} else "#fff"
    return f'<span style="background:{color};color:{fg};padding:2px 8px;border-radius:999px;font-size:12px;white-space:nowrap;">{label}</span>'

def schema_to_columns(schema: dict, df_columns: list[str] | None = None) -> list[dict]:
    """Converte lo schema (dict) in una lista [{name, category, dtype='unknown', sample_values=None}]."""
    if not schema:
        return []
    cat_map = {}
    for c in schema.get("numeriche_continue", []): cat_map[c] = "numeriche_continue"
    for c in schema.get("numeriche_discrete", []): cat_map[c] = "numeriche_discrete"
    for c in schema.get("boolean", []):            cat_map[c] = "boolean"
    for c in schema.get("datetime", []):           cat_map[c] = "datetime"
    for c in schema.get("testo_libero", []):       cat_map[c] = "testo_libero"
    for c in (schema.get("long_text_nlp", {}) or {}).keys(): cat_map[c] = "long_text_nlp"
    for c in (schema.get("categoriche", {}) or {}).keys():   cat_map[c] = "categoriche"

    names = df_columns or list(cat_map.keys())
    out = []
    for name in names:
        out.append({
            "name": name,
            "category": cat_map.get(name, "unknown"),
            "dtype": "unknown",
            "sample_values": None
        })
    # aggiungi anche colonne non presenti in names (se non abbiamo df_columns)
    if not df_columns:
        for name, cat in cat_map.items():
            if name not in {c["name"] for c in out}:
                out.append({"name": name, "category": cat, "dtype": "unknown", "sample_values": None})
    return out

def render_schema(schema: dict):
    if not schema:
        st.info("Nessuno schema disponibile.")
        return
    # conteggi per categoria
    counts = {
        "numeriche_continue": len(schema.get("numeriche_continue", [])),
        "numeriche_discrete": len(schema.get("numeriche_discrete", [])),
        "categoriche": len(schema.get("categoriche", {})),
        "boolean": len(schema.get("boolean", [])),
        "datetime": len(schema.get("datetime", [])),
        "testo_libero": len(schema.get("testo_libero", [])),
        "long_text_nlp": len(schema.get("long_text_nlp", {})),
    }
    chips = []
    for k, v in counts.items():
        col = DEFAULT_PALETTE.get(k, DEFAULT_PALETTE["unknown"])
        chips.append(_badge(f"{k}: {v}", col))
    st.markdown(" ".join(chips), unsafe_allow_html=True)

def render_columns_table(columns: list[dict], palette: dict):
    rows = []
    pal = {**DEFAULT_PALETTE, **(palette or {})}
    for c in columns:
        cat = c.get("category", "unknown")
        color = pal.get(cat, pal["unknown"])
        rows.append({
            "name": c.get("name", ""),
            "category": _badge(cat, color),
            "dtype": c.get("dtype", "unknown"),
            "sample_values": " · ".join(map(str, (c.get("sample_values") or [])[:6])) or "—",
        })
    df = pd.DataFrame(rows, columns=["name", "category", "dtype", "sample_values"])
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Data Explainability", page_icon="📊", layout="wide")
st.title("📊 Data Explainability – MVP")

if "dataset_id" not in st.session_state: st.session_state.dataset_id = None
if "classification" not in st.session_state: st.session_state.classification = None
if "palette" not in st.session_state: st.session_state.palette = DEFAULT_PALETTE
if "columns" not in st.session_state: st.session_state.columns = []

tab1, tab2 = st.tabs(["1) Upload/Selezione", "2) Classificazione & Colonne"])

# --- TAB 1: Upload/Selezione ---
with tab1:
    st.subheader("Carica dataset (CSV o Parquet)")
    with st.form("upload_form", clear_on_submit=False):
        dataset_name = st.text_input("Nome dataset", placeholder="es. my_sales_2025")
        description = st.text_area("Descrizione (opzionale)")
        file = st.file_uploader("Seleziona file", type=["csv", "parquet", "pq"])
        submit = st.form_submit_button("Carica")
    if submit:
        if not dataset_name.strip(): st.error("Inserisci un nome dataset.")
        elif not file: st.error("Seleziona un file.")
        else:
            try:
                with st.spinner("Upload..."):
                    ds = post_upload_dataset(dataset_name.strip(), description.strip() or None, file.getvalue(), file.name)
                st.session_state.dataset_id = ds["id"]
                st.session_state.classification = None
                st.session_state.columns = []
                st.success(f"Upload completato (id={ds['id']}). Vai alla tab 2.")
            except Exception as e:
                st.error(str(e))
    st.divider()
    st.subheader("Seleziona dataset esistente")
    try:
        datasets = list_datasets()
        if datasets:
            opt = {f"{d['id']} — {d['dataset_name']}": d["id"] for d in datasets}
            chosen = st.selectbox("Dataset", options=list(opt.keys()))
            if st.button("Seleziona"):
                st.session_state.dataset_id = opt[chosen]
                st.session_state.classification = None
                st.session_state.columns = []
                st.success(f"Selezionato dataset {st.session_state.dataset_id}")
        else:
            st.info("Nessun dataset salvato.")
    except Exception as e:
        st.warning(f"Lista non disponibile: {e}")

# --- TAB 2: Classificazione ---
with tab2:
    st.subheader("Classificazione & Colonne")
    if not st.session_state.dataset_id:
        st.warning("Carica o seleziona un dataset nella tab precedente.")
        st.stop()

    colA, colB = st.columns([1, 1])
    if colA.button("🔄 Classifica/aggiorna"):
        try:
            with st.spinner("Classificazione in corso..."):
                result = get_classification(st.session_state.dataset_id)
            st.session_state.classification = result
            st.session_state.palette = {**DEFAULT_PALETTE, **(result.get("palette", {}) or {})}
            # columns: se non presenti nel result, derivale dallo schema
            cols = result.get("columns")
            if not cols:
                cols = schema_to_columns(result.get("schema") or {})
            st.session_state.columns = cols
            st.success("Classificazione completata.")
        except Exception as e:
            st.error(str(e))

    result = st.session_state.classification
    palette = st.session_state.palette
    columns = st.session_state.columns

    if result:
        # META
        meta = result.get("meta") or {}
        with st.expander("Meta", expanded=True):
            if meta:
                st.json(meta)
            else:
                st.info("Meta non disponibile da questo endpoint.")

        # SCHEMA (badge per conteggi)
        st.markdown("#### Schema (conteggi per categoria)")
        render_schema(result.get("schema"))

        # WARNINGS
        warns = result.get("warnings") or []
        if warns:
            st.warning("• " + "\n• ".join(warns))

        # COLUMNS tabellare colorata
        st.markdown("#### Colonne")
        if not columns:
            st.info("Nessuna colonna da mostrare.")
        else:
            # filtro categorie
            cats = sorted({c.get("category","unknown") for c in columns})
            chosen = st.multiselect("Filtra per categoria", options=cats, default=cats)
            to_show = [c for c in columns if c.get("category","unknown") in chosen]
            render_columns_table(to_show, palette)
    else:
        st.info("Premi “Classifica/aggiorna” per vedere lo schema.")
