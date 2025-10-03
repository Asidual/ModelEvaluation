# frontend_app.py
import os, io, json, requests
import streamlit as st
import pandas as pd

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------
# HTTP helpers
# ---------------------------
def post_upload_dataset(dataset_name: str, description: str | None, file_bytes: bytes, filename: str, user_name: str | None):
    url = f"{API_BASE_URL}/v1/datasets/upload"
    files = {"file": (filename, io.BytesIO(file_bytes))}
    data = {"dataset_name": dataset_name}
    if description: data["description"] = description
    if user_name:   data["user_name"]   = user_name       # ⬅️ NEW
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

def get_columns_and_palette(dataset_id: int):
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}/columns"
    r = requests.get(url, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", f"Columns failed ({r.status_code})"))
    payload = r.json()
    return payload.get("columns", []), payload.get("category_palette", {})

def get_classification(dataset_id: int):
    """Prova /classify (se lo aggiungerai). Fallback a /columns (columns+palette)."""
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}/classify"
    r = requests.get(url, timeout=180)
    if r.status_code == 200:
        return r.json()
    # fallback a /columns
    cols, pal = get_columns_and_palette(dataset_id)
    return {
        "dataset_id": dataset_id,
        "schema": None,
        "meta": None,
        "warnings": [],
        "columns": cols,
        "palette": pal,
    }

def delete_dataset_api(dataset_id: int) -> bool:
    url = f"{API_BASE_URL}/v1/datasets/{dataset_id}"
    r = requests.delete(url, timeout=60)
    if r.status_code not in (200, 204):
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Delete failed ({r.status_code})")
    return True

# ---------------------------
# UI helpers (palette, rendering)
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
        out.append({"name": name, "category": cat_map.get(name, "unknown"), "dtype": "unknown", "sample_values": None})
    if not df_columns:
        known = {c["name"] for c in out}
        for name, cat in cat_map.items():
            if name not in known:
                out.append({"name": name, "category": cat, "dtype": "unknown", "sample_values": None})
    return out

def render_schema(schema: dict):
    if not schema:
        st.info("Nessuno schema disponibile.")
        return
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

def legend_palette(palette: dict):
    pal = {**DEFAULT_PALETTE, **(palette or {})}
    st.caption("Legenda categorie")
    chips = [_badge(k, v) for k, v in pal.items()]
    st.markdown(" ".join(chips), unsafe_allow_html=True)

# ---------------------------
# App (Fake Auth + Profilo + Upload + Classificazione)
# ---------------------------
st.set_page_config(page_title="Data Explainability", page_icon="📊", layout="wide")
st.title("📊 Data Explainability – MVP")

# Fake auth (solo lato FE)
if "auth" not in st.session_state:
    st.session_state.auth = False
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "classification" not in st.session_state:
    st.session_state.classification = None
if "palette" not in st.session_state:
    st.session_state.palette = DEFAULT_PALETTE
if "columns" not in st.session_state:
    st.session_state.columns = []
if "nav" not in st.session_state:
    st.session_state.nav = "📁 I miei dataset"

# --- Login / Logout ---
with st.sidebar:
    st.header("🔐 Accesso (fake)")
    if not st.session_state.auth:
        u = st.text_input("Username / Email", value=st.session_state.user_name or "")
        if st.button("Accedi"):
            if not u.strip():
                st.warning("Inserisci uno username.")
            else:
                st.session_state.auth = True
                st.session_state.user_name = u.strip()
                st.success(f"Benvenuto, {st.session_state.user_name}!")
                st.rerun()
    else:
        st.write(f"👋 Ciao, **{st.session_state.user_name}**")
        if st.button("Esci"):
            for k in ["auth","user_name","dataset_id","classification","palette","columns"]:
                st.session_state.pop(k, None)
            st.session_state.palette = DEFAULT_PALETTE
            st.rerun()

    st.divider()
    # Navigazione
    st.session_state.nav = st.radio(
        "Navigazione",
        options=["📁 I miei dataset", "⬆️ Nuovo upload", "🔍 Classificazione"],
        index=["📁 I miei dataset", "⬆️ Nuovo upload", "🔍 Classificazione"].index(st.session_state.nav),
    )

if not st.session_state.auth:
    st.info("Effettua l’accesso (fake) dal menu a sinistra per proseguire.")
    st.stop()

# ====================== PAGINA: I miei dataset ======================
if st.session_state.nav == "📁 I miei dataset":
    st.subheader("📁 I miei dataset")
    try:
        datasets = list_datasets()
    except Exception as e:
        st.error(str(e))
        datasets = []

    if not datasets:
        st.info("Non hai ancora dataset. Caricane uno dalla sezione '⬆️ Nuovo upload'.")
    else:
        for d in datasets:
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 3])
                cols[0].markdown(f"**{d.get('dataset_name','(senza nome)')}**")
                cols[1].markdown(f"Righe: **{d.get('n_rows','?')}**")
                cols[2].markdown(f"Colonne: **{d.get('n_cols','?')}**")
                actions = st.columns([1,1,1], gap="small")

                # Azione: Dettaglio (espande JSON)
                with actions[0]:
                    if st.button("Dettaglio", key=f"det-{d['id']}"):
                        try:
                            detail = get_dataset_detail(d["id"])
                            with st.expander(f"Dettaglio dataset #{d['id']}", expanded=True):
                                st.json(detail)
                        except Exception as e:
                            st.error(str(e))

                # Azione: Classifica (vai alla pagina Classificazione)
                with actions[1]:
                    if st.button("Classifica", key=f"class-{d['id']}"):
                        st.session_state.dataset_id = d["id"]
                        st.session_state.classification = None
                        st.session_state.columns = []
                        st.session_state.nav = "🔍 Classificazione"
                        st.rerun()

                # Azione: Elimina
                with actions[2]:
                    if st.button("🗑️ Elimina", key=f"del-{d['id']}"):
                        try:
                            delete_dataset_api(d["id"])
                            st.success(f"Dataset {d['id']} eliminato.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

# ====================== PAGINA: Nuovo upload ======================
elif st.session_state.nav == "⬆️ Nuovo upload":
    st.subheader("⬆️ Carica un nuovo dataset (CSV o Parquet)")
    with st.form("upload_form", clear_on_submit=False):
        dataset_name = st.text_input("Nome dataset", placeholder="es. my_sales_2025")
        description = st.text_area("Descrizione (opzionale)")
        file = st.file_uploader("Seleziona file", type=["csv", "parquet", "pq"])
        submit = st.form_submit_button("Carica")
    if submit:
        if not dataset_name.strip():
            st.error("Inserisci un nome dataset.")
        elif not file:
            st.error("Seleziona un file.")
        else:
            try:
                with st.spinner("Upload..."):
                    ds = post_upload_dataset(dataset_name.strip(), description.strip() or None, file.getvalue(), file.name, user_name=st.session_state.user_name)
                st.success(f"Upload completato (id={ds['id']}).")
                # passa subito alla classificazione
                st.session_state.dataset_id = ds["id"]
                st.session_state.classification = None
                st.session_state.columns = []
                st.session_state.nav = "🔍 Classificazione"
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ====================== PAGINA: Classificazione ======================
elif st.session_state.nav == "🔍 Classificazione":
    st.subheader("🔍 Classificazione & Colonne")
    if not st.session_state.dataset_id:
        st.warning("Seleziona un dataset dalla pagina '📁 I miei dataset' oppure caricane uno nuovo.")
        st.stop()

    colA, colB = st.columns([1, 1])
    if colA.button("🔄 Classifica/aggiorna"):
        try:
            with st.spinner("Classificazione in corso..."):
                result = get_classification(st.session_state.dataset_id)
            st.session_state.classification = result
            st.session_state.palette = {**DEFAULT_PALETTE, **(result.get("palette", {}) or {})}
            cols = result.get("columns")
            if not cols:
                cols = schema_to_columns(result.get("schema") or {})
            st.session_state.columns = cols
            st.success("Classificazione completata.")
        except Exception as e:
            st.error(str(e))

    # mostra dataset corrente
    try:
        ds_detail = get_dataset_detail(st.session_state.dataset_id)
        st.caption(f"Dataset: **{ds_detail.get('dataset_name','?')}** • Righe: **{ds_detail.get('n_rows','?')}** • Colonne: **{ds_detail.get('n_cols','?')}**")
    except Exception:
        pass

    result = st.session_state.classification
    palette = st.session_state.palette
    columns = st.session_state.columns

    if result:
        meta = result.get("meta") or {}
        with st.expander("Meta", expanded=True):
            if meta: st.json(meta)
            else: st.info("Meta non disponibile da questo endpoint.")

        st.markdown("#### Schema (conteggi per categoria)")
        render_schema(result.get("schema"))

        warns = result.get("warnings") or []
        if warns:
            st.warning("• " + "\n• ".join(warns))

        st.markdown("#### Colonne")
        if not columns:
            st.info("Premi “Classifica/aggiorna” per popolare le colonne.")
        else:
            legend_palette(palette)
            cats = sorted({c.get("category","unknown") for c in columns})
            chosen = st.multiselect("Filtra per categoria", options=cats, default=cats)
            to_show = [c for c in columns if c.get("category","unknown") in chosen]
            render_columns_table(to_show, palette)
    else:
        st.info("Premi “Classifica/aggiorna” per vedere lo schema.")
