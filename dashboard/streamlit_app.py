"""
Dashboard métier — acceptation des substitutions.
Données : via l’API FastAPI (`/v1/predictions/files`, `/v1/predictions/content`), qui lit GCS.

  Variables : PREDICTIONS_API_BASE, PREDICTIONS_API_KEY (voir README).
  L’API doit être démarrée et configurée (GOOGLE_APPLICATION_CREDENTIALS côté serveur).

  streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------
# CONFIG
# ----------------------
PREDICTIONS_API_BASE = os.environ.get(
    "PREDICTIONS_API_BASE", "http://127.0.0.1:8000"
).rstrip("/")

INFERENCE_PREFIX = os.environ.get(
    "INFERENCE_GCS_PREFIX", "gs://algo_reco/inference"
).rstrip("/")

DASHBOARD_TITLE = "Substitutions des produits alimentaires en rupture de stock"

st.set_page_config(
    page_title=DASHBOARD_TITLE,
    layout="wide",
)

TARGET = "prediction_estAcceptee_bin"
TARGET_LABEL = "Taux d'acceptation client"
COL_ESTACCEPTEE = "estAcceptee"
COL_PREDICTION = "prediction"
MARQUE = "marqueOriginal"
TYPE_MARQUE = "typeMarqueOriginal"
PRODUIT = "libelleOriginal"
CATEGORIE_ORIGINAL = "categorieOriginal"
DIFF_PRIX = "DiffPrix"
PRODUIT_SUBST = "libelleSubstitution"
MARQUE_SUBST = "marqueSubstitution"

# Effectif minimal par défaut (histogramme + podiums) ; ajustable dans la section produits
MIN_N_PRODUIT_SUBST_DEFAULT = 7

# Palette unique : même échelle « Blues » que « Taux d'acceptation de substitution »
CHART_COLOR_SCALE = "Blues"
BLUE_KPI_BG = px.colors.sequential.Blues[1]
BLUE_KPI_ACCENT = px.colors.sequential.Blues[-1]
PODIUM_ACCEPT_DARK = BLUE_KPI_ACCENT
# Podium « plus refusés » uniquement
PODIUM_REFUS_LINE = "#E65100"
PODIUM_REFUS_FILL_CENTER = "#FFE0B2"
PODIUM_REFUS_TEXT = "#BF360C"
# Largeur ~70 % histogramme / ~30 % podiums (mockup)
_COL_MAIN = [7, 3]
# Sliders compacts (section produits + heatmap) — même largeur relative
_SLIDER_NARROW_COLS = [0.22, 0.78]

# Style injecté une fois : pistes / curseurs des sliders en gris foncé
_SLIDER_DARK_GRAY_CSS = """
<style>
    /* Piste (rail) : gris foncé uniforme, sans dégradé thème */
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background: #4b5563 !important;
        background-image: none !important;
    }
    /* Curseur */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #1f2937 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stSlider"] [role="slider"]:focus-visible {
        box-shadow: 0 0 0 2px #9ca3af !important;
    }
    /* Libellé valeur au-dessus du curseur */
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
        color: #374151 !important;
    }
    [data-testid="stSliderTickBar"] {
        color: #4b5563 !important;
    }
</style>
"""

REQUIRED_COLUMNS = [
    TARGET,
    MARQUE,
    TYPE_MARQUE,
    PRODUIT,
    CATEGORIE_ORIGINAL,
    DIFF_PRIX,
    PRODUIT_SUBST,
    MARQUE_SUBST,
]


def _predictions_api_key() -> str:
    return os.environ.get("PREDICTIONS_API_KEY", "").strip()


@st.cache_data(ttl=60)
def list_inference_files_from_api(api_base: str) -> list[str]:
    """Liste les .csv / .json du préfixe inference configuré sur l’API (source=inference)."""
    key = _predictions_api_key()
    if not key:
        raise RuntimeError("PREDICTIONS_API_KEY manquant (même valeur que sur le serveur API).")
    url = f"{api_base.rstrip('/')}/v1/predictions/files"
    with httpx.Client(timeout=60.0) as client:
        r = client.get(
            url,
            params={"source": "inference"},
            headers={"X-API-Key": key},
        )
        r.raise_for_status()
        body = r.json()
    files = list(body.get("files") or [])
    return sorted(files, reverse=True)


def _date_label_only(uri: str) -> str:
    base = uri.split("/")[-1]
    m = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{4}_\d{2}_\d{2})", base)
    if m2:
        return m2.group(1).replace("_", "-")
    return base


def _inference_choice_labels(all_uris: list[str]) -> dict[str, str]:
    """Libellé = date seule ; si doublon de date, suffixe (1), (2), …"""
    by_date: dict[str, list[str]] = defaultdict(list)
    for u in all_uris:
        by_date[_date_label_only(u)].append(u)
    out: dict[str, str] = {}
    for _d, group in by_date.items():
        group_sorted = sorted(group)
        if len(group_sorted) == 1:
            out[group_sorted[0]] = _date_label_only(group_sorted[0])
        else:
            for i, u in enumerate(group_sorted, start=1):
                out[u] = f"{_date_label_only(u)} ({i})"
    return out


@st.cache_data(ttl=300)
def load_data(gcs_uri: str, api_base: str, row_limit: int) -> pd.DataFrame:
    """Charge un fichier de prédictions via l’API (contenu tabulaire en JSON)."""
    key = _predictions_api_key()
    if not key:
        raise RuntimeError("PREDICTIONS_API_KEY manquant.")
    lim = min(max(row_limit, 1), 50_000)
    url = f"{api_base.rstrip('/')}/v1/predictions/content"
    with httpx.Client(timeout=120.0) as client:
        r = client.get(
            url,
            params={"gs_uri": gcs_uri, "limit": lim},
            headers={"X-API-Key": key},
        )
        if r.status_code == 404:
            raise FileNotFoundError(gcs_uri)
        r.raise_for_status()
        payload = r.json()
    rows = payload.get("rows") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _ensure_target_column(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Si la cible agrégée est absente mais estAcceptee + prediction sont présentes :
    prediction_estAcceptee_bin = 1 ssi estAcceptee == 1 et prediction == 1, sinon 0.
    """
    if TARGET in df.columns:
        return df, False
    if COL_ESTACCEPTEE not in df.columns or COL_PREDICTION not in df.columns:
        return df, False
    out = df.copy()
    a = pd.to_numeric(out[COL_ESTACCEPTEE], errors="coerce").fillna(0).astype(int).eq(1)
    b = pd.to_numeric(out[COL_PREDICTION], errors="coerce").fillna(0).astype(int).eq(1)
    out[TARGET] = (a & b).astype(int)
    return out, True


def _truncate_label(s: str, max_len: int = 32) -> str:
    t = str(s)
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _heatmap_figure(
    pivot: pd.DataFrame,
    text_matrix: list[list[str]],
    *,
    top_n: int,
) -> go.Figure:
    """Heatmap marque × marque : mise en page soignée (gaps, axes, colorbar)."""
    z = pivot.values
    target_h = min(220 + top_n * 30, 920)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=pivot.columns.astype(str).tolist(),
                y=pivot.index.astype(str).tolist(),
                zmin=0,
                zmax=1,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=10, color="#0f172a", family="system-ui, sans-serif"),
                colorscale=CHART_COLOR_SCALE,
                showscale=True,
                xgap=2,
                ygap=2,
                colorbar=dict(
                    title=dict(
                        text=TARGET_LABEL,
                        side="right",
                        font=dict(size=11, color="#475569"),
                    ),
                    tickformat=".0%",
                    tickfont=dict(size=10, color="#64748b"),
                    thickness=14,
                    len=0.55,
                    outlinewidth=0,
                    bgcolor="rgba(255,255,255,0.85)",
                ),
                hovertemplate=(
                    "<b>%{y}</b> → <b>%{x}</b><br>"
                    f"{TARGET_LABEL}: <b>%{{z:.1%}}</b><extra></extra>"
                ),
            )
        ]
    )
    axis_title = dict(font=dict(size=12, color="#64748b"))
    tick_font = dict(size=10, color="#475569")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="system-ui, sans-serif", color="#334155"),
        height=target_h,
        margin=dict(l=8, r=8, t=56, b=112),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#e2e8f0",
            font_size=12,
            font_family="system-ui, sans-serif",
        ),
        xaxis=dict(
            title=dict(text="Marque de substitution", **axis_title),
            tickangle=-42,
            tickfont=tick_font,
            side="bottom",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#e2e8f0",
            mirror=False,
            constrain="domain",
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Marque originale", **axis_title),
            tickfont=tick_font,
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#e2e8f0",
            scaleanchor="x",
            scaleratio=1,
            automargin=True,
        ),
    )
    return fig


def _heatmap_cell_text(rate: float) -> str:
    """Texte affiché dans la cellule ; vide si le taux arrondi à 0 %."""
    if pd.isna(rate):
        return ""
    s = f"{float(rate):.0%}"
    return "" if s == "0%" else s


def _podium_ann_text(product: str, taux: float) -> str:
    """3 lignes : nom sur 2 lignes (équilibré) + taux seul, sans effectif."""
    taux_pct = f"{taux:.1%}"
    label = _truncate_label(product, 64)
    words = label.split()
    if not words:
        return f"<br><br>{taux_pct}"
    if len(words) == 1:
        return f"{words[0]}<br>\u00a0<br>{taux_pct}"
    mid = (len(words) + 1) // 2
    line_a = " ".join(words[:mid])
    line_b = " ".join(words[mid:])
    return f"{line_a}<br>{line_b}<br>{taux_pct}"


def _podium_figure(
    trio: pd.DataFrame,
    *,
    title: str,
    figure_height: int = 340,
    refus: bool = False,
) -> go.Figure | None:
    """
    Podium 2D : marches arrondies. Acceptés : Blues ; refusés : orange (si refus=True).
    Ordre : marche 2, marche 1 (plus haute), marche 3.
    """
    if trio.empty:
        return None
    trio = trio.reset_index(drop=True)
    k = len(trio)
    if refus:
        line_kw = dict(color=PODIUM_REFUS_LINE, width=3)
        fill_center = PODIUM_REFUS_FILL_CENTER
        fill_side = "#FFFFFF"
    else:
        line_kw = dict(color=PODIUM_ACCEPT_DARK, width=3)
        fill_center = PODIUM_ACCEPT_DARK
        fill_side = "#FFFFFF"

    if k >= 3:
        heights_idx = [1, 0, 2]
        x_labels = ["2", "1", "3"]
        y_vals = [0.48, 0.82, 0.34]
        fills = [fill_side, fill_center, fill_side]
    elif k == 2:
        heights_idx = [1, 0]
        x_labels = ["2", "1"]
        y_vals = [0.52, 0.80]
        fills = [fill_side, fill_center]
    else:
        heights_idx = [0]
        x_labels = ["1"]
        y_vals = [0.88]
        fills = [fill_center]

    if refus:
        if k >= 3:
            text_colors = [PODIUM_REFUS_TEXT, PODIUM_REFUS_TEXT, PODIUM_REFUS_TEXT]
        elif k == 2:
            text_colors = [PODIUM_REFUS_TEXT, PODIUM_REFUS_TEXT]
        else:
            text_colors = [PODIUM_REFUS_TEXT]
    elif k >= 3:
        text_colors = [PODIUM_ACCEPT_DARK, "#FFFFFF", PODIUM_ACCEPT_DARK]
    elif k == 2:
        text_colors = [PODIUM_ACCEPT_DARK, "#FFFFFF"]
    else:
        text_colors = ["#FFFFFF"]

    text_px = max(18, min(30, int(figure_height * 0.082)))
    ann_px = max(9, min(12, figure_height // 28))

    annotations: list[dict] = []
    ymax = max(y_vals) * 1.38
    for j, xl in enumerate(x_labels):
        idx = heights_idx[j]
        row = trio.iloc[idx]
        yv = y_vals[j]
        ann_txt = _podium_ann_text(row[PRODUIT_SUBST], float(row["_taux"]))
        annotations.append(
            dict(
                x=xl,
                y=yv + 0.02,
                text=ann_txt,
                showarrow=False,
                yanchor="bottom",
                font=dict(size=ann_px, color="#263238"),
                align="center",
            )
        )

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=y_vals,
                text=x_labels,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=text_px, color=text_colors, family="Arial Black, sans-serif"
                ),
                marker=dict(
                    color=fills,
                    line=line_kw,
                    cornerradius=14,
                ),
                width=0.58,
            )
        ]
    )
    layout = dict(
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(visible=False, range=[0, ymax], fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.22,
        showlegend=False,
        height=figure_height,
        margin=dict(
            t=max(14, figure_height // 16),
            b=max(10, figure_height // 24),
            l=28,
            r=28,
        ),
        annotations=annotations,
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=14))
    fig.update_layout(**layout)
    return fig


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le CSV : {missing}")
        st.stop()
    if len(df) == 0:
        st.warning("Le jeu de données est vide.")
        st.stop()
    if df[TARGET].dropna().isin([0, 1]).mean() < 0.99:
        st.warning(
            "La colonne cible ne semble pas strictement binaire (0/1). "
            "Les KPI « taux » peuvent être moins interprétables."
        )


# ----------------------
# LOAD DATA
# ----------------------
with st.sidebar:
    if not _predictions_api_key():
        st.warning(
            "Définissez `PREDICTIONS_API_KEY` (identique à celle du serveur API)."
        )
    st.caption(f"API : `{PREDICTIONS_API_BASE}`")

    try:
        choices = list_inference_files_from_api(PREDICTIONS_API_BASE)
    except Exception as e:
        choices = []
        if isinstance(e, httpx.ConnectError):
            st.error(
                f"Impossible de joindre l’API à `{PREDICTIONS_API_BASE}` ({e}). "
                "Démarre l’API (ex. `uvicorn` sur le bon port) et vérifie `PREDICTIONS_API_BASE` "
                "(défaut dashboard : 8000, souvent 8001 côté projet)."
            )
        else:
            st.error(f"Impossible de lister les fichiers via l’API : {e}")

    env_override = os.environ.get("DASHBOARD_PREDICTIONS_GCS_URI", "").strip()
    if env_override and env_override.startswith("gs://") and env_override not in choices:
        choices = [env_override] + [c for c in choices if c != env_override]

    if choices:
        _labels = _inference_choice_labels(choices)
        # « Tous » + chaque fichier (libellé = date), dates les plus récentes en premier
        sorted_choices = sorted(choices, key=lambda u: _date_label_only(u), reverse=True)
        all_options = ["__ALL__"] + sorted_choices

        def _format_filtre_option(x: str) -> str:
            if x == "__ALL__":
                return "Tous"
            return _labels.get(x, x)

        picked = st.multiselect(
            "Choisissez une date d'inférence",
            options=all_options,
            default=["__ALL__"],
            format_func=_format_filtre_option,
            help="Cochez « Tous » seul pour tout charger. Décochez « Tous » puis cochez une ou plusieurs dates pour filtrer.",
        )

        if not picked:
            selected_uris = []
        else:
            sans_tous = [p for p in picked if p != "__ALL__"]
            if sans_tous:
                selected_uris = sans_tous
            elif "__ALL__" in picked:
                selected_uris = list(choices)
            else:
                selected_uris = []
    else:
        manual = st.text_input(
            "URI gs:// (aucun fichier listé)",
            value=env_override or f"{INFERENCE_PREFIX}/",
            help="Saisie manuelle si l’API ne retourne aucun fichier ou pour un URI précis.",
        )
        selected_uris = [manual] if manual.strip().startswith("gs://") else []

if not selected_uris:
    st.warning("Sélectionnez au moins un fichier dans la barre latérale.")
    st.stop()

try:
    _row_limit = min(
        int(os.environ.get("STREAMLIT_PREDICTIONS_ROW_LIMIT", "50000")), 50_000
    )
except ValueError:
    _row_limit = 50_000
_row_limit = max(_row_limit, 1)
dfs: list[pd.DataFrame] = []
for uri in selected_uris:
    try:
        part = load_data(uri, PREDICTIONS_API_BASE, _row_limit)
        dfs.append(part)
    except Exception as e:
        if isinstance(e, httpx.ConnectError):
            st.error(
                f"Connexion refusée vers l’API `{PREDICTIONS_API_BASE}` ({e}). "
                "Le dashboard charge les données via HTTP, pas directement depuis `gs://`. "
                "Démarre FastAPI et aligne le port avec `PREDICTIONS_API_BASE`."
            )
        else:
            st.error(f"Impossible de charger `{uri}` : {e}")
        st.caption(
            "Vérifie que l’API tourne, `PREDICTIONS_API_KEY`, `ALLOWED_GCS_PREFIXES` côté API "
            "et que le `gs_uri` pointe vers un fichier `.csv` / `.json` (pas un préfixe dossier seul)."
        )
        st.stop()

df = pd.concat(dfs, ignore_index=True)
df, _ = _ensure_target_column(df)

_validate_df(df)

st.markdown(_SLIDER_DARK_GRAY_CSS, unsafe_allow_html=True)

# ----------------------
# TITLE
# ----------------------
st.markdown(
    f"<h1 style='text-align:center; color:{BLUE_KPI_ACCENT}'>{DASHBOARD_TITLE}</h1>",
    unsafe_allow_html=True,
)

# ----------------------
# KPIs
# ----------------------
col1, col2, col3 = st.columns(3)

nb_transactions = len(df)
nb_produits = df[PRODUIT].nunique()
taux_acceptation = float(df[TARGET].mean())


def kpi(title: str, value: str, color: str | None = None) -> None:
    vcol = color if color is not None else BLUE_KPI_ACCENT
    st.markdown(
        f"""
        <div style='
            background-color: {BLUE_KPI_BG};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        '>
            <div style='font-size:18px; font-weight:600; color:{BLUE_KPI_ACCENT}'>{title}</div>
            <div style='font-size:32px; font-weight:700; color:{vcol}'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col1:
    kpi("Nombre de transactions", f"{nb_transactions:,}")
with col2:
    kpi("Produits originaux substitués", f"{nb_produits:,}")
with col3:
    kpi(TARGET_LABEL, f"{taux_acceptation:.1%}")

st.divider()

# ----------------------
# ACCEPTATION PAR DIMENSIONS
# ----------------------
st.subheader("Taux d'acceptation de substitution")
col1, col2, col3 = st.columns(3)


def taux_acceptation_par(col: str) -> pd.DataFrame:
    return df.groupby(col, dropna=False)[TARGET].mean().reset_index().sort_values(
        TARGET, ascending=False
    )


for c, label, data_col in zip(
    [col1, col2, col3],
    ["Type de marque", "Marque", "Catégorie"],
    [TYPE_MARQUE, MARQUE, CATEGORIE_ORIGINAL],
):
    with c:
        st.markdown(
            f"<h4 style='text-align:center'>Par {label.lower()}</h4>",
            unsafe_allow_html=True,
        )

        data = taux_acceptation_par(data_col)
        if data_col == CATEGORIE_ORIGINAL:
            data = data.head(15)

        fig = px.bar(
            data,
            x=TARGET,
            y=data_col,
            orientation="h",
            labels={TARGET: TARGET_LABEL, data_col: label},
            color=TARGET,
            color_continuous_scale=CHART_COLOR_SCALE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------
# TOP PRODUITS ACCEPTÉS / REFUSÉS
# ----------------------
st.subheader("Produits les plus acceptés / refusés")

_n_per_prod = df.groupby(PRODUIT_SUBST)[TARGET].count()
_n_slider_max = max(int(_n_per_prod.max()) if len(_n_per_prod) else 1, MIN_N_PRODUIT_SUBST_DEFAULT)
_col_n_slider, _col_n_rest = st.columns(_SLIDER_NARROW_COLS, gap="small")
with _col_n_slider:
    _min_n = st.slider(
        "Effectif minimal n (par produit de substitution)",
        min_value=1,
        max_value=_n_slider_max,
        value=min(MIN_N_PRODUIT_SUBST_DEFAULT, _n_slider_max),
        step=1,
        help="Appliqué à l’histogramme et aux podiums Top 3 des plus acceptés et des plus refusés.",
    )

_stats_sub = (
    df.groupby(PRODUIT_SUBST)[TARGET]
    .agg(_taux="mean", n="count")
    .reset_index()
)
_stats_sub = _stats_sub[_stats_sub["n"] >= _min_n]

_max_bars_hist = 60
_hist_df: pd.DataFrame | None = None
hist_height = 420
podium_height = 210
if not _stats_sub.empty:
    _hist_df = _stats_sub.sort_values("_taux", ascending=False).head(_max_bars_hist)
    _hist_df = _hist_df.iloc[::-1]
    hist_height = max(420, 12 * len(_hist_df))
    # Deux podiums ~ moitié-moitié de la hauteur du graphique (mockup)
    _slack_between_podiums = 52
    podium_height = max(160, (hist_height - _slack_between_podiums) // 2)

col_hist_title, col_podium_title = st.columns(_COL_MAIN)
with col_hist_title:
    st.markdown(
        f'<p style="margin:0 0 0.2rem 0;line-height:1.25;"><strong>{TARGET_LABEL} par produit (substitution)</strong> — du plus au moins accepté (n ≥ {_min_n})</p>',
        unsafe_allow_html=True,
    )
with col_podium_title:
    if not _stats_sub.empty:
        st.markdown(
            '<p style="margin:0 0 0.2rem 0;line-height:1.25;"><strong>Top 3 des produits les plus acceptés</strong></p>',
            unsafe_allow_html=True,
        )

col_hist, col_podium = st.columns(_COL_MAIN)

with col_hist:
    if _stats_sub.empty:
        st.caption(
            f"Aucun produit avec n ≥ {_min_n}. Élargis les dates ou la base."
        )
    else:
        assert _hist_df is not None
        fig_hist = px.bar(
            _hist_df,
            x="_taux",
            y=PRODUIT_SUBST,
            orientation="h",
            labels={
                PRODUIT_SUBST: "Produit substitution",
                "_taux": TARGET_LABEL,
            },
            color="_taux",
            color_continuous_scale=CHART_COLOR_SCALE,
        )
        fig_hist.update_layout(
            showlegend=False,
            height=hist_height,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        if len(_stats_sub) > _max_bars_hist:
            st.caption(
                f"Affichage limité aux {_max_bars_hist} produits avec le taux le plus élevé "
                f"({len(_stats_sub)} produits au total après filtre n)."
            )

with col_podium:
    if _stats_sub.empty:
        st.caption(
            f"Aucun produit avec n ≥ {_min_n}. Élargis les dates ou la base."
        )
    else:
        trio_top = _stats_sub.sort_values("_taux", ascending=False).head(3)
        trio_flop = _stats_sub.sort_values("_taux", ascending=True).head(3)
        fig_top = _podium_figure(trio_top, title="", figure_height=podium_height)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)
        st.markdown(
            '<p style="margin:0 0 0.2rem 0;line-height:1.25;"><strong>Top 3 des produits les plus refusés</strong></p>',
            unsafe_allow_html=True,
        )
        fig_flop = _podium_figure(
            trio_flop, title="", figure_height=podium_height, refus=True
        )
        if fig_flop:
            st.plotly_chart(fig_flop, use_container_width=True)

st.divider()

# ----------------------
# ACCEPTATION VS DIFF PRIX
# ----------------------
st.subheader("Taux d'acceptation en fonction de la différence de prix")
df_prix_clean = df.dropna(subset=[DIFF_PRIX])
if len(df_prix_clean) < 2:
    st.info("Pas assez de lignes avec `DiffPrix` renseigné pour ce graphique.")
else:
    df_buck: pd.DataFrame | None = None
    try:
        df_buck = df_prix_clean.assign(
            diff_prix_bucket=pd.cut(df_prix_clean[DIFF_PRIX], bins=10).astype(str)
        )
    except ValueError as e:
        st.warning(f"Impossible de découper `DiffPrix` en bins : {e}")

    if df_buck is not None and not df_buck.empty:
        df_prix_agg = df_buck.groupby("diff_prix_bucket", as_index=False).agg(
            taux_acceptation=(TARGET, "mean"),
            volume=("diff_prix_bucket", "count"),
        )

        fig = px.bar(
            df_prix_agg,
            x="diff_prix_bucket",
            y="taux_acceptation",
            text=df_prix_agg["taux_acceptation"].map(lambda x: f"{x:.1%}"),
            labels={
                "taux_acceptation": TARGET_LABEL,
                "diff_prix_bucket": "Différence de prix (€)",
            },
            color="taux_acceptation",
            color_continuous_scale=CHART_COLOR_SCALE,
        )
        fig.update_layout(
            uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Différence de prix = prix du produit de substitution − prix du produit d’origine."
        )

st.divider()

# ----------------------
# HEATMAP MARQUE × MARQUE (top N)
# ----------------------
st.subheader("La substitution entre marque originale et marque de substitution")
_col_hm_slider, _col_hm_rest = st.columns(_SLIDER_NARROW_COLS, gap="small")
with _col_hm_slider:
    heatmap_top_n = st.slider(
        "Choisissez le nombre de marques (top N)",
        min_value=5,
        max_value=40,
        value=15,
        help="Réduit la matrice marque × marque pour lisibilité et perfs.",
    )
st.caption(
    "Chaque case résume le taux d’acceptation pour une paire marque d’origine × marque proposée "
    "(top N marques par volume sur chaque axe)."
)
top_orig = df[MARQUE].value_counts().head(heatmap_top_n).index
top_subst = df[MARQUE_SUBST].value_counts().head(heatmap_top_n).index
df_hm = df[df[MARQUE].isin(top_orig) & df[MARQUE_SUBST].isin(top_subst)]

if df_hm.empty:
    st.info("Pas de données pour la heatmap avec ces filtres.")
else:
    pivot_marque = df_hm.pivot_table(
        index=MARQUE, columns=MARQUE_SUBST, values=TARGET, aggfunc="mean"
    ).fillna(0)
    text_matrix = [
        [_heatmap_cell_text(pivot_marque.iloc[i, j]) for j in range(pivot_marque.shape[1])]
        for i in range(pivot_marque.shape[0])
    ]
    fig = _heatmap_figure(pivot_marque, text_matrix, top_n=heatmap_top_n)
    st.plotly_chart(fig, use_container_width=True)
