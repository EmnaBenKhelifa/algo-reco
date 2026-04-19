"""
FastAPI service: list and read prediction files from GCS.

OpenAPI / Swagger: /docs and /openapi.json
Auth: header X-API-Key (see PREDICTIONS_API_KEY).
"""
from __future__ import annotations

from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api.config import get_settings
from api.deps import require_api_key
from api import gcs_service

app = FastAPI(
    title="algo-reco predictions API",
    description=(
        "Exposition des prédictions batch stockées dans GCS (chemins type "
        "`predictions/` et `inference/`). Protégé par clé API."
    ),
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str = "ok"


class Source(str, Enum):
    predictions = "predictions"
    inference = "inference"
    all = "all"


class PredictionFile(BaseModel):
    gs_uri: str = Field(description="Full gs:// URI of the file")


class PredictionFilesResponse(BaseModel):
    files: list[str]
    source: str


class PredictionRowsResponse(BaseModel):
    gs_uri: str
    row_count: int
    rows: list[dict]


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse()


@app.get(
    "/v1/predictions/files",
    response_model=PredictionFilesResponse,
    tags=["predictions"],
    dependencies=[Depends(require_api_key)],
)
def list_prediction_files(
    source: Source = Query(
        Source.all,
        description="Which GCS root to list (predictions folder, inference folder, or both).",
    ),
):
    s = get_settings()
    roots: list[str] = []
    if source in (Source.predictions, Source.all):
        roots.append(s["predictions_prefix"])
    if source in (Source.inference, Source.all):
        roots.append(s["inference_prefix"])
    files: list[str] = []
    for root in roots:
        files.extend(gcs_service.list_data_uris(root + "/"))
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return PredictionFilesResponse(files=unique, source=source.value)


@app.get(
    "/v1/predictions/content",
    response_model=PredictionRowsResponse,
    tags=["predictions"],
    dependencies=[Depends(require_api_key)],
)
def get_prediction_content(
    gs_uri: str = Query(
        ...,
        description="Full gs:// URI to a .csv or .json predictions file",
    ),
    limit: int = Query(
        2000,
        ge=1,
        le=50_000,
        description="Max rows to return (CSV only applies nrows; JSON lists are truncated).",
    ),
):
    s = get_settings()
    if not gcs_service.is_allowed(gs_uri, s["allowed_prefixes"]):
        raise HTTPException(
            status_code=403,
            detail="gs_uri is not under an allowed prefix (ALLOWED_GCS_PREFIXES)",
        )
    try:
        rows = gcs_service.read_tabular_json(gs_uri, limit=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Object not found") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GCS read failed: {e!s}")
    return PredictionRowsResponse(gs_uri=gs_uri, row_count=len(rows), rows=rows)
