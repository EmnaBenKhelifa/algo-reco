# algo-reco

End-to-end ML project for food product substitution recommendations during stockouts.

The project includes:
- an orchestration layer with Airflow DAGs (`dags/`),
- a FastAPI service exposing prediction files and content (`api/`),
- a Streamlit dashboard consuming the API (`dashboard/`),
- deployment assets for Cloud Run (`deploy/`) and helper commands in `Makefile`.

## 1) Repository structure

- `api/`: FastAPI app and GCS access helpers.
- `dashboard/`: Streamlit business dashboard.
- `dags/`: Airflow DAGs for ingestion/features/processing/inference and pipeline orchestration.
- `scripts/`: Python business logic used by DAGs.
- `deploy/`: Dockerfiles + Cloud Build configs for API and dashboard.
- `airflow_settings.yaml`: local Airflow variables (ex: `INGESTION_DATES`).
- `Makefile`: build/deploy commands for Cloud Run.

## 2) API endpoints used by Streamlit

The dashboard consumes these API routes:

- `GET /v1/predictions/files?source=inference|predictions|all`
- `GET /v1/predictions/content?gs_uri=...&limit=...`
- `GET /health`

Auth is done with header `X-API-Key`.

Swagger is available at:
- `/docs`
- `/openapi.json`

## 3) Local run (API + Streamlit)

### 3.1 API (FastAPI)

```bash
cd /home/emna/code/algo-reco
export PYTHONPATH=/home/emna/code/algo-reco
export PREDICTIONS_API_KEY="your-local-key"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/sa.json"
uvicorn api.app:app --host 0.0.0.0 --port 8001
```

### 3.2 Dashboard (Streamlit)

```bash
cd /home/emna/code/algo-reco
export PREDICTIONS_API_BASE=http://127.0.0.1:8001
export PREDICTIONS_API_KEY="your-local-key"
streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://localhost:8501`.

## 4) Airflow local run (Astro)

Start the local Airflow stack:

```bash
cd /home/emna/code/algo-reco
astro dev start
```

UI:
- `http://localhost:8080`

`inference_pipeline_dag` triggers in sequence:
- `ingestion_dag` -> `features_dag` -> `processing_dag` -> `inference_dag`

`INGESTION_DATES` is provided through Airflow Variables (JSON), also defined in `airflow_settings.yaml` for local dev.

## 5) Deploy to Cloud Run

The Makefile supports separate services for API and dashboard.

### 5.1 Required env vars (in `.env`, ignored by git)

At minimum:

```env
DEPLOY_PROJECT_ID=your-gcp-project
PREDICTIONS_API_KEY=your-shared-api-key
```

Optional / recommended for custom GCS prefixes:

```env
PREDICTIONS_GCS_PREFIX=gs://<bucket>/predictions
INFERENCE_GCS_PREFIX=gs://<bucket>/inference
ALLOWED_GCS_PREFIXES=gs://<bucket>/predictions,gs://<bucket>/inference
```

### 5.2 Build and deploy

```bash
cd /home/emna/code/algo-reco
make deploy-all
```

Or step by step:

```bash
make build
make deploy-api
make deploy-dashboard
```

Get service URLs:

```bash
make urls
```

## 6) Security notes

- Never commit service account keys (`config/*.json` credentials are ignored).
- Prefer GCP Secret Manager for production secrets.
- Keep `PREDICTIONS_API_KEY` identical between API and dashboard.

## 7) Common troubleshooting

- `401` on API routes: invalid/missing `X-API-Key`.
- `404` on `/v1/predictions/content`: `gs_uri` points to missing object.
- `502` on `/v1/predictions/content`: server-side read issue (permissions, unsupported path, invalid object content).
- Empty Airflow DAG list: check `dags/` files exist and run `airflow dags list-import-errors`.
