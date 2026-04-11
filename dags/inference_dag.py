"""
ML inference DAG (per date).

Loads preprocessed matrices from GCS, full feature rows from BigQuery, runs the
model, and writes ``raw_features_dataset_with_predictions_{date}.csv`` to GCS.

Trigger with conf, for example::

    {"INGESTION_DATES": ["2023-12-20"], "threshold": 0.5}
"""

import logging

import joblib
import pandas as pd
from airflow.sdk import dag, task
from pendulum import datetime

from scripts.ingestion import Ingestion
from scripts.utils import dump_data_gcs, load_data_bq, load_data_gcs
from scripts.inference import run_inference

logger = logging.getLogger(__name__)

DEFAULT_MODEL_GCS = "gs://algo_reco/models/best_model.joblib"
DEFAULT_BEST_PARAMS_GCS = "gs://algo_reco/models/best_params.json"
DEFAULT_PREPROCESSED_PREFIX = "gs://algo_reco/features/inference/x_inference_preprocessed_"
DEFAULT_OUTPUT_GCS = "gs://algo_reco/inference"


@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None,
    default_args={"owner": "ML", "retries": 3},
    catchup=False,
    tags=["ml", "inference"],
    doc_md=__doc__,
)
def inference_dag():

    ingestion = Ingestion()

    @task
    def run_inference_per_date(**context):
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run else {}

        dates = conf.get("INGESTION_DATES", [])
        if not dates:
            raise ValueError(
                "inference_dag requires dag_run.conf['INGESTION_DATES'], "
                'e.g. {"INGESTION_DATES": ["2023-12-20"]}'
            )

        threshold = float(conf.get("threshold", 0.5))
        model_gcs_path = conf.get("model_gcs_path", DEFAULT_MODEL_GCS)
        best_params_gcs_path = conf.get("best_params_gcs_path", DEFAULT_BEST_PARAMS_GCS)
        preprocessed_prefix = conf.get("preprocessed_gcs_prefix", DEFAULT_PREPROCESSED_PREFIX)
        output_gcs_path = conf.get("output_gcs_path", DEFAULT_OUTPUT_GCS)

        project_id = ingestion.project_id
        dataset_id = ingestion.dataset_id

        logger.info("INFERENCE DAG : Loading model from %s", model_gcs_path)
        model = load_data_gcs(model_gcs_path)
        if isinstance(model, str):
            model = joblib.load(model)

        logger.info("INFERENCE DAG : Loading best_params from %s", best_params_gcs_path)
        best_params = load_data_gcs(best_params_gcs_path)
        if not isinstance(best_params, dict):
            raise TypeError("best_params must be a dict")

        summaries = []
        for ds in dates:
            ds_table = ds.replace("-", "_")
            z_path = f"{preprocessed_prefix.rstrip('/')}_{ds}.csv"
            logger.info("INFERENCE DAG : Loading preprocessed features from %s", z_path)
            Z_transformed = load_data_gcs(z_path)
            if not isinstance(Z_transformed, pd.DataFrame):
                raise TypeError("Preprocessed features must be a pandas DataFrame")

            table_name = f"features_inference_{ds_table}"
            logger.info("INFERENCE DAG : Loading raw features from BQ %s.%s.%s", project_id, dataset_id, table_name)
            raw_full = load_data_bq(project_id, dataset_id, table_name)

            if len(raw_full) != len(Z_transformed):
                raise ValueError(
                    f"Row count mismatch for {ds}: BigQuery {table_name} has {len(raw_full)} rows, "
                    f"preprocessed {z_path} has {len(Z_transformed)} rows"
                )

            raw_for_output = raw_full.copy().reset_index(drop=True)

            out = run_inference(
                model=model,
                best_params=best_params,
                Z_transformed=Z_transformed,
                threshold=threshold,
                add_proba=True,
                raw_features_df=raw_for_output,
            )

            out_name = f"raw_features_dataset_with_predictions_{ds}"
            dump_data_gcs(data=out, path=output_gcs_path, filename=out_name)
            summaries.append(f"{ds} -> {output_gcs_path}/{out_name}.csv")

        return " | ".join(summaries)

    run_inference_per_date()


inference_dag()
