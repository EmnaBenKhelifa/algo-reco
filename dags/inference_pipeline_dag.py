# dags/inference_pipeline_dag.py
"""
Full inference pipeline (single entry point).

Enchaîne : ``ingestion_dag`` → ``features_dag`` → ``processing_dag`` → ``inference_dag``.

- **Planifié** via ``schedule`` (défaut : tous les jours à 6h UTC).
- Les dates viennent de la Variable Airflow **JSON** ``INGESTION_DATES``
  (ex. ``["2023-12-22","2023-12-23"]``), injectée dans ``conf`` via le template
  ``var.json.INGESTION_DATES`` — **sans code Python** dans la tâche, compatible
  Airflow 3 (pas d’accès ORM dans le worker).

Configurer ``INGESTION_DATES`` dans l’UI (type JSON) ou ``airflow_settings.yaml``.
"""

from airflow import DAG
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime

# Conf partagée (sauf inference_dag qui n’a pas besoin de INGESTION_MODE)
_CONF_PIPELINE = {
    "INGESTION_MODE": "inference",
    "INGESTION_DATES": "{{ var.json.INGESTION_DATES }}",
}

_CONF_INFERENCE = {
    "INGESTION_DATES": "{{ var.json.INGESTION_DATES }}",
}

with DAG(
    dag_id="inference_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    render_template_as_native_obj=True,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "inference_pipeline"],
    doc_md=__doc__,
) as dag:
    trigger_ingestion = TriggerDagRunOperator(
        task_id="trigger_ingestion",
        trigger_dag_id="ingestion_dag",
        conf=_CONF_PIPELINE,
        wait_for_completion=True,
    )

    trigger_features = TriggerDagRunOperator(
        task_id="trigger_features",
        trigger_dag_id="features_dag",
        conf=_CONF_PIPELINE,
        wait_for_completion=True,
    )

    trigger_processing = TriggerDagRunOperator(
        task_id="trigger_processing",
        trigger_dag_id="processing_dag",
        conf=_CONF_PIPELINE,
        wait_for_completion=True,
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="inference_dag",
        conf=_CONF_INFERENCE,
        wait_for_completion=True,
    )

    trigger_ingestion >> trigger_features >> trigger_processing >> trigger_inference
