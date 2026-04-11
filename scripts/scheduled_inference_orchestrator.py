"""
Sequential trigger of ingestion → features → processing → inference DAGs.

Dates come from the Airflow Variable ``INGESTION_DATES``: a JSON array of
normalized date strings (``YYYY-MM-DD``), e.g. ``["2023-12-20","2023-12-21"]``.

**Not used by** ``inference_pipeline_dag`` **on Airflow 3+**: programmatic
``trigger_dag`` + ORM polling is forbidden inside task workers. The DAG uses
``TriggerDagRunOperator`` with templated ``conf`` instead. This module remains
useful for scripts or Airflow 2.x-only setups.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_ingestion_dates() -> List[str]:
    from airflow.models import Variable

    raw = Variable.get("INGESTION_DATES", deserialize_json=True)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "Variable INGESTION_DATES must be a non-empty JSON array of strings, "
            'e.g. ["2023-12-20", "2023-12-21"]'
        )
    return [str(d).strip() for d in raw]


def _trigger_dag_run(dag_id: str, conf: Dict[str, Any]) -> str:
    """Trigger a DAG run; return ``run_id``."""
    try:
        from airflow.api.common.trigger_dag import trigger_dag
    except ImportError as e:
        raise RuntimeError(
            "trigger_dag not available — use an Airflow version that provides "
            "airflow.api.common.trigger_dag.trigger_dag"
        ) from e

    try:
        dr = trigger_dag(dag_id=dag_id, conf=conf, replace_microseconds=False)
    except TypeError:
        dr = trigger_dag(dag_id=dag_id, conf=conf)
    run_id = dr.run_id
    logger.info("Triggered %s run_id=%s", dag_id, run_id)
    return run_id


def _wait_for_dag_run(
    dag_id: str,
    run_id: str,
    poke_interval: int = 20,
    timeout_sec: int = 7200,
) -> None:
    """
    Poll until the DagRun finishes successfully or fails.
    """
    from airflow import settings
    from airflow.models import DagRun
    from airflow.utils.state import DagRunState

    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        state: Optional[Any] = None
        session = settings.Session()
        try:
            dr = (
                session.query(DagRun)
                .filter(DagRun.dag_id == dag_id, DagRun.run_id == run_id)
                .one_or_none()
            )
            if dr is not None:
                state = dr.state
        finally:
            session.close()

        if state is not None:
            logger.info("DAG %s run %s state=%s", dag_id, run_id, state)
            if state == DagRunState.SUCCESS:
                return
            if state in (
                DagRunState.FAILED,
                DagRunState.UPSTREAM_FAILED,
            ):
                raise RuntimeError(
                    f"DAG {dag_id} run {run_id} finished with state {state!r}"
                )
        time.sleep(poke_interval)

    raise TimeoutError(
        f"Timeout after {timeout_sec}s waiting for {dag_id} run {run_id}"
    )


def run_inference_pipeline(
    poke_interval: int = 20,
    timeout_per_dag_sec: int = 7200,
) -> List[str]:
    """
    Load dates from Variable ``INGESTION_DATES``, then trigger each pipeline DAG
    in order and wait for completion.

    Returns
    -------
    list of str
        Summary lines for each completed step.
    """
    dates = _load_ingestion_dates()
    conf_full: Dict[str, Any] = {
        "INGESTION_MODE": "inference",
        "INGESTION_DATES": dates,
    }
    conf_inference: Dict[str, Any] = {"INGESTION_DATES": dates}

    steps: List[Tuple[str, Dict[str, Any]]] = [
        ("ingestion_dag", conf_full),
        ("features_dag", conf_full),
        ("processing_dag", conf_full),
        ("inference_dag", conf_inference),
    ]

    summaries: List[str] = []
    for dag_id, conf in steps:
        run_id = _trigger_dag_run(dag_id, conf)
        _wait_for_dag_run(
            dag_id,
            run_id,
            poke_interval=poke_interval,
            timeout_sec=timeout_per_dag_sec,
        )
        summaries.append(f"{dag_id}:ok:{run_id}")

    return summaries
