import os
from functools import lru_cache


def _split_prefixes(raw: str) -> list[str]:
    return [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]


@lru_cache
def get_settings() -> dict:
    predictions = os.environ.get(
        "PREDICTIONS_GCS_PREFIX", "gs://algo_reco/predictions"
    ).rstrip("/")
    inference = os.environ.get(
        "INFERENCE_GCS_PREFIX", "gs://algo_reco/inference"
    ).rstrip("/")
    allowed_raw = os.environ.get(
        "ALLOWED_GCS_PREFIXES", f"{predictions},{inference}"
    )
    allowed = _split_prefixes(allowed_raw)
    api_key = os.environ.get("PREDICTIONS_API_KEY", "")
    return {
        "predictions_prefix": predictions,
        "inference_prefix": inference,
        "allowed_prefixes": allowed,
        "api_key": api_key,
    }


def reload_settings() -> None:
    get_settings.cache_clear()
