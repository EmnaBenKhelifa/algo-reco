# scripts/inference.py
from typing import Optional

from airflow.utils.log.logging_mixin import LoggingMixin
import pandas as pd

logger = LoggingMixin().log


def run_inference(
    model,
    best_params: dict,
    Z_transformed: pd.DataFrame,
    threshold: float = 0.5,
    add_proba: bool = True,
    raw_features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run inference on already-loaded artifacts.

    Parameters
    ----------
    model : trained model object
        Loaded in the DAG (joblib.load).
    best_params : dict
        Loaded in the DAG (best_params.json).
    Z_transformed : pd.DataFrame
        Transformed features (model input).
    threshold : float
        Classification threshold.
    add_proba : bool
        Whether to add prediction probability column.
    raw_features_df : pd.DataFrame, optional
        If provided, same row count as ``Z_transformed``; predictions are added to
        this frame (readable columns) instead of the preprocessed matrix.

    Returns
    -------
    pd.DataFrame
        Either ``raw_features_df`` or ``Z_transformed``, enriched with predictions.
    """

    logger.info("INFERENCE : Starting inference")

    # ----------------------------
    # Safety checks
    # ----------------------------
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "INFERENCE : Model does not expose predict_proba()"
        )

    if not isinstance(Z_transformed, pd.DataFrame):
        raise TypeError(
            "INFERENCE : Z_transformed must be a pandas DataFrame"
        )

    model_type = best_params.get("type", "unknown")
    logger.info("INFERENCE : Model type = %s", model_type)
    logger.info("INFERENCE : Z_transformed shape = %s", Z_transformed.shape)
    logger.info("INFERENCE : Using threshold = %.3f", threshold)

    # ----------------------------
    # Prediction
    # ----------------------------
    logger.info("INFERENCE  : Computing prediction probabilities")
    proba = model.predict_proba(Z_transformed)[:, 1]

    if raw_features_df is not None:
        if len(raw_features_df) != len(Z_transformed):
            raise ValueError(
                "INFERENCE : raw_features_df and Z_transformed must have the same "
                f"number of rows ({len(raw_features_df)} vs {len(Z_transformed)})"
            )
        out = raw_features_df.copy().reset_index(drop=True)
    else:
        out = Z_transformed

    if add_proba:
        out["prediction_proba"] = proba

    out["prediction"] = (proba >= threshold).astype(int)

    logger.info("INFERENCE : Inference completed")

    return out
