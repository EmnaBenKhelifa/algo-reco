# dags/processing_dag.py
"""
ML data processing DAG.

Handles TRAIN and INFERENCE processing:
- TRAIN: train/test split + preprocessing
- INFERENCE: preprocessing only (per inference date)
"""

import logging
import pandas as pd
from airflow.sdk import dag, task
from pendulum import datetime

from scripts.ingestion import Ingestion
from scripts.utils import (
    load_data_bq,
    dump_data_gcs,
    load_data_gcs
)
from scripts.processing import (
    get_feature_lists,
    define_target,
    select_features,
    temporal_train_test_split,
    build_preprocessor,
    fit_transform_preprocessor,
    transform_preprocessor,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------
# DAG DEFINITION
# ------------------------------------------------
@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "processing"],
)
def processing_dag():

    ingestion = Ingestion()

    # ------------------------------------------------
    # 1. Read DAG configuration
    # ------------------------------------------------
    @task
    def read_conf(**context):
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run else {}

        # Same keys as dag_run.conf — downstream tasks use INGESTION_MODE / INGESTION_DATES
        return {
            "INGESTION_MODE": conf.get("INGESTION_MODE", "inference"),
            "INGESTION_DATES": conf.get("INGESTION_DATES", []),
        }

    # ------------------------------------------------
    # 2. TRAIN MODE — train/test split
    # ------------------------------------------------
    @task
    def train_split(conf: dict):
        if conf.get("INGESTION_MODE", "inference") != "train":
            logger.info("[PROCESSING] Skip train_split (not in train mode)")
            return "skipped"

        features_table_name = "features_train"
        project_id = ingestion.project_id
        dataset_id = ingestion.dataset_id

        features_num, features_cat = get_feature_lists()
        features_dict = {
            "numerical": features_num,
            "categorical": features_cat,
        }

        # Save feature schema
        dump_data_gcs(features_dict, "gs://algo_reco/features/train", "features")

        # Load features
        features_dataset = load_data_bq(
            project_id, dataset_id, features_table_name
        )

        target = define_target(features_dataset)
        features = select_features(
            features_dataset, features_num, features_cat
        )

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            features, target
        )

        dump_data_gcs(X_train, "gs://algo_reco/features/train", "x_train")
        dump_data_gcs(X_test, "gs://algo_reco/features/train", "x_test")
        dump_data_gcs(pd.DataFrame(y_train), "gs://algo_reco/features/train", "y_train")
        dump_data_gcs(pd.DataFrame(y_test), "gs://algo_reco/features/train", "y_test")

        return "TRAIN SPLIT DONE"

    # ------------------------------------------------
    # 3. TRAIN MODE — preprocessing
    # ------------------------------------------------
    @task
    def train_preprocessing(conf: dict):
        if conf.get("INGESTION_MODE", "inference") != "train":
            logger.info("[PROCESSING] Skip train_preprocessing (not in train mode)")
            return "skipped"

        features_dict = load_data_gcs(
            "gs://algo_reco/features/train/features.json"
        )

        X_train = load_data_gcs("gs://algo_reco/features/train/x_train.csv")
        X_test = load_data_gcs("gs://algo_reco/features/train/x_test.csv")

        preprocessor = build_preprocessor(
            features_dict["numerical"],
            features_dict["categorical"],
        )

        X_train_t, X_test_t, preprocessor = fit_transform_preprocessor(
            preprocessor, X_train, X_test
        )

        dump_data_gcs(
            X_train_t, "gs://algo_reco/features/train", "x_train_preprocessed"
        )
        dump_data_gcs(
            X_test_t, "gs://algo_reco/features/train", "x_test_preprocessed"
        )
        dump_data_gcs(
            preprocessor, "gs://algo_reco/features/train", "preprocessor"
        )

        return "TRAIN PREPROCESSING DONE"

    # ------------------------------------------------
    # 4. INFERENCE MODE — preprocessing per date
    # ------------------------------------------------
    @task
    def inference_preprocessing(conf: dict):
        if conf.get("INGESTION_MODE", "inference") == "train":
            logger.info("[PROCESSING] Skip inference_preprocessing (train mode)")
            return "skipped"

        dates = conf.get("INGESTION_DATES", [])
        if not dates:
            raise ValueError("INFERENCE mode requires INGESTION_DATES")

        project_id = ingestion.project_id
        dataset_id = ingestion.dataset_id

        preprocessor = load_data_gcs(
            "gs://algo_reco/features/train/preprocessor.joblib"
        )

        features_num, features_cat = get_feature_lists()

        for ds in dates:
            ds_table = ds.replace("-", "_")
            logger.info(f"[PROCESSING] Inference for {ds}")

            features_full = load_data_bq(
                project_id,
                dataset_id,
                f"features_inference_{ds_table}",
            )
            X = select_features(features_full, features_num, features_cat)

            X_t = transform_preprocessor(preprocessor, X)

            dump_data_gcs(
                X_t,
                "gs://algo_reco/features/inference",
                f"x_inference_preprocessed_{ds}",
            )

        return "INFERENCE PREPROCESSING DONE"

    # ------------------------------------------------
    # 5. Routing logic
    # ------------------------------------------------
    conf = read_conf()

    train_split_task = train_split(conf)
    train_preprocess_task = train_preprocessing(conf)
    inference_task = inference_preprocessing(conf)

    train_split_task >> train_preprocess_task
    conf >> train_split_task
    conf >> inference_task


# Instantiate DAG
processing_dag()

# -----------------------------------------------------------------------------------------------------------
# # dags/processing_dag.py
# """
# ML data processing data DAG.

# This DAG handles the data processing steps including train/test split and preprocessing.
# It loads the features dataset from BigQuery, performs temporal train/test split,
# applies preprocessing, and stores the processed datasets and preprocessor back to GCS.
# """
# import logging
# from airflow.sdk import dag, task
# from pendulum import datetime
# from scripts.ingestion import Ingestion
# from scripts.utils import load_data_bq, dump_table_into_bq, dump_data_gcs,load_data_gcs
# from scripts.features import *
# from scripts.processing import *

# logger = logging.getLogger(__name__)

# @dag(
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     default_args={"owner": "ML", "retries": 3},
#     catchup=False,
#     tags=["ml", "processing"]
# )

# def processing_dag():
#     ingestion = Ingestion()
#     @task
#     def train_test_split(features_table_name : str, project_id : str, dataset_id : str) :

#         features_num, features_cat = get_feature_lists()
#         features_dict = {"numerical" : features_num, "categorical" : features_cat}
#         dump_data_gcs(features_dict,"gs://algo_reco/features/train","features")
    
#         features_dataset = load_data_bq(project_id, dataset_id, features_table_name)
#         target = define_target(features_dataset)
#         features = select_features(features_dataset, features_num, features_cat)
#         X_train, X_test, y_train, y_test = temporal_train_test_split(features, target)
#         dump_data_gcs(X_train,"gs://algo_reco/features/train","x_train")
#         dump_data_gcs(X_test, "gs://algo_reco/features/train","x_test")
#         dump_data_gcs(pd.DataFrame(y_train), "gs://algo_reco/features/train","y_train")
#         dump_data_gcs(pd.DataFrame(y_test), "gs://algo_reco/features/train","y_test")
#         return f'PROCESSING : Train/Test split dumped into GCS'
    
#     @task
#     def preprocessing():
#         features_dict = load_data_gcs("gs://algo_reco/features/train/features.json")
#         logger.info(f'PROCESSING : features_dict = {features_dict}')
#         X_train = load_data_gcs("gs://algo_reco/features/train/x_train.csv")
#         X_test = load_data_gcs("gs://algo_reco/features/train/x_test.csv")
#         preprocessor = build_preprocessor(features_dict["numerical"],features_dict["categorical"])
#         X_train_transformed, X_test_transformed, preprocessor = fit_transform_preprocessor(preprocessor,X_train,X_test)
#         dump_data_gcs(X_train_transformed,"gs://algo_reco/features/train","x_train_preprocessed")
#         dump_data_gcs(X_test_transformed,"gs://algo_reco/features/train","x_test_preprocessed")
#         dump_data_gcs(preprocessor,"gs://algo_reco/features/train","preprocessor")       
#         return f'PROCESSING : Preprocessed data (X_train_transformed, X_test_transformed, preprocessor) dumped into GCS'
    
#     # BigQuery features table to be loaded
#     project_id = ingestion.project_id
#     dataset_id = ingestion.dataset_id
#     features_table_name = "features_dataset"
#     # Task instantiation
#     split_task = train_test_split(features_table_name, project_id, dataset_id)
#     preprocess_task = preprocessing()

#     # Dependency
#     split_task >> preprocess_task
    
# # Instantiate the DAG
# processing_dag()