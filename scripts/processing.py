# scripts/processing.py
from airflow.utils.log.logging_mixin import LoggingMixin
import pandas as pd
from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logger = LoggingMixin().log

def define_target(df: pd.DataFrame, target_col: str = "estAcceptee_bin") -> pd.Series:
    """
    Define target variable.
    """
    logger.info("PROCESS DATA : Defining target variable '%s' | input shape=%s", target_col, df.shape)
    return df[target_col]


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return numerical and categorical feature lists.
    """
    features_num = [
        "DiffPrix",
        "MemeMarque",
        "MemeNutriscore",
        "MemeBio",
        "prixOriginal",
        "MemeConditionnement",
        "MemeTypeMarque",
        "estBioOriginal",
        "Month",
    ]

    features_cat = [
        "categorieOriginal",
        "marqueOriginal",
        "typeMarqueOriginal",
        "nutriscoreOriginal",
        "origineOriginal",
        "conditionnementOriginal",
        "categorieSubstitution",
        "typeMarqueSubstitution",
        "origineSubstitution",
        "Day_of_week_name",
    ]
    logger.info("PROCESS DATA : Feature lists created | num=%s | cat=%s",
        len(features_num),
        len(features_cat),
    )
    return features_num, features_cat


def select_features(df: pd.DataFrame, features_num: List[str], features_cat: List[str]) -> pd.DataFrame:
    """
    Select feature columns from dataframe.
    """
    X = df[features_num + features_cat]
    logger.info("PROCESS DATA : Features selected | output shape=%s", X.shape)
    
    return X


def temporal_train_test_split(X: pd.DataFrame,y: pd.Series,train_ratio: float = 0.8)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform temporal split (no shuffling).
    """
    cutoff_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:cutoff_idx]
    X_test = X.iloc[cutoff_idx:]

    y_train = y.iloc[:cutoff_idx]
    y_test = y.iloc[cutoff_idx:]
    
    logger.info(
        "PROCESS DATA : Split done | "
        "X_train=%s | X_test=%s | y_train=%s | y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )
    
    return X_train, X_test, y_train, y_test


def build_preprocessor(features_num: List[str],features_cat: List[str],) -> ColumnTransformer:
    """
    Build preprocessing pipeline.
    """
    logger.info(
        "PROCESS DATA : Start building preprocessing pipeline | "
        "num_features=%s | cat_features=%s",
        len(features_num),
        len(features_cat),
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features_num),
            ("cat", categorical_transformer, features_cat),
        ]
    )

    return preprocessor


def fit_transform_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Fit preprocessing on train set and transform train & test.
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    logger.info(
        "PROCESS DATA : Preprocessing completed | "
        "X_train_transformed=%s | X_test_transformed=%s",
        X_train_transformed.shape,
        X_test_transformed.shape,
    )

    return pd.DataFrame(X_train_transformed), pd.DataFrame(X_test_transformed), preprocessor


def transform_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a fitted preprocessor to new rows (inference only).

    Uses transform() — no refit — so one-hot columns match training.
    """
    X_transformed = preprocessor.transform(X)
    logger.info(
        "PROCESS DATA : Inference transform | input=%s | output=%s",
        X.shape,
        getattr(X_transformed, "shape", None),
    )
    return pd.DataFrame(X_transformed)