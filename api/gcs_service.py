from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("URI must start with gs://")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def is_allowed(gs_uri: str, allowed_prefixes: list[str]) -> bool:
    u = gs_uri.rstrip("/")
    for p in allowed_prefixes:
        base = p.rstrip("/")
        if u == base or u.startswith(base + "/"):
            return True
    return False


def list_data_uris(gs_prefix: str, suffixes: tuple[str, ...] = (".csv", ".json")) -> list[str]:
    bucket_name, prefix = parse_gs_uri(gs_prefix)
    client = storage.Client()
    out: list[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        name = blob.name
        if blob.name.endswith("/"):
            continue
        if any(name.lower().endswith(s) for s in suffixes):
            out.append(f"gs://{bucket_name}/{name}")
    return sorted(out)


def read_tabular_json(gs_uri: str, limit: int | None = 2000) -> list[dict]:
    bucket_name, blob_path = parse_gs_uri(gs_uri)
    if not blob_path:
        raise ValueError("URI must point to an object, not only a bucket")
    if blob_path.endswith("/"):
        raise ValueError(
            "URI must be a file (.csv or .json), not a folder (remove trailing / and pick one object)"
        )
    blob = storage.Client().bucket(bucket_name).blob(blob_path)
    try:
        raw = blob.download_as_bytes()
    except NotFound as e:
        raise FileNotFoundError(str(e)) from e
    if gs_uri.lower().endswith(".json"):
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON must be a list or object")
    else:
        df = pd.read_csv(BytesIO(raw), nrows=limit)
        rows = df.to_dict(orient="records")
    if limit is not None and len(rows) > limit:
        return rows[:limit]
    return rows
