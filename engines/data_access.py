"""
Universal Data Access Layer — Layer A

Responsible for ingesting structured and semi-structured data from any source,
including local files, databases, and cloud links, before passing it to the Data Integration layer.
"""
import io
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from datetime import datetime

class DataAccessEngine:
    def __init__(self):
        self.source_metadata: Dict[str, Any] = {}

    def fetch_source_metadata(self, source_id: str, source_type: str, row_count: int, col_count: int) -> Dict[str, Any]:
        """Track provenance and dataset stats"""
        meta = {
            "source_id": source_id,
            "source_type": source_type,
            "fetched_at": datetime.now().isoformat(),
            "row_count": row_count,
            "col_count": col_count
        }
        self.source_metadata[source_id] = meta
        return meta

    def upload_files(self, file_objs: list) -> Dict[str, pd.DataFrame]:
        """Fetch pandas dataframes from uploaded local CSV/JSON file objects."""
        datasets = {}
        for file_obj in file_objs:
            try:
                # streamlit uploaded file
                name = getattr(file_obj, "name", "uploaded_file").lower().replace(" ", "_")
                if name.endswith(".csv"):
                    df = pd.read_csv(file_obj)
                elif name.endswith(".json"):
                    content = file_obj.read()
                    if isinstance(content, str):
                        content = content.encode("utf-8")
                    df = pd.read_json(io.BytesIO(content))
                else:
                    raise ValueError(f"Unsupported local format: {name}")

                dataset_name = name.rsplit(".", 1)[0]
                self.fetch_source_metadata(dataset_name, "local_upload", len(df), len(df.columns))
                datasets[dataset_name] = df
            except Exception as e:
                raise ValueError(f"Failed to load file object: {e}") from e
        return datasets

    def connect_database(self, uri: str, query_or_table: str) -> pd.DataFrame:
        """Connect to SQL DB via SQLAlchemy/pandas."""
        try:
            # Requires relevant drivers to be installed externally (e.g. psycopg2, pyodbc, sqlite3)
            df = pd.read_sql(query_or_table, uri)
            self.fetch_source_metadata(uri, "database", len(df), len(df.columns))
            return df
        except Exception as e:
            raise ValueError(f"Failed to connect or query database '{uri}': {e}") from e

    def connect_cloud_source(self, url: str) -> pd.DataFrame:
        """Connect to purely URL-driven sources (S3, GCS, Blob, HTTP APIs)."""
        try:
            if url.endswith(".csv") or "csv" in url:
                df = pd.read_csv(url)
            elif url.endswith(".json") or "json" in url:
                df = pd.read_json(url)
            elif url.endswith(".parquet") or "parquet" in url:
                df = pd.read_parquet(url)
            else:
                # Default assume CSV for standard API feeds or web tables if no extension
                df = pd.read_csv(url)

            name = hashlib.md5(url.encode()).hexdigest()[:8]
            dataset_name = f"cloud_source_{name}"
            self.fetch_source_metadata(dataset_name, "cloud_link", len(df), len(df.columns))
            return df
        except Exception as e:
            raise ValueError(f"Failed to load cloud source '{url}': {e}") from e
