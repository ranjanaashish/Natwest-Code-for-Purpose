"""Data Integration Engine — Layer A"""
import io, hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class DataIntegrationEngine:
    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.master_df: Optional[pd.DataFrame] = None
        self.schema_metadata: Dict[str, Any] = {}
        self.join_metadata: Dict[str, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum()/max(len(df),1) > 0.8:
                    df[col] = parsed
            except Exception: pass
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum():
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object","category"]).columns:
            if df[col].isnull().sum():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        return df

    def generate_schema_metadata(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        schema: Dict[str, Any] = {}
        for name, df in datasets.items():
            columns: Dict[str, Any] = {}
            for col in df.columns:
                is_num = pd.api.types.is_numeric_dtype(df[col])
                is_dt  = pd.api.types.is_datetime64_any_dtype(df[col])
                null_pct = round(df[col].isnull().sum()/max(len(df),1)*100, 2)
                entry: Dict[str, Any] = {
                    "dtype": str(df[col].dtype), "is_numeric": is_num,
                    "is_datetime": is_dt, "is_categorical": not is_num and not is_dt,
                    "null_pct": null_pct, "unique_count": int(df[col].nunique()),
                }
                if is_num:
                    entry.update({"min": float(df[col].min()), "max": float(df[col].max()),
                                  "mean": float(df[col].mean()), "std": float(df[col].std())})
                if not is_num and not is_dt:
                    # Store top unique values for filter suggestion
                    entry["top_values"] = df[col].value_counts().head(20).index.tolist()
                columns[col] = entry
            schema[name] = {
                "row_count": len(df), "col_count": len(df.columns),
                "columns": columns,
                "numeric_columns":    [c for c,v in columns.items() if v["is_numeric"]],
                "datetime_columns":   [c for c,v in columns.items() if v["is_datetime"]],
                "categorical_columns":[c for c,v in columns.items() if v["is_categorical"]],
            }
        self.schema_metadata = schema
        return schema

    def _find_join_keys(self, df1, df2):
        common = set(df1.columns) & set(df2.columns)
        keys = []
        for col in common:
            if df1[col].dtype != df2[col].dtype: continue
            v1 = set(df1[col].dropna().unique()[:200])
            v2 = set(df2[col].dropna().unique()[:200])
            if v1 & v2: keys.append(col)
        return keys

    def merge_datasets(self, datasets):
        if not datasets: return pd.DataFrame(), {"joins":[],"total_datasets":0}
        items = list(datasets.items())
        if len(items) == 1:
            nm, df = items[0]; master = df.copy(); master["_source"] = nm
            self.master_df = master
            return master, {"joins":[],"total_datasets":1}
        master, master_name = items[0][1].copy(), items[0][0]
        master["_source"] = master_name
        join_records = []
        for name, df in items[1:]:
            candidate = df.copy(); candidate["_source"] = name
            keys = self._find_join_keys(master, candidate)
            try:
                if keys:
                    master = pd.merge(master, candidate, on=keys, how="outer",
                                      suffixes=("", f"_{name}"))
                    join_records.append({"left":master_name,"right":name,"keys":keys,"type":"merge"})
                else:
                    master = pd.concat([master, candidate], ignore_index=True, sort=False)
                    join_records.append({"left":master_name,"right":name,"keys":[],"type":"concat"})
            except Exception:
                master = pd.concat([master, candidate], ignore_index=True, sort=False)
                join_records.append({"left":master_name,"right":name,"keys":[],"type":"concat"})
        self.master_df = master
        return master, {"joins":join_records,"total_datasets":len(datasets),"master_shape":master.shape}

    def process_all(self, imported_datasets: Dict[str, pd.DataFrame]):
        datasets = {}
        for name, df in imported_datasets.items():
            # Apply naming normalization as done previously on ingest
            df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
            df = self.preprocess(df)
            datasets[name] = df
        schema_metadata = self.generate_schema_metadata(datasets)
        master_df, join_metadata = self.merge_datasets(datasets)
        self.datasets = datasets
        return datasets, master_df, schema_metadata, join_metadata

    def compute_hash(self, datasets):
        h = hashlib.md5()
        for name in sorted(datasets):
            df = datasets[name]
            h.update(f"{name}:{df.shape}:{df.columns.tolist()}".encode())
            h.update(df.head(5).to_json().encode())
        return h.hexdigest()
