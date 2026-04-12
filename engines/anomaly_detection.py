"""Anomaly Detection Engine — Layer E2"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class AnomalyEngine:
    Z_THRESHOLD = 2.5
    IQR_FACTOR  = 1.5

    def detect(self, df, columns=None, method="zscore"):
        if df is None or df.empty:
            return pd.DataFrame(), df, {"error": "Empty dataframe"}
        num_cols = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
        target = [c for c in (columns or num_cols) if c in num_cols and c in df.columns]
        if not target:
            return pd.DataFrame(), df, {"error": "No numeric columns"}
        scored = df.copy(); anomaly_mask = pd.Series(False, index=df.index)
        col_stats = {}
        for col in target:
            series = df[col].dropna()
            if method == "zscore":
                z = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                col_anom = pd.Series(z > self.Z_THRESHOLD, index=df.index)
                score_col = z
            else:
                Q1,Q3 = df[col].quantile(0.25),df[col].quantile(0.75)
                IQR = Q3-Q1
                lo,hi = Q1-self.IQR_FACTOR*IQR, Q3+self.IQR_FACTOR*IQR
                col_anom = (df[col]<lo)|(df[col]>hi)
                score_col = np.abs(df[col]-df[col].median())/(IQR+1e-9)
            scored[f"{col}_score"] = score_col
            scored[f"{col}_is_anomaly"] = col_anom
            anomaly_mask |= col_anom
            col_stats[col] = {"anomaly_count":int(col_anom.sum()),"mean":float(series.mean()),
                              "std":float(series.std()),"min":float(series.min()),"max":float(series.max())}
        anomaly_df = df[anomaly_mask].copy()
        if not anomaly_df.empty:
            anomaly_df["_anomaly_cols"] = anomaly_df.apply(
                lambda row: ", ".join(col for col in target if scored.loc[row.name,f"{col}_is_anomaly"]),axis=1)
        summary = {"method":method,"total_rows":len(df),"anomaly_rows":int(anomaly_mask.sum()),
                   "anomaly_pct":round(anomaly_mask.sum()/max(len(df),1)*100,2),
                   "columns_checked":target,"per_column":col_stats}
        return anomaly_df.reset_index(drop=True), scored, summary

    def validate_with_synthetic_gt(self, df, col, n_inject=5):
        if col not in df.columns: return {"error":f"Column '{col}' not found"}
        rng = np.random.default_rng(42)
        df_test = df.copy(); n = len(df_test)
        inject_idx = rng.choice(n, size=min(n_inject,max(1,n//5)), replace=False)
        mean_v = df_test[col].mean(); std_v = df_test[col].std()
        df_test.loc[inject_idx, col] = mean_v+5*std_v
        gt_mask = pd.Series(False,index=df_test.index); gt_mask.iloc[inject_idx]=True
        _, scored, _ = self.detect(df_test, columns=[col])
        if scored.empty or f"{col}_is_anomaly" not in scored.columns:
            return {"error":"Detection failed"}
        pred = scored[f"{col}_is_anomaly"]
        tp=int((gt_mask&pred).sum()); fp=int((~gt_mask&pred).sum())
        fn=int((gt_mask&~pred).sum()); tn=int((~gt_mask&~pred).sum())
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        f1=2*prec*rec/max(prec+rec,1e-9); fpr=fp/max(fp+tn,1)
        return {"Precision":round(prec,4),"Recall":round(rec,4),"F1-Score":round(f1,4),
                "False Positive Rate":round(fpr,4),"True Positives":tp,
                "False Positives":fp,"False Negatives":fn,"Injected Anomalies":len(inject_idx)}

    def auto_detect_column(self, df) -> Optional[str]:
        num_cols = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
        if not num_cols: return None
        covs = {c: df[c].std()/(df[c].mean()+1e-9) for c in num_cols}
        return max(covs, key=covs.get)
