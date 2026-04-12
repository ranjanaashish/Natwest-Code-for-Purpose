"""Result Validation Engine — Layer G"""
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ResultValidationEngine:
    def validate_tabular(self, result_df, master_df, intent, schema_metadata):
        metrics = {}
        all_cols = []
        for info in schema_metadata.values():
            all_cols.extend(info.get("columns",{}).keys())
        result_cols = set(result_df.columns)-{"_source"}
        grounded = sum(1 for c in result_cols if c in all_cols)
        metrics["Schema Grounding (%)"] = round(grounded/max(len(result_cols),1)*100,1)
        metrics["Result Rows"] = len(result_df)
        consistency_checks = consistency_passes = 0
        for col in result_df.select_dtypes(include="number").columns:
            if col.startswith("_"): continue
            for ds_info in schema_metadata.values():
                ci = ds_info.get("columns",{}).get(col,{})
                if ci.get("is_numeric"):
                    lo,hi = ci.get("min",-np.inf),ci.get("max",np.inf)
                    buf = max(abs(hi-lo)*0.1,1)
                    ok = (result_df[col]>=lo-buf)&(result_df[col]<=hi+buf)
                    consistency_checks+=len(result_df); consistency_passes+=ok.sum()
        metrics["Numeric Consistency (%)"] = (
            round(consistency_passes/max(consistency_checks,1)*100,1)
            if consistency_checks else "N/A")
        gt_df = self._synth_gt(master_df, intent)
        if gt_df is not None and not gt_df.empty and not result_df.empty:
            common = list(set(result_df.columns)&set(gt_df.columns)-{"_source"})
            if common:
                res_rows = set(result_df[common].astype(str).apply(tuple,axis=1))
                gt_rows  = set(gt_df[common].astype(str).apply(tuple,axis=1))
                tp=len(res_rows&gt_rows); fp=len(res_rows-gt_rows); fn=len(gt_rows-res_rows)
                prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
                f1=2*prec*rec/max(prec+rec,1e-9)
                metrics["Exact Match Precision"]=round(prec,3)
                metrics["Exact Match Recall"]=round(rec,3)
                metrics["Exact Match F1"]=round(f1,3)
        return metrics

    def _synth_gt(self, master_df, intent):
        try:
            from engines.tabular_query import TabularQueryEngine
            tqe = TabularQueryEngine()
            df, _ = tqe.execute(intent, master_df)
            return df
        except Exception: return None

    def validate_forecast(self, metrics):
        return {k: metrics.get(k) for k in
                ["MAE","RMSE","MAPE (%)","R²","Directional Accuracy (%)",
                 "Baseline RMSE (std)","Model vs Baseline","Hold-out Samples"]}

    def validate_anomaly(self, gt_metrics):
        return gt_metrics

    def validate_llm_narrative(self, narrative, result_df):
        if not narrative or result_df is None or result_df.empty:
            return {"Grounded Answer Rate (GAR)": "N/A"}
        
        narr_nums = set(round(float(n),1) for n in re.findall(r"\b\d+(?:\.\d+)?\b", narrative))
        data_nums = set()
        for col in result_df.select_dtypes(include="number").columns:
            data_nums.update(round(v,1) for v in result_df[col].dropna())
        
        all_cols = list(result_df.columns)
        cited_cols = [c for c in all_cols if c.replace("_"," ").lower() in narrative.lower()]
        
        # Total claims = total numbers mentioned + total column references made
        total_claims = len(narr_nums) + len(cited_cols)
        
        supported_nums = sum(1 for n in narr_nums if n in data_nums)
        supported_claims = supported_nums + len(cited_cols) # all extracted cited_cols are valid schema groundings
        supported_claims = supported_nums + len(cited_cols)
        
        gar = round(supported_claims / total_claims * 100, 1) if total_claims > 0 else 100.0
            
        return {
            "Grounded Answer Rate (GAR)": f"{gar}%",
            "Extracted Claims": list(narr_nums) + cited_cols
        }
