"""Scenario Analysis Engine — Layer E3"""
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from engines.forecasting import ForecastingEngine


class ScenarioEngine:
    PATTERNS = [
        r"(increase|grow|raise|boost)\s+(?:\w+\s+)?by\s+(\d+(?:\.\d+)?)\s*(%|percent)",
        r"(decrease|reduce|cut|drop|lower)\s+(?:\w+\s+)?by\s+(\d+(?:\.\d+)?)\s*(%|percent)",
    ]

    def analyze(self, query, master_df, fe, date_col=None, value_col=None, periods=4):
        if master_df is None or master_df.empty: raise ValueError("No data.")
        if date_col is None or value_col is None:
            dc,vc = fe.auto_detect_columns(master_df,{})
            date_col=date_col or dc; value_col=value_col or vc
        if not date_col or not value_col: raise ValueError("Cannot detect date/value columns.")
        assumptions = self._parse(query, value_col)
        description = self._desc(assumptions, value_col)
        base_fc,_,_ = fe.forecast(master_df, date_col, value_col, periods)
        scen_df = self._apply(master_df.copy(), assumptions, value_col)
        scen_fc,_,_ = fe.forecast(scen_df, date_col, value_col, periods)
        base_fc = base_fc.rename(columns={"yhat":"baseline","yhat_lower":"baseline_lo","yhat_upper":"baseline_hi"})
        scen_fc = scen_fc.rename(columns={"yhat":"scenario","yhat_lower":"scenario_lo","yhat_upper":"scenario_hi"})
        merged = pd.merge(base_fc, scen_fc, on=date_col, how="outer")
        merged["delta"] = merged["scenario"]-merged["baseline"]
        merged["delta_pct"] = (merged["delta"]/(merged["baseline"].abs()+1e-9))*100
        return merged, scen_fc, assumptions, description

    def _parse(self, query, value_col):
        q = query.lower()
        assumptions = {"target_column": value_col}
        for p in self.PATTERNS:
            m = re.search(p, q)
            if m:
                neg = m.group(1) in ("decrease","reduce","cut","drop","lower")
                amt = float(m.group(2))
                assumptions["growth_rate"] = (-amt if neg else amt)/100
                assumptions["direction"] = m.group(1)
                assumptions["amount_pct"] = amt; break
        if re.search(r"add\s+([\d,]+)", q):
            assumptions["flat_add"]=float(re.search(r"add\s+([\d,]+)",q).group(1).replace(",",""))
        if "remove outlier" in q or "without outlier" in q:
            assumptions["remove_outliers"] = True
        return assumptions

    def _apply(self, df, assumptions, value_col):
        if value_col not in df.columns: return df
        if "growth_rate" in assumptions:
            df[value_col] = df[value_col]*(1+assumptions["growth_rate"])
        if "flat_add" in assumptions:
            df[value_col] = df[value_col]+assumptions["flat_add"]
        if assumptions.get("remove_outliers"):
            Q1,Q3 = df[value_col].quantile(0.25),df[value_col].quantile(0.75)
            IQR = Q3-Q1
            df = df[(df[value_col]>=Q1-1.5*IQR)&(df[value_col]<=Q3+1.5*IQR)]
        return df

    def _desc(self, assumptions, value_col):
        parts = [f"Scenario for '{value_col}':"]
        if "growth_rate" in assumptions:
            parts.append(f"  • {assumptions.get('direction','change').capitalize()} by {assumptions['amount_pct']:.1f}%")
        if "flat_add" in assumptions:
            parts.append(f"  • Add flat {assumptions['flat_add']:,.0f}")
        if assumptions.get("remove_outliers"):
            parts.append("  • Outliers removed (IQR)")
        if len(parts)==1: parts.append("  • No transformation detected — showing baseline")
        return "\n".join(parts)
