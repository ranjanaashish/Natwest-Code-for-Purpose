"""Response Planner — Layer H"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd
from engines.intent_router import IntentResult


@dataclass
class ResponsePlan:
    intent: str
    show_text: bool = True
    show_table: bool = False
    show_forecast_chart: bool = False
    show_anomaly_chart: bool = False
    show_scenario_chart: bool = False
    show_metrics_panel: bool = False
    text: str = ""
    result_df: Optional[pd.DataFrame] = None
    forecast_df: Optional[pd.DataFrame] = None
    history_df: Optional[pd.DataFrame] = None
    anomaly_df: Optional[pd.DataFrame] = None
    scored_df: Optional[pd.DataFrame] = None
    scenario_df: Optional[pd.DataFrame] = None
    metrics: Dict[str,Any] = field(default_factory=dict)
    validation_metrics: Dict[str,Any] = field(default_factory=dict)
    llm_validation: Dict[str,Any] = field(default_factory=dict)
    query_plan: str = ""
    scenario_description: str = ""
    value_col: str = ""
    date_col: str = ""
    error: Optional[str] = None


class ResponsePlanner:
    def plan_tabular(self, intent, result_df, text, query_plan, val, lv):
        return ResponsePlan(intent=intent.intent, show_text=True, show_table=True,
                            show_metrics_panel=True, text=text, result_df=result_df,
                            validation_metrics=val, llm_validation=lv, query_plan=query_plan)

    def plan_forecast(self, intent, fdf, hdf, text, mets, val, lv, date_col, value_col):
        return ResponsePlan(intent=intent.intent, show_text=True, show_forecast_chart=True,
                            show_table=True, show_metrics_panel=True, text=text,
                            forecast_df=fdf, history_df=hdf, metrics=mets,
                            validation_metrics=val, llm_validation=lv,
                            date_col=date_col, value_col=value_col)

    def plan_anomaly(self, intent, adf, sdf, text, summary, val, lv, value_col):
        return ResponsePlan(intent=intent.intent, show_text=True, show_anomaly_chart=True,
                            show_table=True, show_metrics_panel=True, text=text,
                            anomaly_df=adf, scored_df=sdf, metrics=summary,
                            validation_metrics=val, llm_validation=lv, value_col=value_col)

    def plan_scenario(self, intent, merged, text, desc, date_col, value_col):
        return ResponsePlan(intent=intent.intent, show_text=True, show_scenario_chart=True,
                            show_table=True, text=text, scenario_df=merged,
                            scenario_description=desc, date_col=date_col, value_col=value_col)

    def plan_text(self, intent, text):
        return ResponsePlan(intent=intent.intent, show_text=True, text=text)

    def plan_error(self, intent_str, error_msg):
        return ResponsePlan(intent=intent_str, show_text=True,
                            text=f"⚠️ {error_msg}", error=error_msg)
