"""
LLM Explanation Engine — Layer F
"""
import re
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

_NO_CODE = (
    "\nCRITICAL RULES (MUST follow):\n"
    "1. NEVER output Python, SQL, pandas, or any code.\n"
    "2. NEVER say 'I cannot' or 'you should run'. Give the answer directly from the data.\n"
    "3. State numbers and facts from the data provided — do NOT guess or extrapolate.\n"
    "4. Answer in plain English sentences only. No code fences (```). No markdown lists with code.\n"
    "5. If numbers are provided, quote them exactly.\n"
)


class LLMExplanationEngine:
    BASE_URL   = "https://openrouter.ai/api/v1"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    MAX_TOKENS = 800
    TEMPERATURE = 0.2

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key  = api_key
        self.model    = model
        self.base_url = (base_url or self.BASE_URL).rstrip("/")

    def _call(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"⚠️ LLM error: {exc}"

    # -----------------------------------------------------------------------
    # Data representation — shows COMPLETE results, no truncation of key data
    # -----------------------------------------------------------------------
    def _full_result_text(self, df: pd.DataFrame) -> str:
        """
        Represent the ENTIRE result (already filtered/aggregated).
        Shows all numeric stats + ALL rows (up to 200).
        Never uses .sample() — the user wants facts from the COMPLETE result.
        """
        if df is None or df.empty:
            return "RESULT: 0 rows matched the filter."
        total = len(df)
        lines = [f"=== COMPLETE QUERY RESULT: {total} total row(s) ===", ""]

        # Numeric summary over ALL rows in the result
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            lines.append("Computed statistics (over ALL result rows):")
            for col in num_cols[:10]:
                s = df[col].dropna()
                lines.append(
                    f"  {col}: SUM={s.sum():,.4f}  MEAN={s.mean():,.4f}  "
                    f"MIN={s.min():,.4f}  MAX={s.max():,.4f}  COUNT={len(s)}"
                )

        # Categorical breakdown over ALL rows
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            lines.append("\nCategorical value counts (ALL rows):")
            for col in cat_cols[:5]:
                vc = df[col].value_counts()
                lines.append(f"  {col}: " +
                             ", ".join(f"{v}={c}" for v, c in vc.head(10).items()))

        # Full row data (all rows if ≤200, else first 200 with note)
        if total <= 200:
            lines.append(f"\nAll {total} rows:")
            lines.append(df.to_string(index=False, max_cols=15))
        else:
            lines.append(f"\nFirst 200 of {total} rows (statistics above cover ALL rows):")
            lines.append(df.head(200).to_string(index=False, max_cols=15))

        return "\n".join(lines)

    def _master_stats(self, master_df: Optional[pd.DataFrame]) -> str:
        """Full statistics of the complete master dataset — no truncation."""
        if master_df is None or master_df.empty:
            return "(no dataset loaded)"
        n = len(master_df)
        num_cols = master_df.select_dtypes(include="number").columns.tolist()
        lines = [f"Full dataset: {n} rows × {len(master_df.columns)} columns"]
        if num_cols:
            lines.append("Numeric column statistics (full dataset):")
            lines.append(master_df[num_cols[:10]].describe().round(4).to_string())
        cat_cols = master_df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols[:5]:
            vc = master_df[col].value_counts()
            lines.append(f"{col} unique values ({master_df[col].nunique()}): "
                         + ", ".join(str(v) for v in vc.head(20).index.tolist()))
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Public explanation methods
    # -----------------------------------------------------------------------

    def explain_table(self, result_df: pd.DataFrame, query: str,
                      plan: str = "", master_df: Optional[pd.DataFrame] = None) -> str:
        system = (
            "You are a senior data analyst. The user asked a question and you have been given "
            "the COMPLETE pre-computed query result below — it is already filtered, sorted, "
            "and aggregated exactly as requested. Your job is to:\n"
            "1. Answer the user's question directly using the numbers in the result.\n"
            "2. Be specific — quote actual values, sums, counts from the result.\n"
            "3. Write 3–5 plain-English sentences.\n"
            + _NO_CODE
        )
        user = (
            f"User's question: {query}\n"
            f"Execution trace: {plan}\n\n"
            f"{self._full_result_text(result_df)}"
        )
        return self._call(system, user)

    def explain_forecast(self, forecast_df, history_df, metrics, query, value_col) -> str:
        system = (
            "You are a forecasting analyst. Explain the forecast in 4–6 plain-English sentences.\n"
            "State the predicted direction, specific forecasted numbers, and model accuracy.\n"
            + _NO_CODE
        )
        metric_text = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
        user = (
            f"User question: {query}\n"
            f"Target metric: {value_col}\n\n"
            f"{self._full_result_text(forecast_df)}\n\n"
            f"Model accuracy:\n{metric_text}"
        )
        return self._call(system, user)

    def explain_anomaly(self, anomaly_df, summary, query) -> str:
        system = (
            "You are a data quality analyst. Explain the anomaly detection results.\n"
            "State exactly how many anomalies, in which columns, what the values are.\n"
            + _NO_CODE
        )
        s = summary
        summary_line = (
            f"{s.get('anomaly_rows',0)} anomalies found out of {s.get('total_rows',0)} rows "
            f"({s.get('anomaly_pct',0):.1f}%), method: {s.get('method','zscore')}, "
            f"columns: {s.get('columns_checked',[])}."
        )
        user = f"Question: {query}\nSummary: {summary_line}\n\n{self._full_result_text(anomaly_df)}"
        return self._call(system, user)

    def explain_scenario(self, scenario_df, assumptions, description, query, value_col) -> str:
        system = (
            "You are a business analyst. Explain the what-if scenario results in 4–6 sentences.\n"
            "State the assumption applied, projected impact in numbers, and direction.\n"
            + _NO_CODE
        )
        user = (
            f"Question: {query}\nScenario:\n{description}\n\n"
            f"Projected outcome for {value_col}:\n{self._full_result_text(scenario_df)}"
        )
        return self._call(system, user)

    def answer_schema_query(self, query: str, schema_text: str, graph_context: Dict) -> str:
        system = (
            "You are a data catalog assistant. Answer using the provided schema. "
            "Name specific columns, types, and datasets in 3–5 sentences.\n"
            + _NO_CODE
        )
        ctx_cols = [c.get("name") for c in graph_context.get("relevant_columns", [])]
        user = (
            f"Question: {query}\n\nDataset schema:\n{schema_text}"
            + (f"\n\nKnowledge graph identified relevant columns: {ctx_cols}" if ctx_cols else "")
        )
        return self._call(system, user)

    def general_answer(self, query: str, master_df: Optional[pd.DataFrame],
                       schema_text: str) -> str:
        system = (
            "You are a helpful data analyst. Answer using the dataset statistics given. "
            "Be specific and quote numbers where relevant. 4–6 sentences.\n"
            + _NO_CODE
        )
        user = (
            f"Question: {query}\n\n"
            f"Schema:\n{schema_text}\n\n"
            f"Full dataset statistics:\n{self._master_stats(master_df)}"
        )
        return self._call(system, user)

    def generate_pandas_code(self, query: str, schema_text: str) -> str:
        system = (
            "You are a Python data analyst. Output EXACTLY ONE line of valid pandas code to answer the user's query.\n"
            "Assume the dataset is loaded as a DataFrame named `df`.\n"
            "Rules:\n"
            "1. Output ONLY the code. No markdown, no code blocks (```python), no intro.\n"
            "2. Do NOT assign variables (e.g., do NOT start with 'result = '). Just write the expression to evaluate.\n"
            "3. Use .reset_index() if grouping so the output is a clean DataFrame where possible.\n"
            "4. Only use standard pandas methods (groupby, sum, mean, nlargest, head, etc.).\n"
            "5. If returning top N, use .nlargest(N, 'col') or sort_values().head().\n"
        )
        user = f"Schema:\n{schema_text}\n\nUser Query: {query}"
        resp = self._call(system, user).strip()
        resp = resp.replace("```python", "").replace("```", "").strip()
        return resp

    # -----------------------------------------------------------------------
    # Model listing
    # -----------------------------------------------------------------------
    @staticmethod
    def fetch_models(api_key: str) -> List[Dict]:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(LLMExplanationEngine.MODELS_URL,
                                headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            models = [
                {"id": m.get("id",""),
                 "name": m.get("name", m.get("id","")),
                 "is_free": ":free" in m.get("id","") or
                             m.get("pricing",{}).get("prompt","1") in ("0","0.0")}
                for m in data if m.get("id")
            ]
            models.sort(key=lambda m: (0 if m["is_free"] else 1, m["name"]))
            return models
        except Exception:
            return [{"id":"google/gemini-flash-1.5",
                     "name":"Gemini Flash 1.5 (default)","is_free":True}]

    @staticmethod
    def format_model_options(models: List[Dict]) -> List[str]:
        return [f"{'🆓 ' if m['is_free'] else '💰 '}{m['name']} ({m['id']})"
                for m in models]

    @staticmethod
    def extract_model_id(label: str) -> str:
        m = re.search(r"\(([^)]+)\)\s*$", label)
        return m.group(1) if m else label
