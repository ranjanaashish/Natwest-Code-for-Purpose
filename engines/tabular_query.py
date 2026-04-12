"""
Tabular Query Engine — Layer D
"""
from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
import numpy as np
from engines.intent_router import IntentResult, IntentRouter


class TabularQueryEngine:

    # -----------------------------------------------------------------------
    # Column resolution — always use fuzzy matching
    # -----------------------------------------------------------------------
    @staticmethod
    def _resolve_col(name: str, df: pd.DataFrame,
                     prefer_numeric: bool = False, kg_context: Dict = None) -> Optional[str]:
        """Return the best matching column from df for a user-supplied name."""
        if not name: return None
        cols = df.select_dtypes(include="number").columns.tolist() if prefer_numeric else df.columns.tolist()
        name_l = name.lower().replace(" ", "_").replace("-", "_")

        # 0. Fast-track KG context
        if kg_context and "columns" in kg_context:
            for kg_c in kg_context["columns"]:
                if name_l in kg_c.lower() and kg_c in cols:
                    return kg_c

        # 1. Exact
        for c in cols:
            if c.lower() == name_l: return c
        # 2. Contains
        for c in cols:
            if name_l in c.lower(): return c
        # 3. Contained-by
        for c in cols:
            if c.lower() in name_l: return c
        # 4. Token overlap
        n_tokens = set(re.split(r"[_\s\-]+", name_l))
        for c in cols:
            c_tokens = set(re.split(r"[_\s\-]+", c.lower()))
            if n_tokens & c_tokens: return c
        return None

    @staticmethod
    def _apply_filter(df: pd.DataFrame, col: str, op: str, value: Any) -> pd.DataFrame:
        """Apply a single filter with case-insensitive string matching."""
        if col not in df.columns:
            return df
        series = df[col]
        try:
            if op == "==" and series.dtype == object:
                # Case-insensitive + strip
                mask = series.astype(str).str.strip().str.upper() == str(value).strip().upper()
                return df[mask]
            elif op == "==":
                return df[series == value]
            elif op == ">":  return df[series > float(value)]
            elif op == ">=": return df[series >= float(value)]
            elif op == "<":  return df[series < float(value)]
            elif op == "<=": return df[series <= float(value)]
        except Exception:
            pass
        return df

    # -----------------------------------------------------------------------
    # Main execute — operates on the FULL master_df, returns only matched rows
    # -----------------------------------------------------------------------
    def execute(self, intent: IntentResult, df: pd.DataFrame,
                kg_context: Dict = None) -> Tuple[pd.DataFrame, str]:
        if df is None or df.empty:
            return pd.DataFrame(), "No data available."

        # Remove internal columns
        display_cols = [c for c in df.columns if not c.startswith("_")]
        result = df[display_cols].copy()
        plan_steps = []
        kg_context = kg_context or {}

        # ── Step 1: Apply ALL filters ────────────────────────────────────────
        pre_filter_len = len(result)
        for f in intent.filters:
            raw_col = f.get("column", "")
            # Resolve column using fuzzy match against current result cols
            col = (raw_col if raw_col in result.columns
                   else self._resolve_col(raw_col, result, kg_context=kg_context))
            if not col:
                continue
            op  = f.get("op", "==")
            val = f.get("value", "")
            before = len(result)
            result = self._apply_filter(result, col, op, val)
            if len(result) != before:
                plan_steps.append(f"filter({col} {op} '{val}')")

        if not plan_steps and len(intent.filters) > 0:
            # Filters were specified but nothing matched — report this clearly
            plan_steps.append(f"filter_attempted(no_rows_matched)")

        # ── Step 2: Select specific columns if requested ─────────────────────
        cols_to_show = []
        for raw in intent.columns:
            resolved = self._resolve_col(raw, result, kg_context=kg_context) if raw not in result.columns else raw
            if resolved: cols_to_show.append(resolved)
        # Always include metrics columns in view
        for raw in intent.metrics:
            resolved = self._resolve_col(raw, result, prefer_numeric=True, kg_context=kg_context) if raw not in result.columns else raw
            if resolved and resolved not in cols_to_show:
                cols_to_show.append(resolved)

        # ── Step 3: Group by + Aggregate ─────────────────────────────────────
        if intent.group_by:
            valid_gb = []
            for g in intent.group_by:
                col = g if g in result.columns else self._resolve_col(g, result, kg_context=kg_context)
                if col: valid_gb.append(col)

            if valid_gb and intent.aggregation:
                # Determine which numeric cols to aggregate
                if intent.metrics:
                    agg_cols = [
                        c if c in result.columns else self._resolve_col(c, result, prefer_numeric=True, kg_context=kg_context)
                        for c in intent.metrics
                    ]
                    agg_cols = [c for c in agg_cols if c and c in result.columns]
                else:
                    agg_cols = result.select_dtypes(include="number").columns.tolist()

                if valid_gb and agg_cols:
                    try:
                        result = (result.groupby(valid_gb)[agg_cols]
                                  .agg(intent.aggregation)
                                  .reset_index())
                        plan_steps.append(f"groupby({valid_gb}).agg({intent.aggregation})")
                        cols_to_show = []  # show all aggregated cols
                    except Exception as e:
                        plan_steps.append(f"groupby_failed({e})")

        # ── Step 4: Pure aggregation (no groupby) ───────────────────────────
        elif intent.aggregation and intent.metrics and not intent.group_by:
            num_cols_avail = result.select_dtypes(include="number").columns.tolist()
            target_cols = [
                c if c in result.columns else self._resolve_col(c, result, prefer_numeric=True, kg_context=kg_context)
                for c in intent.metrics
            ]
            target_cols = [c for c in target_cols if c and c in result.columns]
            if not target_cols:
                target_cols = num_cols_avail[:5]

            if target_cols:
                try:
                    agg_result = {}
                    for col in target_cols:
                        if intent.aggregation == "sum":
                            agg_result[col] = result[col].sum()
                        elif intent.aggregation == "mean":
                            agg_result[col] = result[col].mean()
                        elif intent.aggregation == "count":
                            agg_result[col] = result[col].count()
                        elif intent.aggregation == "max":
                            agg_result[col] = result[col].max()
                        elif intent.aggregation == "min":
                            agg_result[col] = result[col].min()
                        elif intent.aggregation == "median":
                            agg_result[col] = result[col].median()
                    result = pd.DataFrame([agg_result])
                    plan_steps.append(f"agg({intent.aggregation}) on {target_cols}")
                    cols_to_show = []
                except Exception as e:
                    plan_steps.append(f"agg_failed({e})")

        # ── Step 5: Sort ─────────────────────────────────────────────────────
        sort_col = intent.sort_by
        if not sort_col and intent.metrics:
            sort_col = (intent.metrics[0] if intent.metrics[0] in result.columns
                        else self._resolve_col(intent.metrics[0], result, prefer_numeric=True, kg_context=kg_context))
        if not sort_col:
            num_avail = result.select_dtypes(include="number").columns.tolist()
            sort_col = num_avail[0] if num_avail else ""
        if sort_col:
            sort_col = sort_col if sort_col in result.columns else self._resolve_col(sort_col, result, kg_context=kg_context)
        if sort_col and sort_col in result.columns:
            try:
                result = result.sort_values(sort_col, ascending=intent.sort_asc)
                plan_steps.append(f"sort({sort_col}, asc={intent.sort_asc})")
            except Exception:
                pass

        # ── Step 6: Limit ────────────────────────────────────────────────────
        if intent.limit:
            result = result.head(intent.limit)
            plan_steps.append(f"limit({intent.limit})")

        # ── Step 7: Column projection ────────────────────────────────────────
        if cols_to_show:
            valid_show = [c for c in cols_to_show if c in result.columns]
            if valid_show:
                result = result[valid_show]

        query_plan = " → ".join(plan_steps) if plan_steps else "select(all columns)"
        return result.reset_index(drop=True), query_plan

    # -----------------------------------------------------------------------
    # Plain-English summary without LLM
    # -----------------------------------------------------------------------
    def quick_summary(self, df: pd.DataFrame, query: str = "") -> str:
        if df is None or df.empty:
            return ("⚠️ No matching rows found. The filter may not have matched "
                    "any entries in the dataset. Check column names and values.")
        n = len(df)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        parts = [f"Found **{n} row(s)** matching your query."]
        if num_cols:
            for col in num_cols[:4]:
                s = df[col]
                parts.append(
                    f"• **{col}**: total = {s.sum():,.2f}, avg = {s.mean():,.2f}, "
                    f"min = {s.min():,.2f}, max = {s.max():,.2f}"
                )
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols[:3]:
            if df[col].nunique() <= 10:
                vc = df[col].value_counts()
                top = ", ".join(f"**{v}** ({c})" for v, c in vc.head(5).items())
                parts.append(f"• {col}: {top}")
        return "\n".join(parts)
