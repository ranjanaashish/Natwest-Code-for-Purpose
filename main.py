"""
main.py — GraphIntel Self-Service Intelligence Chatbot
Run: streamlit run main.py
"""
import hashlib, sys, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engines.data_access       import DataAccessEngine
from engines.data_access       import DataAccessEngine
from engines.data_integration  import DataIntegrationEngine
from engines.knowledge_graph   import KnowledgeGraphEngine
from engines.intent_router     import IntentRouter, IntentResult
from engines.tabular_query     import TabularQueryEngine
from engines.forecasting       import ForecastingEngine
from engines.anomaly_detection import AnomalyEngine
from engines.scenario_analysis import ScenarioEngine
from engines.llm_explanation   import LLMExplanationEngine
from engines.result_validation import ResultValidationEngine
from engines.response_planner  import ResponsePlanner, ResponsePlan

from ui.components import (
    render_metrics_panel, render_table,
    render_forecast_chart, render_anomaly_chart, render_scenario_chart,
    render_anomaly_distribution, render_zscore_timeline, render_residual_chart,
    render_knowledge_graph, render_intent_badge, section_header,
)

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="GraphIntel · AI Analytics", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
_css = ROOT / "ui" / "styles.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "api_key":"", "selected_model_id":"google/gemini-flash-1.5",
    "endpoint_url":"https://openrouter.ai/api/v1",
    "available_models":[], "models_loaded":False, "last_api_key_hash":"",
    "datasets":{}, "master_df":None, "schema_metadata":{},
    "join_metadata":{}, "data_hash":"", "graph_built":False,
    "chat_history":[], "_pending_query":None,
    "fc_result":None, "fc_params":{},
    "ad_result":None, "ad_params":{},
    "dash_tabular":{}, "dash_forecast":{}, "dash_anomaly":{}, "dash_llm":{},
    "_de":None, "_kg":None, "_ro":None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Engine accessors
# ---------------------------------------------------------------------------
def _da() -> DataAccessEngine:
    if st.session_state.get("_da") is None:
        st.session_state["_da"] = DataAccessEngine()
    return st.session_state["_da"]

def _de() -> DataIntegrationEngine:
    if st.session_state.get("_de") is None:
        st.session_state["_de"] = DataIntegrationEngine()
    return st.session_state["_de"]

def _kg() -> KnowledgeGraphEngine:
    if st.session_state["_kg"] is None:
        st.session_state["_kg"] = KnowledgeGraphEngine(cache_dir=str(ROOT))
    return st.session_state["_kg"]

def _ro() -> IntentRouter:
    if st.session_state["_ro"] is None:
        st.session_state["_ro"] = IntentRouter(st.session_state.get("schema_metadata", {}))
    return st.session_state["_ro"]

def _llm() -> Optional[LLMExplanationEngine]:
    key = st.session_state.get("api_key","").strip()
    if not key: return None
    return LLMExplanationEngine(
        api_key=key,
        model=st.session_state.get("selected_model_id","google/gemini-flash-1.5"),
        base_url=st.session_state.get("endpoint_url","https://openrouter.ai/api/v1"),
    )

# ---------------------------------------------------------------------------
# Column helpers derived from current master_df
# ---------------------------------------------------------------------------
def _numeric_cols() -> List[str]:
    df = st.session_state.get("master_df")
    if df is None or df.empty: return []
    return [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]

def _datetime_cols() -> List[str]:
    df = st.session_state.get("master_df")
    schema = st.session_state.get("schema_metadata", {})
    cols: List[str] = []
    for info in schema.values(): cols.extend(info.get("datetime_columns", []))
    if df is not None:
        for c in df.select_dtypes(include="datetime64").columns:
            if c not in cols: cols.append(c)
    return list(dict.fromkeys(cols))

def _all_cols() -> List[str]:
    df = st.session_state.get("master_df")
    if df is None or df.empty: return []
    return [c for c in df.columns if not c.startswith("_")]

# ---------------------------------------------------------------------------
# OpenRouter model fetch
# ---------------------------------------------------------------------------
def _maybe_fetch_models(api_key: str):
    key_hash = hashlib.md5(api_key.encode()).hexdigest()
    if key_hash == st.session_state["last_api_key_hash"]: return
    with st.spinner("🔄 Fetching models from OpenRouter…"):
        st.session_state["available_models"] = LLMExplanationEngine.fetch_models(api_key)
        st.session_state["models_loaded"] = True
        st.session_state["last_api_key_hash"] = key_hash

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------
def _ingest_datasets(datasets: Dict[str, pd.DataFrame], new_hash: str):
    if new_hash == st.session_state.get("data_hash"): return
    with st.spinner("⚙️ Processing datasets…"):
        try:
            datasets, master_df, schema_meta, join_meta = _de().process_all(datasets)
            content_hash = _de().compute_hash(datasets)
            st.session_state.update({
                "datasets":datasets, "master_df":master_df,
                "schema_metadata":schema_meta, "join_metadata":join_meta,
                "data_hash":new_hash,
                "fc_result":None,"fc_params":{},
                "ad_result":None,"ad_params":{},
                "dash_tabular":{},"dash_forecast":{},"dash_anomaly":{},"dash_llm":{},
            })
            _ro().update_schema(schema_meta)
            kg = _kg()
            kg.load_or_build(datasets, schema_meta, join_meta, content_hash)
            st.session_state["graph_built"] = True
            n = sum(len(d) for d in datasets.values())
            stats = kg.get_stats()
            st.success(
                f"✅ {len(datasets)} dataset(s) · {n:,} total rows · "
                f"KG: {stats['total_nodes']} nodes, {stats['total_edges']} edges"
            )
        except Exception as exc:
            st.error(f"❌ Processing failed: {exc}")

def _process_files(uploaded_files) -> None:
    try:
        h = hashlib.md5()
        for f in uploaded_files:
            h.update(f.name.encode()); h.update(str(f.size).encode())
        new_hash = h.hexdigest()
    except Exception:
        new_hash = str(len(uploaded_files) + hash(str(uploaded_files)))
    for f in uploaded_files: f.seek(0)
    datasets = _da().upload_files(uploaded_files)
    _ingest_datasets(datasets, new_hash)

def _process_url(url: str) -> None:
    try:
        datasets = {"cloud_data": _da().connect_cloud_source(url)}
        _ingest_datasets(datasets, str(hash(url)))
    except Exception as exc:
        st.error(f"❌ URL Fetch Failed: {exc}")

def _process_db(uri: str, query: str) -> None:
    try:
        datasets = {"db_data": _da().connect_database(uri, query)}
        _ingest_datasets(datasets, str(hash(uri+query)))
    except Exception as exc:
        st.error(f"❌ DB Fetch Failed: {exc}")


def _load_sample_data() -> None:
    sample_dir = ROOT / "data" / "samples"
    files = list(sample_dir.glob("*.csv"))
    if not files: st.warning("No sample files found."); return

    class _FF:
        def __init__(self, p: Path):
            self.name=p.name; self.size=p.stat().st_size
            self._b=p.read_bytes(); self._pos=0
        def read(self,n=-1):
            chunk=self._b[self._pos:]if n==-1 else self._b[self._pos:self._pos+n]
            self._pos+=len(chunk); return chunk
        def seek(self,pos): self._pos=pos

    _process_files([_FF(p) for p in files])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar() -> None:
    with st.sidebar:
        st.markdown('<p style="font-size:1.3rem;font-weight:800;color:#ec4764;margin:0">🧠 GraphIntel</p>'
                    '<p style="font-size:0.72rem;color:#9a8a92;margin:0 0 0.8rem">Graph-Aware AI Analytics</p>',
                    unsafe_allow_html=True)
        st.divider()
        st.markdown("### 🔑 LLM Engine")
        st.selectbox("Engine Type", ["OpenAI-Compatible"], key="_engine_type")
        manual = st.checkbox("Enter model manually", key="_manual_model_cb")
        ep = st.text_input("Endpoint URL", value=st.session_state["endpoint_url"],
                           placeholder="https://openrouter.ai/api/v1", key="_ep_input")
        st.session_state["endpoint_url"] = ep.strip()
        ak = st.text_input("API Key", type="password", value=st.session_state["api_key"],
                           placeholder="sk-or-v1-…", key="_ak_input")
        st.session_state["api_key"] = ak.strip()
        if ak.strip(): _maybe_fetch_models(ak.strip())
        if manual:
            mid = st.text_input("Model ID", value=st.session_state["selected_model_id"],
                                placeholder="google/gemini-flash-1.5", key="_mid_input")
            st.session_state["selected_model_id"] = mid.strip()
        else:
            models = st.session_state.get("available_models",[])
            if models:
                options = LLMExplanationEngine.format_model_options(models)
                cur = st.session_state.get("selected_model_id","")
                def_idx = next((i for i,m in enumerate(models) if m["id"]==cur),0)
                sel = st.selectbox("Choose model",options,index=def_idx,key="_model_sel")
                st.session_state["selected_model_id"] = LLMExplanationEngine.extract_model_id(sel)
            elif ak.strip(): st.caption("🔄 Loading models…")
            else: st.caption("⚠️ Enter API key to load models")
        if ak.strip() and st.session_state["models_loaded"]:
            mods = st.session_state["available_models"]
            nf = sum(1 for m in mods if m.get("is_free"))
            st.markdown(f'<small style="color:#9bdeac">✅ Connected · {len(mods)} models ({nf} free 🆓)</small>',
                        unsafe_allow_html=True)
        st.divider()
        st.markdown("### 🗄️ Universal Data Access")
        st.caption("Local, Cloud, or Databases")
        
        load_tabs = st.tabs(["📁 File", "🌐 URL", "🛢️ DB"])
        
        with load_tabs[0]:
            uploaded = st.file_uploader("files", type=["csv","json"], accept_multiple_files=True,
                                        label_visibility="collapsed", key="_uploader")
            if uploaded: _process_files(uploaded)

        with load_tabs[1]:
            cloud_url = st.text_input("Cloud/API URL", placeholder="https://.../data.csv", key="_cloud_url")
            if st.button("Fetch URL", use_container_width=True, key="_btn_url"):
                if cloud_url.strip(): _process_url(cloud_url.strip())
                
        with load_tabs[2]:
            db_uri = st.text_input("Connection String", placeholder="sqlite:///mydb.sqlite", key="_db_uri")
            db_query = st.text_input("Table / Query", placeholder="SELECT * FROM table", key="_db_query")
            if st.button("Query DB", use_container_width=True, key="_btn_db"):
                if db_uri.strip() and db_query.strip(): _process_db(db_uri.strip(), db_query.strip())

        for name, df in st.session_state.get("datasets",{}).items():
            st.markdown(f'<small>🗄️ <b style="color:#ec4764">{name}</b>'
                        f' · {len(df):,} rows · {len(df.columns)} cols</small>', unsafe_allow_html=True)
        if st.button("📂 Load Sample Data", use_container_width=True):
            _load_sample_data(); st.rerun()
        st.divider()
        if st.session_state.get("graph_built"):
            stats = _kg().get_stats()
            st.markdown("### 🕸️ Knowledge Graph")
            st.markdown(f'<small><b style="color:#9bdeac">{stats["total_nodes"]}</b> nodes &nbsp;·&nbsp; '
                        f'<b style="color:#9bdeac">{stats["total_edges"]}</b> edges</small>', unsafe_allow_html=True)
            icons={"dataset":"🗄️","column":"📋","metric":"📊","entity":"🏷️","datatype":"🔤","analysis":"🔬"}
            for t,cnt in stats.get("node_types",{}).items():
                st.markdown(f'<small>{icons.get(t,"·")} {t}: <b style="color:#895d6b">{cnt}</b></small>',
                            unsafe_allow_html=True)
        st.divider()
        st.markdown('<small style="color:#9a8a92">MTP · Graph-Aware Intelligence</small>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core orchestration — uses KG context to improve column resolution
# ---------------------------------------------------------------------------
def _orchestrate(query: str) -> ResponsePlan:
    planner   = ResponsePlanner()
    validator = ResultValidationEngine()
    llm       = _llm()
    ro        = _ro()
    kg        = _kg()

    schema_meta = st.session_state.get("schema_metadata", {})
    master_df   = st.session_state.get("master_df")     # always the FULL dataset
    schema_text = kg.get_schema_text(schema_meta)
    # KG context: top relevant columns for this query
    graph_ctx   = kg.retrieve_context(query, schema_meta)
    intent      = ro.route(query)
    kg.add_analysis_node(query, intent.intent)

    if master_df is None or master_df.empty:
        if intent.intent in ("tabular_query","forecast","anomaly","scenario"):
            return planner.plan_error(intent.intent,
                "No dataset loaded. Upload a CSV/JSON file or click **Load Sample Data**.")

    # ── Tabular ──────────────────────────────────────────────────────────────
    if intent.intent == "tabular_query":
        try:
            if llm:
                try:
                    code = llm.generate_pandas_code(query, schema_text)
                    result_raw = eval(code, {"__builtins__": {}}, {"df": master_df, "pd": pd})
                    if isinstance(result_raw, pd.DataFrame): result_df = result_raw
                    elif isinstance(result_raw, pd.Series): result_df = result_raw.to_frame()
                    else: result_df = pd.DataFrame([{"Result": result_raw}])
                    qplan = f"LLM Generated Pandas: {code}"
                except Exception:
                    tqe = TabularQueryEngine()
                    result_df, qplan = tqe.execute(intent, master_df, kg_context=graph_ctx)
                txt = llm.explain_table(result_df, query, qplan, master_df)
            else:
                tqe = TabularQueryEngine()
                result_df, qplan = tqe.execute(intent, master_df, kg_context=graph_ctx)
                txt = tqe.quick_summary(result_df, query)

            val = validator.validate_tabular(result_df, master_df, intent, schema_meta)
            lv  = validator.validate_llm_narrative(txt, result_df)
            st.session_state["dash_tabular"] = val
            st.session_state["dash_llm"]     = lv
            return planner.plan_tabular(intent, result_df, txt, qplan, val, lv)
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))

    # ── Forecast ─────────────────────────────────────────────────────────────
    elif intent.intent == "forecast":
        try:
            fe = ForecastingEngine()
            dc, vc = fe.auto_detect_columns(master_df, schema_meta)
            if intent.metrics: vc = intent.metrics[0]
            if not dc or not vc:
                return planner.plan_error(intent.intent,
                    "Cannot identify date/value columns. Try: 'Forecast revenue for next 8 weeks'.")
            fdf, hdf, mets = fe.forecast(master_df, dc, vc, intent.time_periods)
            val = validator.validate_forecast(mets)
            if llm:
                txt = llm.explain_forecast(fdf, hdf, mets, query, vc)
            else:
                txt = (f"**Forecast for {vc}** — next {intent.time_periods} periods:\n"
                       f"- Range: {fdf['yhat'].min():,.2f} – {fdf['yhat'].max():,.2f}\n"
                       f"- MAE: {mets.get('MAE','N/A')}, RMSE: {mets.get('RMSE','N/A')}\n"
                       f"- R²: {mets.get('R²','N/A')}, "
                       f"vs Baseline: {mets.get('Model vs Baseline','N/A')}")
            lv = validator.validate_llm_narrative(txt, fdf)
            st.session_state["dash_forecast"] = val
            st.session_state["dash_llm"]      = lv
            return planner.plan_forecast(intent, fdf, hdf, txt, mets, val, lv, dc, vc)
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))

    # ── Anomaly ──────────────────────────────────────────────────────────────
    elif intent.intent == "anomaly":
        try:
            ae       = AnomalyEngine()
            best_col = ae.auto_detect_column(master_df)
            target   = intent.metrics if intent.metrics else None
            adf, sdf, summary = ae.detect(master_df, columns=target)
            col_v    = (target or [best_col])[0] if (target or best_col) else None
            gt       = ae.validate_with_synthetic_gt(master_df, col_v) if col_v else {}
            val      = validator.validate_anomaly(gt)
            if llm:
                txt = llm.explain_anomaly(adf, summary, query)
            else:
                n_anom = summary.get("anomaly_rows",0)
                txt = (f"**Anomaly Detection** ({summary.get('method','zscore').upper()}):\n"
                       f"- Found **{n_anom} anomalies** out of {summary.get('total_rows',0)} rows "
                       f"({summary.get('anomaly_pct',0):.1f}%)\n"
                       f"- Columns analyzed: {', '.join(summary.get('columns_checked',[]))}\n"
                       + (f"- Precision={gt.get('Precision','?')}, "
                          f"Recall={gt.get('Recall','?')}, F1={gt.get('F1-Score','?')}" if gt else ""))
            lv = validator.validate_llm_narrative(txt, adf)
            st.session_state["dash_anomaly"] = val
            st.session_state["dash_llm"]     = lv
            return planner.plan_anomaly(intent, adf, sdf, txt, summary, val, lv, value_col=col_v or "")
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))

    # ── Scenario ─────────────────────────────────────────────────────────────
    elif intent.intent == "scenario":
        try:
            fe = ForecastingEngine(); se = ScenarioEngine()
            dc, vc = fe.auto_detect_columns(master_df, schema_meta)
            if intent.metrics: vc = intent.metrics[0]
            merged, _, assum, desc = se.analyze(query, master_df, fe, dc, vc, intent.time_periods)
            if llm:
                txt = llm.explain_scenario(merged, assum, desc, query, vc)
            else:
                dm = merged["delta"].mean() if "delta" in merged.columns else 0
                txt = f"{desc}\n\n**Projected avg change**: {dm:+,.2f} in {vc}."
            return planner.plan_scenario(intent, merged, txt, desc, dc, vc)
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))

    # ── Schema query ─────────────────────────────────────────────────────────
    elif intent.intent == "schema_query":
        try:
            txt = (llm.answer_schema_query(query, schema_text, graph_ctx)
                   if llm else _format_schema_prose(schema_meta))
            return planner.plan_text(intent, txt)
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))

    # ── General Q&A ──────────────────────────────────────────────────────────
    else:
        try:
            if llm:
                try:
                    # Attempt LLM Pandas eval first to catch complex business logic missed by simple routing
                    code = llm.generate_pandas_code(query, schema_text)
                    result_raw = eval(code, {"__builtins__": {}}, {"df": master_df, "pd": pd})
                    if isinstance(result_raw, pd.DataFrame): result_df = result_raw
                    elif isinstance(result_raw, pd.Series): result_df = result_raw.to_frame()
                    else: result_df = pd.DataFrame([{"Result": result_raw}])
                    qplan = f"LLM Generated Pandas: {code}"
                    
                    txt = llm.explain_table(result_df, query, qplan, master_df)
                    val = validator.validate_tabular(result_df, master_df, intent, schema_meta)
                    lv  = validator.validate_llm_narrative(txt, result_df)
                    st.session_state["dash_tabular"] = val
                    st.session_state["dash_llm"]     = lv
                    intent.intent = "tabular_query"
                    return planner.plan_tabular(intent, result_df, txt, qplan, val, lv)
                except Exception:
                    pass
            
            txt = (llm.general_answer(query, master_df, schema_text)
                   if llm else _no_llm_answer(master_df, schema_meta))
            return planner.plan_text(intent, txt)
        except Exception as e:
            return planner.plan_error(intent.intent, str(e))


def _format_schema_prose(schema_meta: Dict) -> str:
    if not schema_meta: return "No dataset loaded yet."
    lines = []
    for ds_name, info in schema_meta.items():
        rc = info.get("row_count",0)
        num_cols = info.get("numeric_columns",[])
        dt_cols  = info.get("datetime_columns",[])
        cat_cols = info.get("categorical_columns",[])
        lines.append(f"**Dataset: {ds_name}** ({rc:,} rows)")
        if dt_cols:  lines.append(f"- Date columns: {', '.join(dt_cols)}")
        if num_cols: lines.append(f"- Numeric columns: {', '.join(num_cols)}")
        if cat_cols: lines.append(f"- Text columns: {', '.join(cat_cols)}")
        for col in num_cols[:5]:
            ci = info.get("columns",{}).get(col,{})
            lines.append(f"  • {col}: min={ci.get('min',0):.2g}, max={ci.get('max',0):.2g}, mean={ci.get('mean',0):.2g}")
        for col in cat_cols[:4]:
            ci = info.get("columns",{}).get(col,{})
            tv = ci.get("top_values",[])
            if tv: lines.append(f"  • {col} values: {', '.join(str(v) for v in tv[:10])}")
    return "\n".join(lines)


def _no_llm_answer(master_df, schema_meta) -> str:
    if master_df is None or master_df.empty:
        return "No dataset loaded. Upload a CSV/JSON file or click **Load Sample Data**."
    n = len(master_df)
    all_c = [c for c in master_df.columns if not c.startswith("_")]
    num_c = [c for c in master_df.select_dtypes(include="number").columns if not c.startswith("_")]
    lines = [
        f"The dataset has **{n:,} rows** and **{len(all_c)} columns**.",
        f"Columns: {', '.join(all_c[:20])}{'…' if len(all_c)>20 else ''}.",
    ]
    for col in num_c[:4]:
        lines.append(f"- **{col}**: mean={master_df[col].mean():.2f}, "
                     f"min={master_df[col].min():.2f}, max={master_df[col].max():.2f}, "
                     f"sum={master_df[col].sum():.2f}")
    lines.append("\n*Add an API key in the sidebar for AI-powered answers.*")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Response renderer
# ---------------------------------------------------------------------------
def _render_response(plan: ResponsePlan, intent: IntentResult) -> None:
    render_intent_badge(intent.intent, intent.confidence)

    # Show applied filters as chips
    if intent.filters:
        chips = " ".join(
            f'<span class="filter-chip">{f.get("column","")} {f.get("op","")} "{f.get("value","")}"</span>'
            for f in intent.filters
        )
        st.markdown(f"<small>🔎 Filters applied: {chips}</small>", unsafe_allow_html=True)

    st.write("")
    if plan.error:
        st.error(plan.text); return

    if plan.show_text and plan.text:
        st.markdown(plan.text)
    if plan.query_plan:
        st.markdown(f'<div class="plan-text">🔍 {plan.query_plan}</div>', unsafe_allow_html=True)

    _key = uuid.uuid4().hex[:8]

    if plan.show_forecast_chart and plan.forecast_df is not None:
        render_forecast_chart(plan.history_df, plan.forecast_df,
                              plan.date_col, plan.value_col, chart_key=f"resp_{_key}")

    if plan.show_anomaly_chart and plan.scored_df is not None and plan.value_col:
        if plan.value_col in plan.scored_df.columns:
            dc = _get_date_col_schema()
            render_anomaly_chart(plan.scored_df, plan.anomaly_df or pd.DataFrame(),
                                 dc, plan.value_col, chart_key=f"resp_{_key}")

    if plan.show_scenario_chart and plan.scenario_df is not None:
        if plan.scenario_description:
            st.markdown(f'<div class="plan-text">{plan.scenario_description}</div>', unsafe_allow_html=True)
        render_scenario_chart(plan.scenario_df, plan.date_col, plan.value_col,
                              chart_key=f"resp_{_key}")

    # Always show table with actual filtered/aggregated data
    if plan.show_table:
        tbl_key = f"tbl_{_key}"
        if plan.result_df is not None and not plan.result_df.empty:
            render_table(plan.result_df, "Query Result", key_suffix=tbl_key)
        elif plan.forecast_df is not None:
            render_table(plan.forecast_df, "Forecast Table", key_suffix=tbl_key)
        elif plan.anomaly_df is not None:
            render_table(plan.anomaly_df, f"Anomaly Rows ({len(plan.anomaly_df)})", key_suffix=tbl_key)
        elif plan.scenario_df is not None:
            cols_show = [c for c in plan.scenario_df.columns if not c.endswith(("_lo","_hi"))]
            render_table(plan.scenario_df[cols_show], "Scenario vs Baseline", key_suffix=tbl_key)

    if plan.show_metrics_panel:
        with st.expander("📊 Evaluation Metrics", expanded=False):
            tabs_data = []
            if plan.validation_metrics: tabs_data.append(("Output Quality", plan.validation_metrics))
            if plan.metrics:
                scalar = {k:v for k,v in plan.metrics.items() if not isinstance(v,dict)}
                if scalar: tabs_data.append(("Engine Metrics", scalar))
            if plan.llm_validation: tabs_data.append(("LLM Grounding", plan.llm_validation))
            if tabs_data:
                for tab_obj, data in zip(st.tabs([t for t,_ in tabs_data]),
                                         [d for _,d in tabs_data]):
                    with tab_obj: render_metrics_panel(data)


def _get_date_col_schema() -> Optional[str]:
    for info in st.session_state.get("schema_metadata",{}).values():
        if info.get("datetime_columns"): return info["datetime_columns"][0]
    return None

# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------
def _tab_chat() -> None:
    if not st.session_state["chat_history"]:
        st.markdown('<div class="section-header">💡 Try these queries</div>', unsafe_allow_html=True)
        suggestions = [
            "Show top 10 products by revenue",
            "What is the total revenue by region?",
            "Forecast revenue for next 8 weeks",
            "Are there any anomalies in revenue?",
            "What if revenue increases by 15%?",
            "What columns does the dataset have?",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i%3].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending_query"] = s; st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🧠"):
            if msg["role"]=="user": st.markdown(msg["content"])
            else: _render_response(msg["plan"], msg["intent"])

    query = st.chat_input("Ask anything about your data — filters, sums, forecasts, anomalies…")
    if st.session_state.get("_pending_query"):
        query = st.session_state.pop("_pending_query")

    if query:
        st.session_state["chat_history"].append({"role":"user","content":query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔮 Analysing your full dataset…"):
                intent = _ro().route(query)
                plan   = _orchestrate(query)
            _render_response(plan, intent)
        st.session_state["chat_history"].append({"role":"assistant","plan":plan,"intent":intent})
        st.rerun()

# ---------------------------------------------------------------------------
# Tab: Forecasting
# ---------------------------------------------------------------------------
def _tab_forecasting() -> None:
    section_header("📈","Forecasting Engine","Time-Series Prediction")
    master_df = st.session_state.get("master_df")
    if master_df is None or master_df.empty:
        st.markdown('<div class="info-box">Upload a dataset to run forecasts.</div>', unsafe_allow_html=True); return
    dt_cols  = _datetime_cols()
    num_cols = _numeric_cols()
    ctrl, chart = st.columns([1,2.4])
    with ctrl:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Parameters")
        date_col  = st.selectbox("📅 Date Column", dt_cols if dt_cols else _all_cols(), key="fc_date")
        value_col = st.selectbox("📊 Value to Forecast", num_cols, key="fc_val") if num_cols else None
        periods   = st.slider("⏭️ Periods", 1, 52, 8, key="fc_periods")
        run_fc    = st.button("🚀 Run Forecast", use_container_width=True, key="fc_run")
        st.markdown('</div>', unsafe_allow_html=True)
    if run_fc and value_col and date_col:
        with st.spinner("📈 Computing…"):
            try:
                fe = ForecastingEngine()
                fdf, hdf, mets = fe.forecast(master_df, date_col, value_col, periods)
                val = ResultValidationEngine().validate_forecast(mets)
                llm = _llm()
                txt = llm.explain_forecast(fdf, hdf, mets, f"Forecast {value_col}", value_col) if llm else ""
                kid = uuid.uuid4().hex[:8]
                st.session_state["fc_result"] = {
                    "fdf":fdf,"hdf":hdf,"mets":mets,"val":val,"txt":txt,
                    "date_col":date_col,"value_col":value_col,"key_id":kid
                }
                st.session_state["dash_forecast"] = val
            except Exception as exc:
                st.error(f"❌ {exc}"); st.session_state["fc_result"] = None

    res = st.session_state.get("fc_result")
    if res:
        kid = res.get("key_id","fc")
        with chart:
            render_forecast_chart(res["hdf"],res["fdf"],res["date_col"],res["value_col"],chart_key=f"main_{kid}")
        if res.get("txt"): st.markdown(f'<div class="info-box">{res["txt"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
        _metric_row_forecast(res["val"])
        t1,t2,t3 = st.tabs(["📈 Chart","📉 Residuals","📋 Table"])
        with t1: render_forecast_chart(res["hdf"],res["fdf"],res["date_col"],res["value_col"],chart_key=f"tab1_{kid}")
        with t2: render_residual_chart(res["hdf"],res["date_col"],res["value_col"],chart_key=f"resid_{kid}")
        with t3: render_table(res["fdf"],"Forecast Values",key_suffix=f"ftbl_{kid}")
    else:
        with chart:
            st.markdown('<div class="warn-box" style="margin-top:1rem">▶ Set parameters and click <b>Run Forecast</b>.</div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab: Anomaly Detection
# ---------------------------------------------------------------------------
def _tab_anomaly() -> None:
    section_header("🔍","Anomaly Detection Engine","Statistical Outlier Analysis")
    master_df = st.session_state.get("master_df")
    if master_df is None or master_df.empty:
        st.markdown('<div class="info-box">Upload a dataset to detect anomalies.</div>', unsafe_allow_html=True); return
    num_cols = _numeric_cols(); dt_avail = _datetime_cols()
    ctrl, chart = st.columns([1,2.4])
    with ctrl:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Parameters")
        sel_cols  = st.multiselect("📋 Columns", options=num_cols,
                                    default=num_cols[:2] if len(num_cols)>=2 else num_cols, key="ad_cols")
        method    = st.selectbox("🔬 Method", ["Z-Score","IQR"], key="ad_method")
        threshold = st.slider("⚡ Threshold", 1.0, 5.0, 2.5, 0.1, key="ad_thresh")
        dc_opts   = ["(none)"]+dt_avail+[c for c in _all_cols() if c not in dt_avail]
        date_sel  = st.selectbox("📅 Date Column (chart)", dc_opts, key="ad_date")
        dc_ad     = None if date_sel=="(none)" else date_sel
        run_ad    = st.button("🔍 Detect Anomalies", use_container_width=True, key="ad_run")
        st.markdown('</div>', unsafe_allow_html=True)
    if run_ad and sel_cols:
        with st.spinner("🔍 Detecting…"):
            try:
                ae = AnomalyEngine()
                ae.Z_THRESHOLD = threshold if method=="Z-Score" else ae.Z_THRESHOLD
                ae.IQR_FACTOR  = threshold if method=="IQR"     else ae.IQR_FACTOR
                m_str = "zscore" if method=="Z-Score" else "iqr"
                adf, sdf, summary = ae.detect(master_df, columns=sel_cols, method=m_str)
                gt_metrics = ae.validate_with_synthetic_gt(master_df, sel_cols[0])
                latency    = _compute_latency(adf, dc_ad, master_df)
                val        = ResultValidationEngine().validate_anomaly(gt_metrics)
                val["Detection Latency"] = latency
                llm = _llm()
                txt = llm.explain_anomaly(adf, summary, "anomaly detection") if llm else ""
                kid = uuid.uuid4().hex[:8]
                st.session_state["ad_result"] = {
                    "adf":adf,"sdf":sdf,"summary":summary,"val":val,"txt":txt,
                    "sel_cols":sel_cols,"dc":dc_ad,"primary_col":sel_cols[0],
                    "threshold":threshold,"key_id":kid
                }
                st.session_state["dash_anomaly"] = val
            except Exception as exc:
                st.error(f"❌ {exc}"); st.session_state["ad_result"] = None
    res = st.session_state.get("ad_result")
    if res:
        kid = res.get("key_id","ad")
        with chart:
            render_anomaly_chart(master_df,res["adf"],res["dc"],res["primary_col"],chart_key=f"main_{kid}")
        if res.get("txt"): st.markdown(f'<div class="info-box">{res["txt"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
        _metric_row_anomaly(res["val"],res["summary"])
        t1,t2,t3,t4 = st.tabs(["📉 Chart","📊 Distribution","🎯 Score Timeline","📋 Table"])
        with t1: render_anomaly_chart(master_df,res["adf"],res["dc"],res["primary_col"],chart_key=f"tab1_{kid}")
        with t2: render_anomaly_distribution(master_df,res["adf"],res["primary_col"],chart_key=f"dist_{kid}")
        with t3: render_zscore_timeline(res["sdf"],res["primary_col"],res["dc"],res["threshold"],chart_key=f"zscore_{kid}")
        with t4: render_table(res["adf"],f"Anomalies ({len(res['adf'])})",key_suffix=f"atbl_{kid}")
        per_col = res["summary"].get("per_column",{})
        if per_col:
            with st.expander("📋 Per-Column Stats"):
                rows=[{"Column":col,"Anomalies":info.get("anomaly_count",0),
                       "Mean":round(info.get("mean",0),3),"Std":round(info.get("std",0),3),
                       "Min":round(info.get("min",0),3),"Max":round(info.get("max",0),3)}
                      for col,info in per_col.items()]
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    else:
        with chart:
            st.markdown('<div class="warn-box" style="margin-top:1rem">▶ Select columns and click <b>Detect Anomalies</b>.</div>',
                        unsafe_allow_html=True)


def _compute_latency(anomaly_df, date_col, master_df):
    if anomaly_df.empty: return "N/A"
    if date_col and date_col in anomaly_df.columns and date_col in master_df.columns:
        try:
            first = pd.to_datetime(anomaly_df[date_col]).min()
            start = pd.to_datetime(master_df[date_col]).min()
            return f"{(first-start).days}d"
        except Exception: pass
    if len(anomaly_df): return f"idx {anomaly_df.index[0]}"
    return "N/A"

# ---------------------------------------------------------------------------
# Metric block helpers
# ---------------------------------------------------------------------------
def _mblock(col, label, value, subtitle="", good=False, neutral=False):
    with col:
        display = "—" if value is None else (f"{value:.3f}" if isinstance(value,float) else str(value))
        cls = "neutral" if neutral else ("" if good else "danger")
        sub = f'<div class="mc-sub">{subtitle}</div>' if subtitle else ""
        st.markdown(f'<div class="metric-card {cls}"><div class="mc-label">{label}</div>'
                    f'<div class="mc-value">{display}</div>{sub}</div>', unsafe_allow_html=True)

def _sf(v) -> float:
    try: return float(v)
    except: return 0.0

def _metric_row_forecast(val):
    st.markdown('<div class="metrics-section-title">📊 Forecast Accuracy Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    _mblock(c1,"MAE",val.get("MAE"),neutral=True)
    _mblock(c2,"RMSE",val.get("RMSE"),neutral=True)
    _mblock(c3,"MAPE",f'{val.get("MAPE (%)",0):.2f}%',neutral=True)
    _mblock(c4,"R²",val.get("R²"),good=_sf(val.get("R²",0))>0.7)
    _mblock(c5,"Dir. Acc.",f'{val.get("Directional Accuracy (%)",0):.1f}%',good=_sf(val.get("Directional Accuracy (%)",0))>60)
    _mblock(c6,"vs Baseline",val.get("Model vs Baseline","?"),good=val.get("Model vs Baseline","")=="✅ Better")

def _metric_row_anomaly(val, summary):
    st.markdown('<div class="metrics-section-title">📊 Anomaly Detection Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    _mblock(c1,"Precision",val.get("Precision"),good=_sf(val.get("Precision",0))>0.7)
    _mblock(c2,"Recall",val.get("Recall"),good=_sf(val.get("Recall",0))>0.7)
    _mblock(c3,"F1",val.get("F1-Score"),good=_sf(val.get("F1-Score",0))>0.7)
    _mblock(c4,"FPR",val.get("False Positive Rate"),good=_sf(val.get("False Positive Rate",1))<0.2)
    _mblock(c5,"Anomaly %",f'{summary.get("anomaly_pct",0):.1f}%',neutral=True)
    _mblock(c6,"Latency",val.get("Detection Latency","N/A"),neutral=True)

# ---------------------------------------------------------------------------
# Tab: Metrics Dashboard
# ---------------------------------------------------------------------------
def _tab_metrics() -> None:
    section_header("📊","MTP Metrics Dashboard","Live Quality Tracking")
    if not st.session_state.get("graph_built"):
        st.markdown('<div class="info-box">Upload a dataset and run queries to see metrics.</div>', unsafe_allow_html=True); return
    st.markdown('<small style="color:#9a8a92">Updates automatically as you use Chat, Forecasting, and Anomaly tabs.</small>', unsafe_allow_html=True)
    st.write("")
    # Tabular
    dash_tab = st.session_state.get("dash_tabular",{})
    st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
    st.markdown('<div class="metrics-section-title">📤 Tabular Query Metrics</div>', unsafe_allow_html=True)
    if dash_tab:
        c1,c2,c3,c4,c5=st.columns(5)
        _mblock(c1,"Schema Grounding",f'{dash_tab.get("Schema Grounding (%)",0)}%',good=_sf(dash_tab.get("Schema Grounding (%)",0))>80)
        _mblock(c2,"Num. Consistency",f'{dash_tab.get("Numeric Consistency (%)",0)}%',good=_sf(dash_tab.get("Numeric Consistency (%)",0))>90)
        _mblock(c3,"Match F1",dash_tab.get("Exact Match F1","—"),good=_sf(dash_tab.get("Exact Match F1",0))>0.7)
        _mblock(c4,"Precision",dash_tab.get("Exact Match Precision","—"),good=_sf(dash_tab.get("Exact Match Precision",0))>0.7)
        _mblock(c5,"Recall",dash_tab.get("Exact Match Recall","—"),good=_sf(dash_tab.get("Exact Match Recall",0))>0.7)
    else: st.caption("Ask a tabular query in Chat.")
    st.markdown('</div>', unsafe_allow_html=True)
    # Forecast
    dash_fc = st.session_state.get("dash_forecast",{})
    st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
    st.markdown('<div class="metrics-section-title">📈 Forecast Accuracy Metrics</div>', unsafe_allow_html=True)
    if dash_fc: _metric_row_forecast(dash_fc)
    else: st.caption("Run a forecast in the 📈 Forecasting tab or ask in Chat.")
    st.markdown('</div>', unsafe_allow_html=True)
    # Anomaly
    dash_an = st.session_state.get("dash_anomaly",{})
    st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
    st.markdown('<div class="metrics-section-title">🔍 Anomaly Detection Metrics</div>', unsafe_allow_html=True)
    if dash_an:
        c1,c2,c3,c4,c5=st.columns(5)
        _mblock(c1,"Precision",dash_an.get("Precision"),good=_sf(dash_an.get("Precision",0))>0.7)
        _mblock(c2,"Recall",dash_an.get("Recall"),good=_sf(dash_an.get("Recall",0))>0.7)
        _mblock(c3,"F1-Score",dash_an.get("F1-Score"),good=_sf(dash_an.get("F1-Score",0))>0.7)
        _mblock(c4,"FPR",dash_an.get("False Positive Rate"),good=_sf(dash_an.get("False Positive Rate",1))<0.2)
        _mblock(c5,"Latency",dash_an.get("Detection Latency","N/A"),neutral=True)
    else: st.caption("Run anomaly detection in the 🔍 Anomaly tab or ask in Chat.")
    st.markdown('</div>', unsafe_allow_html=True)
    # LLM
    dash_llm = st.session_state.get("dash_llm",{})
    st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
    st.markdown('<div class="metrics-section-title">🤖 LLM Narrative Quality</div>', unsafe_allow_html=True)
    if dash_llm:
        c1,c2 = st.columns(2)
        _mblock(c1,"Grounded Answer Rate (GAR)",dash_llm.get("Grounded Answer Rate (GAR)","N/A"),good=True)
        # Filler column to keep layout
        _mblock(c2,"","","")
    else: st.caption("Ask an LLM-powered question in Chat.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab: Knowledge Graph
# ---------------------------------------------------------------------------
def _tab_kg() -> None:
    section_header("🕸️","Knowledge Graph","Semantic Data Map")
    if not st.session_state.get("graph_built"):
        st.markdown('<div class="info-box">Upload a dataset to build the graph.</div>', unsafe_allow_html=True); return
    kg=_kg(); stats=kg.get_stats()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Nodes",stats["total_nodes"]); c2.metric("Total Edges",stats["total_edges"])
    c3.metric("Node Types",len(stats.get("node_types",{}))); c4.metric("Datasets",stats.get("node_types",{}).get("dataset",0))
    render_knowledge_graph(kg.graph, chart_key=uuid.uuid4().hex[:8])
    with st.expander("📋 Node Types"):
        icons={"dataset":"🗄️","column":"📋","metric":"📊","entity":"🏷️","datatype":"🔤","analysis":"🔬"}
        rows=[{"Type":t,"Icon":icons.get(t,"·"),"Count":cnt} for t,cnt in stats.get("node_types",{}).items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    with st.expander("🔗 Edge Relations"):
        rows=[{"Relation":t,"Count":cnt} for t,cnt in stats.get("edge_types",{}).items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: Data Explorer
# ---------------------------------------------------------------------------
def _tab_explorer() -> None:
    section_header("🗄️","Data Explorer","Uploaded Dataset Preview")
    datasets=st.session_state.get("datasets",{}); master_df=st.session_state.get("master_df")
    schema_meta=st.session_state.get("schema_metadata",{})
    if not datasets:
        st.markdown('<div class="info-box">Upload datasets to explore them here.</div>', unsafe_allow_html=True); return
    options=["📦 master (merged)"]+[f"📄 {n}" for n in datasets.keys()]
    sel=st.selectbox("Select dataset",options,key="de_select")
    df=master_df if sel.startswith("📦") else datasets.get(sel.replace("📄 ",""),pd.DataFrame())
    ds_name=None if sel.startswith("📦") else sel.replace("📄 ","")
    if df is None or df.empty: st.warning("Empty dataset."); return
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows",f"{len(df):,}"); c2.metric("Columns",len(df.columns))
    c3.metric("Numeric Cols",len(df.select_dtypes(include="number").columns))
    c4.metric("Missing (%)",f"{df.isnull().mean().mean()*100:.1f}%")
    st.dataframe(df.head(200),use_container_width=True,hide_index=True,height=350)
    st.markdown("#### 📋 Column Schema")
    if ds_name and ds_name in schema_meta:
        col_info=schema_meta[ds_name].get("columns",{})
    else:
        tmp=_de().generate_schema_metadata({"_view":df})
        col_info=tmp.get("_view",{}).get("columns",{})
    if col_info:
        rows=[]
        for col_name,ci in col_info.items():
            row={"Column":col_name,"Type":ci.get("dtype",""),
                 "Numeric":"✓" if ci.get("is_numeric") else "",
                 "DateTime":"✓" if ci.get("is_datetime") else "",
                 "Null %":f'{ci.get("null_pct",0):.1f}%',"Unique":ci.get("unique_count","")}
            if ci.get("is_numeric"):
                row.update({"Min":f'{ci.get("min",0):.3g}',"Max":f'{ci.get("max",0):.3g}',
                            "Mean":f'{ci.get("mean",0):.3g}',"Std":f'{ci.get("std",0):.3g}'})
            elif ci.get("top_values"):
                row["Top Values"]=", ".join(str(v) for v in ci["top_values"][:6])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    num_df=df.select_dtypes(include="number")
    if not num_df.empty:
        with st.expander("📊 Numeric Summary"):
            desc=num_df.describe().T.reset_index(); desc.columns=["Column"]+list(desc.columns[1:])
            st.dataframe(desc.round(4),hide_index=True,use_container_width=True)
    jm=st.session_state.get("join_metadata",{})
    if jm.get("joins"):
        with st.expander("🔗 Join Metadata"):
            st.json(jm)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _sidebar()
    st.markdown(
        '<div style="text-align:center;padding:1.2rem 0 0.6rem">'
        '<h1 style="font-size:2.4rem;font-weight:900;margin-bottom:0.15rem">'
        '<span class="gradient-text">🧠 GraphIntel</span></h1>'
        '<p style="color:#9a8a92;font-size:0.92rem;margin:0">'
        'Graph-Aware AI &nbsp;·&nbsp; Predictive Forecasting &nbsp;·&nbsp; '
        'Tabular Querying &nbsp;·&nbsp; LLM Analytics</p></div>',
        unsafe_allow_html=True,
    )
    chat_tab,fc_tab,ad_tab,met_tab,kg_tab,data_tab = st.tabs([
        "💬 Chat","📈 Forecasting","🔍 Anomaly Detection",
        "📊 Metrics Dashboard","🕸️ Knowledge Graph","🗄️ Data Explorer",
    ])
    with chat_tab:  _tab_chat()
    with fc_tab:    _tab_forecasting()
    with ad_tab:    _tab_anomaly()
    with met_tab:   _tab_metrics()
    with kg_tab:    _tab_kg()
    with data_tab:  _tab_explorer()


if __name__ == "__main__":
    main()
