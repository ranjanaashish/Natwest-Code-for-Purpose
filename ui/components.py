"""UI Components — unique key on every st.plotly_chart to prevent duplicate IDs."""
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_TEMPLATE = "plotly_dark"
CORAL="#ec4764"; MAUVE="#895d6b"; MINT="#9bdeac"; WARNING="#f5a623"; BLUE="#60a5fa"
BG_PAPER="rgba(12,10,16,0)"; BG_CARD="rgba(27,22,32,0)"


def _uid() -> str: return uuid.uuid4().hex[:10]


def _fig_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=BG_PAPER, plot_bgcolor=BG_CARD,
                      font=dict(family="Inter, sans-serif", color="#f4eff2", size=12),
                      margin=dict(l=30, r=25, t=44, b=30),
                      legend=dict(bgcolor="rgba(27,22,32,0.85)",bordercolor="rgba(137,93,107,0.35)",borderwidth=1),
                      title_font=dict(size=15, color="#f4eff2"))
    fig.update_xaxes(gridcolor="rgba(137,93,107,0.12)", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(137,93,107,0.12)", showgrid=True, zeroline=False)
    return fig


def render_metrics_panel(metrics: Dict[str,Any], cols: int = 4) -> None:
    if not metrics: return
    items = [(k,v) for k,v in metrics.items() if v is not None]
    col_objs = st.columns(min(len(items),cols))
    for i,(key,val) in enumerate(items):
        with col_objs[i%len(col_objs)]:
            st.metric(key, f"{val:.3f}" if isinstance(val,float) else str(val))


def render_table(df: pd.DataFrame, title: str = "Result", key_suffix: str = "") -> None:
    if df is None or df.empty: st.info("No data to display."); return
    st.markdown(f"<small style='color:#9a8a92'>**{title}** &nbsp;·&nbsp; "
                f"{len(df):,} row(s) × {len(df.columns)} col(s)</small>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(500, 44+len(df)*35))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "result.csv", "text/csv", key=f"dl_{key_suffix or _uid()}")


def render_forecast_chart(history_df, forecast_df, date_col, value_col, chart_key=""):
    fig = go.Figure()
    if history_df is not None and not history_df.empty and date_col in history_df.columns:
        fig.add_trace(go.Scatter(x=history_df[date_col], y=history_df["actual"],
                                  name="Historical", line=dict(color=CORAL, width=2.5), mode="lines"))
        fig.add_trace(go.Scatter(x=history_df[date_col], y=history_df["fitted"],
                                  name="Fitted", line=dict(color=MAUVE, width=1.5, dash="dot"), mode="lines"))
    if forecast_df is not None and not forecast_df.empty and date_col in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df[date_col], forecast_df[date_col][::-1]]),
            y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(155,222,172,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
        fig.add_trace(go.Scatter(x=forecast_df[date_col], y=forecast_df["yhat"],
                                  name="Forecast", line=dict(color=MINT, width=2.5, dash="dash"),
                                  mode="lines+markers", marker=dict(size=7, color=MINT)))
    fig.update_layout(title=f"📈 Forecast: {value_col}", xaxis_title=date_col, yaxis_title=value_col)
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"fc_{chart_key or _uid()}")


def render_anomaly_chart(df, anomaly_df, date_col, value_col, chart_key=""):
    if value_col not in df.columns: st.warning(f"Column '{value_col}' not found."); return
    fig = go.Figure()
    x_col = date_col if (date_col and date_col in df.columns) else None
    
    # Protect Streamlit/DOM from freezing by subsampling 'Normal' points if dataset > 1500.
    df_render = df
    if len(df) > 1500:
        anom_idx = set(anomaly_df.index) if not anomaly_df.empty else set()
        is_normal = ~df.index.isin(anom_idx)
        df_norm = df[is_normal].sample(n=min(1500, is_normal.sum()), random_state=42)
        df_render = pd.concat([df_norm, df[~is_normal]]).sort_index()
        
    x_all  = df_render[x_col]  if x_col else df_render.index
    x_anom = (anomaly_df[x_col] if (not anomaly_df.empty and x_col and x_col in anomaly_df.columns)
              else anomaly_df.index if not anomaly_df.empty else pd.Index([]))
    fig.add_trace(go.Scatter(x=x_all, y=df_render[value_col], name="Normal",
                              mode="lines+markers", marker=dict(color=MAUVE, size=4, opacity=0.7),
                              line=dict(color=MAUVE, width=1.8)))
    if not anomaly_df.empty and value_col in anomaly_df.columns:
        fig.add_trace(go.Scatter(x=x_anom, y=anomaly_df[value_col], name="Anomaly",
                                  mode="markers", marker=dict(color=CORAL, size=13, symbol="x-open",
                                                              line=dict(width=2.5, color=WARNING))))
        mv = df[value_col].mean(); sv = df[value_col].std()
        for label, yv in [("Upper 2.5σ", mv+2.5*sv), ("Lower 2.5σ", mv-2.5*sv)]:
            fig.add_hline(y=yv, line_dash="dash", line_color="rgba(245,166,35,0.45)", line_width=1.5,
                          annotation_text=label, annotation_font_color=WARNING)
    fig.update_layout(title=f"🔍 Anomaly: {value_col}", xaxis_title=x_col or "Index", yaxis_title=value_col)
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"ad_{chart_key or _uid()}")


def render_scenario_chart(scenario_df, date_col, value_col, chart_key=""):
    if scenario_df is None or scenario_df.empty: st.info("No scenario data."); return
    fig = go.Figure()
    if "baseline" in scenario_df.columns:
        fig.add_trace(go.Scatter(x=scenario_df[date_col], y=scenario_df["baseline"],
                                  name="Baseline", line=dict(color=BLUE, width=2, dash="dash"), mode="lines+markers"))
    if "scenario" in scenario_df.columns:
        fig.add_trace(go.Scatter(x=scenario_df[date_col], y=scenario_df["scenario"],
                                  name="Scenario", line=dict(color=MINT, width=2.5), mode="lines+markers",
                                  marker=dict(size=7, color=MINT)))
    if "delta" in scenario_df.columns:
        fig.add_trace(go.Bar(x=scenario_df[date_col], y=scenario_df["delta"], name="Δ",
                              marker_color=[MINT if v>=0 else CORAL for v in scenario_df["delta"]],
                              opacity=0.45, yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Delta", overlaying="y", side="right", showgrid=False, color=MAUVE))
    fig.update_layout(title=f"🔀 Scenario: {value_col}", xaxis_title=date_col, yaxis_title=value_col, barmode="overlay")
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"sc_{chart_key or _uid()}")


def render_anomaly_distribution(df, anomaly_df, value_col, chart_key=""):
    if value_col not in df.columns: return
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[value_col], name="All Values", marker_color=MAUVE, opacity=0.6, nbinsx=30))
    if not anomaly_df.empty and value_col in anomaly_df.columns:
        fig.add_trace(go.Histogram(x=anomaly_df[value_col], name="Anomalies", marker_color=CORAL, opacity=0.9, nbinsx=15))
    fig.update_layout(barmode="overlay", title=f"Distribution: {value_col}", xaxis_title=value_col, yaxis_title="Count")
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"dist_{chart_key or _uid()}")


def render_zscore_timeline(scored_df, value_col, date_col, threshold, chart_key=""):
    score_col = f"{value_col}_score"
    if score_col not in scored_df.columns: return
    
    # Subsample to prevent DOM lag on 100k+ rows
    df_render = scored_df
    if len(scored_df) > 1500:
        is_normal = scored_df[score_col] <= threshold
        df_norm = scored_df[is_normal].sample(n=min(1500, is_normal.sum()), random_state=42)
        df_render = pd.concat([df_norm, scored_df[~is_normal]]).sort_index()

    x = df_render[date_col] if (date_col and date_col in df_render.columns) else df_render.index
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=df_render[score_col], name="Score",
                          marker_color=[CORAL if v>threshold else MAUVE for v in df_render[score_col]], opacity=0.8))
    fig.add_hline(y=threshold, line_dash="dash", line_color=WARNING, line_width=1.5,
                  annotation_text=f"Threshold ({threshold})", annotation_font_color=WARNING)
    fig.update_layout(title=f"Score: {value_col}", xaxis_title="Date/Index", yaxis_title="|Score|")
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"zscore_{chart_key or _uid()}")


def render_residual_chart(history_df, date_col, value_col, chart_key=""):
    if history_df is None or history_df.empty: return
    residuals = history_df["actual"] - history_df["fitted"]
    x = history_df[date_col] if date_col in history_df.columns else history_df.index
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=residuals, name="Residuals",
                          marker_color=[MINT if v>=0 else CORAL for v in residuals], opacity=0.8))
    fig.add_hline(y=0, line_color=MAUVE, line_width=1.5)
    fig.update_layout(title="Residuals (Actual − Fitted)", xaxis_title=date_col, yaxis_title="Residual")
    _fig_defaults(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"resid_{chart_key or _uid()}")


def render_knowledge_graph(graph, chart_key=""):
    if graph is None or graph.number_of_nodes() == 0:
        st.markdown('<div class="info-box">Upload a dataset to build the knowledge graph.</div>', unsafe_allow_html=True); return
    try:
        import networkx as nx
        pos = nx.spring_layout(graph, seed=42, k=0.9)
        tc = {"dataset":CORAL,"column":MAUVE,"metric":MINT,"entity":WARNING,"datatype":BLUE,"analysis":"#c084fc"}
        ts = {"dataset":22,"column":13,"metric":15,"entity":10,"datatype":9,"analysis":10}
        nxl,nyl,texts,colors,sizes,hover=[],[],[],[],[],[]
        for node,data in graph.nodes(data=True):
            x,y = pos.get(node,(0,0)); nxl.append(x); nyl.append(y)
            label = data.get("label", data.get("name", node))
            texts.append(label[:22]); ntype = data.get("type","unknown")
            colors.append(tc.get(ntype,"#888")); sizes.append(ts.get(ntype,10))
            hover.append(f"<b>{label}</b><br>Type: {ntype}")
        ex,ey=[],[]
        for u,v in graph.edges():
            xu,yu=pos.get(u,(0,0)); xv,yv=pos.get(v,(0,0))
            ex.extend([xu,xv,None]); ey.extend([yu,yv,None])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(color="rgba(137,93,107,0.28)",width=1),hoverinfo="none",showlegend=False))
        fig.add_trace(go.Scatter(x=nxl,y=nyl,mode="markers+text",text=texts,textposition="top center",
                                  textfont=dict(size=8,color="#c8bec4"),
                                  marker=dict(color=colors,size=sizes,line=dict(width=1.2,color="rgba(244,239,242,0.2)")),
                                  hovertext=hover,hoverinfo="text",showlegend=False))
        fig.update_layout(title="Knowledge Graph",showlegend=False,height=480,
                           xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                           yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True, key=f"kg_{chart_key or _uid()}")
        cols = st.columns(len(tc))
        for i,(t,c) in enumerate(tc.items()):
            with cols[i]:
                st.markdown(f'<span style="color:{c};font-size:1rem">●</span> <small style="color:#9a8a92">{t}</small>', unsafe_allow_html=True)
    except Exception as exc: st.warning(f"Graph error: {exc}")


def render_intent_badge(intent: str, confidence: float) -> None:
    pct = int(confidence*100)
    color = MINT if pct>60 else WARNING if pct>30 else CORAL
    label = intent.replace("_"," ").title()
    st.markdown(f'<span class="intent-badge" style="border-color:{color};color:{color}">⚡ {label} &nbsp;·&nbsp; {pct}% conf</span>', unsafe_allow_html=True)


def section_header(icon: str, title: str, tag: str = "") -> None:
    tag_html = f'<span class="panel-tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="panel-header"><h3>{icon} {title}</h3>{tag_html}</div>', unsafe_allow_html=True)
