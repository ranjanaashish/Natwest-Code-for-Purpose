"""Knowledge Graph Engine — Layer B"""
import hashlib, pickle
from pathlib import Path
from typing import Any, Dict, Optional
import networkx as nx
import pandas as pd


class KnowledgeGraphEngine:
    def __init__(self, cache_dir: str = "."):
        self.graph = nx.DiGraph()
        self._cache_dir  = Path(cache_dir)
        self._cache_file = self._cache_dir / "graph_cache.pkl"
        self._hash_file  = self._cache_dir / "graph_hash.txt"

    def load_or_build(self, datasets, schema_metadata, join_metadata, data_hash):
        cached_hash = self._hash_file.read_text().strip() if self._hash_file.exists() else ""
        if cached_hash == data_hash and self._cache_file.exists():
            try:
                with open(self._cache_file,"rb") as f: self.graph = pickle.load(f); return
            except Exception: pass
        self._build(datasets, schema_metadata, join_metadata)
        self._save_cache(data_hash)

    def _build(self, datasets, schema_metadata, join_metadata):
        self.graph = nx.DiGraph()
        for ds_name, info in schema_metadata.items():
            self.graph.add_node(ds_name, type="dataset", label=ds_name,
                                row_count=info.get("row_count",0))
            for col, ci in info.get("columns",{}).items():
                nid = f"{ds_name}.{col}"
                ntype = ("metric" if ci.get("is_numeric") else
                         "entity" if ci.get("is_datetime") else "column")
                self.graph.add_node(nid, type=ntype, label=col,
                                    dataset=ds_name, **{k:v for k,v in ci.items()
                                                        if not isinstance(v,list)})
                self.graph.add_edge(ds_name, nid, relation="has_column")
                dtype_node = f"dtype_{ci.get('dtype','?')}"
                self.graph.add_node(dtype_node, type="datatype", label=ci.get("dtype","?"))
                self.graph.add_edge(nid, dtype_node, relation="has_type")
        for join in join_metadata.get("joins",[]):
            for key in join.get("keys",[]):
                ln = f"{join['left']}.{key}"; rn = f"{join['right']}.{key}"
                if self.graph.has_node(ln) and self.graph.has_node(rn):
                    self.graph.add_edge(ln, rn, relation="join_key")

    def _save_cache(self, data_hash):
        try:
            with open(self._cache_file,"wb") as f: pickle.dump(self.graph,f)
            self._hash_file.write_text(data_hash)
        except Exception: pass

    def add_analysis_node(self, query, intent):
        nid = f"analysis_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        self.graph.add_node(nid, type="analysis", label=intent[:20], query=query[:80])

    def retrieve_context(self, query: str, schema_metadata: Dict) -> Dict:
        """
        KG-guided context retrieval: find columns most relevant to the query,
        and also return which datasets contain them — accelerates filter resolution.
        """
        q = query.lower()
        relevant_cols = []
        q_words = set(re.split(r'\W+', q)) if q else set()
        for ds_name, info in schema_metadata.items():
            for col, ci in info.get("columns",{}).items():
                col_words = set(re.split(r'[_\s\-]+', col.lower()))
                score = len(col_words & q_words)
                if col.lower() in q or score > 0:
                    relevant_cols.append({
                        "name": col, "dataset": ds_name, "score": score,
                        "is_numeric": ci.get("is_numeric"), "dtype": ci.get("dtype"),
                        "top_values": ci.get("top_values",[])[:10],
                    })
        relevant_cols.sort(key=lambda x: -x["score"])
        return {"relevant_columns": relevant_cols[:12]}

    def get_schema_text(self, schema_metadata: Dict) -> str:
        lines = []
        for ds_name, info in schema_metadata.items():
            lines.append(f"Dataset: {ds_name} ({info.get('row_count',0)} rows)")
            for col, ci in info.get("columns",{}).items():
                kind = "numeric" if ci.get("is_numeric") else "datetime" if ci.get("is_datetime") else "text"
                extra = ""
                if ci.get("is_numeric"):
                    extra = f" [min={ci.get('min',0):.2g}, max={ci.get('max',0):.2g}]"
                elif ci.get("top_values"):
                    vals = ", ".join(str(v) for v in ci["top_values"][:8])
                    extra = f" [values: {vals}]"
                lines.append(f"  - {col} ({kind}){extra}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        node_types: Dict[str,int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("type","unknown"); node_types[t] = node_types.get(t,0)+1
        edge_types: Dict[str,int] = {}
        for _,_,d in self.graph.edges(data=True):
            r = d.get("relation","unknown"); edge_types[r] = edge_types.get(r,0)+1
        return {"total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "node_types": node_types, "edge_types": edge_types}


import re  # needed for retrieve_context
