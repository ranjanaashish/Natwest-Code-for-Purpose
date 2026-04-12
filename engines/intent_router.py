"""
Intent Router — Layer C
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IntentResult:
    intent: str
    confidence: float
    filters: List[Dict]     = field(default_factory=list)
    columns: List[str]      = field(default_factory=list)
    metrics: List[str]      = field(default_factory=list)
    aggregation: str        = ""
    group_by: List[str]     = field(default_factory=list)
    sort_by: str            = ""
    sort_asc: bool          = False
    limit: Optional[int]    = None
    time_periods: int       = 4
    raw_query: str          = ""


class IntentRouter:
    PATTERNS = {
        "forecast": [
            r"\bforecast\b", r"\bpredict\b", r"\bproject\b",
            r"\bnext\s+\d+\s+(day|week|month|quarter|year)",
            r"\btime.?series\b", r"\btrend\b",
        ],
        "anomaly": [
            r"\banomaly\b", r"\banomalies\b", r"\boutlier\b", r"\babnormal\b",
            r"\bunusual\b", r"\bspike\b", r"\bdetect\b",
        ],
        "scenario": [
            r"\bwhat.?if\b", r"\bscenario\b", r"\bsimulat\b",
            r"\bif\s+\w+\s+(increase|decrease|grow|drop|change)",
            r"\bimpact\s+of\b",
        ],
        "schema_query": [
            r"\bwhat\s+column", r"\bwhat\s+field", r"\bschema\b", r"\bstructure\b",
            r"\bdataset.{0,10}have\b", r"\bwhat\s+data\b", r"\blist\s+column",
            r"\bdescribe\s+the\s+data", r"\bwhat\s+is\s+in\b",
        ],
        "tabular_query": [
            r"\bshow\b", r"\blist\b", r"\bfetch\b", r"\bget\b",
            r"\btop\s+\d+\b", r"\bfilter\b", r"\bwhere\b",
            r"\bgroup\s+by\b", r"\bsum\b", r"\btotal\b",
            r"\baverage\b", r"\bavg\b", r"\bcount\b", r"\bmean\b",
            r"\bhighest\b", r"\blowest\b", r"\bsort\b", r"\brank\b",
            r"\baggregate\b", r"\bbreakdown\b", r"\bby\s+\w+\b",
            r"\bper\s+\w+\b", r"\bfrom\s+\w+\b", r"\bin\s+\w+\b",
            r"\bfor\s+\w+\b", r"\bwith\s+\w+\b",
        ],
    }

    AGG_MAP = {
        "sum": "sum", "total": "sum", "add": "sum", "summ": "sum",
        "average": "mean", "avg": "mean", "mean": "mean",
        "count": "count", "how many": "count", "number of": "count",
        "max": "max", "maximum": "max", "highest": "max", "largest": "max",
        "min": "min", "minimum": "min", "lowest": "min", "smallest": "min",
        "median": "median",
    }

    NUMBER_WORDS = {
        "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
        "seven":7,"eight":8,"nine":9,"ten":10,"twenty":20,"fifty":50,"hundred":100,
    }

    BUSINESS_TERMS = {
        "revenue": "price",
        "sales": "price",
        "cost": "price",
        "margin": "price",
        "seller": "seller_id",
        "buyer": "customer_id",
        "customer": "customer_id",
        "user": "customer_id",
        "order": "order_id"
    }

    def __init__(self, schema_metadata: Dict = None):
        self.schema_metadata = schema_metadata or {}
        self._all_cols: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._dt_cols:  List[str] = []
        self._update_cols()

    def update_schema(self, schema_metadata: Dict):
        self.schema_metadata = schema_metadata
        self._update_cols()

    def _update_cols(self):
        self._all_cols = []; self._num_cols = []; self._cat_cols = []; self._dt_cols = []
        for info in self.schema_metadata.values():
            self._all_cols.extend(info.get("columns", {}).keys())
            self._num_cols.extend(info.get("numeric_columns", []))
            self._cat_cols.extend(info.get("categorical_columns", []))
            self._dt_cols.extend(info.get("datetime_columns", []))
        self._all_cols  = list(dict.fromkeys(self._all_cols))
        self._num_cols  = list(dict.fromkeys(self._num_cols))
        self._cat_cols  = list(dict.fromkeys(self._cat_cols))
        self._dt_cols   = list(dict.fromkeys(self._dt_cols))

    # -----------------------------------------------------------------------
    # Fuzzy column finder — used by both filter extraction and tabular engine
    # -----------------------------------------------------------------------
    def find_column(self, word: str, candidates: List[str] = None) -> Optional[str]:
        """
        Find the best matching column name for a user-supplied word.
        Priority: exact → contains → contained-by → remove underscores match.
        """
        cols = candidates or self._all_cols
        w = word.lower().replace(" ", "_").replace("-", "_")
        # Exact
        for c in cols:
            if c.lower() == w: return c
        # Contains  
        for c in cols:
            if w in c.lower(): return c
        # Contained-by
        for c in cols:
            if c.lower() in w: return c
        # Token overlap (any word in the query matches a word in the col name)
        w_tokens = set(re.split(r"[_\s\-]+", w))
        for c in cols:
            c_tokens = set(re.split(r"[_\s\-]+", c.lower()))
            if w_tokens & c_tokens: return c
        return None

    def _detect_hard_ranking(self, original_q: str, mapped_q: str) -> Optional[Dict]:
        """Detect deterministic queries like 'which seller has highest revenue'"""
        patterns = [
            r"(?:highest|top|best|largest|most)\s+(\w[\w\s]*?)\s+(?:by|for|of)?\s*(\w+)",
            r"which\s+(\w+)\s+has\s+(?:the\s+)?(?:highest|most|top)\s+(\w+)"
        ]
        q_options = [mapped_q, original_q]
        for q_src in q_options:
            for pat in patterns:
                for m in re.finditer(pat, q_src):
                    part1, part2 = m.group(1).strip(), m.group(2).strip()
                    col1 = self.find_column(part1) or self.find_column(part1.split()[-1])
                    col2 = self.find_column(part2) or self.find_column(part2.split()[-1])
                    
                    if col1 and col2:
                        is_num1 = col1 in self._num_cols
                        is_num2 = col2 in self._num_cols
                        
                        metric = col1 if is_num1 else col2
                        entity = col2 if is_num1 else col1
                        
                        if metric and entity and metric != entity and metric in self._num_cols:
                            limit_val = self._extract_limit(mapped_q) or 1
                            if limit_val is None and re.search(r"highest|most|best", mapped_q):
                                limit_val = 1
                            return {
                                "group_by": [entity],
                                "aggregation": "sum",
                                "metrics": [metric],
                                "sort_by": metric,
                                "sort_asc": False,
                                "limit": limit_val
                            }
        return None

    # -----------------------------------------------------------------------
    # Intent scoring
    # -----------------------------------------------------------------------
    def route(self, query: str) -> IntentResult:
        q_orig = query.lower().strip()
        q = q_orig
        
        # Apply business terms mapping unconditionally to q
        for term, mapped in self.BUSINESS_TERMS.items():
            q = re.sub(rf"\b{term}\b", mapped, q)

        scores: Dict[str, float] = {}
        for intent, patterns in self.PATTERNS.items():
            scores[intent] = sum(1.5 if re.search(p, q) else 0 for p in patterns)
        best = max(scores, key=scores.get)
        best_score = scores[best]
        if best_score == 0:
            best, best_score = "general_qa", 0.1
        confidence = min(best_score / (len(self.PATTERNS.get(best, [""])) * 1.5), 1.0)
        
        # Hard ranking verification
        rank_override = self._detect_hard_ranking(q_orig, q)
        if rank_override:
            # Strip out any filters that accidentally overlap with our ranking entities
            # This prevents aggressive regex from producing things like "seller_id == 'top 10 sellers'"
            raw_filters = self._extract_filters(q)
            safe_filters = [
                f for f in raw_filters 
                if f["column"] not in rank_override["group_by"] 
                and f["column"] not in rank_override["metrics"]
            ]
            
            return IntentResult(
                intent="tabular_query",
                confidence=1.0,
                filters=safe_filters,
                columns=self._extract_columns(q),
                metrics=rank_override["metrics"],
                aggregation=rank_override["aggregation"],
                group_by=rank_override["group_by"],
                sort_by=rank_override["sort_by"],
                sort_asc=rank_override["sort_asc"],
                limit=rank_override["limit"],
                time_periods=self._extract_periods(q),
                raw_query=query,
            )

        return IntentResult(
            intent=best, confidence=round(confidence, 3),
            filters=self._extract_filters(q),
            columns=self._extract_columns(q),
            metrics=self._extract_metrics(q),
            aggregation=self._extract_aggregation(q),
            group_by=self._extract_group_by(q),
            sort_by=self._extract_sort(q),
            sort_asc=bool(re.search(r"\b(asc|ascending|lowest|smallest|least)\b", q)),
            limit=self._extract_limit(q),
            time_periods=self._extract_periods(q),
            raw_query=query,
        )

    # -----------------------------------------------------------------------
    # Filter extraction — comprehensive NLP handling
    # -----------------------------------------------------------------------
    def _extract_filters(self, q: str) -> List[Dict]:
        filters = []
        seen_cols = set()

        # ── 1. Explicit operator patterns: col > val, col == val, etc. ──────
        op_patterns = [
            (r"\b(\w+)\s*(?:is|=|==|equals?)\s*['\"]?(\w[\w\s]*?)['\"]?(?:\s+(?:and|or|\.|,)|$)", "=="),
            (r"\b(\w+)\s*>\s*(\d+(?:\.\d+)?)", ">"),
            (r"\b(\w+)\s*<\s*(\d+(?:\.\d+)?)", "<"),
            (r"\b(\w+)\s*>=\s*(\d+(?:\.\d+)?)", ">="),
            (r"\b(\w+)\s*<=\s*(\d+(?:\.\d+)?)", "<="),
            (r"\b(\w+)\s*(?:greater than|more than|above|over)\s*(\d+(?:\.\d+)?)", ">"),
            (r"\b(\w+)\s*(?:less than|below|under)\s*(\d+(?:\.\d+)?)", "<"),
            (r"\b(\w+)\s*(?:at least|minimum of)\s*(\d+(?:\.\d+)?)", ">="),
            (r"\b(\w+)\s*(?:at most|maximum of)\s*(\d+(?:\.\d+)?)", "<="),
        ]
        for pattern, op in op_patterns:
            for m in re.finditer(pattern, q):
                raw_col, val = m.group(1).strip(), m.group(2).strip()
                col = self.find_column(raw_col)
                if col and col not in seen_cols:
                    try: val = float(val)
                    except ValueError: pass
                    filters.append({"column": col, "op": op, "value": val})
                    seen_cols.add(col)

        # ── 2. "from/in/for/at STATE_CODE" — 2-letter uppercase codes ───────
        state_m = re.search(
            r'\b(?:from|in|at|for|of)\s+(?:the\s+)?(?:state\s+(?:of\s+)?)?([A-Z]{2})\b', q.upper()
        )
        if state_m:
            code = state_m.group(1).upper()
            # look for a "state" or "region" column
            state_col = (self.find_column("state") or self.find_column("region")
                         or self.find_column("city") or self.find_column("country"))
            if state_col and state_col not in seen_cols:
                filters.append({"column": state_col, "op": "==", "value": code})
                seen_cols.add(state_col)

        # ── 3. "from/in/for/at WORD" where WORD matches a category value ────
        #    e.g. "sales in Electronics", "orders from North region"
        prep_pattern = re.finditer(
            r'\b(?:from|in|at|for|of|within)\s+(?:the\s+)?([A-Za-z][\w\s]{1,30}?)(?:\s+(?:region|state|city|category|department|type|channel|class|area|segment))?(?=\s|$|,|\.)',
            q
        )
        for m in prep_pattern:
            raw_val = m.group(1).strip()
            if len(raw_val.split()) > 4: continue   # too long, likely not a value
            # Try to find a categorical column whose values might match
            col = self._find_col_for_value(raw_val)
            if col and col not in seen_cols:
                filters.append({"column": col, "op": "==", "value": raw_val})
                seen_cols.add(col)

        # ── 4. "where COLUMN is VALUE" / "where COLUMN = VALUE" ─────────────
        where_m = re.finditer(
            r'\bwhere\s+(\w[\w\s]*?)\s+(?:is|=|equals?|==)\s+["\']?(\w[\w\s]*?)["\']?(?=\s|$|,|\.)',
            q
        )
        for m in where_m:
            raw_col, raw_val = m.group(1).strip(), m.group(2).strip()
            col = self.find_column(raw_col)
            if col and col not in seen_cols:
                try: val: Any = float(raw_val)
                except ValueError: val = raw_val
                filters.append({"column": col, "op": "==", "value": val})
                seen_cols.add(col)

        # ── 5. Numeric threshold: "revenue > 1000", "price above 500" ────────
        num_thresh = re.finditer(
            r'\b(\w+)\s+(?:above|below|over|under|greater than|less than|'
            r'more than|exceeds?|below)\s+(\d[\d,]*(?:\.\d+)?)', q
        )
        for m in num_thresh:
            raw_col = m.group(1); val_str = m.group(2).replace(",", "")
            col = self.find_column(raw_col, self._num_cols)
            op_word = m.group(0).split()[1]
            op = ">" if op_word in ("above","over","greater","more","exceeds","exceed") else "<"
            if col and col not in seen_cols:
                filters.append({"column": col, "op": op, "value": float(val_str)})
                seen_cols.add(col)

        return filters

    def _find_col_for_value(self, value: str) -> Optional[str]:
        """Heuristic: find which categorical column likely has 'value' as a possible entry."""
        v_lower = value.lower().strip()
        # 2-letter uppercase → state-like column
        if re.match(r'^[A-Za-z]{2}$', value):
            c = self.find_column("state") or self.find_column("region")
            if c: return c
        # Try to match against known categorical columns
        for col in self._cat_cols:
            # Check if any part of the col name appears in the query around this value
            col_words = re.split(r"[_\-\s]+", col.lower())
            if any(w in v_lower or v_lower in w for w in col_words): return col
        return None

    # -----------------------------------------------------------------------
    # Other extractions
    # -----------------------------------------------------------------------
    def _extract_columns(self, q: str) -> List[str]:
        return [c for c in self._all_cols if c.lower().replace("_"," ") in q
                or c.lower() in q]

    def _extract_metrics(self, q: str) -> List[str]:
        found = []
        for c in self._num_cols:
            clean = c.lower().replace("_", " ")
            if clean in q or c.lower() in q:
                found.append(c)
        return found

    def _extract_aggregation(self, q: str) -> str:
        # Multi-word patterns first
        for phrase, agg in [("how many","count"),("number of","count"),
                             ("how much","sum"),("total of","sum")]:
            if phrase in q: return agg
        for word, agg in self.AGG_MAP.items():
            if re.search(rf"\b{re.escape(word)}\b", q):
                return agg
        return ""

    def _extract_group_by(self, q: str) -> List[str]:
        patterns = [
            r"(?:group\s+by|grouped\s+by|per|for\s+each|by)\s+([\w\s]+?)(?=\s+(?:and|,|$|\.|\?))",
            r"(?:breakdown|break\s+down)\s+by\s+(\w+)",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                raw = m.group(1).strip()
                col = self.find_column(raw)
                if col: return [col]
        return []

    def _extract_sort(self, q: str) -> str:
        m = re.search(r"(?:sort|order|rank|ranked)\s+(?:by\s+)?(\w+)", q)
        if m:
            col = self.find_column(m.group(1))
            if col: return col
        return ""

    def _extract_limit(self, q: str) -> Optional[int]:
        m = re.search(r"top\s+(\d+|" + "|".join(self.NUMBER_WORDS.keys()) + r")\b", q)
        if m:
            raw = m.group(1)
            return self.NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else None)
        return None

    def _extract_periods(self, q: str) -> int:
        m = re.search(
            r"(\d+|" + "|".join(self.NUMBER_WORDS.keys()) + r")\s+"
            r"(day|week|month|quarter|year|period)", q
        )
        if m:
            raw = m.group(1)
            return self.NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else 4)
        return 4
