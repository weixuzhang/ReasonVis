#!/usr/bin/env python
"""Generate ReasonVis predictions for method-style NL2Vis baselines.

The existing full runs are mostly direct model prompting baselines. This script
adds approach baselines that keep the same output contract:

    Visualize [CHART_TYPE] [SQLite SQL]

Implemented methods:
- decomposed: infer chart intent and SQL in a structured NL2Vis plan.
- execution_repair: decomposed generation plus one or more SQL execution repair
  rounds when the predicted SQL is malformed or non-executable.
- nvagent_adapted: a ReasonVis adaptation of the nvAgent workflow
  (schema processor -> VQL composer -> execution validator).
- chat2vis_adapted: a ReasonVis adaptation of the Chat2VIS prompt-engineering
  system.
- lida_adapted: a ReasonVis adaptation of LIDA's summarizer/goal/visualization
  generator pipeline.
- prompt4vis_adapted: a ReasonVis adaptation of Prompt4Vis with leave-database
  example mining and schema filtering.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


CANONICAL_CHARTS = ["BAR", "STACKED BAR", "LINE", "SCATTER", "PIE", "HEATMAP"]

CHART_ALIASES = {
    "BAR CHART": "BAR",
    "BARCHART": "BAR",
    "COLUMN": "BAR",
    "COLUMN CHART": "BAR",
    "STACKED": "STACKED BAR",
    "STACKED_BAR": "STACKED BAR",
    "STACKED BAR CHART": "STACKED BAR",
    "LINE CHART": "LINE",
    "LINEGRAPH": "LINE",
    "LINE GRAPH": "LINE",
    "GROUPED LINE": "LINE",
    "PIE CHART": "PIE",
    "SCATTER PLOT": "SCATTER",
    "SCATTERPLOT": "SCATTER",
    "GROUPED SCATTER": "SCATTER",
    "HEAT MAP": "HEATMAP",
    "HEAT_MAP": "HEATMAP",
}

SYSTEM_PROMPT = """You are an expert system for database-grounded natural-language-to-visualization.

ReasonVis output format:
Visualize [CHART_TYPE] [SQL_QUERY]

Allowed chart types: BAR, STACKED BAR, LINE, SCATTER, PIE, HEATMAP.
SQL must be executable SQLite.
Use only tables and columns in the provided schema.
Do not use non-SQL VQL clauses such as BIN BY.
Do not explain unless the requested JSON schema asks for a short internal field.
"""


_thread_local = threading.local()
_schema_cache: Dict[Tuple[str, str, bool], str] = {}
_schema_lock = threading.Lock()


class LLMCallFailed(RuntimeError):
    """Raised when an LLM call fails after all retry attempts."""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def sorted_sequence_items(data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))


def read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def read_sample_rows(path: Path, limit: int = 2) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: row.get(key, "") for key in list(row.keys())[:6]})
            if len(rows) >= limit:
                break
    return rows


def extract_schema(db_id: str, database_path: Path, include_sample_rows: bool = False) -> str:
    db_dir = database_path / db_id
    if not db_dir.exists():
        return f"Database: {db_id}\nSchema unavailable."

    csv_files = sorted(p for p in db_dir.glob("*.csv") if p.name != "sqlite_sequence.csv")
    table_columns: Dict[str, List[str]] = {}
    lines = [f"Database: {db_id}", "Tables:"]

    for csv_path in csv_files:
        table = csv_path.stem
        columns = read_csv_header(csv_path)
        table_columns[table] = columns
        lines.append(f"- {table}({', '.join(columns)})")
        if include_sample_rows:
            rows = read_sample_rows(csv_path, limit=2)
            if rows:
                lines.append(f"  sample_rows: {json.dumps(rows, ensure_ascii=False)}")

    relationships = []
    table_items = list(table_columns.items())
    for left_table, left_cols in table_items:
        for right_table, right_cols in table_items:
            if left_table >= right_table:
                continue
            common = sorted(set(left_cols).intersection(right_cols))
            key_like = [
                col
                for col in common
                if col.lower() == "id" or col.lower().endswith("_id") or col.lower().endswith(" id")
            ]
            for col in key_like[:6]:
                relationships.append(f"- {left_table}.{col} = {right_table}.{col}")

    if relationships:
        lines.append("Potential joins:")
        lines.extend(relationships[:40])

    return "\n".join(lines)


def table_names_for_db(db_id: str, database_path: Path) -> List[str]:
    db_dir = database_path / db_id
    if not db_dir.exists():
        return []
    return sorted(p.stem for p in db_dir.glob("*.csv") if p.name != "sqlite_sequence.csv")


def filter_schema_text(schema: str, selected_tables: List[str]) -> str:
    selected = {table.lower() for table in selected_tables}
    if not selected:
        return schema
    lines = schema.splitlines()
    out: List[str] = []
    keep_sample = False
    for line in lines:
        table_match = re.match(r"^-\s+([^(]+)\(", line.strip())
        if table_match:
            table_name = table_match.group(1).strip().strip('"')
            keep_sample = table_name.lower() in selected
            if keep_sample:
                out.append(line)
            continue
        if line.startswith("  sample_rows:") and keep_sample:
            out.append(line)
            continue
        if line.startswith("Database:") or line == "Tables:":
            out.append(line)
            continue
        if line.startswith("- ") and "=" in line:
            lowered = line.lower()
            if any(f"{table}." in lowered for table in selected):
                out.append(line)
            continue
    return "\n".join(out) if len(out) > 2 else schema


def cached_schema(db_id: str, database_path: Path, include_sample_rows: bool) -> str:
    cache_key = (db_id, str(database_path), include_sample_rows)
    with _schema_lock:
        if cache_key in _schema_cache:
            return _schema_cache[cache_key]
    schema = extract_schema(db_id, database_path, include_sample_rows)
    with _schema_lock:
        _schema_cache[cache_key] = schema
    return schema


def setup_db(db_id: str, database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    db_dir = database_path / db_id
    for csv_path in sorted(db_dir.glob("*.csv")):
        if csv_path.name == "sqlite_sequence.csv":
            continue
        table_name = csv_path.stem
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            if not headers:
                continue
            columns = ", ".join([f'"{header}" TEXT' for header in headers])
            cursor.execute(f'CREATE TABLE "{table_name}" ({columns})')
            placeholders = ", ".join(["?"] * len(headers))
            for row in reader:
                row = (row + [""] * len(headers))[: len(headers)]
                cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', row)
    conn.commit()
    return conn


def get_db_connection(db_id: str, database_path: Path) -> sqlite3.Connection:
    cache_key = (str(database_path), db_id)
    if not hasattr(_thread_local, "db_connections"):
        _thread_local.db_connections = {}
    if cache_key not in _thread_local.db_connections:
        _thread_local.db_connections[cache_key] = setup_db(db_id, database_path)
    return _thread_local.db_connections[cache_key]


def execute_sql(db_id: str, sql: str, database_path: Path, timeout_seconds: float) -> Tuple[bool, str, int]:
    sql = (sql or "").strip()
    if not sql:
        return False, "missing SQL", 0
    try:
        conn = get_db_connection(db_id, database_path)
    except Exception as exc:
        return False, f"database setup failed: {exc}", 0

    start = time.monotonic()

    def interrupt_if_timed_out() -> int:
        return 1 if timeout_seconds and time.monotonic() - start > timeout_seconds else 0

    if timeout_seconds:
        conn.set_progress_handler(interrupt_if_timed_out, 10000)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return True, "", len(rows)
    except Exception as exc:
        return False, str(exc), 0
    finally:
        if timeout_seconds:
            conn.set_progress_handler(None, 0)


def gold_vql(turn: Dict[str, Any]) -> str:
    return turn.get("query", {}).get("vql") or turn.get("vis_query", {}).get("VQL", "")


def gold_sql(turn: Dict[str, Any]) -> str:
    return (
        turn.get("query", {}).get("canonical_sql")
        or turn.get("vis_query", {}).get("data_part", {}).get("sql_part")
        or ""
    )


def extract_sql_tables(sql: str) -> List[str]:
    tables = []
    for pattern in [
        r'\bFROM\s+"([^"]+)"',
        r'\bJOIN\s+"([^"]+)"',
        r"\bFROM\s+([A-Za-z_][\w]*)",
        r"\bJOIN\s+([A-Za-z_][\w]*)",
    ]:
        for table in re.findall(pattern, sql or "", flags=re.IGNORECASE):
            clean = table.strip().strip('"')
            if clean and clean.lower() not in {"select", "where"} and clean not in tables:
                tables.append(clean)
    return tables


def tokenize(text: str) -> set:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "by",
        "with",
        "show",
        "list",
        "give",
        "me",
        "each",
        "all",
        "in",
        "on",
        "as",
        "from",
        "that",
        "which",
    }
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", (text or "").lower()) if token not in stop}


def context_label(turn: Dict[str, Any]) -> str:
    context = turn.get("context", {})
    if isinstance(context, dict) and context.get("dependency_level"):
        return str(context["dependency_level"])
    dep = turn.get("context_dependency", ["independent"])
    if isinstance(dep, list):
        return str(dep[0]) if dep else "independent"
    return str(dep)


def build_example_bank(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    for seq_id, sequence in sorted_sequence_items(data):
        db_id = sequence.get("db_id", "")
        for turn in sequence.get("conversation", []):
            query = turn.get("nl_query", "")
            sql = gold_sql(turn)
            examples.append(
                {
                    "sequence_id": str(seq_id),
                    "db_id": db_id,
                    "query": query,
                    "tokens": tokenize(query),
                    "chart_type": turn.get("chart_type", ""),
                    "vql": gold_vql(turn),
                    "tables": extract_sql_tables(sql),
                    "context_dependency": context_label(turn),
                    "reasoning_level": turn.get("reasoning_level", ""),
                }
            )
    return examples


def get_example_bank(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not hasattr(args, "_example_bank"):
        args._example_bank = build_example_bank(load_json(Path(args.dataset)))
    return args._example_bank


def select_mined_examples(args: argparse.Namespace, current_query: str, db_id: str, limit: int) -> List[Dict[str, Any]]:
    query_tokens = tokenize(current_query)
    scored = []
    for idx, example in enumerate(get_example_bank(args)):
        if example["db_id"] == db_id:
            continue
        overlap = len(query_tokens & example["tokens"])
        union = len(query_tokens | example["tokens"]) or 1
        score = overlap / union
        if overlap:
            score += 0.02 * min(overlap, 5)
        scored.append((score, -idx, example))
    scored.sort(reverse=True)
    return [example for score, _, example in scored[:limit] if score > 0][:limit]


def format_prompt_examples(examples: List[Dict[str, Any]], include_tables: bool = False) -> str:
    if not examples:
        return "(none)"
    parts = []
    for idx, example in enumerate(examples, start=1):
        parts.append(f"Example {idx} question: {example['query']}")
        if include_tables:
            parts.append(f"Example {idx} target tables: {', '.join(example.get('tables') or [])}")
        parts.append(f"Example {idx} target VQL: {example['vql']}")
    return "\n".join(parts)


def normalize_chart(chart: str) -> str:
    clean = re.sub(r"[_-]+", " ", (chart or "").strip().upper())
    clean = re.sub(r"\s+", " ", clean)
    return CHART_ALIASES.get(clean, clean)


def clean_text_response(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|sql|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = clean_text_response(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_loose_json_generation(text: str) -> Tuple[str, str]:
    text = clean_text_response(text)
    chart_match = re.search(
        r'"(?:chart_type|chart|visualization_type)"\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    sql_match = re.search(r'"(?:sql|sql_query|query)"\s*:\s*"(.*)"\s*\}?\s*$', text, flags=re.IGNORECASE | re.DOTALL)
    if not (chart_match and sql_match):
        return "", ""
    sql = sql_match.group(1).strip()
    sql = re.sub(r'"\s*\}\s*$', "", sql, flags=re.DOTALL).strip()
    return chart_match.group(1), sql


def parse_vql(vql: str) -> Tuple[str, str]:
    text = clean_text_response(vql)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("visualize "):
            text = line
            break

    match = re.match(r"^Visualize\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", ""
    rest = re.sub(r"\s+", " ", match.group(1)).strip()
    for chart in sorted(CANONICAL_CHARTS + list(CHART_ALIASES), key=len, reverse=True):
        pattern = re.compile(rf"^{re.escape(chart)}\s+((?:SELECT|WITH)\b.+)$", re.IGNORECASE | re.DOTALL)
        type_match = pattern.match(rest)
        if type_match:
            return normalize_chart(chart), type_match.group(1).strip()
    generic_match = re.match(r"^(.+?)\s+((?:SELECT|WITH)\b.+)$", rest, flags=re.IGNORECASE | re.DOTALL)
    if generic_match:
        return normalize_chart(generic_match.group(1)), generic_match.group(2).strip()
    return normalize_chart(rest), ""


def strip_disallowed_vql_clauses(sql: str) -> str:
    sql = re.sub(r"\s+BIN\s+.+?\s+BY\s+\w+\s*$", "", sql or "", flags=re.IGNORECASE | re.DOTALL)
    sql = re.sub(r";+\s*$", "", sql).strip()
    return sql


def vql_from_chart_sql(chart: str, sql: str) -> str:
    chart = normalize_chart(chart)
    sql = strip_disallowed_vql_clauses(sql)
    if chart not in CANONICAL_CHARTS:
        chart = "BAR"
    if not re.match(r"^(SELECT|WITH)\b", sql or "", flags=re.IGNORECASE):
        return ""
    return f"Visualize {chart} {sql}"


def normalize_generation(raw: str) -> Tuple[str, Dict[str, Any]]:
    obj = extract_json_object(raw)
    if obj:
        chart = obj.get("chart_type") or obj.get("chart") or obj.get("visualization_type") or ""
        sql = obj.get("sql") or obj.get("sql_query") or obj.get("query") or ""
        vql = obj.get("vql") or obj.get("VQL") or ""
        if not (chart and sql) and vql:
            chart, sql = parse_vql(vql)
        normalized = vql_from_chart_sql(str(chart), str(sql))
        return normalized, {"parsed": obj, "parse_mode": "json"}

    chart, sql = extract_loose_json_generation(raw)
    if chart and sql:
        normalized = vql_from_chart_sql(chart, sql)
        return normalized, {"parse_mode": "loose_json", "chart": chart, "sql": sql}

    chart, sql = parse_vql(raw)
    normalized = vql_from_chart_sql(chart, sql)
    return normalized, {"parse_mode": "vql", "chart": chart, "sql": sql}


def build_history(
    conversation: List[Dict[str, Any]],
    pred_turns: List[Dict[str, Any]],
    turn_index: int,
    history_mode: str,
) -> List[Tuple[str, str]]:
    if history_mode == "none":
        return []
    history: List[Tuple[str, str]] = []
    for prev_idx, prev_turn in enumerate(conversation[:turn_index]):
        prev_query = prev_turn.get("nl_query", "")
        if history_mode == "oracle":
            prev_vql = gold_vql(prev_turn)
        else:
            prev_vql = pred_turns[prev_idx].get("vql", "") if prev_idx < len(pred_turns) else ""
        history.append((prev_query, prev_vql))
    return history


def format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    parts = []
    for idx, (query, vql) in enumerate(history, start=1):
        parts.append(f"Turn {idx} user: {query}")
        parts.append(f"Turn {idx} previous VQL: {vql}")
    return "\n".join(parts)


def get_thread_client(args: argparse.Namespace) -> OpenAI:
    client_key = (args.api_key_env, args.base_url or "", args.request_timeout)
    if not hasattr(_thread_local, "client") or getattr(_thread_local, "client_key", None) != client_key:
        client_kwargs: Dict[str, Any] = {
            "api_key": os.environ.get(args.api_key_env),
            "timeout": args.request_timeout,
        }
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
            client_kwargs["default_headers"] = {
                "HTTP-Referer": args.openrouter_referer,
                "X-Title": args.openrouter_title,
            }
        _thread_local.client = OpenAI(**client_kwargs)
        _thread_local.client_key = client_key
    return _thread_local.client


def call_llm(client: OpenAI, args: argparse.Namespace, prompt: str, temperature: Optional[float] = None) -> str:
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature if temperature is None else temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content or ""


def call_with_retries(client: OpenAI, args: argparse.Namespace, prompt: str) -> Tuple[str, Optional[str]]:
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            return call_llm(client, args, prompt), None
        except Exception as exc:
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (2**attempt))
    raise LLMCallFailed(f"LLM call failed after {args.retries + 1} attempts: {last_error}")


def decomposed_prompt(db_id: str, schema: str, current_query: str, history: List[Tuple[str, str]]) -> str:
    return f"""Run a decomposed NL2Vis method for ReasonVis.

Step 1: infer the user's analytical intent and the best chart type.
Step 2: write one executable SQLite query that returns the data needed by the chart.

Allowed chart_type values: {", ".join(CANONICAL_CHARTS)}.
Return JSON only:
{{"chart_type": "...", "sql": "..."}}

Database id: {db_id}
Schema:
{schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def chat2vis_prompt(db_id: str, schema: str, current_query: str, history: List[Tuple[str, str]]) -> str:
    return f"""Run a Chat2VIS-style NL2Vis baseline adapted to ReasonVis.

Chat2VIS uses prompt engineering to translate natural language requests and table descriptions into visualization code/specifications. For this benchmark, produce the equivalent ReasonVis VQL components.

Rules:
- Return JSON only: {{"chart_type": "...", "sql": "..."}}
- Allowed chart_type values: {", ".join(CANONICAL_CHARTS)}.
- SQL must be executable SQLite over the provided schema.
- Use the conversation history only when the current query refers to prior turns.
- Prefer simple, direct SQL and avoid unsupported visualization clauses.

Database id: {db_id}
Table and column description:
{schema}

Conversation history:
{format_history(history)}

Current user request:
{current_query}
"""


def lida_summary_goal_prompt(db_id: str, schema: str, current_query: str, history: List[Tuple[str, str]]) -> str:
    return f"""Run the Summarizer and Goal Explorer stages of a LIDA-style visualization pipeline adapted to ReasonVis.

LIDA summarizes the dataset, identifies visualization goals, then generates visualization code. For ReasonVis, summarize the schema and produce a concrete SQL-oriented visualization goal.

Return JSON only:
{{
  "data_summary": "compact summary of relevant tables, columns, and relationships",
  "visualization_goal": "what the chart should communicate",
  "likely_tables": ["table", "..."],
  "likely_fields": ["table.column or column", "..."],
  "chart_type": "BAR|STACKED BAR|LINE|SCATTER|PIE|HEATMAP",
  "context_carryover": "needed previous-turn constraints, or none"
}}

Database id: {db_id}
Schema:
{schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def lida_generator_prompt(
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    summary_goal: Dict[str, Any],
) -> str:
    return f"""Run the VisGenerator stage of a LIDA-style visualization pipeline adapted to ReasonVis.

Use the dataset summary and visualization goal to generate a ReasonVis VQL.
Return JSON only:
{{"chart_type": "...", "sql": "..."}}

Requirements:
- Allowed chart_type values: {", ".join(CANONICAL_CHARTS)}.
- SQL must be executable SQLite using only the provided schema.
- The SQL result should contain the fields needed for the chart.
- For STACKED BAR, return x, y, and stack/group columns.
- For HEATMAP, return two categorical dimensions and one numeric value.
- For SCATTER, return two numeric/measure columns when possible.

Database id: {db_id}
Schema:
{schema}

LIDA summary and goal:
{json.dumps(summary_goal, ensure_ascii=False)}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def prompt4vis_schema_filter_prompt(
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    examples: List[Dict[str, Any]],
    table_names: List[str],
) -> str:
    example_text = []
    for idx, example in enumerate(examples, start=1):
        example_text.append(f"Example {idx} question: {example['query']}")
        example_text.append(f"Example {idx} target tables: {', '.join(example.get('tables') or [])}")
    return f"""Run the schema-filtering stage of Prompt4Vis adapted to ReasonVis.

Prompt4Vis filters the database schema before generating a visualization query. Select the minimal useful table set for the current query, preserving tables needed for joins and context carryover.

Return JSON only:
{{"tables": ["table", "..."], "rationale": "brief"}}

Few-shot schema-filter examples from other databases:
{chr(10).join(example_text) if example_text else "(none)"}

Database id: {db_id}
Available tables:
{', '.join(table_names)}

Full schema:
{schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def prompt4vis_generation_prompt(
    db_id: str,
    filtered_schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    examples: List[Dict[str, Any]],
    selected_tables: List[str],
) -> str:
    return f"""Run Prompt4Vis adapted to ReasonVis.

Prompt4Vis uses in-context examples plus a filtered schema to generate a data visualization query. The examples below are mined from other databases to avoid same-database leakage.

Return JSON only:
{{"chart_type": "...", "sql": "..."}}

Allowed chart_type values: {", ".join(CANONICAL_CHARTS)}.
SQL must be executable SQLite over the filtered schema. If the filtered schema is insufficient, use only tables/columns that appear in the full table descriptions shown here.

Mined examples:
{format_prompt_examples(examples, include_tables=False)}

Database id: {db_id}
Selected tables: {', '.join(selected_tables) if selected_tables else '(none)'}
Filtered schema:
{filtered_schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def processor_prompt(db_id: str, schema: str, current_query: str, history: List[Tuple[str, str]]) -> str:
    return f"""You are the Processor agent in an nvAgent-style NL2Vis workflow.

Analyze the query and conversation context. Filter the schema to the tables and columns likely needed, identify possible joins, and state whether the query is single-table or multi-table.

Return JSON only:
{{
  "relevant_schema": {{"table": ["column", "..."]}},
  "relationships": ["table_a.column = table_b.column"],
  "context_carryover": "short description of any needed previous-turn constraints",
  "difficulty": "SINGLE or MULTIPLE",
  "augmented_explanation": "brief schema and reasoning summary"
}}

Database id: {db_id}
Full schema:
{schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def composer_prompt(
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    processor_result: Dict[str, Any],
) -> str:
    return f"""You are the Composer agent in an nvAgent-style NL2Vis workflow.

Use the Processor output to compose the final ReasonVis VQL.
The VQL must be:
Visualize [CHART_TYPE] [SQLite SQL]

Allowed chart types: {", ".join(CANONICAL_CHARTS)}.
Do not use BIN BY or any non-SQL clauses. Use SQLite expressions when dates or bins are required.
For STACKED BAR, return three useful columns: x value, y value, and stack/group column.
For other charts, return at least two useful columns.

Return JSON only:
{{"chart_type": "...", "sql": "..."}}

Database id: {db_id}
Full schema:
{schema}

Processor output:
{json.dumps(processor_result, ensure_ascii=False)}

Conversation history:
{format_history(history)}

Current user query:
{current_query}
"""


def repair_prompt(
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    current_vql: str,
    error: str,
) -> str:
    return f"""You are the Validator/Repair agent for a ReasonVis NL2Vis workflow.

The current VQL has malformed or non-executable SQLite SQL. Fix it while preserving the user's intent and chart type whenever reasonable.

Return JSON only:
{{"chart_type": "...", "sql": "..."}}

Database id: {db_id}
Schema:
{schema}

Conversation history:
{format_history(history)}

Current user query:
{current_query}

Current VQL:
{current_vql}

SQLite error:
{error}
"""


def run_decomposed(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    raw, error = call_with_retries(client, args, decomposed_prompt(db_id, schema, current_query, history))
    vql, parse_info = normalize_generation(raw)
    return vql, {"method_trace": {"raw_generation": raw, "error": error, **parse_info}}


def run_chat2vis_adapted(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    raw, error = call_with_retries(client, args, chat2vis_prompt(db_id, schema, current_query, history))
    vql, parse_info = normalize_generation(raw)
    return vql, {"method_trace": {"raw_generation": raw, "error": error, **parse_info}}


def run_lida_adapted(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    summary_raw, summary_error = call_with_retries(
        client, args, lida_summary_goal_prompt(db_id, schema, current_query, history)
    )
    summary_goal = extract_json_object(summary_raw) or {
        "data_summary": summary_raw[:1200],
        "visualization_goal": current_query,
        "likely_tables": [],
        "likely_fields": [],
        "chart_type": "",
        "context_carryover": "",
    }
    generator_raw, generator_error = call_with_retries(
        client, args, lida_generator_prompt(db_id, schema, current_query, history, summary_goal)
    )
    vql, parse_info = normalize_generation(generator_raw)
    return vql, {
        "method_trace": {
            "summary_raw": summary_raw,
            "summary_error": summary_error,
            "summary_goal": summary_goal,
            "generator_raw": generator_raw,
            "generator_error": generator_error,
            "generator_parse": parse_info,
        }
    }


def normalize_selected_tables(raw_tables: Any, table_names: List[str]) -> List[str]:
    valid = {table.lower(): table for table in table_names}
    out = []
    if isinstance(raw_tables, str):
        raw_iter = re.split(r"[,;\n]+", raw_tables)
    elif isinstance(raw_tables, list):
        raw_iter = raw_tables
    else:
        raw_iter = []
    for item in raw_iter:
        clean = str(item).strip().strip('"`')
        if clean.lower() in valid and valid[clean.lower()] not in out:
            out.append(valid[clean.lower()])
    return out


def run_prompt4vis_adapted(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    database_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    mined_examples = select_mined_examples(args, current_query, db_id, args.example_count)
    filter_examples = mined_examples[: args.schema_filter_examples]
    table_names = table_names_for_db(db_id, database_path)
    filter_raw, filter_error = call_with_retries(
        client,
        args,
        prompt4vis_schema_filter_prompt(db_id, schema, current_query, history, filter_examples, table_names),
    )
    filter_obj = extract_json_object(filter_raw) or {}
    selected_tables = normalize_selected_tables(filter_obj.get("tables", []), table_names)
    filtered_schema = filter_schema_text(schema, selected_tables)
    generation_raw, generation_error = call_with_retries(
        client,
        args,
        prompt4vis_generation_prompt(db_id, filtered_schema, current_query, history, mined_examples, selected_tables),
    )
    vql, parse_info = normalize_generation(generation_raw)
    return vql, {
        "method_trace": {
            "mined_example_ids": [
                {
                    "sequence_id": example["sequence_id"],
                    "db_id": example["db_id"],
                    "tables": example.get("tables", []),
                }
                for example in mined_examples
            ],
            "schema_filter_raw": filter_raw,
            "schema_filter_error": filter_error,
            "schema_filter": filter_obj,
            "selected_tables": selected_tables,
            "generation_raw": generation_raw,
            "generation_error": generation_error,
            "generation_parse": parse_info,
        }
    }


def run_execution_repair(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    database_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    vql, info = run_decomposed(client, args, db_id, schema, current_query, history)
    trace = info["method_trace"]
    repairs = []
    for repair_idx in range(args.repair_rounds + 1):
        chart, sql = parse_vql(vql)
        ok, error, n_rows = execute_sql(db_id, sql, database_path, args.sql_timeout)
        repairs.append({"round": repair_idx, "vql": vql, "exec_ok": ok, "error": error, "n_rows": n_rows})
        if ok and vql:
            break
        raw, call_error = call_with_retries(client, args, repair_prompt(db_id, schema, current_query, history, vql, error))
        new_vql, parse_info = normalize_generation(raw)
        repairs[-1].update({"repair_raw": raw, "repair_call_error": call_error, "repair_parse": parse_info})
        if not new_vql:
            break
        vql = new_vql
    trace["repairs"] = repairs
    return vql, {"method_trace": trace}


def run_nvagent_adapted(
    client: OpenAI,
    args: argparse.Namespace,
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
    database_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    processor_raw, processor_error = call_with_retries(
        client, args, processor_prompt(db_id, schema, current_query, history)
    )
    processor_result = extract_json_object(processor_raw) or {
        "relevant_schema": {},
        "relationships": [],
        "context_carryover": "",
        "difficulty": "",
        "augmented_explanation": processor_raw[:1200],
    }
    composer_raw, composer_error = call_with_retries(
        client, args, composer_prompt(db_id, schema, current_query, history, processor_result)
    )
    vql, parse_info = normalize_generation(composer_raw)
    trace = {
        "processor_raw": processor_raw,
        "processor_error": processor_error,
        "processor_result": processor_result,
        "composer_raw": composer_raw,
        "composer_error": composer_error,
        "composer_parse": parse_info,
        "repairs": [],
    }

    for repair_idx in range(args.repair_rounds + 1):
        chart, sql = parse_vql(vql)
        ok, error, n_rows = execute_sql(db_id, sql, database_path, args.sql_timeout)
        trace["repairs"].append({"round": repair_idx, "vql": vql, "exec_ok": ok, "error": error, "n_rows": n_rows})
        if ok and vql:
            break
        raw, call_error = call_with_retries(client, args, repair_prompt(db_id, schema, current_query, history, vql, error))
        new_vql, repair_parse = normalize_generation(raw)
        trace["repairs"][-1].update({"repair_raw": raw, "repair_call_error": call_error, "repair_parse": repair_parse})
        if not new_vql:
            break
        vql = new_vql
    return vql, {"method_trace": trace}


def generate_sequence_prediction(
    seq_id: str,
    sequence: Dict[str, Any],
    args: argparse.Namespace,
    database_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    db_id = sequence.get("db_id", "")
    schema = cached_schema(db_id, database_path, args.include_sample_rows)
    pred_turns = []
    conversation = sequence.get("conversation", [])
    client = get_thread_client(args)

    for turn_index, turn in enumerate(conversation):
        current_query = turn.get("nl_query", "")
        history = build_history(conversation, pred_turns, turn_index, args.history_mode)
        try:
            if args.method == "decomposed":
                vql, info = run_decomposed(client, args, db_id, schema, current_query, history)
            elif args.method == "execution_repair":
                vql, info = run_execution_repair(client, args, db_id, schema, current_query, history, database_path)
            elif args.method == "nvagent_adapted":
                vql, info = run_nvagent_adapted(client, args, db_id, schema, current_query, history, database_path)
            elif args.method == "chat2vis_adapted":
                vql, info = run_chat2vis_adapted(client, args, db_id, schema, current_query, history)
            elif args.method == "lida_adapted":
                vql, info = run_lida_adapted(client, args, db_id, schema, current_query, history)
            elif args.method == "prompt4vis_adapted":
                vql, info = run_prompt4vis_adapted(
                    client, args, db_id, schema, current_query, history, database_path
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")
        except LLMCallFailed:
            raise
        except Exception as exc:
            vql = ""
            info = {"method_trace": {"fatal_error": str(exc)}}

        turn_prediction = {
            "turn": turn.get("turn", turn_index + 1),
            "vql": vql,
            "method": args.method,
            **info,
        }
        pred_turns.append(turn_prediction)

        if args.sleep:
            time.sleep(args.sleep)

    return seq_id, {
        "db_id": db_id,
        "model": args.model,
        "method": args.method,
        "history_mode": args.history_mode,
        "conversation": pred_turns,
    }


def existing_predictions(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return load_json(path)


def complete_prediction(sequence: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
    expected_turns = len(sequence.get("conversation", []))
    predicted_turns = prediction.get("conversation", [])
    if len(predicted_turns) != expected_turns:
        return False
    for turn in predicted_turns:
        vql = str(turn.get("vql", "")).strip()
        if not vql or not vql.lower().startswith("visualize"):
            return False
    return True


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_json(Path(args.dataset))
    output_path = Path(args.output)
    predictions = existing_predictions(output_path) if args.resume else {}
    database_path = Path(args.database_path)

    items = sorted_sequence_items(data)
    if args.max_sequences is not None:
        items = items[: args.max_sequences]

    pending_items = [
        (seq_id, sequence)
        for seq_id, sequence in items
        if not (args.resume and seq_id in predictions and complete_prediction(sequence, predictions[seq_id]))
    ]

    if args.workers <= 1:
        for seq_index, (seq_id, sequence) in enumerate(tqdm(pending_items, desc=f"{args.method}:{args.history_mode}")):
            seq_id, pred_sequence = generate_sequence_prediction(seq_id, sequence, args, database_path)
            predictions[seq_id] = pred_sequence
            if (seq_index + 1) % args.save_interval == 0:
                save_json(predictions, output_path)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_sequence_prediction, seq_id, sequence, args, database_path): seq_id
                for seq_id, sequence in pending_items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.method}:{args.history_mode}"):
                seq_id, pred_sequence = future.result()
                predictions[seq_id] = pred_sequence
                completed += 1
                if completed % args.save_interval == 0:
                    save_json(predictions, output_path)

    predictions = {seq_id: predictions[seq_id] for seq_id, _ in items if seq_id in predictions}
    save_json(predictions, output_path)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=[
            "decomposed",
            "execution_repair",
            "nvagent_adapted",
            "chat2vis_adapted",
            "lida_adapted",
            "prompt4vis_adapted",
        ],
        required=True,
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--history-mode", choices=["none", "oracle", "predicted"], default="predicted")
    parser.add_argument("--database-path", default="VisEval/viseval_dataset/databases")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-sample-rows", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument("--sql-timeout", type=float, default=20.0)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--schema-filter-examples", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--openrouter-referer", default="https://github.com/weixuzhang/ReasonVis")
    parser.add_argument("--openrouter-title", default="ReasonVis")
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        raise SystemExit(f"{args.api_key_env} is not set.")

    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
