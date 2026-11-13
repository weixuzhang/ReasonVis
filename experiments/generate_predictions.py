#!/usr/bin/env python
"""Generate ReasonVis VQL predictions with OpenAI chat models."""

import argparse
import csv
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert system for conversational natural-language-to-visualization.

Return exactly one VQL statement:

Visualize [CHART_TYPE] [SQL_QUERY]

Allowed chart types: BAR, STACKED BAR, LINE, SCATTER, PIE, HEATMAP.
Use SQLite syntax. Use only tables and columns in the provided schema.
Do not explain your answer.
"""


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
            rows = []
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({key: row.get(key, "") for key in columns[:5]})
                    if len(rows) >= 2:
                        break
            if rows:
                lines.append(f"  sample_rows: {json.dumps(rows, ensure_ascii=False)}")

    relationships = []
    table_items = list(table_columns.items())
    for left_table, left_cols in table_items:
        for right_table, right_cols in table_items:
            if left_table >= right_table:
                continue
            common = sorted(set(left_cols).intersection(right_cols))
            key_like = [col for col in common if col.lower() == "id" or col.lower().endswith("_id") or col.lower().endswith(" id")]
            for col in key_like[:5]:
                relationships.append(f"- {left_table}.{col} = {right_table}.{col}")

    if relationships:
        lines.append("Potential joins:")
        lines.extend(relationships[:30])

    return "\n".join(lines)


def gold_vql(turn: Dict[str, Any]) -> str:
    return turn.get("query", {}).get("vql") or turn.get("vis_query", {}).get("VQL", "")


def clean_vql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("visualize "):
            return line
    return lines[0] if lines else ""


def build_prompt(
    db_id: str,
    schema: str,
    current_query: str,
    history: List[Tuple[str, str]],
) -> str:
    parts = [
        schema,
        "",
        "Conversation history:",
    ]
    if history:
        for idx, (query, vql) in enumerate(history, start=1):
            parts.append(f"Turn {idx} user: {query}")
            parts.append(f"Turn {idx} VQL: {vql}")
    else:
        parts.append("(none)")
    parts.extend([
        "",
        f"Current user query: {current_query}",
        "",
        "Return only the VQL statement.",
    ])
    return "\n".join(parts)


def call_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return clean_vql(response.choices[0].message.content or "")


_thread_local = threading.local()
_schema_cache: Dict[Tuple[str, str, bool], str] = {}
_schema_lock = threading.Lock()


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


def cached_schema(db_id: str, database_path: Path, include_sample_rows: bool) -> str:
    cache_key = (db_id, str(database_path), include_sample_rows)
    with _schema_lock:
        if cache_key in _schema_cache:
            return _schema_cache[cache_key]
    schema = extract_schema(db_id, database_path, include_sample_rows)
    with _schema_lock:
        _schema_cache[cache_key] = schema
    return schema


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

        history: List[Tuple[str, str]] = []
        if args.history_mode != "none":
            for prev_idx, prev_turn in enumerate(conversation[:turn_index]):
                prev_query = prev_turn.get("nl_query", "")
                if args.history_mode == "oracle":
                    prev_vql = gold_vql(prev_turn)
                else:
                    prev_vql = pred_turns[prev_idx].get("vql", "") if prev_idx < len(pred_turns) else ""
                history.append((prev_query, prev_vql))

        prompt = build_prompt(db_id, schema, current_query, history)
        vql = ""
        error = None
        for attempt in range(args.retries + 1):
            try:
                vql = call_openai(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                error = None
                break
            except Exception as exc:
                error = str(exc)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * (2 ** attempt))

        turn_prediction = {
            "turn": turn.get("turn", turn_index + 1),
            "vql": vql,
        }
        if error:
            turn_prediction["error"] = error
            print(f"ERROR seq={seq_id} turn={turn_index + 1}: {error}")
        pred_turns.append(turn_prediction)

        if args.sleep:
            time.sleep(args.sleep)

    return seq_id, {
        "db_id": db_id,
        "model": args.model,
        "history_mode": args.history_mode,
        "conversation": pred_turns,
    }


def existing_predictions(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return load_json(path)


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_json(Path(args.dataset))
    output_path = Path(args.output)
    predictions = existing_predictions(output_path) if args.resume else {}
    database_path = Path(args.database_path)

    items = sorted_sequence_items(data)
    if args.max_sequences is not None:
        items = items[: args.max_sequences]

    pending_items = [(seq_id, sequence) for seq_id, sequence in items if not (args.resume and seq_id in predictions)]

    if args.workers <= 1:
        for seq_index, (seq_id, sequence) in enumerate(tqdm(pending_items, desc=f"{args.model}:{args.history_mode}")):
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
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.model}:{args.history_mode}"):
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
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--history-mode", choices=["none", "oracle", "predicted"], default="predicted")
    parser.add_argument("--database-path", default="VisEval/viseval_dataset/databases")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-sample-rows", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
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
