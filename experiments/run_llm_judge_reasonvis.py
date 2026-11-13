#!/usr/bin/env python
"""Run LLM-judge evaluation for ReasonVis model outputs."""

import argparse
import csv
import json
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator for conversational natural-language-to-visualization.

Score the predicted VQL against the current user request, conversation context,
and gold reference. Return a JSON object only.

Use this 0/1/2 scale:
0 = incorrect or unusable
1 = partially correct, useful but with clear issues
2 = correct and acceptable

Evaluate:
- intent_alignment: whether the prediction answers the user request.
- data_faithfulness: whether the SQL retrieves the intended data.
- chart_suitability: whether the chart type matches the analysis need.
- context_fidelity: whether the prediction carries over relevant prior context.
- overall_acceptability: whether the output is acceptable for evaluation.

Also provide error_category from:
correct, sql_wrong, chart_wrong, context_wrong, invalid_sql, incomplete_output, both_wrong, other.
Keep rationale concise.
"""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sorted_sequence_items(data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))


def context_label(turn: Dict[str, Any]) -> str:
    context = turn.get("context", {})
    if isinstance(context, dict) and context.get("dependency_level"):
        return context["dependency_level"]
    dep = turn.get("context_dependency", ["independent"])
    if isinstance(dep, list):
        return dep[0] if dep else "independent"
    return str(dep)


def gold_vql(turn: Dict[str, Any]) -> str:
    return turn.get("query", {}).get("vql") or turn.get("vis_query", {}).get("VQL", "")


def parse_vql(vql: str) -> Tuple[str, str]:
    text = re.sub(r"\s+", " ", (vql or "")).strip()
    match = re.match(r"^Visualize\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", ""
    rest = match.group(1).strip()
    for vis_type in ["STACKED BAR", "HEATMAP", "SCATTER", "LINE", "PIE", "BAR"]:
        pattern = re.compile(rf"^{re.escape(vis_type)}\s+((?:SELECT|WITH)\b.+)$", re.IGNORECASE | re.DOTALL)
        type_match = pattern.match(rest)
        if type_match:
            return vis_type, type_match.group(1).strip()
    generic = re.match(r"^([A-Za-z0-9_ ]+?)\s+((?:SELECT|WITH)\b.+)$", rest, flags=re.IGNORECASE | re.DOTALL)
    if generic:
        return generic.group(1).strip(), generic.group(2).strip()
    return rest, ""


def setup_db(db_id: str, database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    db_dir = database_path / db_id
    for csv_path in sorted(db_dir.glob("*.csv")):
        if csv_path.name == "sqlite_sequence.csv":
            continue
        table = csv_path.stem
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            columns = ", ".join([f'"{header}" TEXT' for header in headers])
            cursor.execute(f'CREATE TABLE "{table}" ({columns})')
            for row in reader:
                placeholders = ", ".join(["?"] * len(row))
                cursor.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', row)
    conn.commit()
    return conn


def execute_preview(db_id: str, sql: str, database_path: Path, limit: int = 5) -> Dict[str, Any]:
    if not sql:
        return {"ok": False, "error": "missing_sql", "columns": [], "sample_rows": []}
    try:
        conn = setup_db(db_id, database_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(limit)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {
            "ok": True,
            "error": "",
            "columns": columns,
            "sample_rows": [dict(zip(columns, row)) for row in rows],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "columns": [], "sample_rows": []}


def sample_turns(dataset: Dict[str, Any], sample_size: int, seed: int) -> List[Tuple[str, int, Dict[str, Any]]]:
    turns = []
    for seq_id, sequence in sorted_sequence_items(dataset):
        for turn_idx, turn in enumerate(sequence.get("conversation", [])):
            turns.append((seq_id, turn_idx, turn))
    rng = random.Random(seed)
    rng.shuffle(turns)
    return turns[:sample_size] if sample_size else turns


def build_prompt(item: Dict[str, Any]) -> str:
    return json.dumps(item, indent=2, ensure_ascii=False)


def judge_item(client: OpenAI, model: str, item: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(item)},
        ],
        temperature=temperature,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/reasonvis.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database-path", default="VisEval/viseval_dataset/databases")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    dataset = load_json(Path(args.dataset))
    predictions = load_json(Path(args.predictions))
    output_path = Path(args.output)
    done = {row.get("item_id") for row in read_jsonl(output_path)} if args.resume else set()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    database_path = Path(args.database_path)

    for seq_id, turn_idx, turn in tqdm(sample_turns(dataset, args.sample_size, args.seed), desc=args.judge_model):
        item_id = f"{seq_id}-{turn_idx + 1}"
        if item_id in done:
            continue
        sequence = dataset[seq_id]
        pred_turns = predictions.get(seq_id, {}).get("conversation", [])
        pred_turn = pred_turns[turn_idx] if turn_idx < len(pred_turns) else {}
        pred_vql = pred_turn.get("vql", "")
        pred_chart, pred_sql = parse_vql(pred_vql)
        pred_preview = execute_preview(sequence.get("db_id", ""), pred_sql, database_path)

        item = {
            "item_id": item_id,
            "sequence_id": seq_id,
            "turn": turn.get("turn", turn_idx + 1),
            "db_id": sequence.get("db_id", ""),
            "history": [
                {
                    "nl_query": prev.get("nl_query", ""),
                    "gold_vql": gold_vql(prev),
                }
                for prev in sequence.get("conversation", [])[:turn_idx]
            ],
            "current_query": turn.get("nl_query", ""),
            "reasoning_level": turn.get("reasoning_level", ""),
            "context_dependency": context_label(turn),
            "gold_vql": gold_vql(turn),
            "gold_chart_type": turn.get("chart_type", ""),
            "gold_sample_rows": turn.get("execution", {}).get("sample_rows", [])[:5],
            "pred_vql": pred_vql,
            "pred_chart_type": pred_chart,
            "pred_execution_preview": pred_preview,
        }
        try:
            judgment = judge_item(client, args.judge_model, item, args.temperature)
        except Exception as exc:
            judgment = {"error": str(exc)}
        append_jsonl({"item_id": item_id, "input": item, "judgment": judgment}, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
