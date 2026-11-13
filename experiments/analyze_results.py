#!/usr/bin/env python
"""Generate ReasonVis diagnostic tables, figures, errors, and bootstrap CIs."""

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    {
        "run_id": "gpt-4.1-mini_predicted_history",
        "label": "GPT-4.1 mini",
        "family": "OpenAI",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_no_history",
        "label": "GPT-4.1 mini",
        "family": "OpenAI",
        "history": "no_history",
        "prediction": "runs/gpt-4.1-mini_no_history.json",
        "eval": "runs/gpt-4.1-mini_no_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_oracle_history",
        "label": "GPT-4.1 mini",
        "family": "OpenAI",
        "history": "oracle_history",
        "prediction": "runs/gpt-4.1-mini_oracle_history.json",
        "eval": "runs/gpt-4.1-mini_oracle_history_eval",
    },
    {
        "run_id": "gpt-4.1_predicted_history",
        "label": "GPT-4.1",
        "family": "OpenAI",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1_predicted_history.json",
        "eval": "runs/gpt-4.1_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1_no_history",
        "label": "GPT-4.1",
        "family": "OpenAI",
        "history": "no_history",
        "prediction": "runs/gpt-4.1_no_history.json",
        "eval": "runs/gpt-4.1_no_history_eval",
    },
    {
        "run_id": "gpt-4.1_oracle_history",
        "label": "GPT-4.1",
        "family": "OpenAI",
        "history": "oracle_history",
        "prediction": "runs/gpt-4.1_oracle_history.json",
        "eval": "runs/gpt-4.1_oracle_history_eval",
    },
    {
        "run_id": "openrouter_claude-3.5-haiku_predicted_history",
        "label": "Claude 3.5 Haiku",
        "family": "Anthropic/OpenRouter",
        "history": "predicted_history",
        "prediction": "runs/openrouter_claude-3.5-haiku_predicted_history.json",
        "eval": "runs/openrouter_claude-3.5-haiku_predicted_history_eval",
    },
    {
        "run_id": "openrouter_llama-3.3-70b_predicted_history",
        "label": "Llama 3.3 70B",
        "family": "Open-weight/OpenRouter",
        "history": "predicted_history",
        "prediction": "runs/openrouter_llama-3.3-70b_predicted_history.json",
        "eval": "runs/openrouter_llama-3.3-70b_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_decomposed_predicted_history",
        "label": "Decompose",
        "family": "Method/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_decomposed_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_decomposed_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_execution_repair_predicted_history",
        "label": "Decompose+ExecRepair",
        "family": "Method/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_execution_repair_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_execution_repair_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_nvagent_adapted_predicted_history",
        "label": "nvAgent-adapted",
        "family": "Method/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_nvagent_adapted_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_nvagent_adapted_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_chat2vis_adapted_predicted_history",
        "label": "Chat2VIS-adapted",
        "family": "System/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_chat2vis_adapted_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_chat2vis_adapted_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_lida_adapted_predicted_history",
        "label": "LIDA-adapted",
        "family": "System/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_lida_adapted_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_lida_adapted_predicted_history_eval",
    },
    {
        "run_id": "gpt-4.1-mini_prompt4vis_adapted_predicted_history",
        "label": "Prompt4Vis-adapted",
        "family": "System/GPT-4.1 mini",
        "history": "predicted_history",
        "prediction": "runs/gpt-4.1-mini_prompt4vis_adapted_predicted_history.json",
        "eval": "runs/gpt-4.1-mini_prompt4vis_adapted_predicted_history_eval",
    },
]

PREDICTED_RUN_IDS = [
    "gpt-4.1-mini_predicted_history",
    "gpt-4.1_predicted_history",
    "openrouter_claude-3.5-haiku_predicted_history",
    "openrouter_llama-3.3-70b_predicted_history",
]

METHOD_BASELINE_RUN_IDS = [
    "gpt-4.1-mini_predicted_history",
    "gpt-4.1-mini_decomposed_predicted_history",
    "gpt-4.1-mini_execution_repair_predicted_history",
    "gpt-4.1-mini_nvagent_adapted_predicted_history",
]

SYSTEM_BASELINE_RUN_IDS = [
    "gpt-4.1-mini_predicted_history",
    "gpt-4.1-mini_chat2vis_adapted_predicted_history",
    "gpt-4.1-mini_lida_adapted_predicted_history",
    "gpt-4.1-mini_prompt4vis_adapted_predicted_history",
    "gpt-4.1-mini_nvagent_adapted_predicted_history",
]

CHART_ORDER = ["BAR", "STACKED BAR", "LINE", "PIE", "SCATTER", "HEATMAP"]
REASONING_ORDER = ["L1", "L2", "L3", "L4"]
CONTEXT_ORDER = ["independent", "local_followup", "long_range_followup", "multi_turn_compositional"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in columns) + "|")
    return "\n".join(lines)


def latex_table(rows: List[Dict[str, Any]], columns: Sequence[str], path: Path, caption: str, label: str) -> None:
    escaped_cols = [col.replace("_", "\\_") for col in columns]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(escaped_cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [fmt(row.get(col, "")).replace("_", "\\_").replace("%", "\\%") for col in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def sorted_sequence_items(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))


def context_label(turn: Dict[str, Any]) -> str:
    context = turn.get("context", {})
    if isinstance(context, dict) and context.get("dependency_level"):
        return str(context["dependency_level"])
    dep = turn.get("context_dependency", ["independent"])
    if isinstance(dep, list):
        return str(dep[0]) if dep else "independent"
    return str(dep)


def build_metadata(dataset: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    meta = {}
    for seq_id, sequence in sorted_sequence_items(dataset):
        for turn_idx, turn in enumerate(sequence.get("conversation", []), start=1):
            key = (str(seq_id), str(turn_idx))
            meta[key] = {
                "sequence_id": str(seq_id),
                "turn": str(turn_idx),
                "turn_position": turn_idx,
                "db_id": sequence.get("db_id", ""),
                "chart_type": turn.get("chart_type", ""),
                "reasoning_level": turn.get("reasoning_level", ""),
                "hardness": turn.get("hardness", ""),
                "context_dependency": context_label(turn),
                "context_binary": "independent" if context_label(turn) == "independent" else "dependent",
                "operation_tags": ",".join(turn.get("reasoning", {}).get("operation_tags", [])),
            }
    return meta


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
            if not headers:
                continue
            columns = ", ".join([f'"{header}" TEXT' for header in headers])
            cursor.execute(f'CREATE TABLE "{table}" ({columns})')
            for row in reader:
                row = (row + [""] * len(headers))[: len(headers)]
                cursor.execute(f'INSERT INTO "{table}" VALUES ({", ".join(["?"] * len(headers))})', row)
    conn.commit()
    return conn


def execute_status(
    db_id: str,
    sql: str,
    database_path: Path,
    connections: Dict[str, sqlite3.Connection],
    cache: Dict[Tuple[str, str], Tuple[bool, str]],
    timeout_seconds: int = 30,
) -> Tuple[bool, str]:
    sql = (sql or "").strip()
    if not sql:
        return False, "missing_sql"
    cache_key = (db_id, sql)
    if cache_key in cache:
        return cache[cache_key]
    if db_id not in connections:
        connections[db_id] = setup_db(db_id, database_path)
    conn = connections[db_id]
    start = time.monotonic()

    def interrupt_if_timed_out() -> int:
        return 1 if time.monotonic() - start > timeout_seconds else 0

    conn.set_progress_handler(interrupt_if_timed_out, 10000)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        result = (True, "")
    except Exception as exc:
        result = (False, str(exc))
    finally:
        conn.set_progress_handler(None, 0)
    cache[cache_key] = result
    return result


def invalid_sql_cause(error: str) -> str:
    text = (error or "").lower()
    if not text or text == "missing_sql":
        return "missing_sql"
    if "no such column" in text:
        return "no_such_column"
    if "ambiguous column" in text:
        return "ambiguous_column"
    if "no such table" in text:
        return "no_such_table"
    if "syntax error" in text or "unrecognized token" in text or "near " in text:
        return "syntax_error"
    if "only execute one statement" in text:
        return "multiple_statements"
    if "aggregate functions are not allowed" in text or "misuse of aggregate" in text:
        return "aggregate_misuse"
    if "interrupted" in text or "timeout" in text:
        return "timeout"
    return "other_execution_error"


def error_bucket(sql_correct: bool, vis_correct: bool, pred_exec_ok: bool, pred_error: str) -> str:
    if sql_correct and vis_correct:
        return "correct"
    if not (pred_exec_ok or sql_correct):
        if pred_error == "missing_sql":
            return "missing_sql"
        return "invalid_sql"
    if sql_correct and not vis_correct:
        return "chart_wrong_sql_right"
    if (not sql_correct) and vis_correct:
        return "sql_wrong_chart_right"
    return "both_wrong"


def load_run_records(
    run: Dict[str, str],
    metadata: Dict[Tuple[str, str], Dict[str, Any]],
    database_path: Path,
    connections: Dict[str, sqlite3.Connection],
    exec_cache: Dict[Tuple[str, str], Tuple[bool, str]],
) -> List[Dict[str, Any]]:
    detailed = load_json(Path(run["eval"]) / "detailed_results.json")["per_turn"]
    rows = []
    for seq_id, turns in detailed.items():
        for turn, result in turns.items():
            meta = metadata[(str(seq_id), str(turn))]
            pred_sql = result.get("pred_sql", "")
            pred_exec_ok, pred_error = execute_status(meta["db_id"], pred_sql, database_path, connections, exec_cache)
            sql_correct = bool(result.get("sql_correct"))
            vis_correct = bool(result.get("vis_type_correct"))
            turn_success = bool(result.get("turn_success"))
            row = {
                **meta,
                "run_id": run["run_id"],
                "model": run["label"],
                "history": run["history"],
                "sql_correct": sql_correct,
                "vis_type_correct": vis_correct,
                "turn_success": turn_success,
                "pred_vis_type": result.get("pred_vis_type", ""),
                "gt_vis_type": result.get("gt_vis_type", ""),
                "pred_exec_ok": pred_exec_ok,
                "pred_exec_error": pred_error,
                "error_bucket": error_bucket(sql_correct, vis_correct, pred_exec_ok, pred_error),
                "invalid_sql_cause": invalid_sql_cause(pred_error) if not pred_exec_ok else "",
            }
            rows.append(row)
    return rows


def proportion(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else math.nan


def aggregate_group(records: List[Dict[str, Any]], group_name: str, key_fn) -> List[Dict[str, Any]]:
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[key_fn(row)].append(row)
    out = []
    for key, rows in sorted(buckets.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "group": group_name,
                "value": key,
                "n_turns": len(rows),
                "sql_accuracy": proportion(row["sql_correct"] for row in rows),
                "vis_type_accuracy": proportion(row["vis_type_correct"] for row in rows),
                "turn_success_rate": proportion(row["turn_success"] for row in rows),
            }
        )
    return out


def aggregate_by_run(records_by_run: Dict[str, List[Dict[str, Any]]], group_name: str, key_fn) -> List[Dict[str, Any]]:
    rows = []
    for run in RUNS:
        for item in aggregate_group(records_by_run[run["run_id"]], group_name, key_fn):
            rows.append({"run_id": run["run_id"], "model": run["label"], "history": run["history"], **item})
    return rows


def main_results_rows() -> List[Dict[str, Any]]:
    rows = []
    for run in RUNS:
        metrics = load_json(Path(run["eval"]) / "overall_metrics.json")
        rows.append(
            {
                "run_id": run["run_id"],
                "model": run["label"],
                "family": run["family"],
                "history": run["history"],
                "SQL": metrics["sql_accuracy"],
                "Vis": metrics["vis_type_accuracy"],
                "Turn": metrics["turn_success_rate"],
                "SeqComplete": metrics["sequence_complete_success_rate"],
                "Seq50": metrics["sequence_50_percent_success_rate"],
                "ContextDep": metrics["context_dependent_accuracy"],
                "ContextIndep": metrics["context_independent_accuracy"],
                "First": metrics["sequence_first_turn_success_rate"],
                "Last": metrics["sequence_last_turn_success_rate"],
            }
        )
    return rows


def method_baseline_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {row["run_id"]: row for row in main_rows}
    direct = lookup["gpt-4.1-mini_predicted_history"]
    rows = []
    for run_id in METHOD_BASELINE_RUN_IDS:
        row = dict(lookup[run_id])
        row["DeltaTurnVsDirectMini"] = row["Turn"] - direct["Turn"]
        row["DeltaSQLVsDirectMini"] = row["SQL"] - direct["SQL"]
        rows.append(row)
    return rows


def system_baseline_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {row["run_id"]: row for row in main_rows}
    direct = lookup["gpt-4.1-mini_predicted_history"]
    rows = []
    for run_id in SYSTEM_BASELINE_RUN_IDS:
        row = dict(lookup[run_id])
        row["DeltaTurnVsDirectMini"] = row["Turn"] - direct["Turn"]
        row["DeltaSQLVsDirectMini"] = row["SQL"] - direct["SQL"]
        rows.append(row)
    return rows


def plot_grouped_bar(
    rows: List[Dict[str, Any]],
    group_values: Sequence[str],
    run_ids: Sequence[str],
    y_col: str,
    title: str,
    output: Path,
    ylim: Tuple[float, float] = (0.0, 0.42),
) -> None:
    run_lookup = {run["run_id"]: run["label"] for run in RUNS}
    data = {(row["run_id"], str(row["value"])): row[y_col] for row in rows}
    width = 0.18
    x = list(range(len(group_values)))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for idx, run_id in enumerate(run_ids):
        offsets = [pos + (idx - (len(run_ids) - 1) / 2) * width for pos in x]
        ax.bar(offsets, [data.get((run_id, value), 0.0) for value in group_values], width=width, label=run_lookup[run_id])
    ax.set_title(title)
    ax.set_ylabel("Turn success")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(group_values, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_main_results(rows: List[Dict[str, Any]], output: Path) -> None:
    labels = [f"{row['model']}\n{row['history'].replace('_history','').replace('_',' ')}" for row in rows]
    values = [row["Turn"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(range(len(rows)), values, color="#5277a3")
    ax.set_ylabel("Turn success")
    ax.set_ylim(0, 0.32)
    ax.set_title("ReasonVis end-to-end turn success")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_error_taxonomy(rows: List[Dict[str, Any]], output: Path) -> None:
    predicted_rows = [row for row in rows if row["run_id"] in PREDICTED_RUN_IDS]
    buckets = ["correct", "sql_wrong_chart_right", "invalid_sql", "chart_wrong_sql_right", "both_wrong", "missing_sql"]
    labels = [row["model"] for row in predicted_rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bottoms = [0.0] * len(predicted_rows)
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"]
    for bucket, color in zip(buckets, colors):
        values = [row.get(f"rate_{bucket}", 0.0) for row in predicted_rows]
        ax.bar(labels, values, bottom=bottoms, label=bucket.replace("_", " "), color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylabel("Share of turns")
    ax.set_title("Error taxonomy for predicted-history baselines")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def error_taxonomy_rows(records_by_run: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    buckets = ["correct", "sql_wrong_chart_right", "invalid_sql", "chart_wrong_sql_right", "both_wrong", "missing_sql"]
    for run in RUNS:
        records = records_by_run[run["run_id"]]
        counts = Counter(row["error_bucket"] for row in records)
        total = len(records)
        row = {"run_id": run["run_id"], "model": run["label"], "history": run["history"], "n_turns": total}
        for bucket in buckets:
            row[f"count_{bucket}"] = counts.get(bucket, 0)
            row[f"rate_{bucket}"] = counts.get(bucket, 0) / total if total else math.nan
        rows.append(row)
    return rows


def invalid_sql_rows(records_by_run: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for run in RUNS:
        counter = Counter(row["invalid_sql_cause"] for row in records_by_run[run["run_id"]] if row["invalid_sql_cause"])
        total = sum(counter.values())
        for cause, count in counter.most_common():
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["label"],
                    "history": run["history"],
                    "cause": cause,
                    "count": count,
                    "rate_among_invalid_or_missing": count / total if total else math.nan,
                }
            )
    return rows


def chart_confusion_rows(records_by_run: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for run in RUNS:
        counter = Counter()
        for row in records_by_run[run["run_id"]]:
            if row["gt_vis_type"] != row["pred_vis_type"]:
                counter[(row["gt_vis_type"], row["pred_vis_type"] or "EMPTY")] += 1
        total = sum(counter.values())
        for (gold, pred), count in counter.most_common(20):
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["label"],
                    "history": run["history"],
                    "gold_chart": gold,
                    "pred_chart": pred,
                    "count": count,
                    "rate_among_chart_errors": count / total if total else math.nan,
                }
            )
    return rows


def bootstrap_metric(
    records: List[Dict[str, Any]],
    metric_fn,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float]:
    by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_seq[row["sequence_id"]].append(row)
    seq_ids = sorted(by_seq)
    observed = metric_fn(records)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled = []
        for seq_id in rng.choices(seq_ids, k=len(seq_ids)):
            sampled.extend(by_seq[seq_id])
        value = metric_fn(sampled)
        if not math.isnan(value):
            samples.append(value)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return observed, lo, hi


def turn_success(rows: List[Dict[str, Any]]) -> float:
    return proportion(row["turn_success"] for row in rows)


def context_dep_success(rows: List[Dict[str, Any]]) -> float:
    return proportion(row["turn_success"] for row in rows if row["context_binary"] == "dependent")


def context_indep_success(rows: List[Dict[str, Any]]) -> float:
    return proportion(row["turn_success"] for row in rows if row["context_binary"] == "independent")


def context_delta(rows: List[Dict[str, Any]]) -> float:
    return context_dep_success(rows) - context_indep_success(rows)


def bootstrap_ci_rows(records_by_run: Dict[str, List[Dict[str, Any]]], iterations: int, seed: int) -> List[Dict[str, Any]]:
    metrics = [
        ("turn_success", turn_success),
        ("context_dependent_success", context_dep_success),
        ("context_independent_success", context_indep_success),
        ("context_dep_minus_indep", context_delta),
    ]
    rows = []
    for run_idx, run in enumerate(RUNS):
        records = records_by_run[run["run_id"]]
        for metric_name, metric_fn in metrics:
            observed, lo, hi = bootstrap_metric(records, metric_fn, iterations, seed + run_idx * 17)
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["label"],
                    "history": run["history"],
                    "metric": metric_name,
                    "estimate": observed,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return rows


def paired_delta(records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]], seq_ids: Sequence[str]) -> float:
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for row in records_a:
        by_a[row["sequence_id"]].append(row)
    for row in records_b:
        by_b[row["sequence_id"]].append(row)
    sampled_a = [row for seq_id in seq_ids for row in by_a[seq_id]]
    sampled_b = [row for seq_id in seq_ids for row in by_b[seq_id]]
    return turn_success(sampled_a) - turn_success(sampled_b)


def bootstrap_delta_rows(records_by_run: Dict[str, List[Dict[str, Any]]], iterations: int, seed: int) -> List[Dict[str, Any]]:
    comparisons = [
        ("gpt-4.1_predicted_minus_mini_predicted", "gpt-4.1_predicted_history", "gpt-4.1-mini_predicted_history"),
        ("gpt-4.1_oracle_minus_predicted", "gpt-4.1_oracle_history", "gpt-4.1_predicted_history"),
        ("gpt-4.1_oracle_minus_no_history", "gpt-4.1_oracle_history", "gpt-4.1_no_history"),
        ("mini_oracle_minus_predicted", "gpt-4.1-mini_oracle_history", "gpt-4.1-mini_predicted_history"),
        ("mini_oracle_minus_no_history", "gpt-4.1-mini_oracle_history", "gpt-4.1-mini_no_history"),
        (
            "decompose_minus_direct_mini",
            "gpt-4.1-mini_decomposed_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
        (
            "exec_repair_minus_direct_mini",
            "gpt-4.1-mini_execution_repair_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
        (
            "exec_repair_minus_decompose",
            "gpt-4.1-mini_execution_repair_predicted_history",
            "gpt-4.1-mini_decomposed_predicted_history",
        ),
        (
            "nvagent_adapted_minus_direct_mini",
            "gpt-4.1-mini_nvagent_adapted_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
        (
            "chat2vis_adapted_minus_direct_mini",
            "gpt-4.1-mini_chat2vis_adapted_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
        (
            "lida_adapted_minus_direct_mini",
            "gpt-4.1-mini_lida_adapted_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
        (
            "prompt4vis_adapted_minus_direct_mini",
            "gpt-4.1-mini_prompt4vis_adapted_predicted_history",
            "gpt-4.1-mini_predicted_history",
        ),
    ]
    rows = []
    rng = random.Random(seed)
    all_seq_ids = sorted({row["sequence_id"] for rows_ in records_by_run.values() for row in rows_})
    for name, run_a, run_b in comparisons:
        observed = paired_delta(records_by_run[run_a], records_by_run[run_b], all_seq_ids)
        samples = []
        for _ in range(iterations):
            sampled_ids = rng.choices(all_seq_ids, k=len(all_seq_ids))
            samples.append(paired_delta(records_by_run[run_a], records_by_run[run_b], sampled_ids))
        samples.sort()
        rows.append(
            {
                "comparison": name,
                "run_a": run_a,
                "run_b": run_b,
                "metric": "turn_success_delta",
                "estimate": observed,
                "ci95_low": samples[int(0.025 * (len(samples) - 1))],
                "ci95_high": samples[int(0.975 * (len(samples) - 1))],
            }
        )
    return rows


def write_summary(
    output_dir: Path,
    main_rows: List[Dict[str, Any]],
    method_rows: List[Dict[str, Any]],
    system_rows: List[Dict[str, Any]],
    reasoning_rows: List[Dict[str, Any]],
    context_rows: List[Dict[str, Any]],
    error_rows: List[Dict[str, Any]],
    ci_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
) -> None:
    predicted_reasoning = [
        row
        for row in reasoning_rows
        if row["run_id"] in PREDICTED_RUN_IDS and row["value"] in REASONING_ORDER
    ]
    predicted_context = [
        row
        for row in context_rows
        if row["run_id"] in PREDICTED_RUN_IDS and row["value"] in CONTEXT_ORDER
    ]
    lines = [
        "# ReasonVis Analysis Summary",
        "",
        "## Main Results",
        "",
        markdown_table(
            main_rows,
            ["model", "history", "SQL", "Vis", "Turn", "SeqComplete", "Seq50", "ContextDep", "ContextIndep", "First", "Last"],
        ),
        "",
        "## Method Baselines",
        "",
        markdown_table(
            method_rows,
            ["model", "family", "SQL", "Vis", "Turn", "SeqComplete", "Seq50", "DeltaTurnVsDirectMini"],
        ),
        "",
        "## System Baselines",
        "",
        markdown_table(
            system_rows,
            ["model", "family", "SQL", "Vis", "Turn", "SeqComplete", "Seq50", "DeltaTurnVsDirectMini"],
        ),
        "",
        "## Predicted-History Reasoning Breakdown",
        "",
        markdown_table(predicted_reasoning, ["model", "value", "n_turns", "sql_accuracy", "vis_type_accuracy", "turn_success_rate"]),
        "",
        "## Predicted-History Context Breakdown",
        "",
        markdown_table(predicted_context, ["model", "value", "n_turns", "sql_accuracy", "vis_type_accuracy", "turn_success_rate"]),
        "",
        "## Error Taxonomy",
        "",
        markdown_table(
            error_rows,
            [
                "model",
                "history",
                "rate_correct",
                "rate_sql_wrong_chart_right",
                "rate_invalid_sql",
                "rate_chart_wrong_sql_right",
                "rate_both_wrong",
                "rate_missing_sql",
            ],
        ),
        "",
        "## Bootstrap Confidence Intervals",
        "",
        markdown_table(ci_rows, ["model", "history", "metric", "estimate", "ci95_low", "ci95_high"]),
        "",
        "## Pairwise Bootstrap Deltas",
        "",
        markdown_table(delta_rows, ["comparison", "estimate", "ci95_low", "ci95_high"]),
        "",
    ]
    (output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/reasonvis.json")
    parser.add_argument("--database-path", default="VisEval/viseval_dataset/databases")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    dataset = load_json(Path(args.dataset))
    metadata = build_metadata(dataset)
    connections: Dict[str, sqlite3.Connection] = {}
    exec_cache: Dict[Tuple[str, str], Tuple[bool, str]] = {}
    records_by_run = {
        run["run_id"]: load_run_records(run, metadata, Path(args.database_path), connections, exec_cache) for run in RUNS
    }
    for conn in connections.values():
        conn.close()

    main_rows = main_results_rows()
    method_rows = method_baseline_rows(main_rows)
    system_rows = system_baseline_rows(main_rows)
    reasoning_rows = aggregate_by_run(records_by_run, "reasoning_level", lambda row: row["reasoning_level"])
    context_rows = aggregate_by_run(records_by_run, "context_dependency", lambda row: row["context_dependency"])
    chart_rows = aggregate_by_run(records_by_run, "chart_type", lambda row: row["chart_type"])
    turn_position_rows = aggregate_by_run(records_by_run, "turn_position", lambda row: row["turn_position"])
    database_rows = aggregate_by_run(records_by_run, "database", lambda row: row["db_id"])
    error_rows = error_taxonomy_rows(records_by_run)
    invalid_rows = invalid_sql_rows(records_by_run)
    confusion_rows = chart_confusion_rows(records_by_run)
    ci_rows = bootstrap_ci_rows(records_by_run, args.bootstrap_iterations, args.seed)
    delta_rows = bootstrap_delta_rows(records_by_run, args.bootstrap_iterations, args.seed + 101)

    write_csv(main_rows, table_dir / "main_results.csv")
    write_csv(method_rows, table_dir / "method_baselines.csv")
    write_csv(system_rows, table_dir / "system_baselines.csv")
    write_csv(reasoning_rows, table_dir / "breakdown_reasoning_level.csv")
    write_csv(context_rows, table_dir / "breakdown_context_dependency.csv")
    write_csv(chart_rows, table_dir / "breakdown_chart_type.csv")
    write_csv(turn_position_rows, table_dir / "breakdown_turn_position.csv")
    write_csv(database_rows, table_dir / "breakdown_database.csv")
    write_csv(error_rows, table_dir / "error_taxonomy.csv")
    write_csv(invalid_rows, table_dir / "invalid_sql_causes.csv")
    write_csv(confusion_rows, table_dir / "chart_confusions.csv")
    write_csv(ci_rows, table_dir / "bootstrap_ci.csv")
    write_csv(delta_rows, table_dir / "bootstrap_pairwise_deltas.csv")

    latex_table(main_rows, ["model", "history", "SQL", "Vis", "Turn", "SeqComplete", "Seq50"], table_dir / "main_results.tex", "ReasonVis main automatic results.", "tab:reasonvis-main")
    latex_table(method_rows, ["model", "SQL", "Vis", "Turn", "SeqComplete", "Seq50", "DeltaTurnVsDirectMini"], table_dir / "method_baselines.tex", "ReasonVis method-style baselines using the same GPT-4.1-mini backbone.", "tab:reasonvis-method-baselines")
    latex_table(system_rows, ["model", "SQL", "Vis", "Turn", "SeqComplete", "Seq50", "DeltaTurnVsDirectMini"], table_dir / "system_baselines.tex", "ReasonVis adapted system-level baselines using the same GPT-4.1-mini backbone.", "tab:reasonvis-system-baselines")
    latex_table(error_rows, ["model", "history", "rate_correct", "rate_sql_wrong_chart_right", "rate_invalid_sql", "rate_chart_wrong_sql_right", "rate_both_wrong"], table_dir / "error_taxonomy.tex", "ReasonVis error taxonomy.", "tab:reasonvis-errors")
    latex_table(delta_rows, ["comparison", "estimate", "ci95_low", "ci95_high"], table_dir / "bootstrap_pairwise_deltas.tex", "Bootstrap confidence intervals for paired turn-success deltas.", "tab:reasonvis-bootstrap-deltas")

    plot_main_results(main_rows, figure_dir / "main_turn_success.pdf")
    plot_grouped_bar(reasoning_rows, REASONING_ORDER, PREDICTED_RUN_IDS, "turn_success_rate", "Turn success by reasoning level", figure_dir / "reasoning_breakdown.pdf")
    plot_grouped_bar(context_rows, CONTEXT_ORDER, PREDICTED_RUN_IDS, "turn_success_rate", "Turn success by context dependency", figure_dir / "context_breakdown.pdf")
    plot_grouped_bar(chart_rows, CHART_ORDER, PREDICTED_RUN_IDS, "turn_success_rate", "Turn success by chart type", figure_dir / "chart_breakdown.pdf")
    plot_grouped_bar(turn_position_rows, ["1", "2", "3", "4"], PREDICTED_RUN_IDS, "turn_success_rate", "Turn success by turn position", figure_dir / "turn_position_breakdown.pdf")
    plot_error_taxonomy(error_rows, figure_dir / "error_taxonomy.pdf")

    write_json({"runs": RUNS, "records_by_run": {run: len(rows) for run, rows in records_by_run.items()}}, output_dir / "analysis_manifest.json")
    write_summary(output_dir, main_rows, method_rows, system_rows, reasoning_rows, context_rows, error_rows, ci_rows, delta_rows)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(RUNS), "turns_per_run": len(next(iter(records_by_run.values())))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
