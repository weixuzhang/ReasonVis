#!/usr/bin/env python
"""Post-hoc execution-guided repair for ReasonVis prediction files."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from generate_method_baselines import (
    LLMCallFailed,
    build_history,
    cached_schema,
    call_with_retries,
    execute_sql,
    get_thread_client,
    normalize_generation,
    parse_vql,
    repair_prompt,
    save_json,
    sorted_sequence_items,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repair_sequence(
    seq_id: str,
    sequence: Dict[str, Any],
    prediction: Dict[str, Any],
    args: argparse.Namespace,
    database_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    db_id = sequence.get("db_id", "")
    schema = cached_schema(db_id, database_path, args.include_sample_rows)
    client: OpenAI = get_thread_client(args)
    conversation = sequence.get("conversation", [])
    pred_conversation = prediction.get("conversation", [])
    repaired_turns: List[Dict[str, Any]] = []

    for turn_index, gt_turn in enumerate(conversation):
        base_turn = pred_conversation[turn_index] if turn_index < len(pred_conversation) else {}
        current_query = gt_turn.get("nl_query", "")
        vql = base_turn.get("vql", "")
        history = build_history(conversation, repaired_turns, turn_index, args.history_mode)
        trace = {
            "base_vql": vql,
            "repairs": [],
            "source_method": prediction.get("method", ""),
        }

        for repair_idx in range(args.repair_rounds + 1):
            _, sql = parse_vql(vql)
            ok, error, n_rows = execute_sql(db_id, sql, database_path, args.sql_timeout)
            trace["repairs"].append({"round": repair_idx, "vql": vql, "exec_ok": ok, "error": error, "n_rows": n_rows})
            if ok and vql:
                break
            raw, call_error = call_with_retries(
                client,
                args,
                repair_prompt(db_id, schema, current_query, history, vql, error),
            )
            new_vql, parse_info = normalize_generation(raw)
            trace["repairs"][-1].update({"repair_raw": raw, "repair_call_error": call_error, "repair_parse": parse_info})
            if not new_vql:
                break
            vql = new_vql

        repaired_turns.append(
            {
                "turn": gt_turn.get("turn", turn_index + 1),
                "vql": vql,
                "method": args.method_name,
                "method_trace": trace,
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    return seq_id, {
        "db_id": db_id,
        "model": args.model,
        "method": args.method_name,
        "history_mode": args.history_mode,
        "base_prediction": args.input_predictions,
        "conversation": repaired_turns,
    }


def existing_predictions(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return load_json(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-name", default="execution_repair")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--history-mode", choices=["none", "oracle", "predicted"], default="predicted")
    parser.add_argument("--database-path", default="VisEval/viseval_dataset/databases")
    parser.add_argument("--include-sample-rows", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--sql-timeout", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
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

    dataset = load_json(Path(args.dataset))
    base_predictions = load_json(Path(args.input_predictions))
    output_path = Path(args.output)
    repaired_predictions = existing_predictions(output_path) if args.resume else {}
    database_path = Path(args.database_path)

    items = sorted_sequence_items(dataset)
    if args.max_sequences is not None:
        items = items[: args.max_sequences]
    pending_items = [
        (seq_id, sequence)
        for seq_id, sequence in items
        if not (args.resume and seq_id in repaired_predictions)
    ]

    if args.workers <= 1:
        for idx, (seq_id, sequence) in enumerate(tqdm(pending_items, desc=args.method_name)):
            if seq_id not in base_predictions:
                continue
            seq_id, repaired = repair_sequence(seq_id, sequence, base_predictions[seq_id], args, database_path)
            repaired_predictions[seq_id] = repaired
            if (idx + 1) % args.save_interval == 0:
                save_json(repaired_predictions, output_path)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(repair_sequence, seq_id, sequence, base_predictions[seq_id], args, database_path): seq_id
                for seq_id, sequence in pending_items
                if seq_id in base_predictions
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=args.method_name):
                try:
                    seq_id, repaired = future.result()
                except LLMCallFailed:
                    raise
                repaired_predictions[seq_id] = repaired
                completed += 1
                if completed % args.save_interval == 0:
                    save_json(repaired_predictions, output_path)

    repaired_predictions = {seq_id: repaired_predictions[seq_id] for seq_id, _ in items if seq_id in repaired_predictions}
    save_json(repaired_predictions, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
