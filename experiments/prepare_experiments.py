#!/usr/bin/env python
"""Prepare reproducible ReasonVis experiment artifacts."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_DATASET = "data/reasonvis.json"
DEFAULT_OUTPUT_DIR = "runs"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sorted_sequence_items(data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))


def turn_context_label(turn: Dict[str, Any]) -> str:
    context = turn.get("context", {})
    if isinstance(context, dict) and context.get("dependency_level"):
        return context["dependency_level"]
    dependency = turn.get("context_dependency", ["independent"])
    if isinstance(dependency, list):
        return dependency[0] if dependency else "independent"
    return str(dependency)


def make_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    chart_counter = Counter()
    reasoning_counter = Counter()
    context_counter = Counter()
    hardness_counter = Counter()
    turn_len_counter = Counter()
    db_counter = Counter()
    join_turns = 0
    prior_filter_turns = 0

    for _, sequence in data.items():
        conversation = sequence.get("conversation", [])
        turn_len_counter[len(conversation)] += 1
        db_counter[sequence.get("db_id", "unknown")] += 1
        for turn in conversation:
            chart_counter[turn.get("chart_type", "UNKNOWN")] += 1
            reasoning_counter[turn.get("reasoning_level", "UNKNOWN")] += 1
            context_counter[turn_context_label(turn)] += 1
            hardness_counter[turn.get("hardness", "UNKNOWN")] += 1
            sql = (
                turn.get("query", {}).get("canonical_sql")
                or turn.get("vis_query", {}).get("data_part", {}).get("sql_part", "")
            )
            if " join " in f" {sql.lower()} ":
                join_turns += 1
            if turn.get("context_link", {}).get("uses_prior_result_filter"):
                prior_filter_turns += 1

    turn_count = sum(chart_counter.values())
    return {
        "sequence_count": len(data),
        "turn_count": turn_count,
        "database_count": len(db_counter),
        "turn_length_distribution": dict(sorted(turn_len_counter.items())),
        "chart_distribution": dict(sorted(chart_counter.items())),
        "reasoning_distribution": dict(sorted(reasoning_counter.items())),
        "context_dependency_distribution": dict(sorted(context_counter.items())),
        "hardness_distribution": dict(sorted(hardness_counter.items())),
        "turns_with_join": join_turns,
        "turns_with_prior_result_filter": prior_filter_turns,
        "top_databases_by_sequence_count": db_counter.most_common(20),
    }


def labels_for_sequence(sequence: Dict[str, Any]) -> Counter:
    labels = Counter()
    db_id = sequence.get("db_id", "unknown")
    labels[f"db:{db_id}"] += 1
    for turn in sequence.get("conversation", []):
        labels[f"chart:{turn.get('chart_type', 'UNKNOWN')}"] += 1
        labels[f"reasoning:{turn.get('reasoning_level', 'UNKNOWN')}"] += 1
        labels[f"context:{turn_context_label(turn)}"] += 1
        labels[f"hardness:{turn.get('hardness', 'UNKNOWN')}"] += 1
    return labels


def greedy_stratified_sample(data: Dict[str, Any], size: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    items = sorted_sequence_items(data)
    if size >= len(items):
        return {seq_id: sequence for seq_id, sequence in items}

    sequence_labels = {seq_id: labels_for_sequence(sequence) for seq_id, sequence in items}
    global_counts = Counter()
    for labels in sequence_labels.values():
        global_counts.update(labels)

    selected: List[str] = []
    selected_set = set()
    covered = Counter()
    candidates = [seq_id for seq_id, _ in items]
    rng.shuffle(candidates)

    while len(selected) < size and candidates:
        best_seq = None
        best_score = None
        for seq_id in candidates:
            labels = sequence_labels[seq_id]
            novelty = sum(1.0 / max(1, global_counts[label]) for label in labels if covered[label] == 0)
            balance = sum(1.0 / (1 + covered[label]) for label in labels)
            score = novelty * 10.0 + balance
            if best_score is None or score > best_score:
                best_score = score
                best_seq = seq_id
        selected.append(best_seq)
        selected_set.add(best_seq)
        covered.update(sequence_labels[best_seq])
        candidates = [seq_id for seq_id in candidates if seq_id not in selected_set]

    selected_sorted = sorted(selected, key=lambda seq_id: int(seq_id) if seq_id.isdigit() else seq_id)
    return {seq_id: data[seq_id] for seq_id in selected_sorted}


def make_oracle_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    predictions = {}
    for seq_id, sequence in sorted_sequence_items(data):
        pred_turns = []
        for turn in sequence.get("conversation", []):
            vql = turn.get("query", {}).get("vql") or turn.get("vis_query", {}).get("VQL", "")
            pred_turns.append({"turn": turn.get("turn", len(pred_turns) + 1), "vql": vql})
        predictions[seq_id] = {
            "db_id": sequence.get("db_id", ""),
            "conversation": pred_turns,
        }
    return predictions


def write_subset(name: str, data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    subset_dir = output_dir / name
    save_json(data, subset_dir / "dataset.json")
    save_json(make_profile(data), subset_dir / "profile.json")
    save_json(make_oracle_predictions(data), subset_dir / "oracle_predictions.json")
    return {
        "name": name,
        "path": str(subset_dir / "dataset.json"),
        "oracle_predictions": str(subset_dir / "oracle_predictions.json"),
        "profile": str(subset_dir / "profile.json"),
        "sequence_count": len(data),
        "turn_count": sum(len(seq.get("conversation", [])) for seq in data.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-size", type=int, default=10)
    parser.add_argument("--pilot-size", type=int, default=100)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    data = load_json(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(make_profile(data), output_dir / "dataset_profile.json")
    save_json(make_oracle_predictions(data), output_dir / "oracle_predictions_full.json")

    smoke = greedy_stratified_sample(data, args.smoke_size, args.seed)
    pilot = greedy_stratified_sample(data, args.pilot_size, args.seed + 1)

    manifest = {
        "source_dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "artifacts": [
            {
                "name": "full",
                "path": str(dataset_path),
                "oracle_predictions": str(output_dir / "oracle_predictions_full.json"),
                "profile": str(output_dir / "dataset_profile.json"),
                "sequence_count": len(data),
                "turn_count": sum(len(seq.get("conversation", [])) for seq in data.values()),
            },
            write_subset(f"smoke_{args.smoke_size}", smoke, output_dir),
            write_subset(f"pilot_{args.pilot_size}", pilot, output_dir),
        ],
    }
    save_json(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
