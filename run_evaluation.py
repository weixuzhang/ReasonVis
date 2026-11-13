#!/usr/bin/env python
"""
Run the ReasonVis evaluation on prediction data.

This script runs the unified ReasonVis evaluator on a given prediction file,
comparing it to ground truth data and generating evaluation metrics.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

from reasonvis_evaluator import ReasonVisEvaluator


def get_default_eval_config(
    sql_equivalence_threshold: float = 0.95,
    case_insensitive_comparison: bool = True,
    visualization_type_normalization: bool = True,
    save_detailed_results: bool = True,
    generate_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Default evaluation settings aligned with benchmark-style use:
    execution-based SQL comparison when CSV databases are available under database_path.

    For quick runs without VisEval databases, pass evaluation_string_match.json via --config
    or override custom_evaluator_settings.skip_sql_execution to true.
    """
    return {
        "sql_equivalence_threshold": sql_equivalence_threshold,
        "case_insensitive_comparison": case_insensitive_comparison,
        "visualization_type_normalization": visualization_type_normalization,
        "save_detailed_results": save_detailed_results,
        "generate_visualizations": generate_visualizations,
        "evaluations": {
            "sql_accuracy": True,
            "vis_type_correctness": True,
            "exact_vql_match": True,
            "context_dependency": True,
            "hardness_level": True,
            "sequence_success": True,
            "turn_progression": True,
        },
        "custom_evaluator_settings": {
            "skip_sql_execution": False,
            "use_normalized_string_comparison": True,
            "lenient_sql_comparison": False,
            "lenient_vis_type_matching": True,
            "aggressive_sql_normalization": False,
            "lenient_result_comparison": False,
        },
    }


def merge_eval_config(base: Dict[str, Any], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge evaluation-style config (evaluations + custom_evaluator_settings)."""
    if not overlay:
        return deepcopy(base)
    out = deepcopy(base)
    for key, value in overlay.items():
        if key == "evaluations" and isinstance(value, dict):
            out.setdefault("evaluations", {})
            for ek, ev in value.items():
                out["evaluations"][ek] = ev
        elif key == "custom_evaluator_settings" and isinstance(value, dict):
            out.setdefault("custom_evaluator_settings", {})
            out["custom_evaluator_settings"].update(value)
        else:
            out[key] = value
    return out


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ReasonVis evaluation")
    
    parser.add_argument("--ground_truth", "-g", type=str, required=True,
                        help="Path to ground truth dataset file")
    
    parser.add_argument("--predictions", "-p", type=str, required=True,
                        help="Path to prediction file")
    
    parser.add_argument("--database_path", "-d", type=str, 
                        default="VisEval/viseval_dataset/databases",
                        help="Path to database directory")
    
    parser.add_argument("--output_dir", "-o", type=str, 
                        default="evaluation_results",
                        help="Directory to save evaluation results")
    
    parser.add_argument("--config", "-c", type=str,
                        help="Optional JSON configuration file")
    
    parser.add_argument("--sql_equivalence_threshold", type=float, default=0.95,
                        help="Threshold for SQL result equivalence (default: 0.95)")
    
    parser.add_argument("--no_case_sensitive", action="store_false", dest="case_insensitive",
                        help="Disable case-insensitive comparison")
    
    parser.add_argument("--no_vis_normalization", action="store_false", dest="vis_normalization",
                        help="Disable visualization type normalization")
    
    parser.add_argument("--no_detailed_results", action="store_false", dest="save_detailed",
                        help="Disable saving detailed results")
    
    parser.add_argument("--no_visualizations", action="store_false", dest="generate_visualizations",
                        help="Disable generating visualization plots")
    
    parser.add_argument("--disable_metric", "-dm", type=str, action="append",
                        choices=["sql_accuracy", "vis_type_correctness", "exact_vql_match", 
                                "context_dependency", "hardness_level", "sequence_success", 
                                "turn_progression"],
                        help="Disable specific evaluation metrics")
    
    return parser.parse_args()

def main():
    """Run the evaluation."""
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        return 1
    
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        return 1
    
    # Build configuration from shared defaults + command line arguments
    config = get_default_eval_config(
        sql_equivalence_threshold=args.sql_equivalence_threshold,
        case_insensitive_comparison=args.case_insensitive,
        visualization_type_normalization=args.vis_normalization,
        save_detailed_results=args.save_detailed,
        generate_visualizations=args.generate_visualizations,
    )

    # Load config from file if provided (overrides command-line scalar defaults)
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            config = merge_eval_config(config, file_config)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return 1

    # Disable specific metrics if requested (applies after file merge)
    if args.disable_metric:
        for metric in args.disable_metric:
            if metric in config["evaluations"]:
                config["evaluations"][metric] = False
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    evaluator = ReasonVisEvaluator(
        ground_truth_path=args.ground_truth,
        predictions_path=args.predictions,
        database_path=args.database_path,
        output_dir=args.output_dir,
        config=config
    )
    
    print(f"Starting evaluation of {args.predictions} against {args.ground_truth}")
    metrics = evaluator.evaluate_all()
    
    # Print summary of results
    print("\nEvaluation Summary:")
    print(f"SQL Accuracy: {metrics['overall'].get('sql_accuracy', 0):.4f}")
    print(f"Visualization Type Accuracy: {metrics['overall'].get('vis_type_accuracy', 0):.4f}")
    print(f"Turn Success Rate: {metrics['overall'].get('turn_success_rate', 0):.4f}")
    print(f"Exact VQL Match Rate: {metrics['overall'].get('exact_vql_accuracy', 0):.4f}")
    print(f"Context-Dependent Turn Success: {metrics['overall'].get('context_dependent_accuracy', 0):.4f}")
    print(f"Context-Independent Turn Success: {metrics['overall'].get('context_independent_accuracy', 0):.4f}")
    print(f"Sequence Complete Success Rate: {metrics['overall'].get('sequence_complete_success_rate', 0):.4f}")
    print(f"First Turn Success Rate: {metrics['overall'].get('sequence_first_turn_success_rate', 0):.4f}")
    print(f"Last Turn Success Rate: {metrics['overall'].get('sequence_last_turn_success_rate', 0):.4f}")
    print(f"Turn Progression Degradation: {metrics['overall'].get('turn_degradation', 0):.4f}")
    
    print(f"\nDetailed results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 