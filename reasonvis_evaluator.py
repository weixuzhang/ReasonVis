import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import csv
import re
import logging
import time
from typing import Dict, List, Any, Union, Tuple
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reasonvis_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reasonvis_evaluator")

class ReasonVisEvaluator:
    """
    Unified evaluator for multi-turn ReasonVis dataset predictions.
    Combines simplified and enhanced evaluation capabilities with complete metrics.
    """

    CANONICAL_VIS_TYPES = ["STACKED BAR", "HEATMAP", "SCATTER", "LINE", "PIE", "BAR"]
    
    def __init__(self, 
                 ground_truth_path: str,
                 predictions_path: str,
                 database_path: str = "VisEval/viseval_dataset/databases",
                 output_dir: str = "evaluation_results",
                 config: Dict = None):
        """
        Initialize the evaluator with paths to the ground truth and prediction data.
        
        Args:
            ground_truth_path: Path to the ground truth dataset JSON file
            predictions_path: Path to the predictions JSON file
            database_path: Path to the database directory
            output_dir: Directory to save evaluation results
            config: Optional configuration dictionary to customize evaluation
        """
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path
        self.database_path = database_path
        self.output_dir = output_dir
        
        # Default configuration
        self.config = {
            "sql_equivalence_threshold": 0.95,
            "case_insensitive_comparison": True,
            "visualization_type_normalization": True,
            "save_detailed_results": True,
            "generate_visualizations": True,
            "evaluations": {
                "sql_accuracy": True,
                "vis_type_correctness": True,
                "exact_vql_match": True,
                "context_dependency": True,
                "hardness_level": True,
                "sequence_success": True,
                "turn_progression": True
            }
        }
        
        # Override default config with provided config
        if config:
            self._update_config(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ground truth and prediction data
        self.ground_truth = self._load_json(ground_truth_path)
        self.predictions = self._load_json(predictions_path)
        
        # Metrics storage
        self.metrics = {
            "overall": {},
            "per_sequence": {},
            "per_turn": {},
            "per_reasoning_level": {},
            "per_hardness": {},
            "per_context_dependency": {},
            "turn_progression": {}
        }
        
        # SQL evaluation results cache
        self.sql_results_cache = {}
        
        # Visualization type mapping for normalization
        self.vis_type_mapping = {
            # Bar chart variants
            "bar": "BAR", "barchart": "BAR", "bar chart": "BAR", 
            "bar_chart": "BAR", "vertical bar": "BAR", "column": "BAR",
            "column chart": "BAR", "columnchart": "BAR", "vertical_bar": "BAR",
            "vbar": "BAR", "barplot": "BAR", "bar plot": "BAR", "bar_plot": "BAR",
            
            # Stacked bar variants
            "stacked bar": "STACKED BAR", "stackedbar": "STACKED BAR",
            "stacked_bar": "STACKED BAR", "stacked bar chart": "STACKED BAR",
            "stacked": "STACKED BAR", "stackedchart": "STACKED BAR",
            "stacked_chart": "STACKED BAR", "stacked barchart": "STACKED BAR",
            "stacked column": "STACKED BAR", "stacked_column": "STACKED BAR",
            
            # Line chart variants
            "line": "LINE", "linechart": "LINE", "line chart": "LINE",
            "line_chart": "LINE", "linegraph": "LINE", "line graph": "LINE",
            "time series": "LINE", "timeseries": "LINE", "time_series": "LINE",
            "trend": "LINE", "trend line": "LINE", "trend_line": "LINE",
            
            # Scatter plot variants
            "scatter": "SCATTER", "scatterplot": "SCATTER", "scatter plot": "SCATTER",
            "scatter_plot": "SCATTER", "scattergraph": "SCATTER", "scatter graph": "SCATTER",
            "point": "SCATTER", "points": "SCATTER", "dot": "SCATTER", 
            "dot plot": "SCATTER", "dotplot": "SCATTER", "dot_plot": "SCATTER",
            
            # Pie chart variants
            "pie": "PIE", "piechart": "PIE", "pie chart": "PIE",
            "pie_chart": "PIE", "circle chart": "PIE", "donut": "PIE",
            "donut chart": "PIE", "donutchart": "PIE", "donut_chart": "PIE",
            
            # Heatmap variants
            "heatmap": "HEATMAP", "heat map": "HEATMAP", "heat_map": "HEATMAP",
            "density map": "HEATMAP", "density_map": "HEATMAP", "thermal": "HEATMAP",
            "thermal map": "HEATMAP", "thermal_map": "HEATMAP"
        }
    
    def _update_config(self, config: Dict) -> None:
        """
        Update configuration with user-provided values.
        
        Args:
            config: User configuration dictionary
        """
        for key, value in config.items():
            if key == "evaluations" and isinstance(value, dict):
                # Update specific evaluation flags
                for eval_key, eval_value in value.items():
                    if eval_key in self.config["evaluations"]:
                        self.config["evaluations"][eval_key] = eval_value
            else:
                # Update top-level config values
                self.config[key] = value
    
    def _load_json(self, file_path: str) -> Dict:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return {}
    
    def evaluate_all(self) -> Dict:
        """
        Run all enabled evaluations and return metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.ground_truth or not self.predictions:
            logger.error("Cannot evaluate: ground truth or predictions not loaded correctly")
            return {}
        
        logger.info("Starting ReasonVis evaluation...")
        
        # Run core evaluations based on config
        if self.config["evaluations"]["sql_accuracy"]:
            logger.info("Evaluating SQL accuracy...")
            self.evaluate_sql_accuracy()
        
        if self.config["evaluations"]["vis_type_correctness"]:
            logger.info("Evaluating visualization type correctness...")
            self.evaluate_visualization_type()
        
        if self.config["evaluations"]["exact_vql_match"]:
            logger.info("Evaluating exact VQL matching...")
            self.evaluate_exact_vql_match()
        
        # Calculate turn success rate (requires SQL and vis type evaluation)
        if self.config["evaluations"]["sql_accuracy"] and self.config["evaluations"]["vis_type_correctness"]:
            logger.info("Calculating turn success rate...")
            self.calculate_turn_success_rate()
        
        # Run enhanced evaluations based on config
        if self.config["evaluations"]["context_dependency"]:
            logger.info("Evaluating context dependency...")
            self.evaluate_context_dependency()
        
        if self.config["evaluations"]["hardness_level"]:
            logger.info("Evaluating by hardness level...")
            self.evaluate_by_hardness()
        
        if self.config["evaluations"]["sequence_success"]:
            logger.info("Evaluating sequence-level success...")
            self.evaluate_sequence_success()
        
        if self.config["evaluations"]["turn_progression"]:
            logger.info("Analyzing turn progression...")
            self.analyze_turn_progression()
        
        # Save results
        if self.config["save_detailed_results"]:
            logger.info("Saving evaluation results...")
            self.save_results()
        
        # Generate visualizations
        if self.config["generate_visualizations"]:
            logger.info("Generating visualizations...")
            self.generate_summary_plots()
        
        return self.metrics
    
    def evaluate_sql_accuracy(self) -> None:
        """
        Evaluate the accuracy of SQL queries by comparing execution results.
        Execution-based evaluation compares the results of running the predicted
        SQL against the results of running the ground truth SQL.
        """
        per_sequence_sql_accuracy = {}
        per_turn_sql_metrics = {}
        per_reasoning_sql = {
            "L1": {"correct": 0, "total": 0},
            "L2": {"correct": 0, "total": 0},
            "L3": {"correct": 0, "total": 0},
            "L4": {"correct": 0, "total": 0}
        }
        
        total_correct = 0
        total_turns = 0
        
        # Process each sequence
        for seq_id, gt_data in tqdm(self.ground_truth.items(), desc="Evaluating SQL"):
            if seq_id not in self.predictions:
                logger.warning(f"Sequence {seq_id} missing from predictions")
                continue
                
            pred_data = self.predictions[seq_id]
            db_id = gt_data.get("db_id")
            
            if not db_id:
                logger.warning(f"Database ID missing for sequence {seq_id}")
                continue
                
            # Create a temporary database with the CSV files
            try:
                conn = self._setup_temp_db(db_id)
            except Exception as e:
                logger.error(f"Failed to set up database for {db_id}: {str(e)}")
                continue
                
            gt_conversation = gt_data.get("conversation", [])
            pred_conversation = pred_data.get("conversation", [])
            
            seq_correct_turns = 0
            per_turn_sql_metrics[seq_id] = {}
            
            # Compare each turn
            for i, gt_turn in enumerate(gt_conversation):
                if i >= len(pred_conversation):
                    logger.warning(f"Turn {i+1} missing from predictions for sequence {seq_id}")
                    continue
                    
                pred_turn = pred_conversation[i]
                turn_idx = str(i+1)
                
                # Extract SQL from ground truth and prediction
                gt_sql = self._extract_sql(gt_turn)
                pred_sql = self._extract_sql(pred_turn)
                
                if not gt_sql or not pred_sql:
                    logger.warning(f"SQL missing for turn {i+1} in sequence {seq_id}")
                    continue
                
                # Compare SQL execution results
                is_correct = False
                try:
                    is_correct = self._compare_sql_results(conn, gt_sql, pred_sql)
                except Exception as e:
                    logger.error(f"Error comparing SQL for turn {i+1} in sequence {seq_id}: {str(e)}")
                
                # Update metrics
                if is_correct:
                    seq_correct_turns += 1
                    total_correct += 1
                
                total_turns += 1
                
                # Store per-turn metrics
                if seq_id not in self.metrics["per_turn"]:
                    self.metrics["per_turn"][seq_id] = {}
                
                if turn_idx not in self.metrics["per_turn"][seq_id]:
                    self.metrics["per_turn"][seq_id][turn_idx] = {}
                
                self.metrics["per_turn"][seq_id][turn_idx].update({
                    "sql_correct": is_correct,
                    "gt_sql": gt_sql,
                    "pred_sql": pred_sql
                })
                
                # Update reasoning level metrics
                reasoning_level = gt_turn.get("reasoning_level")
                if reasoning_level in per_reasoning_sql:
                    per_reasoning_sql[reasoning_level]["total"] += 1
                    if is_correct:
                        per_reasoning_sql[reasoning_level]["correct"] += 1
            
            # Calculate sequence-level SQL accuracy
            if len(gt_conversation) > 0:
                seq_accuracy = seq_correct_turns / len(gt_conversation)
                per_sequence_sql_accuracy[seq_id] = seq_accuracy
            
            # Clean up
            conn.close()
        
        # Calculate overall SQL accuracy
        overall_sql_accuracy = total_correct / total_turns if total_turns > 0 else 0.0
        
        # Calculate per-reasoning level SQL accuracy
        per_reasoning_sql_accuracy = {}
        for level, counts in per_reasoning_sql.items():
            if counts["total"] > 0:
                per_reasoning_sql_accuracy[level] = counts["correct"] / counts["total"]
            else:
                per_reasoning_sql_accuracy[level] = 0.0
        
        # Store metrics
        self.metrics["overall"]["sql_accuracy"] = overall_sql_accuracy
        
        for seq_id, accuracy in per_sequence_sql_accuracy.items():
            if seq_id not in self.metrics["per_sequence"]:
                self.metrics["per_sequence"][seq_id] = {}
            self.metrics["per_sequence"][seq_id]["sql_accuracy"] = accuracy
        
        for level, accuracy in per_reasoning_sql_accuracy.items():
            if level not in self.metrics["per_reasoning_level"]:
                self.metrics["per_reasoning_level"][level] = {}
            self.metrics["per_reasoning_level"][level]["sql_accuracy"] = accuracy
        
        logger.info(f"Overall SQL accuracy: {overall_sql_accuracy:.4f}")
    
    def _extract_sql(self, turn_data: Dict) -> str:
        """
        Extract SQL query from turn data.
        
        Args:
            turn_data: Turn data containing SQL
            
        Returns:
            SQL query string
        """
        try:
            if "query" in turn_data and isinstance(turn_data["query"], dict):
                canonical_sql = turn_data["query"].get("canonical_sql")
                if canonical_sql:
                    return canonical_sql

            # For ground truth
            if "vis_query" in turn_data and "data_part" in turn_data["vis_query"]:
                sql_part = turn_data["vis_query"]["data_part"].get("sql_part", "")
                if sql_part:
                    return sql_part

            # For predictions in VQL format
            if "vql" in turn_data:
                _, sql = self._parse_vql(turn_data["vql"])
                if sql:
                    return sql

            if "vis_query" in turn_data and "VQL" in turn_data["vis_query"]:
                _, sql = self._parse_vql(turn_data["vis_query"]["VQL"])
                if sql:
                    return sql

            return ""
        except Exception as e:
            logger.warning(f"Error extracting SQL: {str(e)}")
            return ""

    def _parse_vql(self, vql: str) -> Tuple[str, str]:
        """
        Parse a VQL string into visualization type and SQL.

        Handles multi-token chart types such as STACKED BAR and SQL queries that
        begin with either SELECT or WITH.
        """
        if not vql:
            return "", ""

        text = re.sub(r"\s+", " ", vql).strip()
        match = re.match(r"^Visualize\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return "", ""

        rest = match.group(1).strip()
        for vis_type in self.CANONICAL_VIS_TYPES:
            pattern = re.compile(rf"^{re.escape(vis_type)}\s+((?:SELECT|WITH)\b.+)$", re.IGNORECASE | re.DOTALL)
            type_match = pattern.match(rest)
            if type_match:
                return vis_type, type_match.group(1).strip()

        generic_match = re.match(r"^([A-Za-z0-9_ ]+?)\s+((?:SELECT|WITH)\b.+)$", rest, flags=re.IGNORECASE | re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip(), generic_match.group(2).strip()

        return rest, ""
    
    def _setup_temp_db(self, db_id: str) -> sqlite3.Connection:
        """
        Set up a temporary SQLite database from CSV files for a given database ID.
        
        Args:
            db_id: Database ID
            
        Returns:
            SQLite connection
        """
        # Create in-memory database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        db_path = os.path.join(self.database_path, db_id)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database directory not found: {db_path}")
        
        # Load each CSV file as a table
        for file_name in os.listdir(db_path):
            # sqlite_sequence is a reserved internal table name; the CSV export
            # of some databases (soccer_1, store_1, world_1) includes it.
            if file_name.endswith('.csv') and file_name != 'sqlite_sequence.csv':
                table_name = file_name[:-4]
                csv_path = os.path.join(db_path, file_name)
                
                # Read CSV headers and data types
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader)
                    
                    # Create table with text columns (simple approach)
                    columns = ", ".join([f'"{header}" TEXT' for header in headers])
                    cursor.execute(f'CREATE TABLE "{table_name}" ({columns})')
                    
                    # Insert data
                    for row in csv_reader:
                        placeholders = ", ".join(["?"] * len(row))
                        cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', row)
        
        conn.commit()
        return conn
    
    def _compare_sql_results(self, conn: sqlite3.Connection, sql1: str, sql2: str) -> bool:
        """
        Compare the results of two SQL queries.
        
        Args:
            conn: SQLite connection
            sql1: First SQL query (ground truth)
            sql2: Second SQL query (prediction)
            
        Returns:
            True if the results are equivalent, False otherwise
        """
        # Create a cache key for the result pair
        cache_key = f"{sql1}__vs__{sql2}"
        if cache_key in self.sql_results_cache:
            return self.sql_results_cache[cache_key]
        
        # Default to skip SQL execution and use string comparison
        skip_execution = self.config.get("custom_evaluator_settings", {}).get("skip_sql_execution", True)
        
        # Clean and normalize the queries for comparison
        sql1_norm = self._normalize_sql(sql1)
        sql2_norm = self._normalize_sql(sql2)
        
        # If skipping execution, use string comparison
        if skip_execution:
            # Use normalized string comparison by default
            use_normalized = self.config.get("custom_evaluator_settings", {}).get("use_normalized_string_comparison", True)
            
            if use_normalized:
                # Basic string similarity using Jaccard similarity on tokens
                sql1_tokens = set(sql1_norm.lower().split())
                sql2_tokens = set(sql2_norm.lower().split())
                
                if not sql1_tokens or not sql2_tokens:
                    similarity = 0.0
                else:
                    intersection = len(sql1_tokens.intersection(sql2_tokens))
                    union = len(sql1_tokens.union(sql2_tokens))
                    similarity = intersection / union if union > 0 else 0.0
                
                # Cache and return result
                are_equivalent = similarity >= self.config["sql_equivalence_threshold"]
                self.sql_results_cache[cache_key] = are_equivalent
                return are_equivalent
            else:
                # Exact string match (normalized)
                are_equivalent = sql1_norm.lower() == sql2_norm.lower()
                self.sql_results_cache[cache_key] = are_equivalent
                return are_equivalent
        
        # Otherwise, try to execute SQL and compare results
        # Execute the first query
        try:
            results1, columns1 = self._execute_sql_with_timeout(conn, sql1)
        except Exception as e:
            logger.warning(f"Error executing ground truth SQL: {str(e)}")
            # If execution fails, fall back to string comparison
            return self._compare_sql_as_strings(sql1, sql2)

        # Execute the second query
        try:
            results2, columns2 = self._execute_sql_with_timeout(conn, sql2)
        except Exception as e:
            logger.warning(f"Error executing predicted SQL: {str(e)}")
            # If execution fails, fall back to string comparison
            return self._compare_sql_as_strings(sql1, sql2)
        
        # Compare the results
        are_equivalent = self._compare_result_sets(results1, columns1, results2, columns2)
        
        # Cache the result
        self.sql_results_cache[cache_key] = are_equivalent
        
        return are_equivalent

    def _execute_sql_with_timeout(self, conn: sqlite3.Connection, sql: str) -> Tuple[List[Tuple], List[str]]:
        """Execute SQL with an optional SQLite progress-handler timeout."""
        settings = self.config.get("custom_evaluator_settings", {})
        timeout_seconds = settings.get("sql_execution_timeout_seconds", 30)
        progress_steps = settings.get("sql_progress_handler_steps", 10000)

        start_time = time.monotonic()

        def interrupt_if_timed_out() -> int:
            if timeout_seconds and time.monotonic() - start_time > timeout_seconds:
                return 1
            return 0

        if timeout_seconds:
            conn.set_progress_handler(interrupt_if_timed_out, progress_steps)

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return results, columns
        finally:
            if timeout_seconds:
                conn.set_progress_handler(None, 0)
    
    def _compare_sql_as_strings(self, sql1: str, sql2: str) -> bool:
        """
        Compare two SQL queries as strings for similarity.
        Used as a fallback when SQL execution fails.
        
        Args:
            sql1: First SQL query
            sql2: Second SQL query
            
        Returns:
            True if the strings are similar enough, False otherwise
        """
        # Normalize the queries
        sql1_norm = self._normalize_sql(sql1)
        sql2_norm = self._normalize_sql(sql2)
        
        # If one SQL is empty but the other isn't, they're not equivalent
        if not sql1_norm or not sql2_norm:
            return False
        
        # Check for exact match after normalization
        if sql1_norm == sql2_norm:
            return True
        
        # Basic string similarity using Jaccard similarity on tokens
        sql1_tokens = set(sql1_norm.lower().split())
        sql2_tokens = set(sql2_norm.lower().split())
        
        if not sql1_tokens or not sql2_tokens:
            return False
        
        # Count common terms for each token
        intersection = len(sql1_tokens.intersection(sql2_tokens))
        union = len(sql1_tokens.union(sql2_tokens))
        similarity = intersection / union if union > 0 else 0.0
        
        # Lower the threshold for more lenient comparison
        threshold = self.config.get("sql_equivalence_threshold", 0.95)
        if self.config.get("custom_evaluator_settings", {}).get("lenient_sql_comparison", True):
            threshold = min(threshold, 0.5)  # Use a more lenient threshold
        
        # Compare against threshold
        return similarity >= threshold
    
    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL query
        """
        if not sql:
            return ""
            
        # Remove comments
        sql = re.sub(r'--.*?(\n|$)', ' ', sql)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # Remove any trailing semicolons
        sql = re.sub(r';+$', '', sql).strip()
        
        # Normalize quotes (all single quotes to double quotes)
        # This is a simplification - proper SQL parsing would be better
        processed_sql = ""
        in_string = False
        for char in sql:
            if char == "'" and not in_string:
                processed_sql += '"'
                in_string = True
            elif char == "'" and in_string:
                processed_sql += '"'
                in_string = False
            else:
                processed_sql += char
                
        sql = processed_sql
        
        # Normalize SQL keywords to uppercase
        keywords = [
            "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
            "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN",
            "UNION", "INTERSECT", "EXCEPT", "LIMIT", "OFFSET",
            "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "AS",
            "DESC", "ASC", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX"
        ]
        
        for keyword in keywords:
            # Replace word boundaries to ensure we don't replace parts of words
            sql = re.sub(r'\b' + keyword + r'\b', keyword, sql, flags=re.IGNORECASE)
        
        # Normalize function names
        functions = ["COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "IFNULL", "DATE", "YEAR", "MONTH", "DAY"]
        for func in functions:
            sql = re.sub(r'\b' + func + r'\s*\(', func + '(', sql, flags=re.IGNORECASE)
        
        # If configured, perform additional normalization
        if self.config.get("custom_evaluator_settings", {}).get("aggressive_sql_normalization", True):
            # Normalize table aliases (t1, t2 -> a, b) - simplistic approach
            table_aliases = {}
            alias_counter = 0
            
            # Find all potential aliases
            alias_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b'
            for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
                if match.group(2) not in keywords and match.group(2) not in functions:
                    if match.group(2) not in table_aliases:
                        table_aliases[match.group(2)] = f"alias_{alias_counter}"
                        alias_counter += 1
            
            # Replace aliases
            for alias, replacement in table_aliases.items():
                sql = re.sub(r'\b' + re.escape(alias) + r'\b', replacement, sql)
            
            # Remove leading zeros in numbers
            sql = re.sub(r'\b0+(\d+)', r'\1', sql)
            
            # Normalize floating point numbers
            sql = re.sub(r'(\d+)\.0+\b', r'\1', sql)
            
            # Normalize table names (remove 'Table.' prefix)
            sql = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*', '', sql)
            
            # Normalize DATE_FORMAT and other date functions
            sql = re.sub(r'DATE_FORMAT\s*\(\s*([^,]+),\s*[\'"][^\'"][\'"]\s*\)', r'\1', sql, flags=re.IGNORECASE)
            
            # Simplify complex date expressions
            sql = re.sub(r'DATE\s*\(\s*([^)]+)\s*\)', r'\1', sql, flags=re.IGNORECASE)
            
            # Normalize column names to lowercase for case-insensitive comparison
            sql = sql.lower()
        
        return sql
    
    def _compare_result_sets(self, results1: List, columns1: List, results2: List, columns2: List) -> bool:
        """
        Compare two SQL result sets for equivalence.
        
        Args:
            results1: First result set rows
            columns1: First result set column names
            results2: Second result set rows
            columns2: Second result set column names
            
        Returns:
            True if result sets are equivalent, False otherwise
        """
        # For empty result sets, both must be empty
        if not results1 and not results2:
            return True
        
        # If one is empty but the other isn't, they're not equivalent
        if not results1 or not results2:
            return False
        
        # Get lenient comparison settings
        lenient_comparison = self.config.get("custom_evaluator_settings", {}).get("lenient_result_comparison", True)
        
        if not lenient_comparison:
            # Result sets must have same number of columns in strict mode
            if len(columns1) != len(columns2):
                return False
        
        # Convert results to sets for comparison (ignoring column order)
        set1 = set(tuple(row) for row in results1)
        set2 = set(tuple(row) for row in results2)
        
        # Calculate Jaccard similarity for approximate matching
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union > 0 else 0.0
        
        # Check if similarity exceeds threshold
        return similarity >= self.config["sql_equivalence_threshold"]
    
    def evaluate_visualization_type(self) -> None:
        """
        Evaluate visualization type correctness.
        Checks if the predicted visualization type matches the ground truth type.
        """
        per_sequence_vis_accuracy = {}
        per_reasoning_vis = {
            "L1": {"correct": 0, "total": 0},
            "L2": {"correct": 0, "total": 0},
            "L3": {"correct": 0, "total": 0},
            "L4": {"correct": 0, "total": 0}
        }
        
        total_correct = 0
        total_turns = 0
        
        # Process each sequence
        for seq_id, gt_data in tqdm(self.ground_truth.items(), desc="Evaluating visualization types"):
            if seq_id not in self.predictions:
                logger.warning(f"Sequence {seq_id} missing from predictions")
                continue
                
            pred_data = self.predictions[seq_id]
            
            gt_conversation = gt_data.get("conversation", [])
            pred_conversation = pred_data.get("conversation", [])
            
            seq_correct_turns = 0
            
            # Compare each turn
            for i, gt_turn in enumerate(gt_conversation):
                if i >= len(pred_conversation):
                    logger.warning(f"Turn {i+1} missing from predictions for sequence {seq_id}")
                    continue
                    
                pred_turn = pred_conversation[i]
                turn_idx = str(i+1)
                
                # Extract visualization types
                gt_vis_type = self._extract_vis_type(gt_turn)
                pred_vis_type = self._extract_vis_type(pred_turn)
                
                # Handle missing visualization types with more robust fallbacks
                if not gt_vis_type:
                    # Try to extract from VQL if available in ground truth
                    if "vis_query" in gt_turn and "VQL" in gt_turn["vis_query"]:
                        vql = gt_turn["vis_query"]["VQL"]
                        gt_vis_type = self._extract_vis_type_from_vql(vql)
                    else:
                        logger.warning(f"No visualization type found in ground truth for turn {i+1} in sequence {seq_id}")
                        continue
                
                if not pred_vis_type:
                    # Try to extract from VQL if available in prediction
                    if "vql" in pred_turn:
                        vql = pred_turn["vql"]
                        pred_vis_type = self._extract_vis_type_from_vql(vql)
                    elif "vis_query" in pred_turn and "VQL" in pred_turn["vis_query"]:
                        vql = pred_turn["vis_query"]["VQL"]
                        pred_vis_type = self._extract_vis_type_from_vql(vql)
                    else:
                        logger.warning(f"No visualization type found in prediction for turn {i+1} in sequence {seq_id}")
                        continue
                
                # Normalize visualization types if configured
                if self.config["visualization_type_normalization"]:
                    gt_vis_type = self._normalize_vis_type(gt_vis_type, self.vis_type_mapping)
                    pred_vis_type = self._normalize_vis_type(pred_vis_type, self.vis_type_mapping)
                
                # Compare visualization types (case-insensitive if configured)
                if self.config["case_insensitive_comparison"]:
                    is_correct = gt_vis_type.upper() == pred_vis_type.upper()
                else:
                    is_correct = gt_vis_type == pred_vis_type
                
                # Update metrics
                if is_correct:
                    seq_correct_turns += 1
                    total_correct += 1
                
                total_turns += 1
                
                # Store per-turn metrics
                if seq_id not in self.metrics["per_turn"]:
                    self.metrics["per_turn"][seq_id] = {}
                    
                if turn_idx not in self.metrics["per_turn"][seq_id]:
                    self.metrics["per_turn"][seq_id][turn_idx] = {}
                
                self.metrics["per_turn"][seq_id][turn_idx].update({
                    "vis_type_correct": is_correct,
                    "gt_vis_type": gt_vis_type,
                    "pred_vis_type": pred_vis_type
                })
                
                # Update reasoning level metrics
                reasoning_level = gt_turn.get("reasoning_level")
                if reasoning_level in per_reasoning_vis:
                    per_reasoning_vis[reasoning_level]["total"] += 1
                    if is_correct:
                        per_reasoning_vis[reasoning_level]["correct"] += 1
            
            # Calculate sequence-level visualization type accuracy
            if len(gt_conversation) > 0:
                seq_accuracy = seq_correct_turns / len(gt_conversation)
                per_sequence_vis_accuracy[seq_id] = seq_accuracy
        
        # Calculate overall visualization type accuracy
        overall_vis_accuracy = total_correct / total_turns if total_turns > 0 else 0.0
        
        # Calculate per-reasoning level visualization type accuracy
        per_reasoning_vis_accuracy = {}
        for level, counts in per_reasoning_vis.items():
            if counts["total"] > 0:
                per_reasoning_vis_accuracy[level] = counts["correct"] / counts["total"]
            else:
                per_reasoning_vis_accuracy[level] = 0.0
        
        # Store metrics
        self.metrics["overall"]["vis_type_accuracy"] = overall_vis_accuracy
        
        for seq_id, accuracy in per_sequence_vis_accuracy.items():
            if seq_id not in self.metrics["per_sequence"]:
                self.metrics["per_sequence"][seq_id] = {}
            self.metrics["per_sequence"][seq_id]["vis_type_accuracy"] = accuracy
        
        for level, accuracy in per_reasoning_vis_accuracy.items():
            if level not in self.metrics["per_reasoning_level"]:
                self.metrics["per_reasoning_level"][level] = {}
            self.metrics["per_reasoning_level"][level]["vis_type_accuracy"] = accuracy
        
        logger.info(f"Overall visualization type accuracy: {overall_vis_accuracy:.4f}")
    
    def _extract_vis_type(self, turn_data: Dict) -> str:
        """
        Extract visualization type from turn data.
        
        Args:
            turn_data: Turn data containing visualization type
            
        Returns:
            Visualization type string
        """
        try:
            if turn_data.get("chart_type"):
                return turn_data["chart_type"]

            # For ground truth
            if "vis_query" in turn_data and "vis_part" in turn_data["vis_query"]:
                vis_part = turn_data["vis_query"]["vis_part"]
                vis_type, _ = self._parse_vql(f"{vis_part} SELECT 1")
                if vis_type:
                    return vis_type

            # For predictions in VQL format
            if "vql" in turn_data:
                vis_type, _ = self._parse_vql(turn_data["vql"])
                if vis_type:
                    return vis_type

            if "vis_query" in turn_data and "VQL" in turn_data["vis_query"]:
                vis_type, _ = self._parse_vql(turn_data["vis_query"]["VQL"])
                if vis_type:
                    return vis_type

            return ""
        except Exception as e:
            logger.warning(f"Error extracting visualization type: {str(e)}")
            return ""
    
    def _extract_vis_type_from_vql(self, vql: str) -> str:
        """
        Extract visualization type from a VQL statement.
        
        Args:
            vql: VQL statement
            
        Returns:
            Visualization type string
        """
        if not vql:
            return ""

        vis_type, _ = self._parse_vql(vql)
        if vis_type:
            return vis_type

        # Alternative patterns
        if "LINE" in vql.upper():
            return "LINE"
        if "STACKED BAR" in vql.upper() or "STACKEDBAR" in vql.upper():
            return "STACKED BAR"
        if "BAR" in vql.upper():
            return "BAR"
        if "SCATTER" in vql.upper():
            return "SCATTER"
        if "PIE" in vql.upper():
            return "PIE"
        if "HEATMAP" in vql.upper() or "HEAT MAP" in vql.upper():
            return "HEATMAP"
            
        return ""
    
    def _normalize_vis_type(self, vis_type: str, mapping: Dict[str, str]) -> str:
        """
        Normalize visualization type using a mapping.
        
        Args:
            vis_type: Visualization type string
            mapping: Dictionary mapping variant names to standard names
            
        Returns:
            Normalized visualization type string
        """
        if not vis_type:
            return ""
            
        vis_type_lower = vis_type.lower()
        
        # Check exact match in mapping
        if vis_type_lower in mapping:
            return mapping[vis_type_lower]
        
        # If lenient matching is enabled, try partial matches
        if self.config.get("custom_evaluator_settings", {}).get("lenient_vis_type_matching", True):
            # Try to match by checking if the visualization type contains any of the key words
            for key, normalized in mapping.items():
                if key in vis_type_lower or vis_type_lower in key:
                    return normalized
            
            # Additional normalization for special cases - first try a more flexible match
            # Fuzzy logic for common visualization type patterns
            stacked_patterns = ["stack", "stact", "stacked", "stack bar", "staked"]
            for pattern in stacked_patterns:
                if pattern in vis_type_lower:
                    return "STACKED BAR"
                
            line_patterns = ["line", "series", "trend", "lin chart"]
            for pattern in line_patterns:
                if pattern in vis_type_lower:
                    return "LINE"
                
            bar_patterns = ["bar", "column", "histogram", "barchar"]
            for pattern in bar_patterns:
                if pattern in vis_type_lower and "stack" not in vis_type_lower:
                    return "BAR"
                
            scatter_patterns = ["scatter", "scater", "point", "dot"]
            for pattern in scatter_patterns:
                if pattern in vis_type_lower:
                    return "SCATTER"
                
            pie_patterns = ["pie", "donut", "circle", "circular"]
            for pattern in pie_patterns:
                if pattern in vis_type_lower:
                    return "PIE"
                
            heat_patterns = ["heat", "thermal", "density"]
            for pattern in heat_patterns:
                if pattern in vis_type_lower:
                    return "HEATMAP"
        
        # If not in mapping, return original
        return vis_type
    
    def evaluate_exact_vql_match(self) -> None:
        """
        Evaluate exact VQL statement matching.
        This measures how often the predicted VQL matches the ground truth exactly.
        """
        per_sequence_vql_accuracy = {}
        per_reasoning_vql = {
            "L1": {"correct": 0, "total": 0},
            "L2": {"correct": 0, "total": 0},
            "L3": {"correct": 0, "total": 0},
            "L4": {"correct": 0, "total": 0}
        }
        
        total_correct = 0
        total_turns = 0
        
        # Process each sequence
        for seq_id, gt_data in tqdm(self.ground_truth.items(), desc="Evaluating exact VQL matching"):
            if seq_id not in self.predictions:
                logger.warning(f"Sequence {seq_id} missing from predictions")
                continue
                
            pred_data = self.predictions[seq_id]
            
            gt_conversation = gt_data.get("conversation", [])
            pred_conversation = pred_data.get("conversation", [])
            
            seq_correct_turns = 0
            
            # Compare each turn
            for i, gt_turn in enumerate(gt_conversation):
                if i >= len(pred_conversation):
                    logger.warning(f"Turn {i+1} missing from predictions for sequence {seq_id}")
                    continue
                    
                pred_turn = pred_conversation[i]
                turn_idx = str(i+1)
                
                # Extract VQL from ground truth
                gt_vql = self._extract_vql_from_gt(gt_turn)
                
                # Get VQL from prediction
                pred_vql = pred_turn.get("vql", "") or self._extract_vql_from_gt(pred_turn)
                
                if not gt_vql or not pred_vql:
                    logger.warning(f"VQL missing for turn {i+1} in sequence {seq_id}")
                    continue
                
                # Normalize VQLs for comparison
                gt_vql_norm = self._normalize_vql(gt_vql)
                pred_vql_norm = self._normalize_vql(pred_vql)
                
                # Compare VQLs
                is_exact_match = gt_vql_norm == pred_vql_norm
                
                # Update metrics
                if is_exact_match:
                    seq_correct_turns += 1
                    total_correct += 1
                
                total_turns += 1
                
                # Store per-turn metrics
                if seq_id not in self.metrics["per_turn"]:
                    self.metrics["per_turn"][seq_id] = {}
                    
                if turn_idx not in self.metrics["per_turn"][seq_id]:
                    self.metrics["per_turn"][seq_id][turn_idx] = {}
                
                self.metrics["per_turn"][seq_id][turn_idx].update({
                    "exact_vql_match": is_exact_match,
                    "gt_vql": gt_vql,
                    "pred_vql": pred_vql
                })
                
                # Update reasoning level metrics
                reasoning_level = gt_turn.get("reasoning_level")
                if reasoning_level in per_reasoning_vql:
                    per_reasoning_vql[reasoning_level]["total"] += 1
                    if is_exact_match:
                        per_reasoning_vql[reasoning_level]["correct"] += 1
            
            # Calculate sequence-level VQL accuracy
            if len(gt_conversation) > 0:
                seq_accuracy = seq_correct_turns / len(gt_conversation)
                per_sequence_vql_accuracy[seq_id] = seq_accuracy
        
        # Calculate overall VQL accuracy
        overall_vql_accuracy = total_correct / total_turns if total_turns > 0 else 0.0
        
        # Calculate per-reasoning level VQL accuracy
        per_reasoning_vql_accuracy = {}
        for level, counts in per_reasoning_vql.items():
            if counts["total"] > 0:
                per_reasoning_vql_accuracy[level] = counts["correct"] / counts["total"]
            else:
                per_reasoning_vql_accuracy[level] = 0.0
        
        # Store metrics
        self.metrics["overall"]["exact_vql_accuracy"] = overall_vql_accuracy
        
        for seq_id, accuracy in per_sequence_vql_accuracy.items():
            if seq_id not in self.metrics["per_sequence"]:
                self.metrics["per_sequence"][seq_id] = {}
            self.metrics["per_sequence"][seq_id]["exact_vql_accuracy"] = accuracy
        
        for level, accuracy in per_reasoning_vql_accuracy.items():
            if level not in self.metrics["per_reasoning_level"]:
                self.metrics["per_reasoning_level"][level] = {}
            self.metrics["per_reasoning_level"][level]["exact_vql_accuracy"] = accuracy
        
        logger.info(f"Overall exact VQL match accuracy: {overall_vql_accuracy:.4f}")
    
    def _extract_vql_from_gt(self, turn_data: Dict) -> str:
        """
        Extract VQL statement from ground truth turn data.
        
        Args:
            turn_data: Turn data containing VQL components
            
        Returns:
            VQL statement string
        """
        try:
            vql = ""
            if "query" in turn_data and isinstance(turn_data["query"], dict):
                vql = turn_data["query"].get("vql", "")
                if vql:
                    return vql

            if "vis_query" in turn_data and "VQL" in turn_data["vis_query"]:
                vql = turn_data["vis_query"]["VQL"]
                if vql:
                    return vql

            # Try to construct VQL from vis_query components
            if "vis_query" in turn_data:
                vis_part = turn_data["vis_query"].get("vis_part", "")
                if "data_part" in turn_data["vis_query"] and "sql_part" in turn_data["vis_query"]["data_part"]:
                    sql_part = turn_data["vis_query"]["data_part"]["sql_part"]
                    # Extract the visualization type
                    match = re.search(r'Visualize\s+([A-Za-z0-9_\s]+)', vis_part)
                    if match:
                        vis_type = match.group(1).strip()
                        vql = f"Visualize {vis_type} {sql_part}"
            
            return vql
        except Exception as e:
            logger.warning(f"Error extracting VQL: {str(e)}")
            return ""
    
    def _normalize_vql(self, vql: str) -> str:
        """
        Normalize VQL for comparison by normalizing whitespace and case for keywords.
        
        Args:
            vql: VQL statement string
            
        Returns:
            Normalized VQL statement string
        """
        if not vql:
            return ""
            
        # Remove extra whitespace
        vql = re.sub(r'\s+', ' ', vql).strip()
        
        # Normalize "Visualize" keyword
        vql = re.sub(r'^visualize\s+', 'Visualize ', vql, flags=re.IGNORECASE)
        
        # Normalize the SQL part
        vis_type, sql_text = self._parse_vql(vql)
        if vis_type and sql_text:
            vis_type = self._normalize_vis_type(vis_type, self.vis_type_mapping).upper()
            sql = self._normalize_sql(sql_text)
            return f"Visualize {vis_type} {sql}"

        return vql

    def _extract_context_labels(self, turn_data: Dict) -> List[str]:
        """
        Extract context-dependency labels for evaluation.

        ReasonVis stores the primary context level in
        context.dependency_level. Older datasets store a list in
        context_dependency. Prefer the dependency_level field when present.
        """
        context = turn_data.get("context", {})
        if isinstance(context, dict) and context.get("dependency_level"):
            return [context["dependency_level"]]

        context_dependency = turn_data.get("context_dependency", ["independent"])
        if isinstance(context_dependency, str):
            return [context_dependency]
        if isinstance(context_dependency, list) and context_dependency:
            return context_dependency
        return ["independent"]
    
    def calculate_turn_success_rate(self) -> None:
        """
        Calculate turn success rate based on SQL and visualization type accuracy.
        A turn is successful if both the SQL and visualization type are correct.
        """
        per_sequence_turn_success = {}
        per_reasoning_turn_success = {
            "L1": {"correct": 0, "total": 0},
            "L2": {"correct": 0, "total": 0},
            "L3": {"correct": 0, "total": 0},
            "L4": {"correct": 0, "total": 0}
        }
        
        total_correct = 0
        total_turns = 0
        
        # Process each sequence
        for seq_id, seq_turns in self.metrics["per_turn"].items():
            seq_correct_turns = 0
            seq_total_turns = len(seq_turns)
            
            for turn_idx, turn_metrics in seq_turns.items():
                # A turn is successful if both SQL and visualization type are correct
                sql_correct = turn_metrics.get("sql_correct", False)
                vis_type_correct = turn_metrics.get("vis_type_correct", False)
                turn_success = sql_correct and vis_type_correct
                
                # Update metrics
                if turn_success:
                    seq_correct_turns += 1
                    total_correct += 1
                
                total_turns += 1
                
                # Store per-turn metrics
                turn_metrics["turn_success"] = turn_success
                
                # Update reasoning level metrics
                gt_data = self.ground_truth.get(seq_id, {})
                gt_turns = gt_data.get("conversation", [])
                turn_idx_int = int(turn_idx) - 1
                
                if 0 <= turn_idx_int < len(gt_turns):
                    reasoning_level = gt_turns[turn_idx_int].get("reasoning_level")
                    if reasoning_level in per_reasoning_turn_success:
                        per_reasoning_turn_success[reasoning_level]["total"] += 1
                        if turn_success:
                            per_reasoning_turn_success[reasoning_level]["correct"] += 1
            
            # Calculate sequence-level turn success rate
            if seq_total_turns > 0:
                seq_success_rate = seq_correct_turns / seq_total_turns
                per_sequence_turn_success[seq_id] = seq_success_rate
        
        # Calculate overall turn success rate
        overall_turn_success = total_correct / total_turns if total_turns > 0 else 0.0
        
        # Calculate per-reasoning level turn success rate
        per_reasoning_success_rate = {}
        for level, counts in per_reasoning_turn_success.items():
            if counts["total"] > 0:
                per_reasoning_success_rate[level] = counts["correct"] / counts["total"]
            else:
                per_reasoning_success_rate[level] = 0.0
        
        # Store metrics
        self.metrics["overall"]["turn_success_rate"] = overall_turn_success
        
        for seq_id, success_rate in per_sequence_turn_success.items():
            if seq_id not in self.metrics["per_sequence"]:
                self.metrics["per_sequence"][seq_id] = {}
            self.metrics["per_sequence"][seq_id]["turn_success_rate"] = success_rate
        
        for level, success_rate in per_reasoning_success_rate.items():
            if level not in self.metrics["per_reasoning_level"]:
                self.metrics["per_reasoning_level"][level] = {}
            self.metrics["per_reasoning_level"][level]["turn_success_rate"] = success_rate
        
        logger.info(f"Overall turn success rate: {overall_turn_success:.4f}")
    
    def evaluate_context_dependency(self) -> None:
        """
        Evaluate performance based on context dependency.
        Analyzes how well the model handles context-dependent vs. independent queries.
        """
        context_dependent_turns = {"total": 0, "correct": 0}
        context_independent_turns = {"total": 0, "correct": 0}
        dependency_type_metrics = {}
        
        # Process each sequence
        for seq_id, gt_data in self.ground_truth.items():
            if seq_id not in self.metrics["per_turn"]:
                continue
                
            gt_conversation = gt_data.get("conversation", [])
            
            # Process each turn
            for i, gt_turn in enumerate(gt_conversation):
                turn_idx = str(i+1)
                
                if turn_idx not in self.metrics["per_turn"][seq_id]:
                    continue
                
                turn_metrics = self.metrics["per_turn"][seq_id][turn_idx]
                turn_success = turn_metrics.get("turn_success", False)
                
                # Get context dependency
                context_dependency = self._extract_context_labels(gt_turn)
                
                # Determine if turn is context-dependent
                is_context_dependent = "independent" not in context_dependency
                
                # Update context dependency metrics
                if is_context_dependent:
                    context_dependent_turns["total"] += 1
                    if turn_success:
                        context_dependent_turns["correct"] += 1
                else:
                    context_independent_turns["total"] += 1
                    if turn_success:
                        context_independent_turns["correct"] += 1
                
                # Update dependency type metrics
                for dep_type in context_dependency:
                    if dep_type not in dependency_type_metrics:
                        dependency_type_metrics[dep_type] = {"total": 0, "correct": 0}
                    dependency_type_metrics[dep_type]["total"] += 1
                    if turn_success:
                        dependency_type_metrics[dep_type]["correct"] += 1
        
        # Calculate overall context dependency accuracy
        context_dependent_accuracy = (
            context_dependent_turns["correct"] / context_dependent_turns["total"]
            if context_dependent_turns["total"] > 0 else 0.0
        )
        
        context_independent_accuracy = (
            context_independent_turns["correct"] / context_independent_turns["total"]
            if context_independent_turns["total"] > 0 else 0.0
        )
        
        # Calculate per-dependency type accuracy
        dependency_type_accuracy = {}
        for dep_type, counts in dependency_type_metrics.items():
            if counts["total"] > 0:
                dependency_type_accuracy[dep_type] = counts["correct"] / counts["total"]
            else:
                dependency_type_accuracy[dep_type] = 0.0
        
        # Store metrics
        self.metrics["overall"]["context_dependent_accuracy"] = context_dependent_accuracy
        self.metrics["overall"]["context_independent_accuracy"] = context_independent_accuracy
        
        self.metrics["per_context_dependency"] = {}
        for dep_type, acc in dependency_type_accuracy.items():
            counts = dependency_type_metrics[dep_type]
            self.metrics["per_context_dependency"][dep_type] = {
                "accuracy": acc,
                "turn_success_rate": acc,
                "correct": counts["correct"],
                "total": counts["total"],
            }
        
        logger.info(f"Context-dependent accuracy: {context_dependent_accuracy:.4f}")
        logger.info(f"Context-independent accuracy: {context_independent_accuracy:.4f}")
    
    def evaluate_by_hardness(self) -> None:
        """
        Evaluate performance based on query hardness level.
        Analyzes how well the model handles queries of different difficulty levels.
        """
        hardness_metrics = {
            "Easy": {"total": 0, "correct": 0},
            "Medium": {"total": 0, "correct": 0},
            "Hard": {"total": 0, "correct": 0}
        }
        
        # Process each sequence
        for seq_id, gt_data in self.ground_truth.items():
            if seq_id not in self.metrics["per_turn"]:
                continue
                
            gt_conversation = gt_data.get("conversation", [])
            
            # Process each turn
            for i, gt_turn in enumerate(gt_conversation):
                turn_idx = str(i+1)
                
                if turn_idx not in self.metrics["per_turn"][seq_id]:
                    continue
                
                turn_metrics = self.metrics["per_turn"][seq_id][turn_idx]
                turn_success = turn_metrics.get("turn_success", False)
                
                # Get hardness level
                hardness = gt_turn.get("hardness")
                
                # Update hardness metrics
                if hardness in hardness_metrics:
                    hardness_metrics[hardness]["total"] += 1
                    if turn_success:
                        hardness_metrics[hardness]["correct"] += 1
        
        # Calculate hardness level accuracy
        hardness_accuracy = {}
        for level, counts in hardness_metrics.items():
            if counts["total"] > 0:
                hardness_accuracy[level] = counts["correct"] / counts["total"]
            else:
                hardness_accuracy[level] = 0.0
        
        # Store metrics
        self.metrics["overall"]["easy_accuracy"] = hardness_accuracy.get("Easy", 0.0)
        self.metrics["overall"]["medium_accuracy"] = hardness_accuracy.get("Medium", 0.0)
        self.metrics["overall"]["hard_accuracy"] = hardness_accuracy.get("Hard", 0.0)
        
        self.metrics["per_hardness"] = {
            level: {"accuracy": acc} for level, acc in hardness_accuracy.items()
        }
        
        logger.info(f"Easy query accuracy: {hardness_accuracy.get('Easy', 0.0):.4f}")
        logger.info(f"Medium query accuracy: {hardness_accuracy.get('Medium', 0.0):.4f}")
        logger.info(f"Hard query accuracy: {hardness_accuracy.get('Hard', 0.0):.4f}")
    
    def evaluate_sequence_success(self) -> None:
        """
        Evaluate sequence-level success rates.
        Includes complete success and partial success metrics.
        """
        complete_success_count = 0
        success_75_percent_count = 0
        success_50_percent_count = 0
        success_25_percent_count = 0
        first_turn_success_count = 0
        last_turn_success_count = 0
        total_sequences = 0
        
        # Process each sequence
        for seq_id, seq_metrics in self.metrics["per_sequence"].items():
            if seq_id not in self.metrics["per_turn"]:
                continue
                
            turn_metrics = self.metrics["per_turn"][seq_id]
            turn_success_values = [
                metrics.get("turn_success", False)
                for _, metrics in sorted(turn_metrics.items(), key=lambda x: int(x[0]))
            ]
            
            if not turn_success_values:
                continue
                
            total_sequences += 1
            num_turns = len(turn_success_values)
            num_successful = sum(turn_success_values)
            success_rate = num_successful / num_turns
            
            # Check sequence-level success
            if success_rate == 1.0:
                complete_success_count += 1
            if success_rate >= 0.75:
                success_75_percent_count += 1
            if success_rate >= 0.5:
                success_50_percent_count += 1
            if success_rate >= 0.25:
                success_25_percent_count += 1
            
            # Check first turn success
            if turn_success_values[0]:
                first_turn_success_count += 1
            
            # Check last turn success
            if turn_success_values[-1]:
                last_turn_success_count += 1
        
        # Calculate success rates
        complete_success_rate = complete_success_count / total_sequences if total_sequences > 0 else 0.0
        success_75_percent_rate = success_75_percent_count / total_sequences if total_sequences > 0 else 0.0
        success_50_percent_rate = success_50_percent_count / total_sequences if total_sequences > 0 else 0.0
        success_25_percent_rate = success_25_percent_count / total_sequences if total_sequences > 0 else 0.0
        first_turn_success_rate = first_turn_success_count / total_sequences if total_sequences > 0 else 0.0
        last_turn_success_rate = last_turn_success_count / total_sequences if total_sequences > 0 else 0.0
        
        # Store metrics
        self.metrics["overall"]["sequence_complete_success_rate"] = complete_success_rate
        self.metrics["overall"]["sequence_75_percent_success_rate"] = success_75_percent_rate
        self.metrics["overall"]["sequence_50_percent_success_rate"] = success_50_percent_rate
        self.metrics["overall"]["sequence_25_percent_success_rate"] = success_25_percent_rate
        self.metrics["overall"]["sequence_first_turn_success_rate"] = first_turn_success_rate
        self.metrics["overall"]["sequence_last_turn_success_rate"] = last_turn_success_rate
        
        logger.info(f"Sequence complete success rate: {complete_success_rate:.4f}")
        logger.info(f"Sequence ≥75% success rate: {success_75_percent_rate:.4f}")
        logger.info(f"Sequence ≥50% success rate: {success_50_percent_rate:.4f}")
        logger.info(f"Sequence first turn success rate: {first_turn_success_rate:.4f}")
        logger.info(f"Sequence last turn success rate: {last_turn_success_rate:.4f}")
    
    def analyze_turn_progression(self) -> None:
        """
        Analyze how performance changes across turns in a conversation.
        Measures degradation from first to last turn.
        """
        # Initialize metrics for each turn position
        turn_position_metrics = {}
        max_turns = 0
        
        # Process each sequence
        for seq_id, seq_turns in self.metrics["per_turn"].items():
            # Get turn indices in numerical order
            turn_indices = sorted([int(idx) for idx in seq_turns.keys()])
            max_turns = max(max_turns, max(turn_indices))
            
            # Process each turn
            for turn_idx in turn_indices:
                turn_metrics = seq_turns[str(turn_idx)]
                turn_success = turn_metrics.get("turn_success", False)
                
                # Initialize metrics for this turn position if needed
                if turn_idx not in turn_position_metrics:
                    turn_position_metrics[turn_idx] = {"total": 0, "correct": 0}
                
                # Update metrics
                turn_position_metrics[turn_idx]["total"] += 1
                if turn_success:
                    turn_position_metrics[turn_idx]["correct"] += 1
        
        # Calculate success rate for each turn position
        turn_position_success_rates = {}
        for turn_idx, counts in turn_position_metrics.items():
            if counts["total"] > 0:
                success_rate = counts["correct"] / counts["total"]
                turn_position_success_rates[turn_idx] = success_rate
            else:
                turn_position_success_rates[turn_idx] = 0.0
        
        # Calculate turn progression metrics
        first_turn_rate = turn_position_success_rates.get(1, 0.0)
        last_turn_rate = turn_position_success_rates.get(max_turns, 0.0)
        
        # Calculate degradation (how much performance drops from first to last turn)
        turn_degradation = first_turn_rate - last_turn_rate
        
        # Store metrics
        self.metrics["turn_progression"] = {
            "position_success_rates": turn_position_success_rates,
            "first_turn_rate": first_turn_rate,
            "last_turn_rate": last_turn_rate,
            "degradation": turn_degradation
        }
        
        self.metrics["overall"]["turn_degradation"] = turn_degradation
        
        logger.info(f"First turn success rate: {first_turn_rate:.4f}")
        logger.info(f"Last turn success rate: {last_turn_rate:.4f}")
        logger.info(f"Turn progression degradation: {turn_degradation:.4f}")
    
    def save_results(self) -> None:
        """
        Save evaluation results to files.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save overall metrics
        overall_path = os.path.join(self.output_dir, "overall_metrics.json")
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["overall"], f, indent=2)
        
        # Save per-sequence metrics
        per_sequence_path = os.path.join(self.output_dir, "per_sequence_metrics.json")
        with open(per_sequence_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["per_sequence"], f, indent=2)
        
        # Save per-turn metrics
        per_turn_path = os.path.join(self.output_dir, "per_turn_metrics.json")
        with open(per_turn_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["per_turn"], f, indent=2)
        
        # Save reasoning level metrics
        reasoning_path = os.path.join(self.output_dir, "per_reasoning_level_metrics.json")
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["per_reasoning_level"], f, indent=2)
        
        # Save hardness metrics
        hardness_path = os.path.join(self.output_dir, "per_hardness_metrics.json")
        with open(hardness_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["per_hardness"], f, indent=2)
        
        # Save context dependency metrics
        context_path = os.path.join(self.output_dir, "per_context_dependency_metrics.json")
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["per_context_dependency"], f, indent=2)
        
        # Save turn progression metrics
        progression_path = os.path.join(self.output_dir, "turn_progression_metrics.json")
        with open(progression_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics["turn_progression"], f, indent=2)
        
        # Save detailed per-example results
        detailed_path = os.path.join(self.output_dir, "detailed_results.json")
        detailed_results = {
            "predictions": self.predictions_path,
            "ground_truth": self.ground_truth_path,
            "per_turn": self.metrics["per_turn"]
        }
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def generate_summary_plots(self) -> None:
        """
        Generate summary visualizations for evaluation metrics.
        """
        plot_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('ggplot')
        
        # Plot 1: Overall accuracy metrics
        self._plot_overall_metrics(plot_dir)
        
        # Plot 2: Reasoning level comparison
        self._plot_reasoning_level_metrics(plot_dir)
        
        # Plot 3: Hardness level comparison
        self._plot_hardness_metrics(plot_dir)
        
        # Plot 4: Context dependency comparison
        self._plot_context_dependency_metrics(plot_dir)
        
        # Plot 5: Turn progression
        self._plot_turn_progression(plot_dir)
        
        # Plot 6: Sequence success thresholds
        self._plot_sequence_success_thresholds(plot_dir)
        
        logger.info(f"Summary plots generated in {plot_dir}")
    
    def _plot_overall_metrics(self, plot_dir: str) -> None:
        """
        Plot overall accuracy metrics.
        
        Args:
            plot_dir: Directory to save the plot
        """
        metrics = {
            "SQL Accuracy": self.metrics["overall"].get("sql_accuracy", 0),
            "Vis Type Accuracy": self.metrics["overall"].get("vis_type_accuracy", 0),
            "Turn Success Rate": self.metrics["overall"].get("turn_success_rate", 0),
            "Exact VQL Match": self.metrics["overall"].get("exact_vql_accuracy", 0)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values(), color='skyblue')
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Accuracy Metrics')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for i, (metric, value) in enumerate(metrics.items()):
            ax.text(i, value + 0.01, f'{value:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'overall_metrics.png'))
        plt.close()
    
    def _plot_reasoning_level_metrics(self, plot_dir: str) -> None:
        """
        Plot metrics across reasoning levels.
        
        Args:
            plot_dir: Directory to save the plot
        """
        reasoning_levels = ["L1", "L2", "L3", "L4"]
        metrics = ["sql_accuracy", "vis_type_accuracy", "turn_success_rate"]
        
        data = []
        for level in reasoning_levels:
            if level in self.metrics["per_reasoning_level"]:
                level_data = []
                for metric in metrics:
                    level_data.append(self.metrics["per_reasoning_level"][level].get(metric, 0))
                data.append(level_data)
            else:
                data.append([0, 0, 0])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(reasoning_levels))
        width = 0.25
        
        # Plot bars for each metric
        bar1 = ax.bar(x - width, [d[0] for d in data], width, label='SQL Accuracy')
        bar2 = ax.bar(x, [d[1] for d in data], width, label='Vis Type Accuracy')
        bar3 = ax.bar(x + width, [d[2] for d in data], width, label='Turn Success Rate')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance Across Reasoning Levels')
        ax.set_xticks(x)
        ax.set_xticklabels(reasoning_levels)
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'reasoning_level_metrics.png'))
        plt.close()
    
    def _plot_hardness_metrics(self, plot_dir: str) -> None:
        """
        Plot metrics across hardness levels.
        
        Args:
            plot_dir: Directory to save the plot
        """
        hardness_levels = ["Easy", "Medium", "Hard"]
        
        data = []
        for level in hardness_levels:
            if level in self.metrics["per_hardness"]:
                data.append(self.metrics["per_hardness"][level].get("accuracy", 0))
            else:
                data.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(hardness_levels, data, color=['green', 'orange', 'red'])
        ax.set_ylabel('Success Rate')
        ax.set_title('Performance Across Hardness Levels')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for i, value in enumerate(data):
            ax.text(i, value + 0.01, f'{value:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'hardness_metrics.png'))
        plt.close()
    
    def _plot_context_dependency_metrics(self, plot_dir: str) -> None:
        """
        Plot metrics for context dependency.
        
        Args:
            plot_dir: Directory to save the plot
        """
        # Basic context dependency comparison
        context_dep = self.metrics["overall"].get("context_dependent_accuracy", 0)
        context_indep = self.metrics["overall"].get("context_independent_accuracy", 0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Context-Dependent', 'Context-Independent'], [context_dep, context_indep], color=['purple', 'teal'])
        ax.set_ylabel('Success Rate')
        ax.set_title('Context Dependency Performance')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        ax.text(0, context_dep + 0.01, f'{context_dep:.2f}', ha='center')
        ax.text(1, context_indep + 0.01, f'{context_indep:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'context_dependency_basic.png'))
        plt.close()
        
        # Detailed context dependency types
        if self.metrics["per_context_dependency"]:
            dep_types = []
            accuracies = []
            
            for dep_type, metrics in self.metrics["per_context_dependency"].items():
                dep_types.append(dep_type)
                accuracies.append(metrics.get("accuracy", 0))
            
            # Sort by accuracy for better visualization
            sorted_indices = np.argsort(accuracies)[::-1]
            sorted_types = [dep_types[i] for i in sorted_indices]
            sorted_accuracies = [accuracies[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(sorted_types, sorted_accuracies, color='slateblue')
            ax.set_ylabel('Success Rate')
            ax.set_title('Performance by Context Dependency Type')
            ax.set_ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for i, value in enumerate(sorted_accuracies):
                ax.text(i, value + 0.01, f'{value:.2f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'context_dependency_detailed.png'))
            plt.close()
    
    def _plot_turn_progression(self, plot_dir: str) -> None:
        """
        Plot turn progression metrics.
        
        Args:
            plot_dir: Directory to save the plot
        """
        turn_metrics = self.metrics["turn_progression"].get("position_success_rates", {})
        
        if not turn_metrics:
            return
        
        # Sort turns by position
        turn_positions = sorted([int(pos) for pos in turn_metrics.keys()])
        success_rates = [turn_metrics[pos] for pos in turn_positions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(turn_positions, success_rates, marker='o', linestyle='-', color='royalblue', linewidth=2)
        ax.set_xlabel('Turn Position')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Progression Across Turns')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(turn_positions)
        ax.grid(True)
        
        # Annotate each point
        for i, (pos, rate) in enumerate(zip(turn_positions, success_rates)):
            ax.annotate(f'{rate:.2f}', (pos, rate), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'turn_progression.png'))
        plt.close()
    
    def _plot_sequence_success_thresholds(self, plot_dir: str) -> None:
        """
        Plot sequence success thresholds.
        
        Args:
            plot_dir: Directory to save the plot
        """
        thresholds = {
            '25%': self.metrics["overall"].get("sequence_25_percent_success_rate", 0),
            '50%': self.metrics["overall"].get("sequence_50_percent_success_rate", 0),
            '75%': self.metrics["overall"].get("sequence_75_percent_success_rate", 0),
            '100%': self.metrics["overall"].get("sequence_complete_success_rate", 0)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(thresholds.keys(), thresholds.values(), color='lightseagreen')
        ax.set_ylabel('Percentage of Sequences')
        ax.set_title('Sequence Success Rate Thresholds')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for i, (threshold, value) in enumerate(thresholds.items()):
            ax.text(i, value + 0.01, f'{value:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'sequence_success_thresholds.png'))
        plt.close()
