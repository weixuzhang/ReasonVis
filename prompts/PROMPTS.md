# Prompts

All prompts are defined in `experiments/`; the exact strings below are copied from the code.

## Direct prompting (`generate_predictions.py`)

System:

```
You are an expert system for conversational natural-language-to-visualization.

Return exactly one VQL statement:

Visualize [CHART_TYPE] [SQL_QUERY]

Allowed chart types: BAR, STACKED BAR, LINE, SCATTER, PIE, HEATMAP.
Use SQLite syntax. Use only tables and columns in the provided schema.
Do not explain your answer.
```

User:

```
Database: <db_id>
Tables:
- <table>(<col1>, <col2>, ...)
Potential joins:
- <table_a>.<col> = <table_b>.<col>

Conversation history:
Turn 1 user: <query>
Turn 1 VQL: <vql>              # gold under --history-mode oracle, model's own under predicted; "(none)" if empty

Current user query: <query>

Return only the VQL statement.
```

Schema lines come from the CSV headers; join hints are columns named `id` / `*_id` shared by two tables. `--include-sample-rows` appends two sample rows per table (used for method baselines, not for direct prompting). Temperature 0.2, max_tokens 1000.

## Method baselines (`generate_method_baselines.py`)

Shared system prompt:

```
You are an expert system for database-grounded natural-language-to-visualization.

ReasonVis output format:
Visualize [CHART_TYPE] [SQL_QUERY]

Allowed chart types: BAR, STACKED BAR, LINE, SCATTER, PIE, HEATMAP.
SQL must be executable SQLite.
Use only tables and columns in the provided schema.
Do not use non-SQL VQL clauses such as BIN BY.
Do not explain unless the requested JSON schema asks for a short internal field.
```

Each stage receives the schema, conversation history and current query and returns JSON `{"chart_type": ..., "sql": ...}` that is converted to VQL. Stage prompts are the functions named here:

| `--method` | Prompt functions |
|---|---|
| `decomposed` | `decomposed_prompt` |
| `execution_repair` | `decomposed_prompt` → `repair_prompt` (given the SQLite error) |
| `chat2vis_adapted` | `chat2vis_prompt` |
| `lida_adapted` | `lida_summary_goal_prompt` → `lida_generator_prompt` |
| `prompt4vis_adapted` | `prompt4vis_schema_filter_prompt` → `prompt4vis_generation_prompt` (few-shot examples from other databases) |
| `nvagent_adapted` | `processor_prompt` → `composer_prompt` → `repair_prompt` |

## LLM judge (`run_llm_judge_reasonvis.py`)

```
You are a strict but fair evaluator for conversational natural-language-to-visualization.

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
```

## Output parsing

Code fences and explanations are stripped and the first line starting with `Visualize` is kept; chart aliases (`BAR CHART`, `STACKED_BAR`, `HEAT MAP`) normalize to the six labels; the SQL must start with `SELECT` or `WITH`; execution errors, timeouts and missing SQL count as wrong.
