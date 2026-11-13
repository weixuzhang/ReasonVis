# ReasonVis

Evaluation code and prompts for conversational NL2VIS. Each turn's target is `Visualize [CHART_TYPE] [SQLite SQL]`; SQL is judged by execution against the grounded database, chart type by label match.

```
reasonvis_evaluator.py / run_evaluation.py   evaluator and CLI
experiments/generate_predictions.py          direct prompting (any OpenAI-compatible endpoint)
experiments/generate_method_baselines.py     Decompose, ExecRepair, nvAgent, Chat2VIS, LIDA, Prompt4Vis
experiments/repair_predictions_execution.py  execution-guided repair of a prediction file
experiments/analyze_results.py               tables, error taxonomy, bootstrap CIs
experiments/run_llm_judge_reasonvis.py       LLM judge
experiments/configs/                         evaluator configs
prompts/PROMPTS.md                           prompts
scripts/download_databases.sh                grounded CSV databases
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_databases.sh      # VisEval/viseval_dataset/databases/<db_id>/*.csv
# put the benchmark file at data/reasonvis.json
export OPENAI_API_KEY=...
```

## Run

```bash
# predictions (--history-mode none|oracle|predicted; --base-url for vLLM/OpenRouter)
python experiments/generate_predictions.py --dataset data/reasonvis.json \
  --output runs/gpt-4.1.json --model gpt-4.1 --history-mode predicted --temperature 0.2

# method baselines: same flags plus --method decomposed|execution_repair|nvagent_adapted|chat2vis_adapted|lida_adapted|prompt4vis_adapted --include-sample-rows
python experiments/generate_method_baselines.py --dataset data/reasonvis.json \
  --output runs/decomposed.json --method decomposed --model gpt-4.1-mini --history-mode predicted --include-sample-rows

# evaluation
python run_evaluation.py --ground_truth data/reasonvis.json --predictions runs/gpt-4.1.json \
  --database_path VisEval/viseval_dataset/databases --output_dir runs/gpt-4.1_eval \
  --config experiments/configs/eval_execution_no_plots.json

# aggregate (edit RUNS at the top of the script)
python experiments/analyze_results.py --dataset data/reasonvis.json --output-dir results
```

Prediction file: JSON keyed by sequence id, each with `db_id` and `conversation: [{"turn": 1, "vql": "Visualize BAR SELECT ..."}, ...]`.

## License

MIT. Databases come from Spider / nvBench via VisEval under their original licenses.
