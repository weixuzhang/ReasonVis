#!/usr/bin/env bash
# Download the grounded CSV databases used by ReasonVis.
# They are the nvBench/Spider databases as packaged by VisEval (microsoft/VisEval).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/VisEval"
cd "$ROOT/VisEval"
if [ ! -f viseval_dataset.zip ]; then
  curl -L -o viseval_dataset.zip https://github.com/microsoft/VisEval/raw/main/viseval_dataset.zip
fi
unzip -q -o viseval_dataset.zip
# The archive unpacks as visEval_dataset; scripts expect the lowercase name.
if [ -d visEval_dataset ] && [ ! -d viseval_dataset ]; then
  mv visEval_dataset viseval_dataset
fi
echo "Databases ready: $ROOT/VisEval/viseval_dataset/databases ($(ls viseval_dataset/databases | wc -l) databases)"
