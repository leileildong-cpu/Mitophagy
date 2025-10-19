#!/usr/bin/env bash
set -e

# 可选：自动激活conda环境（若conda已在PATH）
if command -v conda &> /dev/null
then
  eval "$(conda shell.bash hook)"
  conda activate mitophagy || true
fi

python src/run.py --config config.yaml
echo "Done. See outputs/ for results."