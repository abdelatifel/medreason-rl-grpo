#!/usr/bin/env bash
# Full pipeline: base -> SFT -> {DPO, GRPO}, then evaluate every stage.
set -euo pipefail

CONFIG=configs/qwen1.5b.yaml

python -m medreason.evaluate --config "$CONFIG"
python -m medreason.sft --config "$CONFIG"
python -m medreason.evaluate --config "$CONFIG" --adapter outputs/sft

python -m medreason.build_preferences --config "$CONFIG" --policy outputs/sft
python -m medreason.dpo --config "$CONFIG" --init outputs/sft
python -m medreason.grpo --config "$CONFIG" --init outputs/sft

# DPO and GRPO adapters were trained on the merged SFT weights, so SFT is merged first.
python -m medreason.evaluate --config "$CONFIG" --adapter outputs/sft outputs/dpo
python -m medreason.evaluate --config "$CONFIG" --adapter outputs/sft outputs/grpo
