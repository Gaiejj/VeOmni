#!/bin/bash
# 8-GPU smoke test for Wan2.1 DiT training
# Uses toy model + dummy data, no real data/weights needed
set -x
set -o pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN

torchrun \
  --standalone \
  --nproc-per-node=8 \
  tests/train_scripts/train_dit_test.py \
  configs/dit/wan_smoke_test_8gpu.yaml \
  2>&1 | tee /tmp/veomni_smoke_test.log

echo "Exit code: $?"
