#!/bin/bash
# Multi-node pre-training launch script for Wan2.1
# Usage:
#   On each node: bash scripts/launch_pretrain_multinode.sh <config.yaml> [--node-rank N]
#
# Environment variables (set before running):
#   MASTER_ADDR: IP of node 0 (required for multi-node)
#   MASTER_PORT: Port for rendezvous (default: 12345)
#   NNODES: Number of nodes (default: 1)
#   NODE_RANK: Rank of this node (default: 0, or from --node-rank)
#   NPROC_PER_NODE: GPUs per node (default: auto-detect)

set -e

CONFIG=${1:-configs/dit/wan_pretrain_80h20.yaml}
shift || true

# Parse optional --node-rank argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --node-rank)
            NODE_RANK=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# NCCL configuration for multi-node InfiniBand
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

# Defaults
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-12345}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}

echo "========================================="
echo "Wan2.1 Pre-training Launch"
echo "========================================="
echo "Config:         $CONFIG"
echo "Nodes:          $NNODES"
echo "Node Rank:      $NODE_RANK"
echo "GPUs per node:  $NPROC_PER_NODE"
echo "Master:         ${MASTER_ADDR:-localhost}:${MASTER_PORT}"
echo "========================================="

if [ "$NNODES" -eq 1 ]; then
    torchrun \
        --standalone \
        --nproc-per-node=$NPROC_PER_NODE \
        tasks/train_dit.py \
        $CONFIG
else
    if [ -z "$MASTER_ADDR" ]; then
        echo "ERROR: MASTER_ADDR must be set for multi-node training"
        exit 1
    fi
    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv_backend=c10d \
        tasks/train_dit.py \
        $CONFIG
fi
