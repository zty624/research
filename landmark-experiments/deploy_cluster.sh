#!/bin/bash
# deploy_cluster.sh — Set up all landmark experiments on the 4090 cluster
#
# Usage:
#   1. Ensure SSH tunnel is running (~/Desktop/ghost/script/tunnel.sh)
#   2. Run: bash deploy_cluster.sh
#
# Cluster paths:
#   Code:  /inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/cc
#   Data:  /inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/data
#   GPU utilization target: >= 30%

set -euo pipefail

SSH_HOST="tengyue@localhost"
SSH_PORT=10022
BASE_DIR="/inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/cc"
DATA_DIR="/inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/data"

ssh_cmd="ssh -p ${SSH_PORT} ${SSH_HOST}"

echo "=== Checking SSH connection ==="
${ssh_cmd} "echo 'Connected to cluster'" || { echo "Failed to connect. Is tunnel running?"; exit 1; }

echo "=== Creating base directories ==="
${ssh_cmd} "mkdir -p ${BASE_DIR} ${DATA_DIR}"

echo "=== Syncing experiment code ==="
# Use rsync to copy landmark-experiments to cluster
rsync -avz --progress \
    -e "ssh -p ${SSH_PORT}" \
    /mnt/data/Arch/workspace/research/landmark-experiments/ \
    ${SSH_HOST}:${BASE_DIR}/landmark-experiments/ \
    --exclude='__pycache__' \
    --exclude='results/' \
    --exclude='vae-experiments/' \
    --exclude='.git' \
    --exclude='*.pyc'

echo "=== Setting up uv environments ==="
# Each experiment topic gets its own folder with uv
for topic in ai-for-science speech-audio-language-models ai-math-theorem-proving \
             ai-ethics-fairness-governance generative-models multimodal-vlm \
             robotics-foundation-models ai-agent-rag-vla ai-compiler-hardware-codesign \
             ai-safety-interpretability efficient-finetuning-merging \
             llm-reasoning-rl-alignment long-context-data-curation \
             new-architectures vae-representation-learning diffusion \
             ai-infra-operator-optimization; do
    echo "  Setting up: ${topic}"
    ${ssh_cmd} "cd ${BASE_DIR}/landmark-experiments/${topic} && \
        uv init --no-readme 2>/dev/null || true && \
        uv add torch numpy matplotlib scikit-learn 2>/dev/null || true"
done

echo "=== Deployment complete ==="
echo "Run experiments with: bash run_cluster.sh [topic] [experiment_number]"
