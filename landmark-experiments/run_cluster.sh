#!/bin/bash
# run_cluster.sh — Run landmark experiments on the 4090 cluster
#
# Usage:
#   bash run_cluster.sh                    # Run all new experiments
#   bash run_cluster.sh <topic>            # Run all experiments in a topic
#   bash run_cluster.sh <topic> <number>   # Run a specific experiment
#   bash run_cluster.sh gpu_status         # Check GPU utilization
#
# GPU utilization target: >= 30%

set -euo pipefail

SSH_HOST="tengyue@localhost"
SSH_PORT=10022
BASE_DIR="/inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/cc"
DATA_DIR="/inspire/qb-ilm/project/video-generation/chenxie-25019/tengyue/data"

ssh_cmd="ssh -p ${SSH_PORT} ${SSH_HOST}"

# New experiments (68-104) mapped to topics
declare -A EXP_MAP=(
    [68]="ai-for-science/68_pinn"
    [69]="ai-for-science/69_fno"
    [70]="ai-for-science/70_deeponet"
    [71]="ai-for-science/71_molecular_gen"
    [72]="speech-audio-language-models/72_whisper"
    [73]="speech-audio-language-models/73_encodec"
    [74]="speech-audio-language-models/74_valle"
    [75]="ai-math-theorem-proving/75_lean_miniprover"
    [76]="ai-math-theorem-proving/76_mcts_prover"
    [77]="ai-ethics-fairness-governance/77_bias_detection"
    [78]="ai-ethics-fairness-governance/78_fairness_metrics"
    [79]="generative-models/79_gaussian_splatting"
    [80]="generative-models/80_vae_gan_compare"
    [81]="multimodal-vlm/81_llava"
    [82]="multimodal-vlm/82_clip_zeroshot"
    [83]="robotics-foundation-models/83_rt_action"
    [84]="robotics-foundation-models/84_openvla"
    [85]="ai-agent-rag-vla/85_toolformer"
    [86]="ai-agent-rag-vla/86_hugginggpt"
    [87]="ai-compiler-hardware-codesign/87_tvm_autotune"
    [88]="ai-compiler-hardware-codesign/88_triton_kernel"
    [89]="ai-safety-interpretability/89_red_team"
    [90]="ai-safety-interpretability/90_sae"
    [91]="efficient-finetuning-merging/91_dora"
    [92]="efficient-finetuning-merging/92_model_merging_ties"
    [93]="llm-reasoning-rl-alignment/93_dpo_detailed"
    [94]="llm-reasoning-rl-alignment/94_prm"
    [95]="long-context-data-curation/95_flash_attention_impl"
    [96]="long-context-data-curation/96_ntk_aware_rope"
    [97]="new-architectures/97_kan_detailed"
    [98]="new-architectures/98_moe_router"
    [99]="vae-representation-learning/99_ijepa"
    [100]="vae-representation-learning/100_byol"
    [101]="diffusion/101_ldm"
    [102]="diffusion/102_sde_solver"
    [103]="ai-infra-operator-optimization/103_paged_attention"
    [104]="ai-infra-operator-optimization/104_mixed_precision"
)

# Topic to experiment numbers
declare -A TOPIC_MAP=(
    [ai-for-science]="68 69 70 71"
    [speech-audio-language-models]="72 73 74"
    [ai-math-theorem-proving]="75 76"
    [ai-ethics-fairness-governance]="77 78"
    [generative-models]="79 80"
    [multimodal-vlm]="81 82"
    [robotics-foundation-models]="83 84"
    [ai-agent-rag-vla]="85 86"
    [ai-compiler-hardware-codesign]="87 88"
    [ai-safety-interpretability]="89 90"
    [efficient-finetuning-merging]="91 92"
    [llm-reasoning-rl-alignment]="93 94"
    [long-context-data-curation]="95 96"
    [new-architectures]="97 98"
    [vae-representation-learning]="99 100"
    [diffusion]="101 102"
    [ai-infra-operator-optimization]="103 104"
)

run_experiment() {
    local exp_path="$1"
    local topic=$(dirname "$exp_path")
    local script=$(basename "$exp_path")

    echo ">>> Running: ${exp_path}"

    # Run with uv in tmux, redirect data to DATA_DIR, pin to GPU
    ${ssh_cmd} "cd ${BASE_DIR}/landmark-experiments/${topic} && \
        tmux new-session -d -s ${script} 2>/dev/null || true && \
        tmux send-keys -t ${script} \
            'CUDA_VISIBLE_DEVICES=0 DATA_DIR=${DATA_DIR} uv run python ${script}.py 2>&1 | tee ${BASE_DIR}/landmark-experiments/${topic}/results/${script}.log' Enter"
}

check_gpu() {
    echo "=== GPU Status ==="
    ${ssh_cmd} "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
}

case "${1:-all}" in
    gpu_status)
        check_gpu
        ;;
    all)
        echo "=== Running all new experiments (68-104) ==="
        for num in $(seq 68 104); do
            if [[ -v EXP_MAP[$num] ]]; then
                run_experiment "${EXP_MAP[$num]}"
                sleep 2  # Stagger launches
            fi
        done
        echo ""
        echo "All experiments launched. Check status with: bash run_cluster.sh gpu_status"
        ;;
    *)
        # Check if it's a topic name
        if [[ -v TOPIC_MAP[$1] ]]; then
            echo "=== Running all experiments in: $1 ==="
            for num in ${TOPIC_MAP[$1]}; do
                run_experiment "${EXP_MAP[$num]}"
                sleep 2
            done
        # Check if it's a specific experiment number
        elif [[ -v EXP_MAP[$1] ]]; then
            run_experiment "${EXP_MAP[$1]}"
        else
            echo "Unknown topic or experiment: $1"
            echo "Available topics: ${!TOPIC_MAP[*]}"
            echo "Available experiment numbers: ${!EXP_MAP[*]}"
            exit 1
        fi
        ;;
esac
