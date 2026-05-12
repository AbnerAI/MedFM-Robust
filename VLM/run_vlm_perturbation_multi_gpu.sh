#!/bin/bash
#SBATCH -o job.out
#SBATCH --partition=a100
#SBATCH -J vlm-final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3

# ===================================================
# Model Run Switches (true=run, false=skip)
# ===================================================
RUN_LLAVA=false
RUN_MEDGEMMA=false
RUN_MEDGEMMA15=false
RUN_GEMINI=true    # Only run Gemini Proxy

echo "=========================================="
echo "Medical VLM Perturbation Evaluation - Multi-GPU Parallel Version"
echo "=========================================="
echo "Will run the following models:"
$RUN_LLAVA && echo "  - LLaVA-Med (GPU 0)"
$RUN_MEDGEMMA && echo "  - MedGemma (GPU 1)"
$RUN_MEDGEMMA15 && echo "  - MedGemma 1.5 (GPU 2)"
$RUN_GEMINI && echo "  - Gemini Proxy (API, no GPU required)"
echo "=========================================="

# watch tail logs/*.log

# ============================================
# Configuration Parameters
# ============================================
SAMPLE_NUM=100
SEVERITIES="1,3,5"
SEED=42
FINETUNE_EPOCHS=20

# DeepSeek API Configuration
DEEPSEEK_API_KEY="API"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
USE_DEEPSEEK="--use_deepseek"

# Gemini Proxy API Configuration
GEMINI_PROXY_API_KEY="jobgpu_with_finetune.sh"
GEMINI_PROXY_BASE_URL="https://////////"
GEMINI_PROXY_MODEL="gemini-2.5-flash"

# Environment Paths
ENV_PERTURBATION="/mnt/fast/nobackup/scratch4weeks/xxx/envs/for_perb_except_seg"
ENV_LLAVA="/mnt/fast/nobackup/scratch4weeks/xxx/envs/llava-med"
ENV_MEDGEMMA="/mnt/fast/nobackup/scratch4weeks/xxx/envs/medgemma"
ENV_MEDGEMMA15="/mnt/fast/nobackup/scratch4weeks/xxx/envs/medgemma"

# Model Paths
LLAVA_MODEL_PATH="../models/llava-med-v1.5-mistral-7b"
MEDGEMMA_MODEL_PATH="/mnt/fast/nobackup/scratch4weeks/xxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/Multi-scenario/Pipeline_for_All/models/medgemma-4b-it"
MEDGEMMA15_MODEL_PATH="/mnt/fast/nobackup/scratch4weeks/xxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/Multi-scenario/Pipeline_for_All/models/medgemma-1.5-4b-it"

# Fine-tuning Output Directory
FINETUNE_OUTPUT="./outputs/finetune"

# Script Paths
SCRIPT_EVAL="./eval_vlm_perturbation.py"

# Log Directory
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# ============================================
# Define Evaluation Functions for Each Model
# ============================================

run_llava_med() {
    echo "[GPU 0] LLaVA-Med evaluation started..."

    source ~/miniconda3/bin/activate
    conda activate ${ENV_LLAVA}

    export CUDA_VISIBLE_DEVICES=0

    # OmniMedVQA VQA
    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${LLAVA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${LLAVA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # ROCO Caption
    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${LLAVA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${LLAVA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # MeCoVQA Grounding (Fine-tuning + Evaluation)
    FINETUNED_MODEL="${FINETUNE_OUTPUT}/llava-med/mecovqa_grounding/final"
    if [ ! -d "$FINETUNED_MODEL" ]; then
        python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --finetune \
            --model_type llava-med --model_path ${LLAVA_MODEL_PATH} \
            --finetune_output ${FINETUNE_OUTPUT} --epochs ${FINETUNE_EPOCHS}
    fi

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}
	
    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type llava-med --model_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    echo "[GPU 0] LLaVA-Med evaluation completed!"
}

run_medgemma() {
    echo "[GPU 1] MedGemma evaluation started..."

    source ~/miniconda3/bin/activate
    conda activate ${ENV_MEDGEMMA}

    export CUDA_VISIBLE_DEVICES=1

    # OmniMedVQA VQA
    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${MEDGEMMA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${MEDGEMMA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # ROCO Caption
    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${MEDGEMMA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${MEDGEMMA_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # MeCoVQA Grounding (Fine-tuning + Evaluation)
    FINETUNED_MODEL="${FINETUNE_OUTPUT}/medgemma/mecovqa_grounding/final"
    if [ ! -d "$FINETUNED_MODEL" ]; then
        python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --finetune \
            --model_type medgemma --medgemma_path ${MEDGEMMA_MODEL_PATH} \
            --finetune_output ${FINETUNE_OUTPUT} --epochs ${FINETUNE_EPOCHS}
    fi

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type medgemma --medgemma_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    echo "[GPU 1] MedGemma evaluation completed!"
}

run_medgemma15() {
    echo "[GPU 2] MedGemma 1.5 evaluation started..."

    source ~/miniconda3/bin/activate
    conda activate ${ENV_MEDGEMMA15}

    export CUDA_VISIBLE_DEVICES=2

    # OmniMedVQA VQA
    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${MEDGEMMA15_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${MEDGEMMA15_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # ROCO Caption
    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${MEDGEMMA15_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${MEDGEMMA15_MODEL_PATH} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # MeCoVQA Grounding (Fine-tuning + Evaluation)
    FINETUNED_MODEL="${FINETUNE_OUTPUT}/medgemma15/mecovqa_grounding/final"
    if [ ! -d "$FINETUNED_MODEL" ]; then
        python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --finetune \
            --model_type medgemma15 --medgemma15_path ${MEDGEMMA15_MODEL_PATH} \
            --finetune_output ${FINETUNE_OUTPUT} --epochs ${FINETUNE_EPOCHS}
    fi

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type medgemma15 --medgemma15_path ${FINETUNED_MODEL} \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations
	
    echo "[GPU 2] MedGemma 1.5 evaluation completed!"
}

run_gemini_proxy() {
    echo "[API] Gemini Proxy evaluation started..."

    source ~/miniconda3/bin/activate
    conda activate ${ENV_MEDGEMMA}  # Use environment with torch

    # OmniMedVQA VQA
    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset omnimedvqa --task vqa --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # ROCO Caption
    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset roco --task caption --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    # MeCoVQA Grounding (Closed-source API does not support fine-tuning, direct zero-shot evaluation)
    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK}

    python ${SCRIPT_EVAL} --dataset mecovqa --task grounding --sample_num ${SAMPLE_NUM} \
        --model_type gemini-proxy \
        --gemini_proxy_api_key "${GEMINI_PROXY_API_KEY}" \
        --gemini_proxy_base_url "${GEMINI_PROXY_BASE_URL}" \
        --gemini_proxy_model "${GEMINI_PROXY_MODEL}" \
        --deepseek_api_key "${DEEPSEEK_API_KEY}" --deepseek_base_url "${DEEPSEEK_BASE_URL}" ${USE_DEEPSEEK} \
        --perturbed --all_perturbations

    echo "[API] Gemini Proxy evaluation completed!"
}

# ============================================
# Execute Selected Models in Parallel
# ============================================
echo ""
echo "=========================================="
echo "Starting parallel evaluation..."
echo "=========================================="
echo ""

# Record started processes
PIDS=""

# Start corresponding models based on switches
if $RUN_LLAVA; then
    run_llava_med > ${LOG_DIR}/llava_med.log 2>&1 &
    PID_LLAVA=$!
    PIDS="$PIDS $PID_LLAVA"
    echo "LLaVA-Med started (PID: $PID_LLAVA, GPU: 0)"
fi

if $RUN_MEDGEMMA; then
    run_medgemma > ${LOG_DIR}/medgemma.log 2>&1 &
    PID_MEDGEMMA=$!
    PIDS="$PIDS $PID_MEDGEMMA"
    echo "MedGemma started (PID: $PID_MEDGEMMA, GPU: 1)"
fi

if $RUN_MEDGEMMA15; then
    run_medgemma15 > ${LOG_DIR}/medgemma15.log 2>&1 &
    PID_MEDGEMMA15=$!
    PIDS="$PIDS $PID_MEDGEMMA15"
    echo "MedGemma 1.5 started (PID: $PID_MEDGEMMA15, GPU: 2)"
fi

if $RUN_GEMINI; then
    run_gemini_proxy > ${LOG_DIR}/gemini_proxy.log 2>&1 &
    PID_GEMINI=$!
    PIDS="$PIDS $PID_GEMINI"
    echo "Gemini Proxy started (PID: $PID_GEMINI, API call, no GPU required)"
fi

echo ""
echo "Started processes: $PIDS"
echo "View logs: tail -f ${LOG_DIR}/*.log"
echo ""

# Wait for all background processes to complete
if $RUN_LLAVA; then
    wait $PID_LLAVA
    STATUS_LLAVA=$?
    echo "LLaVA-Med completed (exit code: $STATUS_LLAVA)"
fi

if $RUN_MEDGEMMA; then
    wait $PID_MEDGEMMA
    STATUS_MEDGEMMA=$?
    echo "MedGemma completed (exit code: $STATUS_MEDGEMMA)"
fi

if $RUN_MEDGEMMA15; then
    wait $PID_MEDGEMMA15
    STATUS_MEDGEMMA15=$?
    echo "MedGemma 1.5 completed (exit code: $STATUS_MEDGEMMA15)"
fi

if $RUN_GEMINI; then
    wait $PID_GEMINI
    STATUS_GEMINI=$?
    echo "Gemini Proxy completed (exit code: $STATUS_GEMINI)"
fi

# ============================================
# Merge Results and Generate Paper Tables
# ============================================
echo ""
echo "=========================================="
echo "Merging results and generating paper tables"
echo "=========================================="

source ~/miniconda3/bin/activate
conda activate ${ENV_LLAVA}

SCRIPT_MERGE="./merge_results_to_table.py"
OUTPUT_DIR="./outputs/vlm_perturbation_results"
TABLE_DIR="./outputs/paper_tables"
mkdir -p ${TABLE_DIR}

RESULT_COUNT=$(find ${OUTPUT_DIR} -name "results_*.json" -type f 2>/dev/null | wc -l)

if [ "$RESULT_COUNT" -gt 0 ]; then
    echo "Found ${RESULT_COUNT} result files, generating paper tables..."

    python ${SCRIPT_MERGE} --input_dir ${OUTPUT_DIR} --task vqa \
        --output ${TABLE_DIR}/table_vqa_all_models.xlsx \
        --output_csv ${TABLE_DIR}/table_vqa_all_models.csv

    python ${SCRIPT_MERGE} --input_dir ${OUTPUT_DIR} --task caption \
        --output ${TABLE_DIR}/table_caption_all_models.xlsx \
        --output_csv ${TABLE_DIR}/table_caption_all_models.csv

    python ${SCRIPT_MERGE} --input_dir ${OUTPUT_DIR} --task grounding \
        --output ${TABLE_DIR}/table_grounding_all_models.xlsx \
        --output_csv ${TABLE_DIR}/table_grounding_all_models.csv

    python ${SCRIPT_MERGE} --input_dir ${OUTPUT_DIR} \
        --output ${TABLE_DIR}/table_all_tasks_all_models.xlsx \
        --output_csv ${TABLE_DIR}/table_all_tasks_all_models.csv

    echo "Paper tables saved to: ${TABLE_DIR}/"
fi

echo ""
echo "=========================================="
echo "All completed!"
echo "=========================================="
echo ""
echo "Log files:"
echo "  ${LOG_DIR}/llava_med.log"
echo "  ${LOG_DIR}/medgemma.log"
echo "  ${LOG_DIR}/medgemma15.log"
echo "  ${LOG_DIR}/gemini_proxy.log"
echo ""
