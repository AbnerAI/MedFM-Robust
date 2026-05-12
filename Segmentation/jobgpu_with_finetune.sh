#!/bin/bash
#SBATCH -o job.out
#SBATCH --partition=a100
#SBATCH -J FMs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# source ~/miniconda3/bin/activate
# conda activate /mnt/fast/nobackup/scratch4weeks/xxx/envs/medsam

# # bbox is shared by both perturbation types.
# # 1. For sammed2d model, data needs to be manually resized to 256x256 during adversarial attack, done by resize_tools.
# # 2. Then manual perturbation code (implemented at BNU, custom_perb.py), since it's original data without resize,
# # the pipeline handles the resize. Note: since bbox is already 256x256 during adv (step 1), no need to resize again here.

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate /mnt/fast/nobackup/scratch4weeks/xxx/envs/medsam
# conda activate /mnt/fast/nobackup/scratch4weeks/xxx/envs/sammed2d

# generate perb
# just run once!
# python segmentation_generate_perb_all_V9_adpative_efficient.py --all_datasets --adaptive

# generate perb. One-click control of disturbance intensity!!!
# python segmentation_generate_perb_all_V10_adpative_efficient.py --all_datasets --adaptive --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'

# python debug_inference_minimal_for_sam_med2d_manual.py --file ISIC_0010862.jpg
# python debug_inference_minimal_for_sam_med2d.py --file ISIC_0000030.jpg

# python debug_sam_comparison.py --file ISIC_0000030.jpg
# exit 0

# python generate_perb_all.py --all_datasets
# exit 0

# bbox is shared by both perturbation types.
# 1. For sammed2d model, data needs to be manually resized to 256x256 during adversarial attack, done by resize_tools.
# 2. Then manual perturbation code (implemented at BNU, custom_perb.py), since it's original data without resize,
# the pipeline handles the resize. Note: since bbox is already 256x256 during adv (step 1), no need to resize again here.

# Debug
# python sammed2d_gradient_tracer.py
# python min_sammed2_test.py

# General settings
TARGET_SIZE="256 256"
RESIZE_SCRIPT="../resize_tools/1.mask2pos_img_and_gt.py"
BBOX_SCRIPT="../resize_tools/2.mask2pos_for_bbox_coordinates.py"

# Function: Process single dataset
process_dataset() {
    local dataset_name=$1
    local image_folder=$2
    local gt_folder=$3
    local out_image_folder=$4
    local out_gt_folder=$5
    local bbox_output=$6
    
    echo "========================================"
    echo "Processing dataset: ${dataset_name}"
    echo "========================================"
    
    # Step 1: Resize images and ground truth
    echo "[1/2] Resizing images and masks..."
    python ${RESIZE_SCRIPT} \
        --image_folder "${image_folder}" \
        --gt_folder "${gt_folder}" \
        --out_image_folder "${out_image_folder}" \
        --out_gt_folder "${out_gt_folder}" \
        --target_size ${TARGET_SIZE} \
        --dataset_name "${dataset_name}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Resize step failed for ${dataset_name}"
        return 1
    fi
    
    # Step 2: Generate bbox coordinates
    echo "[2/2] Generating bbox coordinates..."
    python ${BBOX_SCRIPT} \
        --mask_folder "${gt_folder}" \
        --target_size ${TARGET_SIZE} \
        --output_json "${bbox_output}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Bbox generation failed for ${dataset_name}"
        return 1
    fi
    
    echo "[OK] ${dataset_name} processing completed"
    echo ""
}

# ========================================
# Dataset Configuration
# ========================================

# Base paths
BASE_PATH="/mnt/fast/nobackup/scratch4weeks/xxx"
MEDSAM_PATH="${BASE_PATH}/MedSAM/work_dir/medsam-vit-base"
DATA_PATH="${BASE_PATH}/data/Segmentation_Data_2025"

# ================================================================================================
# ======================================== Fine-tuning Configuration (Optional) ==================
# ================================================================================================
#
# Fine-tuning switch: Set to true to enable fine-tuning, set to false to skip fine-tuning and run pipeline directly
# 
ENABLE_FINETUNE=true

# Fine-tuning strategy options (supports multiple strategies, separated by spaces):
#   - decoder_only:       Train decoder only (fastest, ~8GB VRAM, suitable for small datasets <500 images)
#   - decoder_prompt:     Train decoder+prompt (recommended default, ~10GB VRAM)
#   - adapter_only:       Train adapter layers (SAM-Med2D specific, ~12GB VRAM)
#   - encoder_partial:    Train partial encoder (~16GB VRAM, requires more data)
#   - full:               Full fine-tuning (~24GB VRAM, requires large dataset)
#   - lora:               LoRA fine-tuning (parameter efficient, ~10GB VRAM)
#   - lora_plus_encoder:  LoRA+ fine-tune encoder (different learning rates for A/B matrices, ~12GB VRAM)
#   - lora_plus_decoder:  LoRA+ fine-tune decoder (different learning rates for A/B matrices, ~10GB VRAM)
#
# Examples:
#   Single strategy:   FINETUNE_STRATEGIES=("decoder_only")
#   Multiple strategies:   FINETUNE_STRATEGIES=("decoder_only" "lora" "encoder_partial")
#   All strategies: FINETUNE_STRATEGIES=("decoder_only" "decoder_prompt" "encoder_partial" "lora" "lora_plus_encoder" "lora_plus_decoder" "full")
#
# FINETUNE_STRATEGIES=("decoder_only" "decoder_prompt" "encoder_partial" "lora" "full")
# FINETUNE_STRATEGIES=("decoder_only")
FINETUNE_STRATEGIES=("decoder_only" "decoder_prompt" "encoder_partial" "lora" "lora_plus_encoder" "lora_plus_decoder" "full")
# FINETUNE_STRATEGIES=("full")
FINETUNE_EPOCHS=20 # default 10 -> 20
FINETUNE_LR=1e-5
FINETUNE_BATCH_SIZE=64 # full 8; other 64

# [New] LoRA Configuration
LORA_R=16
LORA_ALPHA=32

# [New] LoRA+ Configuration (B matrix learning rate = A matrix learning rate * LORA_PLUS_LR_RATIO)
# Recommended value from paper: 16
LORA_PLUS_LR_RATIO=16

# Base timestamp (all strategies share the same batch identifier)
BASE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ================================================================================================
# ======================================== Common Processing Functions ===========================
# ================================================================================================

# Set output directories for current strategy
setup_output_dirs() {
    local strategy=$1
    
    if [ "$ENABLE_FINETUNE" = true ]; then
        EXPERIMENT_OUTPUT_BASE="./results/${BASE_TIMESTAMP}_${strategy}"
    else
        EXPERIMENT_OUTPUT_BASE="./results/${BASE_TIMESTAMP}_pretrained"
    fi
    FINETUNE_OUTPUT_BASE="${EXPERIMENT_OUTPUT_BASE}/finetune_checkpoints"
    
    echo "========================================"
    echo "Experiment output directory: ${EXPERIMENT_OUTPUT_BASE}"
    echo "========================================"
    mkdir -p "${EXPERIMENT_OUTPUT_BASE}"
}

# Common fine-tune + evaluation function (supports multi-strategy loop)
# Usage: run_dataset_all_strategies <model_name> <dataset_name> <dataset_name_perturbed> <data_path> <img_dir> <mask_dir> <bbox_json>
run_dataset_all_strategies() {
    local model_name=$1
    local dataset_name=$2
    local dataset_perturbed=$3
    local data_path=$4
    local img_dir=$5
    local mask_dir=$6
    local bbox_json=$7

    local current_batch_size=${FINETUNE_BATCH_SIZE}
    
    echo ""
    echo "########################################################################"
    echo "# Starting dataset processing: ${dataset_name} (${model_name})"
    echo "# Strategy list: ${FINETUNE_STRATEGIES[*]}"
    echo "# Number of strategies: ${#FINETUNE_STRATEGIES[@]}"
    echo "########################################################################"
    echo ""
    
    if [ "$ENABLE_FINETUNE" = true ]; then
        # Iterate through all fine-tuning strategies
        for CURRENT_STRATEGY in "${FINETUNE_STRATEGIES[@]}"; do
            echo ""
            echo "================================================================"
            echo ">>> Current strategy: ${CURRENT_STRATEGY} (${dataset_name})"
            echo "================================================================"
            echo ""

            if [ "${CURRENT_STRATEGY}" == "full" ]; then
                current_batch_size=8
            else
                current_batch_size=${FINETUNE_BATCH_SIZE}  # Keep default 64
            fi

            # Set output directory for current strategy
            setup_output_dirs "${CURRENT_STRATEGY}"
            
            local finetune_dir="${FINETUNE_OUTPUT_BASE}/${model_name}_${dataset_name}"
            local pipeline_output="${EXPERIMENT_OUTPUT_BASE}/pipeline_${model_name}_${dataset_name}"
            
            echo ">> Starting fine-tuning ${model_name} on ${dataset_name} [Strategy: ${CURRENT_STRATEGY}]..."
            
            # Build base command
            FINETUNE_CMD="python finetune.py \
                --model_name ${model_name} \
                --model_config model_config.json \
                --strategy ${CURRENT_STRATEGY} \
                --data_path \"${data_path}\" \
                --img_dir \"${img_dir}\" \
                --mask_dir \"${mask_dir}\" \
                --bbox_json \"${bbox_json}\" \
                --num_epochs ${FINETUNE_EPOCHS} \
                --batch_size ${current_batch_size} \
                --learning_rate ${FINETUNE_LR} \
                --use_amp \
                --output_dir \"${finetune_dir}\""
            
            # If LoRA or LoRA+ strategy, add LoRA parameters
            if [[ "${CURRENT_STRATEGY}" == "lora"* ]]; then
                FINETUNE_CMD="${FINETUNE_CMD} \
                --lora_r ${LORA_R} \
                --lora_alpha ${LORA_ALPHA}"
            fi
            
            # If LoRA+ strategy, add LoRA+ specific parameters
            if [[ "${CURRENT_STRATEGY}" == "lora_plus"* ]]; then
                FINETUNE_CMD="${FINETUNE_CMD} \
                --lora_plus_lr_ratio ${LORA_PLUS_LR_RATIO}"
            fi
            
            # Execute fine-tuning command
            eval ${FINETUNE_CMD}
            
            echo ">> Evaluating fine-tuned model on test set [Strategy: ${CURRENT_STRATEGY}]..."
            python pipeline.py \
                --model_name ${model_name} \
                --model_config model_config.json \
                --dataset_config dataset_config.json \
                --eval_mode both \
                --dataset_name ${dataset_name} ${dataset_perturbed} \
                --attack_types fgsm pgd \
                --levels 1 2 3 \
                --debug \
                --finetune_checkpoint "${finetune_dir}/best_model.pth" \
                --data_split_json "${finetune_dir}/data_split.json" \
                --output_root "${pipeline_output}"
            
            echo ""
            echo "[OK] Strategy ${CURRENT_STRATEGY} (${dataset_name}) completed"
            echo "================================================================"
            echo ""
        done
    else
        # No fine-tuning, directly evaluate all data (run only once)
        setup_output_dirs "pretrained"
        local pipeline_output="${EXPERIMENT_OUTPUT_BASE}/pipeline_${model_name}_${dataset_name}"
        
        echo ">> Directly evaluating pretrained model (all data)..."
        python pipeline.py \
            --model_name ${model_name} \
            --model_config model_config.json \
            --dataset_config dataset_config.json \
            --eval_mode both \
            --dataset_name ${dataset_name} ${dataset_perturbed} \
            --attack_types fgsm pgd \
            --levels 1 2 3 \
            --debug \
            --output_root "${pipeline_output}"
    fi
    
    echo ""
    echo "########################################################################"
    echo "# ${dataset_name} (${model_name}) all strategies completed"
    echo "########################################################################"
    echo ""
}

# ================================================================================================
# ========================================MedSAM Pipeline =========================================
# ================================================================================================

conda activate /mnt/fast/nobackup/scratch4weeks/xxx/envs/medsam

# --------------------------------------------------------------------------------
# Dataset 1: isic_2016 (MedSAM)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "medsam" \
    "isic_2016" \
    "isic_2016_perturbed" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/images_256" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/masks_256" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 2: nikhilroxtomar-brain-tumor (MedSAM)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "medsam" \
    "nikhilroxtomar-brain-tumor" \
    "nikhilroxtomar-brain-tumor_perturbed" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/images_256" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/masks_256" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 3: kelkalot-the-hyper-kvasir-dataset (MedSAM)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "medsam" \
    "kelkalot-the-hyper-kvasir-dataset" \
    "kelkalot-the-hyper-kvasir-dataset_perturbed" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/images_256" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/masks_256" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 4: deathtrooper-multichannel-glaucoma-disc (MedSAM)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "medsam" \
    "deathtrooper-multichannel-glaucoma-disc" \
    "deathtrooper-multichannel-glaucoma-disc_perturbed" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/images_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/masks_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 5: deathtrooper-multichannel-glaucoma-cup (MedSAM)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "medsam" \
    "deathtrooper-multichannel-glaucoma-cup" \
    "deathtrooper-multichannel-glaucoma-cup_perturbed" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/images_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/masks_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/bbox_coordinates_256.json"

# ================================================================================================
# ========================================SAM-Med2D Pipeline ======================================
# ================================================================================================

conda activate /mnt/fast/nobackup/scratch4weeks/xxx/envs/sammed2d

# SAM-Med2D strategy configuration (supports multiple strategies)
# Note: adapter_only is SAM-Med2D specific strategy
# Available strategies: "decoder_only" "decoder_prompt" "adapter_only" "encoder_partial" "lora" "lora_plus_encoder" "lora_plus_decoder" "full"
FINETUNE_STRATEGIES_SAMMED2D=("decoder_only" "decoder_prompt" "adapter_only" "encoder_partial" "lora" "lora_plus_encoder" "lora_plus_decoder" "full")

# Temporarily save MedSAM strategies, switch to SAM-Med2D strategies
FINETUNE_STRATEGIES_BACKUP=("${FINETUNE_STRATEGIES[@]}")
FINETUNE_STRATEGIES=("${FINETUNE_STRATEGIES_SAMMED2D[@]}")

# --------------------------------------------------------------------------------
# Dataset 1: isic_2016 (SAM-Med2D)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "sammed2d" \
    "isic_2016" \
    "isic_2016_perturbed" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/images_256" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/masks_256" \
    "${DATA_PATH}/Part-1-Lesion-Segmentation-2016/Training/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 2: nikhilroxtomar-brain-tumor (SAM-Med2D)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "sammed2d" \
    "nikhilroxtomar-brain-tumor" \
    "nikhilroxtomar-brain-tumor_perturbed" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/images_256" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/masks_256" \
    "${DATA_PATH}/nikhilroxtomar-brain-tumor-segmentation/1/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 3: kelkalot-the-hyper-kvasir-dataset (SAM-Med2D)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "sammed2d" \
    "kelkalot-the-hyper-kvasir-dataset" \
    "kelkalot-the-hyper-kvasir-dataset_perturbed" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/images_256" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/masks_256" \
    "${DATA_PATH}/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 4: deathtrooper-multichannel-glaucoma-disc (SAM-Med2D)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "sammed2d" \
    "deathtrooper-multichannel-glaucoma-disc" \
    "deathtrooper-multichannel-glaucoma-disc_perturbed" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/images_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/masks_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/bbox_coordinates_256.json"

# --------------------------------------------------------------------------------
# Dataset 5: deathtrooper-multichannel-glaucoma-cup (SAM-Med2D)
# --------------------------------------------------------------------------------
run_dataset_all_strategies "sammed2d" \
    "deathtrooper-multichannel-glaucoma-cup" \
    "deathtrooper-multichannel-glaucoma-cup_perturbed" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/images_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/masks_256" \
    "${DATA_PATH}/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/bbox_coordinates_256.json"

# Restore MedSAM strategies (if needed later)
FINETUNE_STRATEGIES=("${FINETUNE_STRATEGIES_BACKUP[@]}")

echo "========================================"
echo "All tasks completed!"
echo "========================================"
