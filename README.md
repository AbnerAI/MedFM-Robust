# 🏥 MedFM-Robust: Benchmarking Robustness of Medical Foundation Models

A comprehensive framework for evaluating the robustness of medical foundation models under realistic perturbations, supporting both segmentation and vision-language tasks.

## 🌐 Project Page

👉 [XXXX-1-/](XXXX-1-/)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026%20(under%20review)-green.svg)](#citation)

**Key Highlights:**

- 🔬 Systematic robustness evaluation across 8 medical imaging modalities
- 🎯 SSIM-guided perturbation generation with calibrated severity levels
- 🧠 Benchmarking of Med-VLMs (LLaVA-Med, MedGemma) and SAM-based segmentation models
- ⚡ Multiple fine-tuning strategies including LoRA and adapter-based methods
- 📊 Comprehensive metrics for segmentation (IoU, Dice) and VLM tasks (Accuracy, BLEU, CIDEr)

---

## Overview

<img width="720" height="420" alt="framework_01" src="https://github.com/user-attachments/assets/23e6a3d7-fd8f-4cdc-ad68-6355d5378708" />

We present a comprehensive robustness benchmark for medical image AI models, covering:

- **Adversarial robustness**: FGSM and PGD attacks at 5 perturbation levels on medical segmentation models
- **Natural corruption robustness**: 40 perturbation types (12 base + 28 medical-specific) across 5 SSIM-calibrated severity levels
- **Fine-tuning strategies**: Full, Decoder-only, Encoder-partial, LoRA, and Adapter — robustness comparison across all strategies
- **VLM evaluation**: Robustness of medical VLMs (LLaVA-Med, MedGemma, MedGemma-1.5, GPT-4o-mini, Gemini-2.5-flash) on VQA, captioning, and visual grounding tasks

**Supported segmentation models:** MedSAM, SAM-Med2D

**Supported VLMs:** LLaVA-Med, MedGemma, MedGemma 1.5, GPT-4o-mini, Gemini-2.5-flash

**Supported datasets:** ISIC 2016, Brain Tumor MRI, Breast Ultrasound, Endoscopy, Retinal, Pathology, OmniMedVQA, ROCO, MeCoVQA

<img width="720" height="218" alt="intro_01" src="https://github.com/user-attachments/assets/bd0bf1b1-1657-4f0e-a31e-26f116a8d372" />

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Perturbation Types](#perturbation-types)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Repository Structure

```
.
├── Segmentation/
│   ├── pipeline.py                                        # Main evaluation pipeline
│   ├── model_zoo.py                                       # Model wrappers (MedSAM, SAM-Med2D)
│   ├── finetune.py                                        # Fine-tuning training script
│   ├── finetune_utils.py                                  # Fine-tuning utilities (LoRA, Adapter, Loss)
│   ├── segmentation_generate_perb_all_V10_adaptive_efficient.py  # Perturbation dataset generator
│   ├── extract_csv_results.py                             # CSV result extraction and organization
│   ├── jobgpu_with_finetune.sh                            # GPU job submission script
│   ├── model_config.json                                  # Model configuration
│   ├── dataset_config.json                                # Dataset configuration
│   └── data_record.json                                   # Data split records
│
├── VLM/
│   ├── eval_vlm_perturbation.py                           # VLM robustness evaluation
│   ├── generate_perturbation.py                           # Perturbation dataset generator (VLM)
│   ├── merge_results_to_table.py                          # Merge results into paper-ready tables
│   └── run_vlm_perturbation_multi_gpu.py                  # Multi-GPU VLM evaluation runner
│
├── Tools/
│   ├── 1.mask2pos_img_and_gt.py                           # Convert masks to positional image & GT
│   └── 2.mask2pos_for_bbox_coordinates.py                 # Convert masks to bounding box coordinates
│
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Create conda environment
conda create -n med-robust python=3.10 -y
conda activate med-robust

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft tqdm pandas opencv-python matplotlib scikit-image scipy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Quick Start

### 1. Generate Perturbation Datasets

**For segmentation datasets:**
```bash
# Generate perturbations for a single dataset
python Segmentation/segmentation_generate_perb_all_V10_adaptive_efficient.py \
    --dataset_name isic_2016 \
    --adaptive

# Generate perturbations for all datasets
python Segmentation/segmentation_generate_perb_all_V10_adaptive_efficient.py \
    --all_datasets --adaptive

# Custom level intensity multipliers
python Segmentation/segmentation_generate_perb_all_V10_adaptive_efficient.py \
    --dataset_name isic_2016 \
    --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
```

**For VLM datasets:**
```bash
python VLM/generate_perturbation.py --dataset omnimedvqa --sample_num 50
python VLM/generate_perturbation.py --all --sample_num 50
```

### 2. Evaluate Segmentation Model Robustness

**Adversarial attack evaluation:**
```bash
python Segmentation/pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016 \
    --eval_mode adversarial \
    --attack_types fgsm pgd \
    --levels 1 2 3 4 5
```

**Corruption robustness evaluation:**
```bash
python Segmentation/pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016_perturbed \
    --eval_mode perturbation \
    --perturbation_path /path/to/perturbed_datasets
```

**Both modes simultaneously:**
```bash
python Segmentation/pipeline.py \
    --model_name sammed2d \
    --dataset_name isic_2016 isic_2016_perturbed \
    --eval_mode both
```

### 3. Fine-tune for Robustness

```bash
# Decoder-only fine-tuning (fastest, recommended for small datasets)
python Segmentation/finetune.py \
    --model_name medsam \
    --strategy decoder_only \
    --data_path /path/to/data \
    --num_epochs 10

# LoRA fine-tuning (parameter-efficient)
python Segmentation/finetune.py \
    --model_name medsam \
    --strategy lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --data_path /path/to/data \
    --num_epochs 20

# Adapter fine-tuning (SAM-Med2D)
python Segmentation/finetune.py \
    --model_name sammed2d \
    --strategy adapter_only \
    --data_path /path/to/data \
    --num_epochs 20

# Partial encoder fine-tuning
python Segmentation/finetune.py \
    --model_name medsam \
    --strategy encoder_partial \
    --unfreeze_encoder_layers 4 \
    --data_path /path/to/data
```

### 4. Evaluate Fine-tuned Model

```bash
python Segmentation/pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016 \
    --eval_mode both \
    --finetune_checkpoint ./checkpoints/medsam_lora_best.pth \
    --data_split_json ./checkpoints/data_split.json
```

### 5. Evaluate VLM Robustness

```bash
# VQA task with LLaVA-Med
python VLM/eval_vlm_perturbation.py \
    --dataset omnimedvqa \
    --task vqa \
    --model_type llava-med \
    --sample_num 100

# Grounding task with MedGemma (with fine-tuning)
python VLM/eval_vlm_perturbation.py \
    --dataset mecovqa \
    --task grounding \
    --model_type medgemma \
    --finetune
```

### 6. Aggregate and Export Results

```bash
# Merge all results into paper-ready tables
python VLM/merge_results_to_table.py --results_dir ./results

# Extract and organize CSV files
python Segmentation/extract_csv_results.py \
    --results_dir ./results \
    --output_dir ./extracted_csv \
    --summary_only \
    --merge_summary
```

---

## Configuration

Model and dataset paths are configured via JSON files:

- `Segmentation/model_config.json`: Model weights paths, image sizes, and normalization parameters
- `Segmentation/dataset_config.json`: Dataset image/mask directories, bounding box annotations, and output settings

---

## Perturbation Types

| Category | Types |
|----------|-------|
| Noise | Gaussian noise, Salt-and-pepper, Speckle noise |
| Blur | Gaussian blur, Motion blur, Defocus blur |
| Digital | JPEG compression, Pixelation, Contrast, Brightness, Saturation |
| Geometric | Rotation, Scaling, Translation |
| CT | Metal artifacts, Beam-hardening cupping |
| MRI | Bias-field inhomogeneity, Ghosting / k-space motion |
| Ultrasound | Acoustic shadowing, Reverberation |
| Pathology | Stain variation (HSV) |
| Endoscopy | Specular reflection, Bubbles |
| OCT | Shadow, Blink artifacts, Defocus |
| X-ray | Scatter, Exposure variation, Grid patterns |
| Angiography | Haze |

Each type has **5 severity levels** (1 = mild, 5 = severe), generated with adaptive SSIM-calibrated intensity (Level 1: SSIM 0.90–0.98 → Level 5: SSIM 0.50–0.59).

---

## Evaluation Metrics

| Metric | Task | Description |
|--------|------|-------------|
| IoU | Segmentation | Primary metric; Intersection over Union |
| Dice | Segmentation | Secondary metric; F1 score over masks |
| IoU Drop / Dice Drop | Segmentation | Performance degradation under perturbation |
| Accuracy | VQA / Grounding | % correctly answered / localized |
| Acc@IoU≥0.5 | Grounding | Bounding box overlap threshold |
| BLEU-4 | Captioning | n-gram precision with brevity penalty |
| CIDEr | Captioning | TF-IDF weighted consensus score |

---

## Results

### Segmentation — Strategy Ranking

| Rank | Strategy | Mean IoU Drop |
|------|----------|--------------|
| 1 | **Full fine-tuning** | **0.025** |
| 2 | Dec-Only | 0.029 |
| 2 | Enc-Partial | 0.029 |
| 2 | Dec-Prompt | 0.029 |
| 5 | Adapter | 0.033 |
| 6 | LoRA | 0.048 |

### VLM — Task Robustness

| Task | Setting | Drop |
|------|---------|------|
| Captioning | Zero-shot | < 0.02 BLEU |
| VQA (medical models) | Zero-shot | < 8 points |
| VQA (Gemini-2.5-flash) | Zero-shot | 36.1 points (54% relative) |
| Visual Grounding | LoRA fine-tuned | > 40 points |

Results are automatically saved to `./results/` with:
- Per-image detailed CSV files
- `*_SUMMARY.csv`: Aggregated statistics per corruption type
- `*_STATS_BY_LEVEL.csv`: Breakdown by severity level

---

## License

This code is released for academic use only.
