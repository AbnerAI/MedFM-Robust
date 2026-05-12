#!/usr/bin/env python3
"""
CSV Results Extraction Tool
===========================
Extract all CSV files from results folder, preserving complete context information:
- Dataset name (dataset)
- Model name (model)
- Fine-tuning strategy (strategy)
- Evaluation type (adversarial/perturbation)

Output structure:
extracted_csv/
├── index.csv                   # Index file, records metadata for all CSVs
├── medsam/
│   ├── isic_2016/
│   │   ├── decoder_only/
│   │   │   ├── results_adversarial_xxx.csv
│   │   │   └── results_adversarial_xxx_SUMMARY.csv
│   │   └── lora/
│   │       └── ...
│   └── brain-tumor/
│       └── ...
└── sammed2d/
    └── ...

Usage:
    python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv
    python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat  # Flat output
"""

# # Basic usage - maintain hierarchy structure
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv

# # Extract summary files only (most practical - small files)
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only

# # Flat output (all files in one directory, filename contains all info)
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat --summary_only

# # Also generate merged comparison table
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only --merge_summary
# ```

# **Output structure:**
# ```
# extracted_csv/
# ├── index.csv              # Index file, records metadata for all CSVs
# ├── index.json             # JSON format index
# ├── merged_summary_adversarial.csv   # (generated with --merge_summary)
# ├── merged_summary_perturbation.csv  # (generated with --merge_summary)
# └── medsam/
#     └── isic_2016/
#         └── decoder_only/
#             └── results_adversarial_xxx_SUMMARY.csv

import os
import re
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
import csv


def parse_experiment_path(csv_path: str) -> dict:
    """
    Parse experiment metadata from CSV path.

    Path pattern examples:
    ./results/20250120_143000_decoder_only/pipeline_medsam_isic_2016/results/results_adversarial_xxx.csv
    ./results/20250120_143000_pretrained/pipeline_sammed2d_brain-tumor/results/results_perturbation_xxx.csv
    """
    path_parts = Path(csv_path).parts
    info = {
        'csv_path': csv_path,
        'csv_name': os.path.basename(csv_path),
        'strategy': 'unknown',
        'model': 'unknown',
        'dataset': 'unknown',
        'eval_type': 'unknown',
        'timestamp': 'unknown'
    }

    # 1. Extract timestamp and strategy from main directory name
    # Format: {timestamp}_{strategy} or {timestamp}_pretrained
    for part in path_parts:
        # Match format like 20250120_143000_decoder_only
        match = re.match(r'^(\d{8}_\d{6})_(.+)$', part)
        if match:
            info['timestamp'] = match.group(1)
            info['strategy'] = match.group(2)
            break

    # 2. Extract model and dataset from pipeline_* directory name
    # Format: pipeline_{model}_{dataset}
    for part in path_parts:
        match = re.match(r'^pipeline_([^_]+)_(.+)$', part)
        if match:
            info['model'] = match.group(1)
            info['dataset'] = match.group(2)
            break

    # 3. Extract eval_type from CSV filename
    # Format: results_adversarial_xxx.csv or results_perturbation_xxx.csv
    csv_name = info['csv_name']
    if 'adversarial' in csv_name.lower():
        info['eval_type'] = 'adversarial'
    elif 'perturbation' in csv_name.lower():
        info['eval_type'] = 'perturbation'

    # 4. Check if it's a summary file
    info['is_summary'] = 'SUMMARY' in csv_name or 'STATS' in csv_name

    return info


def find_all_csv_files(results_dir: str) -> list:
    """Recursively find all CSV files."""
    csv_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files


def generate_output_filename(info: dict, flat: bool = False) -> str:
    """
    Generate output filename.

    Hierarchical mode: {model}/{dataset}/{strategy}/{original_filename}
    Flat mode: {model}__{dataset}__{strategy}__{eval_type}__{original_filename}
    """
    if flat:
        # Flat naming for quick viewing
        parts = [
            info['model'],
            info['dataset'],
            info['strategy'],
            info['eval_type']
        ]
        base_name = info['csv_name']
        return f"{'__'.join(parts)}__{base_name}"
    else:
        # Hierarchical structure
        return os.path.join(
            info['model'],
            info['dataset'],
            info['strategy'],
            info['csv_name']
        )


def extract_csv_files(results_dir: str, output_dir: str, flat: bool = False,
                       summary_only: bool = False, detail_only: bool = False):
    """
    Main extraction function.

    Args:
        results_dir: Results directory (./results)
        output_dir: Output directory
        flat: Whether to use flat naming
        summary_only: Only extract summary files (*_SUMMARY.csv, *_STATS*.csv)
        detail_only: Only extract detailed result files (non-summary files)
    """
    print(f"[SCAN] Scanning directory: {results_dir}")
    csv_files = find_all_csv_files(results_dir)
    print(f"[OK] Found {len(csv_files)} CSV files")

    if not csv_files:
        print("[WARN] No CSV files found")
        return

    # Parse metadata for all CSV files
    all_info = []
    for csv_path in csv_files:
        info = parse_experiment_path(csv_path)
        all_info.append(info)

    # Filter
    if summary_only:
        all_info = [info for info in all_info if info['is_summary']]
        print(f"[FILTER] Summary files: {len(all_info)}")
    elif detail_only:
        all_info = [info for info in all_info if not info['is_summary']]
        print(f"[FILTER] Detail files: {len(all_info)}")

    if not all_info:
        print("[WARN] No files to extract after filtering")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy files
    copied_count = 0
    index_data = []

    for info in all_info:
        src_path = info['csv_path']
        rel_output = generate_output_filename(info, flat)
        dst_path = os.path.join(output_dir, rel_output)

        # Create target directory
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy file
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1

            # Record index
            index_data.append({
                'model': info['model'],
                'dataset': info['dataset'],
                'strategy': info['strategy'],
                'eval_type': info['eval_type'],
                'is_summary': info['is_summary'],
                'timestamp': info['timestamp'],
                'output_path': rel_output,
                'original_path': src_path
            })

            print(f"[OK] [{info['model']}][{info['dataset']}][{info['strategy']}] {info['csv_name']}")

        except Exception as e:
            print(f"[ERROR] Copy failed: {src_path} -> {e}")

    # Save index file
    index_path = os.path.join(output_dir, 'index.csv')
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        if index_data:
            writer = csv.DictWriter(f, fieldnames=index_data[0].keys())
            writer.writeheader()
            writer.writerows(index_data)

    # Also save JSON format index
    index_json_path = os.path.join(output_dir, 'index.json')
    with open(index_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Generate summary statistics
    print(f"\n{'='*60}")
    print(f"[OK] Extraction complete!")
    print(f"{'='*60}")
    print(f"   Total copied: {copied_count} files")
    print(f"   Output directory: {output_dir}")
    print(f"   Index file: {index_path}")

    # Statistics by dimension
    models = set(info['model'] for info in index_data)
    datasets = set(info['dataset'] for info in index_data)
    strategies = set(info['strategy'] for info in index_data)

    print(f"\n[STATS] Dimension statistics:")
    print(f"   Models: {len(models)} - {', '.join(sorted(models))}")
    print(f"   Datasets: {len(datasets)} - {', '.join(sorted(datasets))}")
    print(f"   Strategies: {len(strategies)} - {', '.join(sorted(strategies))}")
    print(f"{'='*60}\n")

    return index_data


def create_merged_summary(output_dir: str, index_data: list):
    """
    Create merged summary table for quick comparison of different strategies' performance.

    Output:
    - merged_summary_adversarial.csv: All adversarial attack summary data
    - merged_summary_perturbation.csv: All perturbation evaluation summary data
    """
    import pandas as pd

    print(f"\n[MERGE] Generating merged summary tables...")

    # Only process summary files
    summary_files = [info for info in index_data if info['is_summary'] and 'SUMMARY' in info['output_path']]

    if not summary_files:
        print("[WARN] No summary files found, skipping merge")
        return

    # Process adversarial and perturbation separately
    for eval_type in ['adversarial', 'perturbation']:
        type_files = [info for info in summary_files if info['eval_type'] == eval_type]

        if not type_files:
            continue

        merged_rows = []
        for info in type_files:
            csv_path = os.path.join(output_dir, info['output_path'])
            try:
                df = pd.read_csv(csv_path)
                # Add metadata columns
                df['Model'] = info['model']
                df['Dataset'] = info['dataset']
                df['Strategy'] = info['strategy']
                df['Timestamp'] = info['timestamp']
                merged_rows.append(df)
            except Exception as e:
                print(f"[WARN] Read failed: {csv_path} - {e}")

        if merged_rows:
            merged_df = pd.concat(merged_rows, ignore_index=True)
            # Reorder columns, metadata first
            cols = ['Model', 'Dataset', 'Strategy', 'Timestamp'] + \
                   [c for c in merged_df.columns if c not in ['Model', 'Dataset', 'Strategy', 'Timestamp']]
            merged_df = merged_df[cols]

            output_path = os.path.join(output_dir, f'merged_summary_{eval_type}.csv')
            merged_df.to_csv(output_path, index=False, float_format='%.4f')
            print(f"[OK] Merged summary table: {output_path} ({len(merged_df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CSV files from results directory, preserving dataset/model/strategy info",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
---------------
# Basic usage - hierarchical directory structure
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv

# Flat output - all files in one directory, filename contains all info
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat

# Extract summary files only (SUMMARY.csv)
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only

# Extract detailed result files only (excluding SUMMARY)
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --detail_only

# Generate merged summary table
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --merge_summary
        """
    )

    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory path (default: ./results)')
    parser.add_argument('--output_dir', type=str, default='./extracted_csv',
                        help='Output directory path (default: ./extracted_csv)')
    parser.add_argument('--flat', action='store_true',
                        help='Use flat naming (all files in same directory)')
    parser.add_argument('--summary_only', action='store_true',
                        help='Extract summary files only (*_SUMMARY.csv, *_STATS*.csv)')
    parser.add_argument('--detail_only', action='store_true',
                        help='Extract detailed result files only (non-summary files)')
    parser.add_argument('--merge_summary', action='store_true',
                        help='Generate merged summary table (requires pandas)')

    args = parser.parse_args()

    # Validate arguments
    if args.summary_only and args.detail_only:
        print("[ERROR] --summary_only and --detail_only cannot be used together")
        return

    if not os.path.exists(args.results_dir):
        print(f"[ERROR] Results directory does not exist: {args.results_dir}")
        return

    # Execute extraction
    index_data = extract_csv_files(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        flat=args.flat,
        summary_only=args.summary_only,
        detail_only=args.detail_only
    )

    # Optional: generate merged summary table
    if args.merge_summary and index_data:
        try:
            create_merged_summary(args.output_dir, index_data)
        except ImportError:
            print("[WARN] Merged summary requires pandas library, please install: pip install pandas")


if __name__ == "__main__":
    main()
