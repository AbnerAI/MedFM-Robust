#!/usr/bin/env python3
"""
Merge evaluation results from multiple models to generate complete paper-style tables

Reference REOBench paper table format:
- Table 1: Scene classification
- Table 2: Semantic segmentation
- Table 3: Object detection
- Table 4: Image captioning
- Table 5: VQA
- Table 6: Visual grounding

Usage:
    python merge_results_to_table.py --input_dir ./outputs --task vqa --output paper_table.xlsx

    Or specify multiple JSON files:
    python merge_results_to_table.py --files results1.json results2.json --output paper_table.xlsx
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_results(json_file):
    """Load a single result file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_metrics_from_stats(stats, task_name):
    """Extract metrics from statistics."""
    if task_name == 'grounding':
        metric_key = 'accuracy@0.5'
        is_percentage = True
    elif task_name == 'vqa':
        metric_key = 'accuracy'
        is_percentage = True
    elif task_name == 'caption':
        metric_key = 'mean_score'
        is_percentage = False
    else:
        metric_key = 'accuracy'
        is_percentage = True

    return metric_key, is_percentage


def process_single_result(result_data):
    """Process a single result file and extract key metrics."""
    config = result_data.get('config', {})
    stats = result_data.get('statistics', {})

    model_type = config.get('model_type', 'unknown')
    dataset = config.get('dataset', 'unknown')
    task = config.get('task', 'unknown')

    metric_key, is_percentage = extract_metrics_from_stats(stats, task)

    # Extract data for each perturbation type
    perturbation_data = {}
    clean_metric = None
    overall_metric = None

    for key, s in stats.items():
        if key == 'all_perturbations_overall':
            overall_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
            continue

        if key == 'none_s0':
            clean_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
        else:
            # Skip non-perturbation statistics keys
            if key in ['caption_standard_metrics'] or not key.endswith(tuple(f'_s{i}' for i in range(10))):
                continue
            # Parse perturbation type and severity
            parts = key.split('_s')
            if len(parts) == 2:
                perturb_type = parts[0]
                severity = int(parts[1])

                if perturb_type not in perturbation_data:
                    perturbation_data[perturb_type] = {}

                perturbation_data[perturb_type][severity] = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))

    # Calculate average for each perturbation type
    perturb_avg = {}
    for perturb_type, severity_data in perturbation_data.items():
        if severity_data:
            perturb_avg[perturb_type] = np.mean(list(severity_data.values()))

    return {
        'model_type': model_type,
        'dataset': dataset,
        'task': task,
        'clean': clean_metric,
        'overall_avg': overall_metric,
        'perturbation_avg': perturb_avg,
        'perturbation_detail': perturbation_data,
        'is_percentage': is_percentage
    }


def merge_same_setting(results_list):
    """
    Merge multiple results with the same (model_type, dataset, task) into a single complete record.

    Description:
    - Your pipeline may run clean and perturbed evaluations separately for the same configuration,
      resulting in two results_*.json files (one with only Clean, one with only perturbation columns).
    - This function merges them to avoid many '-' placeholders in the final CSV/Excel.

    Merge strategy (keeping original core functionality unchanged):
    - clean / overall_avg: Take non-None value (prefer existing)
    - perturbation_detail: Deep merge by perturbation type and severity
    - perturbation_avg: Recalculate mean based on merged detail
    - is_percentage: Must be consistent, otherwise raise error (avoid mixing different task metrics)
    """
    if not results_list:
        return results_list

    merged = {}

    for r in results_list:
        key = (r.get('model_type'), r.get('dataset'), r.get('task'))

        if key not in merged:
            merged[key] = {
                'model_type': r.get('model_type'),
                'dataset': r.get('dataset'),
                'task': r.get('task'),
                'clean': None,
                'overall_avg': None,
                'perturbation_avg': {},
                'perturbation_detail': {},
                'is_percentage': r.get('is_percentage', True),
            }

        m = merged[key]

        # Metric type must be consistent
        if m.get('is_percentage') != r.get('is_percentage', True):
            raise ValueError(
                f"is_percentage mismatch for {key}: {m.get('is_percentage')} vs {r.get('is_percentage')}"
            )

        # clean / overall: take non-None value
        if m.get('clean') is None and r.get('clean') is not None:
            m['clean'] = r.get('clean')

        if m.get('overall_avg') is None and r.get('overall_avg') is not None:
            m['overall_avg'] = r.get('overall_avg')

        # Deep merge perturbation details
        detail = r.get('perturbation_detail') or {}
        for pt, sev_map in detail.items():
            if pt not in m['perturbation_detail']:
                m['perturbation_detail'][pt] = {}
            if isinstance(sev_map, dict):
                m['perturbation_detail'][pt].update(sev_map)

    # Recalculate perturbation_avg based on detail
    for m in merged.values():
        pert_avg = {}
        for pt, sev_map in (m.get('perturbation_detail') or {}).items():
            if isinstance(sev_map, dict) and len(sev_map) > 0:
                pert_avg[pt] = float(np.mean(list(sev_map.values())))
        m['perturbation_avg'] = pert_avg

    return list(merged.values())


def format_value(val, is_pct=True, decimals=2):
    """Format numeric value."""
    if val is None:
        return '-'
    if is_pct:
        return round(val * 100, decimals)
    return round(val, decimals + 2)


def generate_merged_table(results_list, output_excel, output_csv=None):
    """
    Merge results from multiple models to generate paper-format table.

    Args:
        results_list: List of processed results
        output_excel: Output Excel file path
        output_csv: Output CSV file path (optional)
    """
    if not results_list:
        print("[ERROR] No results to merge")
        return

    # Get all perturbation types
    all_perturb_types = set()
    for r in results_list:
        all_perturb_types.update(r['perturbation_avg'].keys())

    all_perturb_types = sorted(list(all_perturb_types))

    # Build table data
    table_data = []

    for r in results_list:
        row = {
            'Model': r['model_type'],
            'Dataset': r['dataset'],
            'Task': r['task'],
            'Clean': format_value(r['clean'], r['is_percentage']),
        }

        # Add average column for each perturbation type
        for perturb_type in all_perturb_types:
            val = r['perturbation_avg'].get(perturb_type)
            row[perturb_type] = format_value(val, r['is_percentage'])

        # Add overall average
        row['Avg'] = format_value(r['overall_avg'], r['is_percentage'])

        # Calculate delta (clean - avg)
        if r['clean'] is not None and r['overall_avg'] is not None:
            delta = r['clean'] - r['overall_avg']
            row['ΔTP'] = format_value(delta, r['is_percentage'])
        else:
            row['ΔTP'] = '-'

        table_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(table_data)

    # Sort
    df = df.sort_values(['Task', 'Dataset', 'Model'])

    # Save Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

    print(f"[OK] Excel table saved: {output_excel}")

    # Save CSV (optional)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"[OK] CSV table saved: {output_csv}")


def find_result_files(input_dir, task_filter=None, dataset_filter=None):
    """Find result files in directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []

    result_files = list(input_dir.rglob("results_*.json"))

    # Filter by task and dataset
    filtered_files = []
    for f in result_files:
        try:
            data = load_results(f)
            config = data.get('config', {})

            if task_filter and config.get('task') != task_filter:
                continue

            if dataset_filter and config.get('dataset') != dataset_filter:
                continue

            filtered_files.append(str(f))
        except:
            continue

    return filtered_files


def main():
    parser = argparse.ArgumentParser(description="Merge multiple model evaluation results to generate table")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing result JSON files")
    parser.add_argument("--files", nargs='+', type=str, default=None,
                        help="Directly specify list of result JSON files")
    parser.add_argument("--task", type=str, default=None,
                        choices=['vqa', 'caption', 'grounding'],
                        help="Filter by specific task")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by specific dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output Excel filename")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV filename")

    args = parser.parse_args()

    # Collect result files
    result_files = []

    if args.files:
        result_files = args.files
    elif args.input_dir:
        result_files = find_result_files(args.input_dir, args.task, args.dataset)
    else:
        print("[ERROR] Please specify --input_dir or --files")
        return

    if not result_files:
        print("[ERROR] No result files found")
        return

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"   - {f}")

    # Load and process results
    results_list = []
    for filepath in result_files:
        try:
            data = load_results(filepath)
            processed = process_single_result(data)
            results_list.append(processed)
            print(f"   [OK] Loaded: {processed['model_type']} / {processed['dataset']} / {processed['task']}")
        except Exception as e:
            print(f"   [WARN] Failed to load {filepath}: {e}")

    if not results_list:
        print("[ERROR] No valid results")
        return

    # Merge results with same (model/dataset/task) (e.g., clean-run and perturbed-run)
    results_list = merge_same_setting(results_list)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_excel = args.output
    else:
        task_str = args.task or 'all'
        output_excel = f"paper_table_{task_str}_{timestamp}.xlsx"

    output_csv = args.output_csv or output_excel.replace('.xlsx', '.csv')

    # Generate table
    generate_merged_table(results_list, output_excel, output_csv)


if __name__ == "__main__":
    main()
