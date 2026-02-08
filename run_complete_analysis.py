#!/usr/bin/env python3
"""
Master Run Script
=================================

Usage:
    python run_complete_analysis.py

Pre-requisites:
    - Run complete_preprocessing.py first (unchanged from v2)
    - processed_data_v2/ directory must exist with features.csv, targets/, loso_splits/

What this addresses:
    E#2:  Age normalization rationale → age_sensitivity analysis
    E#3:  Narrow cohort → manuscript rewrite (not code)
    E#4:  Spot measures vs CGM → manuscript rewrite (not code)
    E#5:  Neural network overstated → softened in code comments
    E#6:  Low R² clinical relevance → back_transformed_errors
    E#7:  CV hygiene → FIXED: in-fold feature selection + scaling
    E#8:  Terminology → manuscript rewrite (not code)

    R1 #34:     Add CGM references → manuscript rewrite
    R1 #39:     ECG preprocessing detail → manuscript rewrite

    R2:         Consistent terminology → manuscript rewrite
    R2:         Position as validation study → manuscript rewrite

    R3 #1:      Take-home in first paragraph → manuscript rewrite
    R3 #2:      Cohort accounting flow → manuscript rewrite
    R3 #3:      Age normalization sensitivity → age_sensitivity
    R3 #4:      CV hygiene confirmed → FIXED in all code
    R3 #5:      Back-transformed MAE → back_transformed_errors
    R3 #6:      Soften neural network claims → code comments updated
    R3 #7:      Emphasize uncertainty → bootstrap with subject-level resampling
"""

import subprocess
import sys
import os
from pathlib import Path


def run_script(script_name, description):
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  Script: {script_name}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n ERROR in {script_name} (return code: {result.returncode})")
        return False

    print(f"\n {script_name} completed successfully")
    return True


def main():
    print()
    print("=" * 70)
    print("REVISED ANALYSIS PIPELINE")
    print("=" * 70)
    print()

    # Check prerequisites
    data_dir = Path("processed_data_v2")
    if not data_dir.exists():
        print("   processed_data_v2/ not found!")
        print("   Please run complete_preprocessing.py first.")
        sys.exit(1)

    print(" Data directory found: processed_data_v2/")
    print()

    # Scripts to run in order
    scripts = [
        (
            'comprehensive_baseline_revised.py',
            'STEP 1: Baseline comparison + age adjustment + sensitivity + back-transformed errors\n'
            '  FIXES: CV hygiene (in-fold selection+scaling), age sensitivity (R3#3),\n'
            '         back-transformed MAE (R3#5), feature selection stability (R3#4)'
        ),
        (
            'ablation_study_revised.py',
            'STEP 2: Ablation study with proper CV hygiene\n'
            '  FIXES: CV hygiene (in-fold selection+scaling for all configurations)'
        ),
        (
            'validation_framework_revised.py',
            'STEP 3: Statistical validation (permutation, bootstrap, residuals)\n'
            '  FIXES: Pipeline-based CV, subject-level bootstrap (R3#7)'
        ),
    ]

    success_count = 0
    for script_name, description in scripts:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"\n  Stopping due to error in {script_name}")
            break

    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETE: {success_count}/{len(scripts)} scripts successful")
    print("=" * 70)

    if success_count == len(scripts):
        print()
        print(" Output directories:")
        print("   analysis_results_v3/     — Baseline comparisons, age adjustment, sensitivity")
        print("   ablation_results_v3/     — Ablation study results")
        print("   validation_results_v3/   — Statistical validation reports")

if __name__ == "__main__":
    main()