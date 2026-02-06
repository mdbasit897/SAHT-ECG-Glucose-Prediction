#!/usr/bin/env python3
"""
Master Run Script v3.0 (REVISED)
=================================
Runs the complete revised analysis pipeline addressing all reviewer/editor comments.

Usage:
    python run_complete_analysis_revised.py

Pre-requisites:
    - Run complete_preprocessing.py first (unchanged from v2)
    - processed_data_v2/ directory must exist with features.csv, targets/, loso_splits/

What this addresses:
    Editor #1:  Prediction target clarity → manuscript rewrite (not code)
    Editor #2:  Age normalization rationale → age_sensitivity analysis (NEW)
    Editor #3:  Narrow cohort → manuscript rewrite (not code)
    Editor #4:  Spot measures vs CGM → manuscript rewrite (not code)
    Editor #5:  Neural network overstated → softened in code comments
    Editor #6:  Low R² clinical relevance → back_transformed_errors (NEW)
    Editor #7:  CV hygiene → FIXED: in-fold feature selection + scaling
    Editor #8:  Terminology → manuscript rewrite (not code)

    R1 #34:     Add CGM references → manuscript rewrite
    R1 #39:     ECG preprocessing detail → manuscript rewrite

    R2:         Consistent terminology → manuscript rewrite
    R2:         Position as validation study → manuscript rewrite

    R3 #1:      Take-home in first paragraph → manuscript rewrite
    R3 #2:      Cohort accounting flow → manuscript rewrite
    R3 #3:      Age normalization sensitivity → age_sensitivity (NEW)
    R3 #4:      CV hygiene confirmed → FIXED in all code
    R3 #5:      Back-transformed MAE → back_transformed_errors (NEW)
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
        print(f"\n❌ ERROR in {script_name} (return code: {result.returncode})")
        return False

    print(f"\n✅ {script_name} completed successfully")
    return True


def main():
    print()
    print("=" * 70)
    print("REVISED ANALYSIS PIPELINE v3.0")
    print("Addressing all reviewer and editor comments")
    print("=" * 70)
    print()

    # Check prerequisites
    data_dir = Path("processed_data_v2")
    if not data_dir.exists():
        print("⚠️  processed_data_v2/ not found!")
        print("   Please run complete_preprocessing.py first.")
        print("   The preprocessing code is UNCHANGED from v2.")
        sys.exit(1)

    print("✅ Data directory found: processed_data_v2/")
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
            print(f"\n⚠️  Stopping due to error in {script_name}")
            break

    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETE: {success_count}/{len(scripts)} scripts successful")
    print("=" * 70)

    if success_count == len(scripts):
        print()
        print("📁 Output directories:")
        print("   analysis_results_v3/     — Baseline comparisons, age adjustment, sensitivity")
        print("   ablation_results_v3/     — Ablation study results")
        print("   validation_results_v3/   — Statistical validation reports")
        print()
        print("📊 Key files for manuscript revision:")
        print("   */baseline_comparison.csv     → Table 1 (updated)")
        print("   */age_adjustment.csv          → Table 2 (updated)")
        print("   */age_sensitivity.csv         → Supplementary Table S2 (NEW)")
        print("   */feature_importance.csv      → Supplementary Table S1")
        print("   */feature_selection_stability.csv → Supplementary Table S4 (NEW)")
        print("   */summary.json               → Back-transformed errors for Section 3")
        print("   */validation_report.json     → Permutation + bootstrap for Section 3.4")
        print()
        print("📝 Remaining tasks (manuscript text only, no code needed):")
        print("   1. Rewrite Abstract/Intro with 'glycemic status estimation' terminology")
        print("   2. Add cohort flow: 64 → 60 → 43 → HbA1c n=29 / FBG n=38")
        print("   3. Expand Section 2.3 (ECG preprocessing details)")
        print("   4. Add Section 2.7 note: 'feature selection and standardization")
        print("      were performed within each LOSO training fold'")
        print("   5. Soften neural network language in Section 3.1 and 4")
        print("   6. Add back-transformed MAE sentence in Section 3")
        print("   7. Add clinical relevance paragraph in Discussion")
        print("   8. Expand limitations paragraph")
        print("   9. Add Reviewer 1's suggested CGM references to Discussion")
        print("  10. Proofread: grammar, math spacing, section capitalization")


if __name__ == "__main__":
    main()