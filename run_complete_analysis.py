import subprocess
import sys

scripts = [
    'complete_preprocessing.py',
    'baseline_implementation.py', 
    'ablation_study.py',
    'validation_framework.py',
    'generate_figures.py'
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print('='*50)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Error in {script}. Stopping.")
        break