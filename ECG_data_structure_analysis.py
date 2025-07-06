import scipy.io
import numpy as np

# Check a sample ECG file
sample_ecg = scipy.io.loadmat('Dataset_on_electrocardiograph/dataset_ecg/ECG/19070921.mat')
print("ECG file structure:")
for key in sample_ecg.keys():
    if not key.startswith('__'):
        data = sample_ecg[key]
        print(f"{key}: shape {data.shape}, dtype {data.dtype}")

        # Calculate duration (assuming 250 Hz)
        if len(data.shape) == 1:
            duration_hours = len(data) / 250 / 3600
            print(f"  Duration: {duration_hours:.1f} hours")
            print(f"  Samples: {len(data):,}")