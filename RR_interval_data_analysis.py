import scipy.io
import numpy as np

# Check RR-interval structure
sample_rr = scipy.io.loadmat('Dataset_on_electrocardiograph/Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus/RR-interval/19070921.mat')
print("RR-interval file structure:")
for key in sample_rr.keys():
    if not key.startswith('__'):
        data = sample_rr[key]
        print(f"{key}: shape {data.shape}, dtype {data.dtype}")
        if data.size > 0:
            print(f"  Values: {data.flatten()[:10]}...")  # First 10 values