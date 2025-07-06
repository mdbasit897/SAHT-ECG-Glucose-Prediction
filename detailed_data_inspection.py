# Check what's inside the Excel files
import pandas as pd

# Clinical indicators
clinical = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/clinical_indicators.xlsx')
print('Clinical Indicators:')
print(f'Shape: {clinical.shape}')
print(f'Columns: {list(clinical.columns)}')
print(clinical.head())

print('\n' + '='*50 + '\n')

# Objective sleep quality
obj_sleep = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/objective_sleep_quality.xlsx')
print('Objective Sleep Quality:')
print(f'Shape: {obj_sleep.shape}')
print(f'Columns: {list(obj_sleep.columns)}')
print(obj_sleep.head())

print('\n' + '='*50 + '\n')

# Subjective sleep quality
subj_sleep = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/subjective_sleep_quality.xlsx')
print('Subjective Sleep Quality:')
print(f'Shape: {subj_sleep.shape}')
print(f'Columns: {list(subj_sleep.columns)}')
print(subj_sleep.head())