import pandas as pd
import numpy as np
import torch
np.random.seed(42)
data_size = 1000
# Generate synthetic medical data
data={
    'Patient_ID': range(1, data_size + 1),
    'Age': np.random.randint(20, 80, size=data_size),
    'HR': np.random.normal(80, 15, data_size), # Heart Rate
    'MAP': np.random.normal(70,20, data_size), # Mean Arterial Pressure
    'Sp02': np.random.normal(95, 5, data_size), # Oxygen Saturation
    'Comorbidity_Score': np.random.randint(0, 5, size=data_size)
}
df = pd.DataFrame(data)
#2. Add "Clinical Noise"(Missing values- a classic MD-AI problem)
# Simulate 5% missing Spo2 data (sensor slip)
df.loc[df.sample(frac=0.05).index, 'Sp02'] = np.nan
#3. Clinical Preprocessing(cleaning)
#Logic: if SpO2 is missing we impute it with the median value
print("Cleaning Data...")
df['SpO2']=df['Sp02'].fillna(df['Sp02'].median())
#4.  Define our target : "Critical Event"
# Logic: if MAP < 60, mark as high risk(1), else low risk(0)
df['Hypotension_Risk'] = (df['MAP'] < 60).astype(int)
#5. Convert to PyTorch Tensors for RTX 5060
features = df[['Age', 'HR', 'MAP', 'SpO2', 'Comorbidity_Score']].values
labels = df['Hypotension_Risk'].values
X=torch.tensor(features, dtype=torch.float32).to('cuda')
Y=torch.tensor(labels, dtype=torch.float32).to('cuda')
print(f"Processed {len(df)} patient records.")
print(f"Data pushed to GPU: {X.device}")
print("\nFirst 5 records of processed data:")
print(df.head())
