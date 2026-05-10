import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
#1. Setup Device
device=torch.device("cuda")
#2. Create the dataset (The 'Medical Records')
np.random.seed(42)
data_size=1000
features=np.random.rand(data_size, 4) # Random Age, HR, SpO2, Comorbidity_Score
#Logic : if 2nd col (HR) is high and 3rd col (SpO2) is low, Risk is 1 (high risk)
labels=((features[:,1]>0.6) & (features[:,2]<0.4)).astype(float)
X = torch.tensor(features, dtype=torch.float32).to(device)
Y= torch.tensor(labels, dtype=torch.float32).view(-1,1).to(device)
#3. Define the model
class SimpleMD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
model=SimpleMD().to(device)
optimizer=optim.Adam(model.parameters(), lr=0.01)
criterion=nn.BCELoss() # Binary Cross Entropy (standard for Yes/No)
#4. Train the model
print("Training the Medical AI Model...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs=model(X)
    loss=criterion(outputs, Y)
    loss.backward() # Backpropagation
    optimizer.step()
    if (epoch+1)%10==0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
print("Training complete.")
#5. Final Test : A "new patient" with specific vitals
sick_patient=torch.tensor([[0.2, 0.9, 0.1, 0.5]], dtype=torch.float32).to(device)
with torch.no_grad():
    risk=model(sick_patient).item() * 100
    print(f"New Analysis-Predicted Hypotension Risk: {risk:.2f}%")

