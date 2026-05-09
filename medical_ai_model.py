import torch
import torch.nn as nn
import torch.optim as optim
#1. Architecture : Defining the "Doctor AI"
class HypotensionModel(nn.Module):
    def __init__(self):
        super(HypotensionModel, self).__init__()
        self.layer1 = nn.Linear(4, 16)  # Input features: Age, HR, SpO2, Comorbidity_Score
        self.layer2= nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)   # Output: Hypotension Risk (0 or 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
#2. Setup GPU and model
device=torch.device("cuda")
model=HypotensionModel().to(device)
#3. Use the data from medical_data_clean.py
#Features: Age, HR, SpO2, Comorbidity_Score
example_patient=torch.tensor([[65.0, 110.0, 92.0, 3.0]], dtype=torch.float32).to(device)
#4. Run Inference(prediction)
model.eval()
with torch.no_grad():
    prediction=model(example_patient)
    risk_score=prediction.item()
print(f"Clinical Analysis for Patient...")
print(f"Predicted Hypotension Risk Score: {risk_score:.2f}%")
