@echo off
echo Starting AI Environment setup...
python -m pip install --upgrade pip
echo Installing Data Science tools...
pip install numpy pandas matplotlib seaborn scikit-learn jupyter tqdm flask fastapi uvicorn requests
echo AI Environment setup complete!
pause