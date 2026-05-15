echo "Starting AI setup..."
python -m pip install --upgrade pip
echo "Installing Data Science Libraries..."
pip install numpy pandas matplotlib seaborn scikit-learn
echo "Installing Jupyter & Interactive Tools..."
pip install jupyter notebook tqdm
echo "Installing Web tools..."
pip install flask fastapi uvicorn requests
echo "Setup Complete! Your RTX 5060 is ready for data work"
