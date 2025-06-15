#!/bin/bash

# Check if models exist, if not train them first
if [ ! -f "models/LightGBM.pkl" ] || [ ! -f "models/LogisticRegression.pkl" ] || [ ! -f "models/NaiveBayes.pkl" ] || [ ! -f "models/RandomForest.pkl" ]; then
    echo "Models not found. Training models first..."
    python main.py
    echo "Model training completed."
fi

# Start Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0