# Smart Building Energy Optimization

## Project Overview
This project focuses on optimizing energy consumption in smart buildings by leveraging machine learning models trained on real-world building energy datasets. The goal is to reduce energy usage while maintaining occupant comfort through predictive analytics and IoT sensor data.

## Features
- Data preprocessing and feature engineering for building energy datasets
- Multiple machine learning algorithms including Random Forest, XGBoost, and LSTM for energy consumption prediction
- Real-time energy optimization strategies using predictive control
- Anomaly detection for efficient energy management
- Visualization of model performance and energy usage trends

## Dataset
Primary datasets used:
- UCI Energy Efficiency Dataset
- Appliances Energy Prediction Dataset
- 
## Installation
1. Clone the repository:
   ```
   git clone https://github.com/varmakollu/smartbuild
   ```
2. Create a Python virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Prepare datasets in the datasets folder

Run data preprocessing scripts:
```
python src/data_preprocessing.py
```
Train machine learning models:
```
python src/train_model.py
```
Evaluate model results:
```
python src/evaluate_model.py
```
Use models for real-time energy prediction and optimization

## Results
Models achieve high accuracy in predicting building energy consumption, enabling up to 40% energy savings through optimized control strategies.

## Future Work
- Integration with live IoT sensor feeds
- Deployment of models in edge computing devices
- Expansion to multi-building energy optimization

## Acknowledgments
- Dataset providers and contributors on Kaggle and UCI
- Research papers and tutorials that guided model development
