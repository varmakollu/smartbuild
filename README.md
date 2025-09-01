# Smart Building Energy Optimization using Machine Learning

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
- IoT Smart Building Sensor Data from Kaggle

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
- Prepare datasets by running the data preprocessing scripts in the `/data` folder.
- Train models using scripts in `/models` folder.
- Run prediction and optimization scripts in `/optimization` folder.
- Visualize results using notebooks in `/notebooks`.

## Project Structure
- `/data`: Datasets and preprocessing scripts
- `/models`: Model training and evaluation scripts
- `/optimization`: Real-time prediction and control algorithms
- `/notebooks`: Jupyter notebooks for exploratory data analysis and visualization
- `requirements.txt`: Project dependencies

## Results
Models achieve high accuracy in predicting building energy consumption, enabling up to 40% energy savings through optimized control strategies.

## Future Work
- Integration with live IoT sensor feeds
- Deployment of models in edge computing devices
- Expansion to multi-building energy optimization

## Acknowledgments
- Dataset providers and contributors on Kaggle and UCI
- Research papers and tutorials that guided model development
