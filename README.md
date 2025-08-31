# Smart Building Energy Optimization with Machine Learning

## Overview
This project implements a machine learning framework to optimize energy consumption in smart buildings. Using synthetic and real-world-inspired data, the system predicts energy demand and recommends optimizations to reduce consumption while maintaining occupant comfort.

## Features
- Synthetic dataset generation simulating building operational and environmental conditions
- Multiple ML models: Random Forest, Neural Networks, and time series forecasting
- Real-time energy consumption prediction and scenario-based optimization
- IoT sensor data integration for smart building system control
- Model performance evaluation and feature importance analysis

## Getting Started

### Prerequisites
- Python 3.8+
- Recommended packages in `requirements.txt`

### Installation
```
pip install -r requirements.txt
```

### Running the Project
```
python smart_building_energy_optimization.py
```

## Usage
1. Generate or load building energy dataset  
2. Train ML models and evaluate performance  
3. Use the best model for real-time energy optimization  
4. Integrate with IoT sensor streams for live recommendations

## Results Summary
- Random Forest achieved best R² score (~93%) predicting building energy consumption
- Occupancy, outdoor temperature, and hour of day are key predictive features
- Real-time optimization scenarios demonstrated up to 6.5% potential energy savings
- IoT integration enables adaptive environmental and system control for efficiency

## UCI Energy Efficiency Dataset
This benchmark dataset contains 768 samples with 8 building characteristics. It's perfect for getting started and includes features like:

- Relative compactness and surface area

- Wall area and roof area

- Overall height and orientation

- Glazing area and distribution

## Real-World Building Data
For advanced applications, consider datasets like the Cambridge University Estates archive or the smart company facility dataset, which provides 6 years of real operational data including:

- Energy consumption from 72 meters

- Weather data from on-site stations

- HVAC operational data

- Photovoltaic system production data
