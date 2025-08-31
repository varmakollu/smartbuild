
#!/usr/bin/env python3
"""
Smart Building Energy Optimization using Machine Learning
This script demonstrates a complete ML-based energy optimization system for smart buildings.
Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SmartBuildingEnergyOptimizer:
    """
    A comprehensive machine learning system for smart building energy optimization
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.feature_columns = []

    def generate_synthetic_data(self, n_samples=8760):
        """Generate synthetic smart building data"""
        np.random.seed(42)

        # Time-based features
        hours = np.arange(n_samples) % 24
        days = np.arange(n_samples) // 24 % 365
        months = (days // 30) % 12

        # Environmental features
        outdoor_temp = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, n_samples)
        humidity = 40 + 20 * np.sin(2 * np.pi * days / 365 + np.pi/4) + np.random.normal(0, 5, n_samples)
        solar_radiation = np.maximum(0, 500 * np.sin(2 * np.pi * hours / 24) * 
                                    np.sin(2 * np.pi * days / 365 + np.pi/6) + 
                                    np.random.normal(0, 50, n_samples))

        # Building operational features
        base_occupancy = 20
        work_hours = ((hours >= 9) & (hours <= 17)).astype(int) * 2
        weekend_factor = ((days % 7) < 5).astype(int) * 0.8 + 0.2
        occupancy = base_occupancy + work_hours * 30 * weekend_factor + np.random.normal(0, 5, n_samples)
        occupancy = np.maximum(0, occupancy)

        # Energy loads
        lighting_load = occupancy * 0.5 + (hours >= 18).astype(int) * 10 + np.random.normal(0, 2, n_samples)
        hvac_load = 20 + 0.8 * np.abs(outdoor_temp - 22) + 0.1 * occupancy + np.random.normal(0, 3, n_samples)
        appliance_load = 5 + 0.05 * occupancy + np.random.normal(0, 1, n_samples)

        # Total energy consumption
        cooling_load = np.maximum(0, 0.2 * (outdoor_temp - 24) * occupancy * 0.01)
        heating_load = np.maximum(0, 0.3 * (18 - outdoor_temp) * occupancy * 0.01)
        energy_consumption = (hvac_load + lighting_load + appliance_load + 
                             cooling_load + heating_load + np.random.normal(0, 2, n_samples))

        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'outdoor_temperature': outdoor_temp,
            'humidity': humidity,
            'solar_radiation': solar_radiation,
            'occupancy': occupancy,
            'hour': hours,
            'day_of_year': days,
            'month': months,
            'day_of_week': days % 7,
            'is_weekend': (days % 7 >= 5).astype(int),
            'lighting_load': lighting_load,
            'hvac_load': hvac_load,
            'appliance_load': appliance_load,
            'energy_consumption': energy_consumption
        })

    def prepare_data(self, df):
        """Prepare data for machine learning"""
        self.feature_columns = ['outdoor_temperature', 'humidity', 'solar_radiation', 'occupancy', 
                               'hour', 'day_of_year', 'month', 'day_of_week', 'is_weekend']

        X = df[self.feature_columns]
        y = df['energy_consumption']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, shuffle=False)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def train_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """Train multiple ML models"""

        models_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'use_scaling': False
            },
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'use_scaling': False
            },
            'Neural Network': {
                'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
                'use_scaling': True
            }
        }

        results = {}

        for name, config in models_config.items():
            model = config['model']

            if config['use_scaling']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'use_scaling': config['use_scaling']
            }

            self.models[name] = model

        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.best_model_scaling = results[best_model_name]['use_scaling']

        return results

    def optimize_energy(self, current_conditions):
        """Real-time energy optimization"""

        # Prepare input for prediction
        input_data = pd.DataFrame([current_conditions])

        if self.best_model_scaling:
            input_scaled = self.scalers['standard'].transform(input_data[self.feature_columns])
            predicted = self.best_model.predict(input_scaled)[0]
        else:
            predicted = self.best_model.predict(input_data[self.feature_columns])[0]

        # Generate optimization recommendations
        recommendations = []

        # Temperature-based recommendations
        if current_conditions['outdoor_temperature'] > 25:
            recommendations.append("Enable efficient cooling systems")
        elif current_conditions['outdoor_temperature'] < 18:
            recommendations.append("Optimize heating efficiency")

        # Occupancy-based recommendations
        if current_conditions['occupancy'] > 60:
            recommendations.append("Increase ventilation for high occupancy")
        elif current_conditions['occupancy'] < 10:
            recommendations.append("Switch to energy-saving mode")

        # Time-based recommendations
        if current_conditions['hour'] in [22, 23, 0, 1, 2, 3, 4, 5]:
            recommendations.append("Reduce lighting and non-essential systems")

        return {
            'predicted_consumption': predicted,
            'recommendations': recommendations
        }

def main():
    """Main execution function"""

    # Initialize optimizer
    optimizer = SmartBuildingEnergyOptimizer()

    print("=== Smart Building Energy Optimization System ===")
    print("1. Generating synthetic building data...")

    # Generate data
    df = optimizer.generate_synthetic_data()
    print(f"   Generated {len(df)} hourly data points")

    print("2. Preparing data for machine learning...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = optimizer.prepare_data(df)

    print("3. Training machine learning models...")
    results = optimizer.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

    print("\n=== Model Performance ===")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  - R² Score: {metrics['R²']:.4f}")
        print(f"  - MAE: {metrics['MAE']:.4f}")
        print(f"  - RMSE: {np.sqrt(metrics['MSE']):.4f}")

    print(f"\nBest Model: {optimizer.best_model_name} (R² = {results[optimizer.best_model_name]['R²']:.4f})")

    print("\n4. Testing real-time optimization...")

    # Example optimization scenario
    current_conditions = {
        'outdoor_temperature': 28.5,
        'humidity': 65,
        'solar_radiation': 450,
        'occupancy': 75,
        'hour': 14,
        'day_of_year': 180,
        'month': 6,
        'day_of_week': 2,
        'is_weekend': 0
    }

    optimization_result = optimizer.optimize_energy(current_conditions)

    print("Current Conditions:")
    for key, value in current_conditions.items():
        print(f"  {key}: {value}")

    print(f"\nPredicted Energy Consumption: {optimization_result['predicted_consumption']:.2f} kWh")
    print("\nOptimization Recommendations:")
    for i, rec in enumerate(optimization_result['recommendations'], 1):
        print(f"  {i}. {rec}")

    print("\n=== System Ready for Deployment ===")

if __name__ == "__main__":
    main()
