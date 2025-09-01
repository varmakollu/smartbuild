import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (update path as appropriate)
df = pd.read_csv('ENB2012_data.csv')

# Preview data
print(df.head())
print(df.info())
print(df.describe())

# Define features and target (assuming last two columns are Heating Load and Cooling Load)
X = df.iloc[:, :-2]  # All columns except last two
y = df.iloc[:, -2]   # Heating Load (second last column)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title('Feature Importance for Heating Load Prediction')
plt.show()
