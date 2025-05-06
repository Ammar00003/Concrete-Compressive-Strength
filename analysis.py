import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Data/Concrete_Data.xls', engine='xlrd')

# Display basic information
#print(df.info())
#print(df.describe())

# Define features and target
X = df.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength(MPa, megapascals) ']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R²: {r2:.3f}')
print(f'RMSE: {rmse:.3f}')

# Save the trained model
joblib.dump(model, 'concrete_compressive_model.pkl')

# Plotting the graph

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Compressive Strength')
plt.show()
plt.savefig("Graph/Concrete_Graph.png")
