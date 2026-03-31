import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Model training
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.title("Linear Regression")
plt.legend()
plt.show()
