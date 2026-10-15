
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

y_actual = np.array([10, 20, 15, 30, 25])
y_predicted = np.array([8, 22, 15, 27, 28])

residuals = y_actual - y_predicted

print("Residuals:", residuals)
print("MAE:", round(mean_absolute_error(y_actual, y_predicted), 2))

plt.scatter(y_predicted, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Price (lakhs)")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
