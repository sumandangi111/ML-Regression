# How to prevent data leakage
# Follow these rules:
# 1. Split the data first into training and test sets.
# 2. Fit preprocessing steps (scaling, imputation, feature selection) using training data only.
# 3. Apply the fitted transformations to test data without refitting them.
# 4. Use a Scikit-learn Pipeline to keep preprocessing and modeling together.
# 5. For time-series problems, split chronologically rather than randomly when that matches the real prediction task.
# 6. Remove features that reveal information unavailable at prediction time.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Test MSE:",
      mean_squared_error(y_test, predictions))
