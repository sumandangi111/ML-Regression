import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# Feature: house area in square feet
X = np.array([
    [600], [700], [800], [800],
    [900], [1000], [1000], [1000],
    [1100], [1200], [1200], [1300],
    [1400], [1500], [1600], [1800]
])

# Target: selling price in lakh rupees
y = np.array([
    15, 18, 20, 22,
    23, 25, 28, 32,
    31, 35, 42, 39,
    45, 50, 54, 65
])

quantiles = [0.10, 0.50, 0.90]
models = {}

for q in quantiles:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_iter=100,
        random_state=42
    )

    model.fit(X, y)
    models[q] = model

# Predict the price for a 1,000 sq. ft. house
new_house = np.array([[1000]])

for q, model in models.items():
    prediction = model.predict(new_house)[0]
    print(f"Q{int(q * 100)}: ₹{prediction:.2f} lakh")