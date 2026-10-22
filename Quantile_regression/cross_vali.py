import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
data={
    "area": [800, 1000, 1200, 1500, 1800, 2000,
             900, 1100, 1400, 1700, 1900, 2200],
    "bedrooms": [2, 2, 3, 3, 4, 4,
                 2, 3, 3, 4, 4, 5],
    "location_score": [6, 8, 7, 9, 8, 10,
                       5, 7, 8, 9, 9, 10],
    "age": [10, 5, 8, 3, 6, 2,
            12, 7, 4, 5, 3, 1],
    "price": [20, 28, 35, 50, 54, 70,
              22, 30, 43, 52, 65, 78]
}
df=pd.DataFrame(data)
X=df[["area","bedrooms","location_score","age"]]
y=df["price"]
quantiles = [0.10, 0.50, 0.90]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for q in quantiles:
    fold_losses = []

    for train_idx, val_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=2,
            l2_regularization=1.0,
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        loss = mean_pinball_loss(
            y_val,
            predictions,
            alpha=q
        )

        fold_losses.append(loss)

    print(f"\nQ{int(q * 100)}")
    print("Fold losses:", np.round(fold_losses, 3))
    print("Mean loss:", round(np.mean(fold_losses), 3))
    print("Std deviation:", round(np.std(fold_losses), 3))