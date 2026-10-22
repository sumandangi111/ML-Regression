import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from itertools import product
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

X_train, X_val, y_train, y_val=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

best_loss = float("inf")
best_params = None

for learning_rate, max_iter, max_leaf_nodes in product(
    [0.05, 0.1],
    [50, 100],
    [3, 7]
):
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.50,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=2,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(X_train,y_train)
    predictions=model.predict(X_val)

    loss=mean_pinball_loss(
        y_val, predictions, alpha=0.50
    )

    print(
        f"Rate={learning_rate}, Iterations={max_iter}, "
        f"Leaves={max_leaf_nodes}, Loss={loss:.3f}"
    )

    if loss<best_loss:
        best_loss=loss
        best_params={
            "learning_rate":learning_rate,
            "max_iter":max_iter,
            "max_leaf_nodes":max_leaf_nodes
        }
print("\nBest parameters:", best_params)
print("Best validation loss:", best_loss)