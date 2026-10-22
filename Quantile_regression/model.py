from sklearn.ensemble import HistGradientBoostingClassifier
quantiles=[0.10,0.50,0.90]
models={}

for q in quantiles:
    model=HistGradientBoostingClassifier(
        loss="quantile",
        quantile=q,
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X,y)
    models[q]=model

print("ALL quantile models trained!")