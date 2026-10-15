import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint, uniform


def load_data():
    return fetch_california_housing(as_frame=True).frame


def split_data(df):
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(y_true, y_pred, label="Model"):
    print(f"\n{label}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R2  : {r2_score(y_true, y_pred):.4f}")


def topic_1(df):
    """Multiple Linear Regression."""
    print("\n" + "=" * 72 + "\nTOPIC 1: MULTIPLE LINEAR REGRESSION\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    model = LinearRegression().fit(Xtr, ytr)
    evaluate(yte, model.predict(Xte), "Linear Regression")
    print("\nCoefficients:")
    print(pd.Series(model.coef_, index=Xtr.columns).sort_values(ascending=False))
    print("Intercept:", model.intercept_)
    evaluate(yte, np.full(len(yte), ytr.mean()), "Mean baseline")


def topic_2(df):
    """Regression assumptions and diagnostics."""
    print("\n" + "=" * 72 + "\nTOPIC 2: REGRESSION ASSUMPTIONS AND DIAGNOSTICS\n" + "=" * 72)
    import statsmodels.api as sm
    Xtr, Xte, ytr, yte = split_data(df)
    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)
    residuals = yte - pred
    evaluate(yte, pred)
    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals, alpha=0.25)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs Predicted Values")
    plt.tight_layout()
    plt.show()
    sm.qqplot(residuals, line="45", fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.show()
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()
    print("Residual mean:", residuals.mean())
    print("Look for random scatter around zero; patterns may indicate nonlinearity or unequal variance.")


def topic_3(df):
    """Feature engineering and feature selection."""
    print("\n" + "=" * 72 + "\nTOPIC 3: FEATURE ENGINEERING AND FEATURE SELECTION\n" + "=" * 72)
    X = df.drop(columns="MedHouseVal").copy()
    y = df["MedHouseVal"]
    X["RoomsPerBedroom"] = X["AveRooms"] / X["AveBedrms"].replace(0, np.nan)
    X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"].replace(0, np.nan)
    X["BedroomsPerRoom"] = X["AveBedrms"] / X["AveRooms"].replace(0, np.nan)
    X = X.replace([np.inf, -np.inf], np.nan)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k=8)),
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ])
    pipe.fit(Xtr, ytr)
    evaluate(yte, pipe.predict(Xte), "Linear Regression with Top 8 Selected Features")

    imputer = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_imp = pd.DataFrame(imputer.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    selector = SelectKBest(score_func=f_regression, k=8).fit(Xtr_imp, ytr)
    print("Selected features:", list(Xtr.columns[selector.get_support()]))

    lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=10000))
    ]).fit(Xtr, ytr)
    print("\nLasso coefficients:")
    print(pd.Series(lasso.named_steps["model"].coef_, index=Xtr.columns).sort_values())


def topic_4(df):
    """Multicollinearity and VIF demo."""
    print("\n" + "=" * 72 + "\nTOPIC 4: MULTICOLLINEARITY AND VIF\n" + "=" * 72)
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    rng = np.random.default_rng(42)
    area = rng.normal(100, 20, 500)
    X = pd.DataFrame({
        "Area": area,
        "Area_Square_Meters": area * 0.092903,  # intentionally redundant
        "Rooms": rng.normal(5, 1, 500),
        "Age": rng.normal(20, 8, 500)
    })
    print("Correlation matrix:\n", X.corr().round(3))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    Xc = sm.add_constant(X)
    vif = pd.DataFrame({
        "Feature": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    })
    print("\nVIF table:\n", vif)
    print("Area and Area_Square_Meters are intentionally redundant; VIF may be extremely large or infinite.")


def topic_5(df):
    """Polynomial regression using synthetic nonlinear data."""
    print("\n" + "=" * 72 + "\nTOPIC 5: POLYNOMIAL REGRESSION\n" + "=" * 72)
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 300).reshape(-1, 1)
    y = 4 + 2 * X[:, 0] + 3 * X[:, 0] ** 2 + rng.normal(0, 2, 300)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rows = []
    for degree in [1, 2, 3, 5, 10]:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        ).fit(Xtr, ytr)
        rows.append({
            "Degree": degree,
            "Train_RMSE": np.sqrt(mean_squared_error(ytr, model.predict(Xtr))),
            "Test_RMSE": np.sqrt(mean_squared_error(yte, model.predict(Xte))),
            "Test_R2": r2_score(yte, model.predict(Xte))
        })
    print(pd.DataFrame(rows).to_string(index=False))
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
    ).fit(Xtr, ytr)
    xp = np.linspace(-3, 3, 300).reshape(-1, 1)
    plt.scatter(X, y, alpha=0.35, label="Data")
    plt.plot(xp, model.predict(xp), color="red", label="Degree 2 fit")
    plt.title("Polynomial Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def topic_6(df):
    """Ridge and Lasso."""
    print("\n" + "=" * 72 + "\nTOPIC 6: RIDGE AND LASSO REGRESSION\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=10000))
        ])
    }
    for name, model in models.items():
        model.fit(Xtr, ytr)
        evaluate(yte, model.predict(Xte), name)

    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge())
    ])
    search = GridSearchCV(
        ridge_pipe,
        {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        scoring="neg_root_mean_squared_error",
        cv=5, n_jobs=-1
    ).fit(Xtr, ytr)
    print("\nBest Ridge parameters:", search.best_params_)
    print("Best CV RMSE:", -search.best_score_)
    evaluate(yte, search.predict(Xte), "Tuned Ridge")


def topic_7(df):
    """Decision Tree Regressor."""
    print("\n" + "=" * 72 + "\nTOPIC 7: DECISION TREE REGRESSOR\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    for depth in [3, 8, 15, None]:
        model = DecisionTreeRegressor(
            max_depth=depth, min_samples_split=10,
            min_samples_leaf=5, random_state=42
        ).fit(Xtr, ytr)
        evaluate(ytr, model.predict(Xtr), f"Train — max_depth={depth}")
        evaluate(yte, model.predict(Xte), f"Test — max_depth={depth}")

    model = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=5, random_state=42
    ).fit(Xtr, ytr)
    print("\nFeature importance:")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False))
    plt.figure(figsize=(18, 8))
    plot_tree(model, feature_names=Xtr.columns, filled=True, fontsize=7)
    plt.title("Decision Tree (max_depth=4)")
    plt.tight_layout()
    plt.show()


def topic_8(df):
    """Random Forest Regressor."""
    print("\n" + "=" * 72 + "\nTOPIC 8: RANDOM FOREST REGRESSOR\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    model = RandomForestRegressor(
        n_estimators=150, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features=1.0,
        random_state=42, n_jobs=-1
    ).fit(Xtr, ytr)
    evaluate(ytr, model.predict(Xtr), "Random Forest — Train")
    evaluate(yte, model.predict(Xte), "Random Forest — Test")
    print("\nFeature importance:")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False))


def topic_9(df):
    """Gradient Boosting Regressor."""
    print("\n" + "=" * 72 + "\nTOPIC 9: GRADIENT BOOSTING REGRESSOR\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        min_samples_leaf=5, random_state=42
    ).fit(Xtr, ytr)
    evaluate(ytr, model.predict(Xtr), "Gradient Boosting — Train")
    evaluate(yte, model.predict(Xte), "Gradient Boosting — Test")
    print("\nFeature importance:")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False))


def topic_10(df):
    """XGBoost, optional dependency."""
    print("\n" + "=" * 72 + "\nTOPIC 10: XGBOOST REGRESSION\n" + "=" * 72)
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost is not installed. Install with: pip install xgboost")
        return
    Xtr, Xte, ytr, yte = split_data(df)
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="rmse",
        random_state=42, n_jobs=-1
    ).fit(Xtr, ytr)
    evaluate(ytr, model.predict(Xtr), "XGBoost — Train")
    evaluate(yte, model.predict(Xte), "XGBoost — Test")
    print("\nTop feature importances:")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(10))


def topic_11(df):
    """Cross-validation."""
    print("\n" + "=" * 72 + "\nTOPIC 11: CROSS-VALIDATION\n" + "=" * 72)
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        scores = -cross_val_score(
            model, X, y, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        print(f"\n{name}")
        print("Fold RMSE:", np.round(scores, 4))
        print("Mean RMSE:", round(scores.mean(), 4))
        print("Std RMSE:", round(scores.std(), 4))


def topic_12(df):
    """GridSearchCV and RandomizedSearchCV."""
    print("\n" + "=" * 72 + "\nTOPIC 12: HYPERPARAMETER TUNING\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 15, None],
            "min_samples_leaf": [1, 2, 5],
            "max_features": [0.7, 1.0]
        },
        scoring="neg_root_mean_squared_error",
        cv=3, n_jobs=-1, verbose=1
    ).fit(Xtr, ytr)
    print("Grid best parameters:", grid.best_params_)
    print("Grid best CV RMSE:", -grid.best_score_)
    evaluate(yte, grid.predict(Xte), "GridSearchCV Test")

    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": randint(100, 401),
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_split": randint(2, 15),
            "min_samples_leaf": randint(1, 8),
            "max_features": uniform(0.5, 0.5)
        },
        n_iter=20, scoring="neg_root_mean_squared_error",
        cv=3, random_state=42, n_jobs=-1, verbose=1
    ).fit(Xtr, ytr)
    print("\nRandomized best parameters:", random_search.best_params_)
    print("Randomized best CV RMSE:", -random_search.best_score_)
    evaluate(yte, random_search.predict(Xte), "RandomizedSearchCV Test")


def topic_13(df):
    """Compare multiple models on one test split."""
    print("\n" + "=" * 72 + "\nTOPIC 13: MODEL COMPARISON\n" + "=" * 72)
    Xtr, Xte, ytr, yte = split_data(df)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10000)),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": np.sqrt(mean_squared_error(yte, pred)),
            "R2": r2_score(yte, pred)
        })
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))


TOPICS = {
    1: topic_1, 2: topic_2, 3: topic_3, 4: topic_4,
    5: topic_5, 6: topic_6, 7: topic_7, 8: topic_8,
    9: topic_9, 10: topic_10, 11: topic_11, 12: topic_12,
    13: topic_13
}


def main():
    parser = argparse.ArgumentParser(description="Run Advanced Regression topic examples.")
    parser.add_argument("--topic", type=int, choices=range(1, 14),
                        help="Run one topic (1-13). Omit to run all topics.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Reserved for future use; plotting topics currently show plots.")
    args = parser.parse_args()

    print("Loading California Housing dataset...")
    df = load_data()
    if args.topic:
        TOPICS[args.topic](df)
    else:
        for number, function in TOPICS.items():
            function(df)


if __name__ == "__main__":
    main()
