#src/models/logistic_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(df, target_col="Delayed", test_size=0.2, random_state=42):
    """
    Trains a Logistic Regression model
    """

    # Split features & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # important if delays are imbalanced
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def show_feature_importance(model, feature_names, top_n=10):
    """
    Displays top positive & negative features
    """
    import pandas as pd

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })

    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    print("\nTop Risk Increasing Features")
    print(coef_df.head(top_n)[["Feature", "Coefficient"]])

    print("\nTop Risk Reducing Features")
    print(coef_df.tail(top_n)[["Feature", "Coefficient"]])