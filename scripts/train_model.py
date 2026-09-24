import json

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)


df = pd.read_csv(
    "data/tourism.csv"
)

df = df.drop_duplicates()


if "Unnamed: 0" in df.columns:

    df = df.drop(
        columns=["Unnamed: 0"]
    )


X = df.drop(
    columns=[
        "ProdTaken",
        "CustomerID"
    ]
)

y = df["ProdTaken"]


categorical = X.select_dtypes(
    include=["object"]
).columns.tolist()


numerical = [
    c for c in X.columns
    if c not in categorical
]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]),

            numerical
        ),

        (
            "cat",

            Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )

            ]),

            categorical
        )

    ]
)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=42
)


pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        RandomForestClassifier(

            n_estimators=400,

            class_weight="balanced",

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1

        )
    )

])


pipeline.fit(
    X_train,
    y_train
)


probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


threshold = 0.35


predictions = (
    probabilities >= threshold
).astype(int)


metrics = {

    "roc_auc":
        float(
            roc_auc_score(
                y_test,
                probabilities
            )
        ),

    "f1":
        float(
            f1_score(
                y_test,
                predictions
            )
        ),

    "precision":
        float(
            precision_score(
                y_test,
                predictions
            )
        ),

    "recall":
        float(
            recall_score(
                y_test,
                predictions
            )
        ),

    "threshold":
        threshold
}


Path(
    "deployment"
).mkdir(
    exist_ok=True
)


joblib.dump(

    pipeline,

    "deployment/model.joblib"

)


with open(
    "deployment/model_metadata.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(metrics)