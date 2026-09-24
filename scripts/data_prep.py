import pandas as pd
from pathlib import Path


source = Path("data/tourism.csv")

output = Path(
    "data/tourism_clean.csv"
)


df = pd.read_csv(source)

df = df.drop_duplicates()


if "Unnamed: 0" in df.columns:

    df = df.drop(
        columns=["Unnamed: 0"]
    )


df.to_csv(
    output,
    index=False
)


print(
    "Prepared dataset:",
    df.shape
)