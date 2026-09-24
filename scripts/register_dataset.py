import os

from huggingface_hub import HfApi


token = os.environ["HF_TOKEN"]

repo_id = os.environ["HF_DATASET_REPO"]


api = HfApi(token=token)


api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
    token=token
)


api.upload_file(

    path_or_fileobj="data/tourism.csv",

    path_in_repo="tourism.csv",

    repo_id=repo_id,

    repo_type="dataset",

    token=token

)


print("Dataset registered successfully.")