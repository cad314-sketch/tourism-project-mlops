import pandas as pd
import os
# Import necessary libraries from huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Define repository details
REPO   = f"{"cad314-sketch"}/{REPO_NAME}"
repo_id = "cool/tourism_project_dataset"
repo_type = "dataset"

# Initialize API client using the Hugging Face token from environment variables
api = HfApi(token=os.getenv("HF_TKN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload the data folder to the repo
# Corrected path to point to 'data' folder relative to the repo root
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
