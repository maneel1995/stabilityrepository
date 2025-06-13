import argparse
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def upload_to_huggingface(output_dir: str, repo_id: str, private: bool = True):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="dataset",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local Ray dataset output to Hugging Face as private dataset.")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face dataset repo name (e.g., your-username/image_predictions)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="Path to the output directory with Parquet files (default: data/output/)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default is private)"
    )

    args = parser.parse_args()
    upload_to_huggingface(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        private=not args.public
    )
