import os
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return f"✔ Downloaded: {output_path}"
    except Exception as e:
        return f"✖ Failed: {url} - {e}"


def generate_urls(start: int, end: int, base_url: str, output_dir: str):
    urls_outputs = []
    for i in range(start, end):
        filename = f"train-{i:05d}-of-00330.parquet"
        url = f"{base_url}{filename}"
        output_path = os.path.join(output_dir, filename)
        urls_outputs.append((url, output_path))
    return urls_outputs


def download_dataset(full: bool, max_workers: int):
    base_url = "https://huggingface.co/datasets/wikimedia/wit_base/resolve/main/data/"
    output_dir = "data/huggingface_data"

    if full:
        start, end = 0, 330
    else:
        start, end = 0, 2

    download_jobs = generate_urls(start, end, base_url, output_dir)

    print(f"🔽 Starting download of {end - start} files with {max_workers} threads...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, path): (url, path) for url, path in download_jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading", ncols=100):
            result = future.result()
            tqdm.write(result)

    print("\n✅ All downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WIT dataset files from HuggingFace.")
    parser.add_argument("--full", action="store_true", help="Download the full dataset (330 files)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download threads (default: 4)")

    args = parser.parse_args()
    download_dataset(full=args.full, max_workers=args.workers)
