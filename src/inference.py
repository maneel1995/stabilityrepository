import argparse
import ray
from model import ImageClassifier


def main(concurrency: int, batch_size: int):
    # Disable progress bars and verbose logs
    context = ray.data.DataContext.get_current()
    context.enable_progress_bars = False
    context.verbose = False

    # Load dataset
    ds = ray.data.read_parquet('data/huggingface_data')

    # Drop column
    ds_without_wit_features = ds.drop_columns(['wit_features'])

    # Run inference
    predictions = ds_without_wit_features.map_batches(
        ImageClassifier, concurrency=concurrency, batch_size=batch_size
    )
    final_table = predictions.filter(lambda row: row["truth_values"])

    # Write output
    final_table.write_parquet("data/output/")
    print(
        f"✅ Predictions written to data/output/ (batch_size={batch_size}, concurrency={concurrency})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ImageClassifier on Ray Dataset with custom concurrency and batch size."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent inference workers (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for inference (default: 5)"
    )

    args = parser.parse_args()
    main(concurrency=args.concurrency, batch_size=args.batch_size)
