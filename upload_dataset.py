"""
ЭТАП 1 — Upload and version the IMDB sentiment dataset in ClearML.

Usage:
    python upload_dataset.py

After running, note the printed dataset_id — you'll need it in train.py.
"""

import os
import sys

from clearml import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from data.download_data import download_imdb_subset

DATASET_PROJECT = "Sentiment Classification"
DATASET_NAME = "IMDB Sentiment"
DATASET_VERSION = "1.0"


def main():
    # ── 1. Download data ──────────────────────────────────────────────────────
    data_dir = download_imdb_subset(n_train=1000, n_test=300, output_dir="./data/raw")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # ── 2. Create ClearML Dataset ─────────────────────────────────────────────
    dataset = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=DATASET_PROJECT,
        dataset_version=DATASET_VERSION,
        description="IMDB binary sentiment dataset (1000 train / 300 test subset)",
    )

    # ── 3. Add files ──────────────────────────────────────────────────────────
    dataset.add_files(path=train_path, dataset_path="train.csv")
    dataset.add_files(path=test_path, dataset_path="test.csv")

    # ── 4. Log preview to the dataset ────────────────────────────────────────
    import pandas as pd

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger = dataset.get_logger()
    logger.report_table(
        title="Train Preview",
        series="Head",
        table_plot=train_df.head(10),
    )
    logger.report_histogram(
        title="Label Distribution",
        series="Train",
        values=train_df["label"].value_counts().values.tolist(),
        xlabels=["negative (0)", "positive (1)"],
        yaxis="Count",
    )

    # ── 5. Upload + finalize ──────────────────────────────────────────────────
    dataset.upload()
    dataset.finalize()

    print("\n" + "=" * 60)
    print(f"Dataset uploaded successfully!")
    print(f"  Project : {DATASET_PROJECT}")
    print(f"  Name    : {DATASET_NAME}")
    print(f"  Version : {DATASET_VERSION}")
    print(f"  ID      : {dataset.id}")
    print("=" * 60)
    print("\nPaste the dataset ID into train.py → args['dataset_id']")


if __name__ == "__main__":
    main()
