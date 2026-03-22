"""
Download a subset of the IMDB sentiment dataset and save as CSV files.

Usage:
    python data/download_data.py
"""

import os
import pandas as pd


def download_imdb_subset(
    n_train: int = 1000,
    n_test: int = 300,
    output_dir: str = "./data/raw",
    random_state: int = 42,
) -> str:
    """Download IMDB dataset and save a small subset to CSV."""
    try:
        from datasets import load_dataset

        print("Downloading IMDB dataset via HuggingFace datasets...")
        dataset = load_dataset("imdb", trust_remote_code=True)

        train_df = (
            dataset["train"]
            .to_pandas()
            .sample(n=n_train, random_state=random_state)
            .reset_index(drop=True)
        )
        test_df = (
            dataset["test"]
            .to_pandas()
            .sample(n=n_test, random_state=random_state)
            .reset_index(drop=True)
        )
    except ImportError:
        print("'datasets' library not found — generating synthetic data fallback.")
        train_df, test_df = _generate_synthetic(n_train, n_test, random_state)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df[["text", "label"]].to_csv(train_path, index=False)
    test_df[["text", "label"]].to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} train samples → {train_path}")
    print(f"Saved {len(test_df)} test samples  → {test_path}")

    label_dist = train_df["label"].value_counts().to_dict()
    print(f"Label distribution (train): {label_dist}")

    return output_dir


def _generate_synthetic(n_train: int, n_test: int, random_state: int):
    """Minimal fallback — obviously fake but structurally correct."""
    import random

    random.seed(random_state)

    positive_templates = [
        "This movie was absolutely fantastic and I loved every minute.",
        "An incredible film with outstanding performances.",
        "One of the best movies I have ever seen in my life.",
        "Brilliantly directed and wonderfully acted throughout.",
        "A masterpiece of modern cinema that moved me deeply.",
    ]
    negative_templates = [
        "This film was a complete waste of time and money.",
        "Terrible acting and a boring, predictable plot.",
        "I could not wait for this awful movie to end.",
        "The worst movie of the year by a wide margin.",
        "Dull, lifeless, and utterly devoid of originality.",
    ]

    def make_rows(n):
        rows = []
        for _ in range(n):
            label = random.randint(0, 1)
            text = random.choice(positive_templates if label == 1 else negative_templates)
            rows.append({"text": text, "label": label})
        return pd.DataFrame(rows)

    return make_rows(n_train), make_rows(n_test)


if __name__ == "__main__":
    download_imdb_subset()
