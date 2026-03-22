import os
import random
import pandas as pd


def download_imdb_subset(n_train=1000, n_test=300, output_dir="./data/raw", random_state=42):
    try:
        from datasets import load_dataset
        print("Downloading IMDB from HuggingFace...")
        ds = load_dataset("imdb", trust_remote_code=True)
        train_df = ds["train"].to_pandas().sample(n=n_train, random_state=random_state).reset_index(drop=True)
        test_df = ds["test"].to_pandas().sample(n=n_test, random_state=random_state).reset_index(drop=True)
    except ImportError:
        print("datasets not found, generating synthetic data")
        train_df, test_df = _synthetic(n_train, n_test, random_state)

    os.makedirs(output_dir, exist_ok=True)
    train_df[["text", "label"]].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df[["text", "label"]].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Saved: train={len(train_df)}, test={len(test_df)}")
    return output_dir


def _synthetic(n_train, n_test, seed):
    random.seed(seed)
    pos = [
        "Great film, loved every minute of it.",
        "Amazing performances, highly recommend.",
        "One of the best movies I have ever seen.",
    ]
    neg = [
        "Terrible waste of time and money.",
        "Boring and completely predictable.",
        "Awful acting, would not watch again.",
    ]

    def make(n):
        rows = []
        for _ in range(n):
            lbl = random.randint(0, 1)
            rows.append({"text": random.choice(pos if lbl else neg), "label": lbl})
        return pd.DataFrame(rows)

    return make(n_train), make(n_test)


if __name__ == "__main__":
    download_imdb_subset()
