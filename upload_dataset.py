import os
import sys
import pandas as pd
from clearml import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from data.download_data import download_imdb_subset

PROJECT = "Sentiment Classification"
DS_NAME = "IMDB Sentiment"
DS_VERSION = "1.0"

data_dir = download_imdb_subset(n_train=1000, n_test=300, output_dir="./data/raw")

dataset = Dataset.create(
    dataset_name=DS_NAME,
    dataset_project=PROJECT,
    dataset_version=DS_VERSION,
    description="IMDB binary sentiment subset (1000/300)",
)

dataset.add_files(path=os.path.join(data_dir, "train.csv"), dataset_path="train.csv")
dataset.add_files(path=os.path.join(data_dir, "test.csv"), dataset_path="test.csv")

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

log = dataset.get_logger()
log.report_table("Train Preview", "Head", table_plot=train_df.head(10))
log.report_histogram(
    "Label Distribution", "Train",
    values=train_df["label"].value_counts().values.tolist(),
    xlabels=["negative (0)", "positive (1)"],
    yaxis="Count",
)

dataset.upload()
dataset.finalize()

print(f"\nDataset ID: {dataset.id}")
print("Paste into train.py → args['dataset_id']")
