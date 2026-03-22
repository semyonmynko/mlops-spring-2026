"""
ЭТАП 2 — Train a TF-IDF + LogisticRegression sentiment classifier via ClearML Agent.

To run two experiments with different parameters, clone the task in the ClearML UI
and change the hyperparameters (e.g. max_features, C, ngram_max) before re-running.

Usage (local):
    python train.py

Usage (remote via agent):
    The script calls task.execute_remotely() which enqueues it to the 'students' queue.
    The ClearML Agent picks it up and runs it in its environment.

Required: set args['dataset_id'] to the ID printed by upload_dataset.py.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from clearml import Dataset, Logger, OutputModel, Task

# ── Task init ─────────────────────────────────────────────────────────────────
task = Task.init(
    project_name="Sentiment Classification",
    task_name="TF-IDF + LogisticRegression",
    output_uri=True,
)

logger: Logger = task.get_logger()
output_model = OutputModel(task=task, framework="scikit-learn")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Change these in the ClearML UI before cloning a task to create experiment 2.
args = {
    # --- Dataset ---
    "dataset_id": "8af7f2fc364b41b992aaae9e4a57f7d4",
    "dataset_name": "IMDB Sentiment",
    "dataset_project": "Sentiment Classification",
    "dataset_version": "1.0",
    # --- TF-IDF ---
    "max_features": 5000,
    "ngram_min": 1,
    "ngram_max": 1,
    "analyzer": "word",
    "sublinear_tf": True,
    # --- LogisticRegression ---
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    # --- General ---
    "random_state": 42,
    "test_size": 0.2,
}
task.connect(args)  # values become editable in the ClearML UI

# ── Remote execution ──────────────────────────────────────────────────────────
# Comment this line out to run locally.
task.execute_remotely(queue_name="students", clone=False, exit_process=True)

# Everything below runs on the agent ──────────────────────────────────────────

# ── Load dataset from ClearML ─────────────────────────────────────────────────
print("Loading dataset from ClearML...")
if args["dataset_id"] != "REPLACE_WITH_YOUR_DATASET_ID":
    ds = Dataset.get(dataset_id=args["dataset_id"])
else:
    ds = Dataset.get(
        dataset_name=args["dataset_name"],
        dataset_project=args["dataset_project"],
        dataset_version=args["dataset_version"],
    )

local_path = ds.get_local_copy()
train_df = pd.read_csv(os.path.join(local_path, "train.csv"))
test_df = pd.read_csv(os.path.join(local_path, "test.csv"))

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

X_train = train_df["text"].fillna("").astype(str).tolist()
y_train = train_df["label"].tolist()
X_test = test_df["text"].fillna("").astype(str).tolist()
y_test = test_df["label"].tolist()

# ── Build pipeline ────────────────────────────────────────────────────────────
pipe = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=int(args["max_features"]),
                ngram_range=(int(args["ngram_min"]), int(args["ngram_max"])),
                analyzer=args["analyzer"],
                sublinear_tf=bool(args["sublinear_tf"]),
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=float(args["C"]),
                max_iter=int(args["max_iter"]),
                solver=args["solver"],
                random_state=int(args["random_state"]),
            ),
        ),
    ]
)

print("Training...")
pipe.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="binary")

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")

# Log scalar metrics
logger.report_scalar(title="accuracy", series="accuracy", value=accuracy, iteration=0)
logger.report_scalar(title="f1", series="f1", value=f1, iteration=0)

# Log classification report as table
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df.columns = ["class"] + list(report_df.columns[1:])
logger.report_table(
    title="Classification Report",
    series="metrics",
    table_plot=report_df,
    iteration=0,
)
output_model.report_table(
    "Classification Report", "Metrics", table_plot=report_df
)

# ── Confusion matrix as image ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()

logger.report_matplotlib_figure(
    title="Confusion Matrix",
    series="test",
    figure=fig,
    iteration=0,
)
output_model.report_confusion_matrix(
    "Confusion Matrix",
    "ConfusionMatrix",
    matrix=cm,
    xlabels=["Negative", "Positive"],
    ylabels=["Negative", "Positive"],
)
plt.close(fig)

# ── Save model artifact ───────────────────────────────────────────────────────
model_path = "sentiment_model.pkl"
joblib.dump(pipe, model_path, compress=3)
print(f"Model saved to {model_path}")

output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
task.upload_artifact(name="model", artifact_object=model_path)

print("\nTask complete.")
print(f"  accuracy = {accuracy:.4f}")
print(f"  f1       = {f1:.4f}")

task.close()
