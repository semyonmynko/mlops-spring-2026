import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from clearml import Dataset, OutputModel, Task

task = Task.init(
    project_name="Sentiment Classification",
    task_name="TF-IDF + LogisticRegression",
    output_uri=True,
)

args = task.connect({
    "dataset_id": "ea10b62755614e09acebc6bb091eea09",
    "dataset_name": "IMDB Sentiment",
    "dataset_project": "Sentiment Classification",
    "dataset_version": "1.0",
    "max_features": 5000,
    "ngram_min": 1,
    "ngram_max": 1,
    "analyzer": "word",
    "sublinear_tf": True,
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
})

task.execute_remotely(queue_name="students", clone=False, exit_process=True)

logger = task.get_logger()
out_model = OutputModel(task=task, framework="scikit-learn")

print("Loading dataset...")
if args["dataset_id"] != "REPLACE_WITH_YOUR_DATASET_ID":
    ds = Dataset.get(dataset_id=args["dataset_id"])
else:
    ds = Dataset.get(
        dataset_name=args["dataset_name"],
        dataset_project=args["dataset_project"],
        dataset_version=args["dataset_version"],
    )

local_path = ds.get_local_copy()


def find_csv(base, name):
    p = os.path.join(base, name)
    if os.path.isfile(p):
        return p
    nested = os.path.join(base, name, name)
    if os.path.isfile(nested):
        return nested
    raise FileNotFoundError(f"{name} not found under {base}")


train_df = pd.read_csv(find_csv(local_path, "train.csv"))
test_df = pd.read_csv(find_csv(local_path, "test.csv"))
print(f"train={len(train_df)}, test={len(test_df)}")

X_train = train_df["text"].fillna("").astype(str).tolist()
y_train = train_df["label"].tolist()
X_test = test_df["text"].fillna("").astype(str).tolist()
y_test = test_df["label"].tolist()

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=int(args["max_features"]),
        ngram_range=(int(args["ngram_min"]), int(args["ngram_max"])),
        analyzer=args["analyzer"],
        sublinear_tf=bool(args["sublinear_tf"]),
    )),
    ("clf", LogisticRegression(
        C=float(args["C"]),
        max_iter=int(args["max_iter"]),
        solver=args["solver"],
        random_state=int(args["random_state"]),
    )),
])

print("Training...")
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="binary")
print(f"acc={acc:.4f}  f1={f1:.4f}")

logger.report_scalar("accuracy", "accuracy", acc, 0)
logger.report_scalar("f1", "f1", f1, 0)

report = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).transpose().reset_index()
report.columns = ["class"] + list(report.columns[1:])
logger.report_table("Classification Report", "metrics", table_plot=report, iteration=0)
out_model.report_table("Classification Report", "Metrics", table_plot=report)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
logger.report_matplotlib_figure("Confusion Matrix", "test", figure=fig, iteration=0)
out_model.report_confusion_matrix(
    "Confusion Matrix", "test",
    matrix=cm.tolist(),
    xaxis="Predicted", yaxis="Actual",
    xlabels=["Negative", "Positive"],
    ylabels=["Negative", "Positive"],
)
plt.close(fig)

model_path = "sentiment_model.pkl"
joblib.dump(pipe, model_path)
task.upload_artifact("model", artifact_object=model_path)
out_model.update_weights(weights_filename=model_path)
print("Done")
