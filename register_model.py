"""
ЭТАП 3 — Find the best completed training task and publish its model to
the ClearML Model Registry.

Usage:
    # Auto-select best by F1:
    python register_model.py

    # Specify task ID explicitly:
    python register_model.py --task-id <TASK_ID>
"""

import argparse
import os
import sys

from clearml import Model, OutputModel, Task


PROJECT_NAME = "Sentiment Classification"
MODEL_NAME = "sentiment-classifier"
MODEL_VERSION = "1.0"


def find_best_task() -> Task:
    """Return the completed training task with the highest F1 score."""
    tasks = Task.get_tasks(
        project_name=PROJECT_NAME,
        task_name="TF-IDF + LogisticRegression",
        task_filter={"status": ["completed"]},
    )
    if not tasks:
        raise RuntimeError(
            "No completed training tasks found in project "
            f"'{PROJECT_NAME}'. Run train.py first."
        )

    def get_f1(t: Task) -> float:
        try:
            metrics = t.get_last_scalar_metrics()
            return metrics["f1"]["f1"]["last"]
        except (KeyError, TypeError):
            return 0.0

    best = max(tasks, key=get_f1)
    print(f"Best task: '{best.name}'  id={best.id}  f1={get_f1(best):.4f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=None, help="ClearML task ID to register")
    args = parser.parse_args()

    # ── 1. Find source task ───────────────────────────────────────────────────
    if args.task_id:
        source_task = Task.get_task(task_id=args.task_id)
        print(f"Using task id={source_task.id}")
    else:
        source_task = find_best_task()

    # ── 2. Get model weights from artifact ───────────────────────────────────
    # Prefer the OutputModel stored on the task; fall back to artifact upload.
    models = source_task.get_models()
    output_models = models.get("output", [])

    if output_models:
        source_model: Model = output_models[0]
        local_weights = source_model.get_local_copy()
        print(f"Got model weights from OutputModel: {local_weights}")
    else:
        artifact_path = source_task.artifacts.get("model")
        if artifact_path is None:
            raise RuntimeError("No model artifact found on task.")
        local_weights = artifact_path.get_local_copy()
        print(f"Got model weights from artifact: {local_weights}")

    # ── 3. Create a registration task ────────────────────────────────────────
    reg_task = Task.init(
        project_name=PROJECT_NAME,
        task_name=f"Register {MODEL_NAME} v{MODEL_VERSION}",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    # ── 4. Publish to Model Registry ─────────────────────────────────────────
    registered = OutputModel(
        task=reg_task,
        name=MODEL_NAME,
        framework="scikit-learn",
        tags=["production", "sentiment", "tfidf", "logistic-regression"],
    )
    registered.update_weights(weights_filename=local_weights, auto_delete_file=False)

    # Copy metrics from the source task
    metrics = source_task.get_last_scalar_metrics()
    try:
        accuracy = metrics["accuracy"]["accuracy"]["last"]
        f1 = metrics["f1"]["f1"]["last"]
        registered.report_table(
            "Model Metrics",
            "Summary",
            table_plot={
                "Metric": ["accuracy", "f1"],
                "Value": [round(accuracy, 4), round(f1, 4)],
            },
        )
        reg_task.get_logger().report_scalar("accuracy", "accuracy", accuracy, 0)
        reg_task.get_logger().report_scalar("f1", "f1", f1, 0)
    except (KeyError, TypeError) as e:
        print(f"Warning: could not copy metrics — {e}")

    # Publish makes the model visible in the Model Registry
    registered.publish()

    reg_task.close()

    print("\n" + "=" * 60)
    print("Model published to ClearML Model Registry!")
    print(f"  Name    : {MODEL_NAME}")
    print(f"  Version : {MODEL_VERSION}")
    print(f"  ID      : {registered.id}")
    print("=" * 60)
    print("\nNext: run clearml-serving to deploy this model.")
    print("  See the README for the exact CLI commands.")


if __name__ == "__main__":
    main()
