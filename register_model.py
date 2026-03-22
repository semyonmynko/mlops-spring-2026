import argparse
import pandas as pd
from clearml import OutputModel, Task


PROJECT = "Sentiment Classification"


def get_best_task():
    tasks = Task.get_tasks(
        project_name=PROJECT,
        task_filter={"status": ["completed"]},
    )
    tasks = [t for t in tasks if "Register" not in t.name]
    if not tasks:
        raise RuntimeError("No completed training tasks found")

    def f1(t):
        try:
            return t.get_last_scalar_metrics()["f1"]["f1"]["last"]
        except Exception:
            return 0.0

    best = max(tasks, key=f1)
    print(f"Best: {best.name}  id={best.id}  f1={f1(best):.4f}")
    return best


parser = argparse.ArgumentParser()
parser.add_argument("--task-id", default=None)
cli = parser.parse_args()

src = Task.get_task(task_id=cli.task_id) if cli.task_id else get_best_task()

output_models = src.get_models().get("output", [])
if output_models:
    weights = output_models[0].get_local_copy()
else:
    weights = src.artifacts["model"].get_local_copy()
print(f"Weights: {weights}")

reg_task = Task.init(
    project_name=PROJECT,
    task_name="Register sentiment-classifier v1.0",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)

model = OutputModel(
    task=reg_task,
    name="sentiment-classifier",
    framework="scikit-learn",
    tags=["production", "tfidf", "lr", "v1.0"],
)
model.update_weights(weights_filename=weights, auto_delete_file=False)

metrics = src.get_last_scalar_metrics()
try:
    acc = metrics["accuracy"]["accuracy"]["last"]
    f1_val = metrics["f1"]["f1"]["last"]
    model.report_table("Metrics", "Summary", table_plot=pd.DataFrame({
        "Metric": ["accuracy", "f1"],
        "Value": [round(acc, 4), round(f1_val, 4)],
    }))
    reg_task.get_logger().report_scalar("accuracy", "accuracy", acc, 0)
    reg_task.get_logger().report_scalar("f1", "f1", f1_val, 0)
except Exception as e:
    print(f"metrics copy failed: {e}")

model.publish()
reg_task.close()

print(f"\nModel ID: {model.id}")
