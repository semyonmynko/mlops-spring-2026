# MLOps Spring 2026 — Sentiment Classification with ClearML

Full ML lifecycle: dataset versioning → remote training → model registry → inference endpoint → Streamlit UI.

**Task**: Binary sentiment classification on IMDB reviews (negative / positive).
**Model**: TF-IDF + Logistic Regression (scikit-learn pipeline).

---

## Project structure

```
.
├── data/
│   └── download_data.py      # Downloads IMDB subset → data/raw/
├── upload_dataset.py         # ЭТАП 1 — Creates & uploads ClearML Dataset
├── train.py                  # ЭТАП 2 — Training script (runs via ClearML Agent)
├── register_model.py         # ЭТАП 3 — Publishes best model to Model Registry
├── serving/
│   └── preprocess.py         # ЭТАП 4 — ClearML Serving preprocessing class
├── ui/
│   └── app.py                # ЭТАП 5 — Streamlit UI (HTTP only, no local model)
└── requirements.txt
```

---

## ЭТАП 0 — Infrastructure

### 1. Deploy ClearML Server (Docker Compose)

```bash
git clone https://github.com/allegroai/clearml-server
cd clearml-server/docker
docker-compose up -d
```

Open the UI at `http://localhost:8080`.

### 2. Configure ClearML SDK

```bash
pip install clearml
clearml-init
```

Paste the credentials from **Settings → Workspace → Create new credentials**.

### 3. Start ClearML Agent

```bash
pip install clearml-agent
clearml-agent init           # point to your server
clearml-agent daemon --queue students --detached
```

Verify the agent appears in **Orchestration → Workers**.

### 4. Create the queue

In the ClearML UI: **Orchestration → Queues → New Queue → name it `students`**.

---

## ЭТАП 1 — Dataset

```bash
pip install -r requirements.txt
python upload_dataset.py
```

**Note the `dataset_id`** printed at the end — paste it into `train.py`:

```python
args = {
    "dataset_id": "<YOUR_DATASET_ID>",
    ...
}
```

Verify the dataset appears in **Datasets** in the ClearML UI with version `1.0`.

---

## ЭТАП 2 — Training via Agent

### Run experiment 1 (defaults)

```bash
python train.py
```

The script calls `task.execute_remotely(queue_name="students")` and exits locally.
The agent picks it up and runs the full training remotely.

### Run experiment 2 (different hyperparameters)

In the ClearML UI:
1. Open the completed task from experiment 1.
2. Click **Clone**.
3. In the cloned task, go to **Configuration → Hyperparameters** and change:
   - `max_features`: `10000`
   - `C`: `0.1`
   - `ngram_max`: `2`
4. Click **Enqueue** → queue `students`.

Or re-run `train.py` locally after editing the defaults in `args = {...}`.

**Expected result**: 2 completed tasks in ClearML UI with different params and metrics.

---

## ЭТАП 3 — Model Registry

```bash
python register_model.py
```

This automatically finds the task with the best F1 score and publishes its model to the Model Registry.

To specify a task manually:
```bash
python register_model.py --task-id <TASK_ID>
```

Verify the model appears in **Models** with status **Published**.

---

## ЭТАП 4 — Inference Endpoint

### 1. Install clearml-serving

```bash
pip install clearml-serving
```

### 2. Create a serving service

```bash
clearml-serving create --name "Sentiment Serving"
# → note the SERVICE_ID printed
```

### 3. Deploy the model

```bash
clearml-serving --id <SERVICE_ID> model add \
  --engine sklearn \
  --endpoint "sentiment" \
  --published \
  --project "Sentiment Classification" \
  --name "sentiment-classifier" \
  --max-versions 1 \
  --preprocess "serving/preprocess.py"
```

### 4. Start the serving container

```bash
clearml-serving --id <SERVICE_ID> serve --port 8080
```

### 5. Test the endpoint

```bash
curl -X POST http://localhost:8080/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
# → {"label": "positive", "label_id": 1}

curl -X POST http://localhost:8080/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible film, complete waste of time."}'
# → {"label": "negative", "label_id": 0}
```

---

## ЭТАП 5 — Streamlit UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

The UI:
- Accepts text input
- Sends a POST request to the ClearML Serving endpoint
- Displays the predicted label and request latency
- Shows an error message if the endpoint is unavailable
- Does **not** load any model file directly

To change the endpoint URL, use the sidebar or set an environment variable:

```bash
SERVING_URL=http://my-server:8080/serve/sentiment streamlit run ui/app.py
```

---

## Quick reference — ClearML API used

| Operation | API call |
|---|---|
| Create task | `Task.init(project_name=..., task_name=..., output_uri=True)` |
| Log hyperparams | `task.connect(dict)` |
| Remote execution | `task.execute_remotely(queue_name="students")` |
| Get dataset | `Dataset.get(dataset_id=...)` |
| Create dataset | `Dataset.create(...)` |
| Upload dataset | `dataset.add_files(...)` → `dataset.upload()` → `dataset.finalize()` |
| Register model | `OutputModel(task=..., framework=...)` + `.update_weights(...)` |
| Publish model | `output_model.publish()` |
| Log metrics | `logger.report_scalar(...)` |
| Log confusion matrix | `logger.report_matplotlib_figure(...)` |
