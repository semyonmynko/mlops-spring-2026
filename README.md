# MLOps Spring 2026 — Sentiment Classification

Курсовой проект по MLOps. Задача — бинарная классификация тональности отзывов (IMDB, positive/negative) с полным циклом через ClearML: загрузка данных, обучение через агента, регистрация модели, инференс, UI.

**Модель:** TF-IDF + Logistic Regression (sklearn pipeline)

## Структура

```
.
├── data/
│   └── download_data.py      # скачивает подмножество IMDB в data/raw/
├── upload_dataset.py         # загружает датасет в ClearML
├── train.py                  # обучение (запускается через агента)
├── register_model.py         # публикует лучшую модель в Model Registry
├── serving/
│   └── preprocess.py         # pre/postprocess для ClearML Serving
├── ui/
│   └── app.py                # Streamlit UI
├── docker-compose.yml        # ClearML Server + Serving + UI
├── Dockerfile.serving
├── Dockerfile.ui
├── scripts/
│   └── setup.sh              # первичная настройка сервера
└── .github/
    └── workflows/
        └── deploy.yml        # CI/CD через GitHub Actions
```

## Этап 0 — Инфраструктура

### Запустить ClearML Server

```bash
docker compose up -d mongo elasticsearch redis
sleep 30
docker compose up -d apiserver fileserver webserver async_delete
```

UI будет на `http://localhost:8080` (логин: `admin` / `clearml1234`).

### Настроить SDK

Создай credentials в UI: **Settings → Workspace → Create new credentials**, затем:

```bash
pip install clearml clearml-agent
```

Сохрани конфиг в `~/clearml.conf`:
```
api {
  web_server: http://<HOST>:8080/
  api_server: http://<HOST>:8008
  files_server: http://<HOST>:8081
  credentials {
    "access_key" = "..."
    "secret_key" = "..."
  }
}
```

### Запустить агента

```bash
clearml-agent daemon --queue students --create-queue --foreground &
```

Агент появится в **Orchestration → Workers**.

## Этап 1 — Датасет

```bash
pip install -r requirements.txt
python upload_dataset.py
```

Скрипт скачивает 1000 train / 300 test примеров из IMDB и загружает их в ClearML Dataset версии 1.0. Напечатанный `dataset_id` нужно вставить в `train.py`.

## Этап 2 — Обучение

```bash
python train.py
```

Скрипт создаёт ClearML Task, логирует гиперпараметры, отправляет задачу в очередь `students` и завершается локально. Агент подхватывает её и запускает обучение.

Логируется: accuracy, f1, confusion matrix (изображение), classification report, artifact модели.

**Второй эксперимент** — клонировать задачу в UI (правая кнопка → Clone), изменить параметры (например `C`, `max_features`) и поставить в очередь.

## Этап 3 — Model Registry

```bash
python register_model.py
```

Находит задачу с лучшим f1, скачивает веса и публикует модель в Model Registry. Модель получает теги `production`, `v1.0` и метрики.

Указать конкретную задачу:
```bash
python register_model.py --task-id <TASK_ID>
```

## Этап 4 — Inference Endpoint

```bash
pip install clearml-serving fastapi uvicorn grpcio

# создать serving service
clearml-serving create --name "Sentiment Serving"

# добавить модель
clearml-serving --id <SERVICE_ID> model add \
  --engine sklearn \
  --endpoint "sentiment" \
  --published \
  --project "Sentiment Classification" \
  --name "sentiment-classifier" \
  --preprocess "serving/preprocess.py"

# запустить
CLEARML_SERVING_TASK_ID=<SERVICE_ID> uvicorn clearml_serving.serving.main:app --port 8082
```

Тест:
```bash
curl -X POST http://localhost:8082/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was great"}'
# {"label":"positive","label_id":1}

curl -X POST http://localhost:8082/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible waste of time"}'
# {"label":"negative","label_id":0}
```

## Этап 5 — UI

```bash
streamlit run ui/app.py
```

Открыть `http://localhost:8501`.

UI принимает текст, отправляет POST-запрос на serving endpoint, показывает метку и latency. Модель не загружается локально.

Адрес endpoint можно поменять в сайдбаре или через переменную окружения:
```bash
SERVING_URL=http://<HOST>:8082/serve/sentiment streamlit run ui/app.py
```

## Деплой на сервер

### Первичная настройка

```bash
bash scripts/setup.sh
```

Заполнить `.env`:
```env
CLEARML_API_ACCESS_KEY=
CLEARML_API_SECRET_KEY=
CLEARML_AGENT_GIT_USER=
CLEARML_AGENT_GIT_PASS=
CLEARML_SERVING_TASK_ID=
SERVER_HOST=<IP>
```

Затем:
```bash
docker compose up -d serving ui
```

### CI/CD

При каждом пуше в `main` GitHub Actions пересобирает и перезапускает контейнеры `serving` и `ui` на сервере.

Добавить в **Settings → Secrets** репозитория:
- `SERVER_HOST` — IP сервера
- `SERVER_USER` — пользователь (обычно `root`)
- `SERVER_SSH_KEY` — приватный SSH-ключ
