# RAG система

## Структура сервисов

В проекте поднимаются три сервиса:

* **API** — основное приложение (FastAPI / backend)
* **Qdrant** — векторное хранилище
* **Redis** — кэш / вспомогательное хранилище

### Описание переменных

```env
REDIS_URL=
BASE_LLM_URL=
API_KEY=
LLM_MODEL=
EMBEDDING_MODEL=
RERANKER_MODEL=
QDRANT_URL=
COLLECTION_NAME=
```

* **REDIS_URL** — URL для подключения к Redis
  Пример: `redis://redis:6379/0`

* **BASE_LLM_URL** — базовый URL LLM-сервиса (OpenAI-совместимый API или свой endpoint)

* **API_KEY** — API-ключ для доступа к LLM

* **LLM_MODEL** — модель для генерации ответов
  Пример: `gpt-4o`, `llama-3`, и т.п.

* **EMBEDDING_MODEL** — модель для генерации эмбеддингов

* **RERANKER_MODEL** — модель для rerank-этапа (если используется в пайплайне)

* **QDRANT_URL** — адрес Qdrant
  Для Docker Compose: `http://qdrant:6333`

* **COLLECTION_NAME** — имя коллекции в Qdrant, используемой приложением

## Запуск проекта

Из корня проекта выполните:

```bash
docker compose up --build
```

Или в фоне:

```bash
docker compose up -d --build
```

После запуска:

* API будет доступно по адресу: **[http://localhost:8000](http://localhost:8000)**
* Swagger / OpenAPI: **[http://localhost:8000/docs](http://localhost:8000/docs)**
* Qdrant API: **[http://localhost:6333](http://localhost:6333)**