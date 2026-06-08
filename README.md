# llm-wiki

LLM-driven wiki/knowledge base для BI AQYL. Бэкенд (этот репо) + фронтенд (репо `llm-wiki-frontend`).

## Стек

- **Backend**: FastAPI · Python 3.12 · uv · Celery · ChromaDB · SQLite · OpenAI SDK
- **Frontend**: React 19 · Vite 6 · TypeScript · Tailwind 4 · Zustand · i18next
- **Связь**: REST + SSE-стриминг под префиксом `/api/v1/`, заголовки `X-User-*` для контекста пользователя

## Архитектура связки

```
┌──────────────────────┐                    ┌────────────────────────────┐
│ llm-wiki-frontend    │   /api/v1/*        │ llm-wiki                   │
│ React 19 + Vite      │ ────────────────►  │ FastAPI                    │
│ Zustand · i18next    │   SSE для chat     │  ├── api/routes.py (legacy)│
│ ported UI BI AQYL    │   /search          │  └── api/v1/  ← адаптер    │
│ :5173                │ ◄──────────────    │     :8000                  │
└──────────────────────┘                    │                            │
                                            │  ChromaDB · SQLite · OpenAI│
                                            └────────────────────────────┘
```

## Запуск (dev)

```bash
# backend
cd llm-wiki
uv sync
uv run uvicorn llm_wiki.main:app --reload --port 8000

# frontend (в отдельном терминале)
cd llm-wiki-frontend
npm install
npm run dev      # :5173, проксирует /api → :8000
```

## Роадмап интеграции с фронтендом

Делается фазами. Каждая фаза = 1–2 промпта Курсору. Ставим `[x]` после готовности.

### Phase 0 — Инфраструктура

- [x] CORS, папка `src/llm_wiki/api/v1/`, dep `get_current_user`
- [x] Frontend: vite proxy, hardcoded session, dev запускается
- [x] README с роадмапом

### Phase 1 — Materials (список и детали)

- [x] `GET /api/v1/documents` — список из FileRecord с маппингом в schema Material
- [x] `GET /api/v1/documents/{id}` — деталь материала
- [x] `GET /api/v1/tags` → `[]` (заглушка под фазу 6)
- [x] `DELETE /api/v1/documents/{id}` — soft delete (admin)
- [x] `GET /api/v1/documents/{id}/sources` — список исходников
- [x] `GET /api/v1/documents/{id}/dossier` — саммари + page_count
- [x] `GET /api/v1/documents/{id}/related` → `[]` (под фазу 4)
- [x] `GET /api/v1/files/{file_id}/raw` — отдача оригинала
- [x] **Результат:** на фронте Workspace показывает реальные материалы, открывается модалка с саммари

### Phase 2 — Upload / Status

- [x] `POST /api/v1/uploads` — single-file DropZone endpoint (returns `{ document_id, title, content_type, path, status }`)
- [x] `POST /api/v1/materials/upload` — batch multipart, обёртка над существующим ingestion, returns `{ uploaded, skipped }`
- [x] `GET /api/v1/documents/{id}/status` — polling статуса ingestion
- [x] `DELETE /api/v1/documents/{id}/sources/{source_id}` — удалить исходник
- [x] **Результат:** DropZone в модалке грузит файл → статус прогрессирует → итог: done

### Phase 3 — Chat (SSE)

- [x] `POST /api/v1/documents/{id}/ask` — RAG-чат по документу; JSON (default) + SSE (`?stream=true`)
- [x] `POST /api/v1/cards/{card_id}/ask` — алиас на `/documents/{id}/ask`
- [x] Промпт `prompts/chat_document.md` (ru/en/kk, режимы library/expert/advisor)
- [x] `stream_completion()` в `LLMClient` для инкрементального стриминга токенов
- [x] `slug_filter` в `ChunkStore.query()` для скоупа чанков на конкретный документ
- [x] Тесты `tests/unit/api/v1/test_phase3.py` (11 тестов — JSON + SSE + insufficient_evidence)
- [x] **Результат:** чат в модалке отвечает через `apiFetch` (JSON) + готов к SSE через `useSSEStream`

### Phase 4 — Search & Advisor (SSE)

- [ ] `POST /api/v1/search` (SSE) с режимами `library` / `expert` / `advisor`
- [ ] `GET /api/v1/documents/{id}/related` — реальный rerank через Chroma
- [ ] Промпты `prompts/search_expert.md`, `prompts/advisor.md`
- [ ] Rate-limit 10 advisor/min на email
- [ ] **Результат:** глобальный поиск + advisor стримит, related-материалы в модалке заполняются

### Phase 5 — Studio (артефакты)

- [ ] `POST /api/v1/studio/presentation` — JSON слайдов
- [ ] `POST /api/v1/studio/flashcards` — JSON карточек
- [ ] `POST /api/v1/studio/test` — JSON теста
- [ ] Кэш в SQLite, TTL 24ч
- [ ] `POST /api/v1/studio/export` — PDF/DOCX/PPTX
- [ ] **Результат:** в Studio-колонке генерируются и экспортируются артефакты

### Phase 6 — Tags / Notifications / Guidelines

- [ ] Таблицы `tags`, `document_tags`, `notifications` в SQLite
- [ ] `GET/POST /api/v1/tags`, `PUT /api/v1/documents/{id}/tags`, `GET /api/v1/documents/{id}/tag-suggestions`
- [ ] `GET /api/v1/notifications`, `POST /api/v1/notifications/{id}/read`, триггер из ingestion
- [ ] `GET /api/v1/guidelines`, `GET /api/v1/guidelines/{section_id}` (из `data/guidelines.yaml`)
- [ ] **Результат:** теги, колокольчик, вкладка Guidelines живые

### Phase 7 — Авторизация и прод

- [ ] `POST /api/v1/auth/login` — реальная аутентификация
- [ ] Убрать hardcoded session во фронте, вернуть LoginPage
- [ ] Жёсткий `get_current_user` (401 без заголовков)
- [ ] `Dockerfile` для фронта (nginx multi-stage), `nginx.conf`
- [ ] Обновить `docker-compose.yml` сервисом `frontend`
- [ ] Smoke-чек-лист
- [ ] **Результат:** прод-сборка, `docker compose up` поднимает всю систему

## Контракты API

Все эндпоинты под `/api/v1/`. Заголовки в каждом запросе:
`X-User-Email`, `X-User-Role`, `X-Business-Unit`, `X-User-Geo`, `X-User-Position`, `Accept-Language`.

Pydantic-схемы — в `src/llm_wiki/api/v1/schemas.py`. Сверять 1:1 с TS-интерфейсами фронта в `llm-wiki-frontend/src/stores/*.ts` и `src/features/**/*.tsx`.

## Тесты

`uv run pytest tests/`. На каждую фазу — свой подкаталог `tests/unit/api/v1/test_{phase}.py`.
