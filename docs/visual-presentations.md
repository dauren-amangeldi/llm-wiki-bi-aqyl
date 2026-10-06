# Visual presentations: implementation and rollout

`presentation_visual` is independent of `presentation`. Both count as one presentation on a case card. A visual deck has 1–8 slides, a chosen `ru`, `kk` or `en` language, private PNGs/thumbnails and prepared PDF/PPTX files. Visual PowerPoint contains slide images and speaker notes; its visible text is not editable. The existing editable renderer is retained.

## Libraries and rendering

- Existing `openai` SDK / `LLMClient`: structured story and notes from selected processed sources.
- Images API, configurable `VISUAL_IMAGE_MODEL`, initially `gpt-image-2.5-flare`, medium quality, native 1536×864. Also supports the configured 2048×1152 size; rejects unexpected dimensions. No 3:2-to-16:9 crop.
- Pillow: compose exact text over text-free artwork, font measurements, four layouts, thumbnails. DejaVu Sans in Docker contains Kazakh glyphs. Layout is checked before buying images; another geometry can be selected without changing copy. If none fits, generation fails rather than truncating text.
- `python-pptx` and `fpdf2`: the same completed PNGs become the export pages. Notes and numbered source names go into PowerPoint notes.
- Existing PostgreSQL, Celery/Redis and object-store abstraction. Production must use private S3; temporary files inside a pod are not the durable result.

Full-image text generation was compared against hybrid composition on six RU/KK/EN samples. Hybrid was selected to preserve letters, numbers and spacing. Generated artwork remains illustrative, not evidence or a data chart. This does not automatically verify every semantic claim made by the text model.

## Durable workflow

`POST → source snapshot → queued plan → slide units → export unit → atomic ready revision + notification`.

`visual_jobs` stores the attempt, immutable input snapshot and deadline. `visual_units` stores plan (index 0), slides (1–8), exports (9), delivery token, lease and completed result. `artifact_revisions` stores successful manifests and source fingerprints; current `artifacts.versions` remains compatible with existing consumers.

The queued database row is the dispatch outbox. Beat dispatches bounded work every 10 seconds. A broker failure does not lose the accepted job. Short PostgreSQL advisory locks coordinate admission and dispatch across replicas; no database connection stays open during the model or image call. Unique delivery tokens fence duplicate/stale messages. Every attempt writes immutable object paths containing its token.

Completed units can be reused after an explicit retry with the same source snapshot, language, prompt version and image settings. Retry waits for prior running units to finish. If the worker is lost or provider outcome is unknown, no automatic paid retry occurs. A transient 429/5xx can retry once; provider SDK image retries are disabled. Completed work survives an export-only failure. A failed update preserves the last successful revision.

The visual dispatcher owns its deadlines; the old generic janitor excludes this kind. An ops purge that marks an artifact failed also stops future visual dispatch. `/ops/celery` additionally exposes durable visual unit counts and oldest pending time: an empty Redis queue does not mean the database queue is empty.

## Limits and operations

Feature starts **disabled**. Deploying these files alone does not turn paid visual generation on. Configure API, workers and beat consistently; keep one beat scheduler. New task names require updated workers before enabling the API switch.

| Setting | Initial value | Meaning |
|---|---:|---|
| `VISUAL_PRESENTATIONS_ENABLED` | false | Enable admission and dispatch |
| `VISUAL_MAX_ACTIVE` | 2 | Global dispatched/running visual units, including planning/export |
| `VISUAL_MAX_QUEUED` | 20 | Maximum pending decks; one pending deck per user |
| `IMAGE_GLOBAL_CONCURRENCY` | 2 | Shared image-provider leases, including infographic calls when visual is enabled |
| `IMAGE_DAILY_REQUEST_LIMIT` | 80 | Attempted image requests per UTC day across replicas |
| `VISUAL_UNIT_TIMEOUT_S` | 240 | Entire unit, including provider and storage |
| `VISUAL_JOB_TIMEOUT_S` | 1800 | Queue wait + complete attempt; a deadline, not an ETA |
| `VISUAL_MAX_SOURCE_CHARS` | 100000 | Reject larger input instead of silently clipping sources |

Image slots and daily request counts are atomic Redis operations. Redis unavailability blocks new calls. Unknown provider outcomes hold a slot until its 15-minute lease expires. Redis must preserve this state: clearing its budget keys resets request counters. The limiter is a **request cap, not a USD budget**. Configure an approved `PRICE_TABLE` tariff to estimate money; missing prices stay `null/unknown_model`, never zero. Media telemetry records usage and provider request IDs.

Celery units use 330/360-second soft/hard limits and a 400-second work lease. The provider timeout is the unit timeout minus 30 seconds. Delivery leases expire after 60 seconds and can be republished. Queued backoff does not occupy a worker slot. With one Celery worker slot, resource isolation from other tasks cannot be guaranteed; a queue name does not create dedicated capacity. Probe settings are not changed by this feature.

Start on test with one deck, then two, while measuring API/readyz latency, queue age, memory, 429s and image usage. Do not infer production capacity from the local sample. For a rollback, disable the feature on API and workers/beat; already-published ready decks remain readable. Keep additive tables and S3 objects.

## Schema and retention

Additive SQL: `scripts/migrations/20261005_visual_presentations.sql`. Existing startup `create_all` also creates the three tables. No data rewrite or queue purge is required. Apply with DDL-capable credentials before rollout. Existing editable versions are preserved lazily before their first overwrite; numeric revisions are assigned thereafter. Editable revisions do not imply a change to the editable design.

Keep the latest ten successful revisions per artifact/language. Hourly cleanup expires resumable attempt payloads/units 24 hours after their deadline. Small idempotency tombstones remain until the artifact is deleted. Orphaned objects older than 24 hours are removed only when absent from both retained revisions and resumable units. A resumed job can reference an earlier attempt's objects without losing them during cleanup. Current access checks apply even to historical slides, thumbnails and exports. Deletion/private source changes block further reads; an already-downloaded file cannot be recalled.

## API contract

- `GET /api/v1/studio/capabilities`: `visual_presentations`, `presentation_revisions`, supported languages.
- `POST /api/v1/studio/generate`: `{document_id, kind:"presentation_visual", language, source_doc_ids?, request_key, resume?}` → 202 `{artifact_id, kind, status, generation_id}`. Reuse `request_key` when recovering a lost HTTP response; use a new key for an intentional new attempt.
- `GET /api/v1/artifacts?document_id=...&language=...`: summaries include `has_content`, `slide_count`, available languages and visual `generation`.
- `GET /api/v1/artifacts/{id}?language=kk&revision=2`: the selected successful version plus current attempt status. An absent language returns `404 detail.reason=version_not_found`; it never substitutes Russian. Without `language`, legacy editable consumers can still request all versions.
- `GET /api/v1/artifacts/{id}/revisions?language=...`: accessible retained history.
- `GET /api/v1/artifacts/{id}/assets/{language}/{revision}/{index}?variant=full|thumb`: authenticated image response, private/no-store. No public S3 URL or base64 in polling responses.
- `GET /api/v1/artifacts/{id}/export?format=pdf|pptx&language=...&revision=...`: exact revision. Visual exports are already assembled; download does not regenerate images.

Progress stages: queued, planning, rendering, exporting, ready, failed; rendering carries `slides_done/slides_total`. `generation.status` describes the attempt; `versions` may still contain an older successful result. Stage/count progress is not a time estimate.

Errors have stable reasons: missing/changed/oversized sources, disabled feature, full queue, user limit, retry wait, invalid deck, text overflow, bad/missing image, daily limit, provider unavailable/unknown outcome, deadline, worker interruption and unavailable assets. Schema/provider error bodies are not exposed as user-facing text. Fail-fast invalid inputs do not create a fake generation notification. Accepted attempts end in the requester's notification only, with kind/language/revision metadata.

## Explicit boundaries

- No wiki/case translation project, editable BI redesign, individual slide editor, automatic generation of all languages or new production containers.
- History supports opening/downloading older revisions. Promoting an old revision as current is not exposed yet.
- CPU layout/export and synchronous object-store operations run off the API event loop. This does not resolve pre-existing readiness issues under load.
- Images are generated from short source-derived briefs; OpenAI remains the external processor already used by the app. Private assets are served by the authorized backend.
- The `APP_ENVIRONMENT=test` load-test auth guard was restored on 2026-10-06. Before rollout verify the effective settings in API/worker/beat: test must explicitly use `APP_ENVIRONMENT=test` when test auth is enabled, and stage/production must use `LOAD_TEST_AUTH_ENABLED=false`. Domain/secret/allowlist checks remain mandatory (repository `AGENTS.md`).

## Verification (2026-10-05)

Real isolated Postgres/Redis/Celery/API run produced three slides and downloadable PDF/PPTX from a synthetic case. The first layout attempt failed before image calls; after deterministic layout fallback the next attempt completed. Six earlier native 16:9 image samples covered RU/KK/EN full-image versus hybrid composition. An individual image took about 9–11 seconds in that local pilot; this is not a deck SLA.

Automated checks cover duplicate HTTP/delivery requests, dispatch across replicas, broker failure, worker loss, retry reuse/wait, empty/changed sources, private historical images, cancellation, current-versus-old content, strict language selection, editable history/retention, orphan cleanup, shared Redis caps and 16:9 exports with exact image bytes/notes. Frontend checks cover mode/language/revision selection, retaining previous content, downloads, private blob lifecycle and visual-only menu state.

The broad backend run passed 367 tests and failed three existing tests. The final retry-boundary run passed all 13 visual API tests, including late plan, image and export results; renderer/export checks cover all four layouts in RU/KK/EN at both native resolutions. Redis integration checks use a real isolated broker. The additive migration was applied to the original schema twice successfully.

Frontend regression after the UI follow-up: 133 test files, 1284 passed and 2 skipped; TypeScript and production build passed. The build retains existing large Mermaid chunk warnings. Mode-tab keyboard navigation is checked inside the enclosing viewer. An existing double-click test now waits for the inventory to enable the button before clicking; sending the initial GET alone did not guarantee the rendered ready state.

The three failures were reproduced on the unmodified base commit in a separate database: `test_rate_limit::test_window_expiry_allows_new_requests`, `test_sensitive_access::test_cases_list_hides_others_private` and `test_sensitive_phase1::test_create_case_persists_privacy`. The latter two expect empty cases to be public despite the existing ready-material rule. These are not masked by changing application behavior in this feature.

Interactive Chrome acceptance subsequently passed on the normal local stack at `http://localhost:5173`, with `VISUAL_PRESENTATIONS_ENABLED=true` applied only to the gitignored local environment and local API/worker/beat processes. Real UI requests produced a 2-slide visual RU deck and a 6-slide editable EN deck. Progress completed without page reload; PDF/PPTX downloads were checked on disk. Mode/language selection, history/revision selection, keyboard navigation and opening a visual revision from a notification over its case were exercised. Closing the artifact left the case open. The earlier permission-denied isolated page on port 5174 was not used for this pass. Deployment and production load/cost acceptance remain separate.
