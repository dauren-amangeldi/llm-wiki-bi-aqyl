"""Бэкфилл BUG-02: вычистить «призраков», накопленных ДО каскадного удаления.

До фикса DELETE /cases/{id} удалял одну строку кейса — файлы, вики-страницы,
эмбеддинги, артефакты, твин-сессии и чат оставались жить и находились поиском
(UAT-ревизия показала это на проде: материал удалённого кейса — первый
результат поиска, счётчик не уменьшался).

Призрак = файл, не входящий в doc_ids ни одного живого кейса, плюс сироты
второго порядка (артефакты/твины/чат несуществующих кейсов, страницы без
файла-источника).

Запуск (в контейнере api, где есть env и зависимости):
  docker compose exec api uv run python scripts/purge_ghosts.py            # dry-run
  docker compose exec api uv run python scripts/purge_ghosts.py --apply
  docker compose exec api uv run python scripts/purge_ghosts.py --apply --purge-s3

По умолчанию — dry-run: печатает, ЧТО будет удалено, ничего не трогая.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="реально удалить (без флага — dry-run)")
    parser.add_argument("--purge-s3", action="store_true", help="удалить и сырые объекты в S3")
    args = parser.parse_args()

    from llm_wiki.config import settings

    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        # 1. Файлы-призраки: не входят в doc_ids ни одного кейса.
        ghost_files = conn.execute(text("""
            SELECT f.file_id, f.original_name, f.raw_key,
                   COALESCE(f.created_pages, '[]'::json)::text AS pages
            FROM files f
            WHERE NOT EXISTS (
                SELECT 1 FROM cases c
                WHERE c.doc_ids::jsonb ? f.file_id
            )
        """)).fetchall()
        ghost_ids = [r.file_id for r in ghost_files]

        import json as _json
        ghost_slugs: list[str] = []
        for r in ghost_files:
            try:
                ghost_slugs.extend(s for s in _json.loads(r.pages) if s)
            except Exception:  # noqa: BLE001
                pass

        # 2. Сироты второго порядка: артефакты/твины/чат несуществующих кейсов.
        ghost_case_artifacts = conn.execute(text("""
            SELECT artifact_id FROM artifacts a
            WHERE a.document_id LIKE 'case-%'
              AND NOT EXISTS (SELECT 1 FROM cases c WHERE c.id = a.document_id)
        """)).fetchall()
        ghost_doc_artifacts = conn.execute(text("""
            SELECT artifact_id FROM artifacts a
            WHERE a.document_id NOT LIKE 'case-%'
              AND NOT EXISTS (SELECT 1 FROM files f WHERE f.file_id = a.document_id)
        """)).fetchall()
        ghost_sessions = conn.execute(text("""
            SELECT id FROM twin_sessions t
            WHERE NOT EXISTS (SELECT 1 FROM cases c WHERE c.id = t.case_id)
        """)).fetchall()
        ghost_chat = conn.execute(text("""
            SELECT count(*) AS n FROM chat_messages m
            WHERE m.scope_type = 'case'
              AND NOT EXISTS (SELECT 1 FROM cases c WHERE c.id = m.scope_id)
        """)).scalar()

        # 3. Страницы без файла-источника (файл уже удалён, страница осталась).
        orphan_pages = conn.execute(text("""
            SELECT w.slug FROM wiki_fts w
            WHERE NOT EXISTS (
                SELECT 1 FROM files f WHERE f.created_pages::jsonb ? w.slug
            )
        """)).fetchall()
        orphan_page_slugs = [r.slug for r in orphan_pages]

        print(f"Файлов-призраков:            {len(ghost_ids)}")
        for r in ghost_files[:20]:
            print(f"  · {r.file_id}  {r.original_name}")
        if len(ghost_files) > 20:
            print(f"  … и ещё {len(ghost_files) - 20}")
        print(f"Их вики-страниц:             {len(ghost_slugs)}")
        print(f"Страниц вовсе без файла:     {len(orphan_page_slugs)}")
        print(f"Артефактов мёртвых кейсов:   {len(ghost_case_artifacts)}")
        print(f"Артефактов мёртвых доков:    {len(ghost_doc_artifacts)}")
        print(f"Твин-сессий мёртвых кейсов:  {len(ghost_sessions)}")
        print(f"Чат-реплик мёртвых кейсов:   {ghost_chat}")

        if not args.apply:
            print("\nDRY-RUN: ничего не удалено. Запустите с --apply.")
            return 0

        all_slugs = list(dict.fromkeys(ghost_slugs + orphan_page_slugs))
        if all_slugs:
            conn.execute(text("DELETE FROM wiki_fts WHERE slug = ANY(:s)"), {"s": all_slugs})
            conn.execute(text("DELETE FROM chunk_embeddings WHERE slug = ANY(:s)"), {"s": all_slugs})
        if ghost_ids:
            conn.execute(text("DELETE FROM chunk_embeddings WHERE file_id = ANY(:f)"), {"f": ghost_ids})
            conn.execute(text("DELETE FROM artifacts WHERE document_id = ANY(:f)"), {"f": ghost_ids})
            conn.execute(text("DELETE FROM files WHERE file_id = ANY(:f)"), {"f": ghost_ids})
        art_ids = [r.artifact_id for r in ghost_case_artifacts + ghost_doc_artifacts]
        if art_ids:
            conn.execute(text("DELETE FROM artifacts WHERE artifact_id = ANY(:a)"), {"a": art_ids})
        sess_ids = [r.id for r in ghost_sessions]
        if sess_ids:
            conn.execute(text("DELETE FROM twin_messages WHERE session_id = ANY(:s)"), {"s": sess_ids})
            conn.execute(text("DELETE FROM twin_sessions WHERE id = ANY(:s)"), {"s": sess_ids})
        conn.execute(text("""
            DELETE FROM chat_messages
            WHERE scope_type = 'case'
              AND NOT EXISTS (SELECT 1 FROM cases c WHERE c.id = chat_messages.scope_id)
        """))
        print("\nБД вычищена.")

    if args.apply and args.purge_s3:
        from llm_wiki.storage.object_store import get_object_store, legacy_raw_key

        store = get_object_store()
        deleted = failed = 0
        for r in ghost_files:
            key = r.raw_key
            if not key and r.original_name and "." in r.original_name:
                key = legacy_raw_key(r.file_id, "." + r.original_name.rsplit(".", 1)[1])
            if not key:
                continue
            try:
                store.delete(key)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  S3 не удалился {key}: {exc}", file=sys.stderr)
        print(f"S3: удалено {deleted}, ошибок {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
