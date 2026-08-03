"""Clarifying-question generation for the AI advisor.

The advisor asks a few multiple-choice questions before giving a recommendation.
Questions are **generated dynamically per situation** (``generate_questions``):
one LLM call tailors 3–5 questions to the user's actual dilemma — the way a sharp
consultant probes what actually changes *this* decision — and also classifies the
decision type. The curated static sets (``QUESTION_SETS`` / ``questions_for``) are
kept as a graceful fallback when the LLM call fails, so the flow never breaks.
The frontend always adds a free-text "своими словами" option on top.
"""

from __future__ import annotations

import json
from typing import Any

from llm_wiki.llm.client import LLMClient

# Decision types + their human labels (shown as the classified type).
DECISION_TYPE_LABELS: dict[str, str] = {
    "partnership": "Партнёрство / совместный бизнес",
    "market_entry": "Выход на новый рынок",
    "divestment": "Продажа доли или актива",
    "investment": "Инвестиция / новое направление",
    "generic": "Управленческое решение",
}


def _q(qid: str, text: str, options: list[str], multi: bool = False) -> dict[str, Any]:
    return {"id": qid, "text": text, "options": options, "multi": multi}


# One curated set per type. `partnership` mirrors the reference screens 2-6.
QUESTION_SETS: dict[str, list[dict[str, Any]]] = {
    "partnership": [
        _q("proved", "Что продукт уже доказал?",
           ["Эффект внутри компании", "Внешний спрос от клиентов",
            "Экономику продаж вне компании", "Пока неизвестно"]),
        _q("partner_gives", "Что даёт потенциальный партнёр?",
           ["Доступ к клиентам и канал продаж", "Инвестиции и разделение затрат",
            "Технологию или экспертизу"], multi=True),
        _q("partner_wants", "Что партнёр хочет получить?",
           ["Долю в будущем бизнесе", "Эксклюзивность на рынке",
            "Процент от выручки", "Пока неизвестно"]),
        _q("priority", "Что для вас сейчас важнее?",
           ["Быстро проверить рынок", "Сохранить контроль над продуктом",
            "Получить деньги сейчас"]),
        _q("exclusivity", "На что распространяется эксклюзивность?",
           ["Один рынок (например, Казахстан)", "Отдельный сегмент клиентов",
            "Весь продукт без ограничений", "Пока неизвестно"]),
    ],
    "market_entry": [
        _q("demand", "Что вы знаете о спросе на новом рынке?",
           ["Подтверждён данными", "Есть гипотезы", "Только предположения", "Неизвестно"]),
        _q("mode", "Как планируете выходить?",
           ["Самостоятельно", "Через локального партнёра",
            "Покупка местного игрока", "Ещё не решили"]),
        _q("timing", "Насколько критичны сроки?",
           ["Нужно быстро", "Есть время", "Не срочно"]),
        _q("priority", "Что для вас важнее?",
           ["Скорость входа", "Контроль", "Минимизация рисков"]),
    ],
    "divestment": [
        _q("why", "Почему рассматриваете продажу?",
           ["Непрофильный актив", "Нужны деньги сейчас",
            "Актив не растёт", "Стратегический фокус"], multi=True),
        _q("profit", "Актив прибыльный?",
           ["Да, стабильно", "Нестабильно", "Убыточный", "Неизвестно"]),
        _q("buyers", "Есть ли покупатели?",
           ["Да, конкретные", "Есть интерес", "Пока нет"]),
        _q("priority", "Что важнее?",
           ["Максимальная цена", "Быстрая сделка", "Условия для бизнеса и команды"]),
    ],
    "investment": [
        _q("evidence", "Что подтверждает потенциал?",
           ["Данные или пилот", "Рыночный тренд", "Экспертная оценка", "Пока гипотеза"]),
        _q("scale", "Масштаб инвестиции?",
           ["Небольшой пилот", "Среднее вложение", "Крупная ставка"]),
        _q("horizon", "Горизонт окупаемости?",
           ["До года", "1–3 года", "Более 3 лет", "Неизвестно"]),
        _q("priority", "Что важнее?",
           ["Быстрый возврат", "Стратегический эффект", "Ограничение риска"]),
    ],
    "generic": [
        _q("urgency", "Насколько срочно нужно решение?",
           ["Очень срочно", "Есть время", "Не срочно"]),
        _q("priority", "Что важнее в этом решении?",
           ["Скорость", "Контроль", "Минимизация рисков", "Финансовый результат"], multi=True),
        _q("reversible", "Насколько обратимо решение?",
           ["Легко откатить", "Частично", "Необратимо", "Неизвестно"]),
        _q("data", "Какими данными вы располагаете?",
           ["Достаточно", "Частично", "Мало", "Почти нет"]),
    ],
}


def questions_for(decision_type: str) -> list[dict[str, Any]]:
    """Curated fallback question set for a type (used when generation fails)."""
    return QUESTION_SETS.get(decision_type, QUESTION_SETS["generic"])


_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision_type": {"type": "string", "enum": list(DECISION_TYPE_LABELS)},
    },
    "required": ["decision_type"],
}


async def classify_decision_type(llm: LLMClient, query: str) -> str:
    """Classify a situation into one of the decision types (LLM, one small call).

    Falls back to ``"generic"`` on any error so the flow never breaks.
    """
    prompt = (
        "Classify the management situation below into exactly one decision type.\n"
        f"Types: {', '.join(DECISION_TYPE_LABELS)}.\n"
        "- partnership: joint venture, partner, co-selling, sharing a stake/exclusivity.\n"
        "- market_entry: entering a new market/region/country.\n"
        "- divestment: selling a share, an asset, or exiting a business.\n"
        "- investment: investing in / launching a new direction or product.\n"
        "- generic: anything else.\n\n"
        f"Situation:\n{query}\n\n"
        'Return JSON: {"decision_type": "<one type>"}.'
    )
    try:
        text, _usage = await llm.complete(
            prompt=prompt,
            system="You are a precise classifier. Return only valid JSON.",
            file_id="advisor-classify",
            agent_type="advisor",
            response_format="json",
            json_schema=_CLASSIFY_SCHEMA,
            schema_name="decision_type",
        )
        import json

        data = json.loads(text)
        dt = str(data.get("decision_type", "")).strip()
        return dt if dt in DECISION_TYPE_LABELS else "generic"
    except Exception:  # noqa: BLE001
        return "generic"


# ── Dynamic question generation ──────────────────────────────────────────────
# Strict Structured Outputs schema. Note: count/length constraints (minItems,
# maxItems, …) are NOT supported in OpenAI strict mode, so counts are enforced in
# the prompt and clamped in `_normalize_questions` instead.
_GENERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision_type": {"type": "string", "enum": list(DECISION_TYPE_LABELS)},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "multi": {"type": "boolean"},
                },
                "required": ["id", "text", "options", "multi"],
            },
        },
    },
    "required": ["decision_type", "questions"],
}


def _normalize_questions(raw: Any) -> list[dict[str, Any]]:
    """Coerce the model's questions into clean FlowQuestion dicts.

    Drops malformed entries, caps options at 4 and questions at 5, requires at
    least 2 options, and de-duplicates ids. Returns ``[]`` if nothing survives
    (which triggers the static fallback in ``generate_questions``).
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        opts_raw = item.get("options")
        options = (
            [str(o).strip() for o in opts_raw if str(o).strip()]
            if isinstance(opts_raw, list)
            else []
        )[:4]
        if not text or len(options) < 2:
            continue
        qid = str(item.get("id", "")).strip() or f"q{len(out) + 1}"
        if qid in seen:
            qid = f"{qid}_{len(out) + 1}"
        seen.add(qid)
        out.append(
            {"id": qid, "text": text, "options": options, "multi": bool(item.get("multi"))}
        )
        if len(out) >= 5:
            break
    return out


async def generate_questions(
    llm: LLMClient, query: str, language: str = "ru"
) -> dict[str, Any]:
    """Generate clarifying questions tailored to the user's situation.

    One LLM call returns the decision type + 3–5 situation-specific questions.
    Falls back to classify + the static set on any error, so the flow is robust.
    Returns ``{"decision_type": str, "questions": list[FlowQuestion]}``.
    """
    prompt = (
        "You are the BI AQYL management advisor. Below is a real management "
        "situation described by a user. Generate the clarifying questions you "
        "need answered before giving a grounded recommendation — tailored to "
        "THIS situation, the way a sharp consultant probes what actually changes "
        "this specific decision. Never ask generic filler.\n\n"
        "Rules:\n"
        "- 3 to 5 questions, ordered most to least decision-relevant.\n"
        "- Each question gets 2–3 concrete answer options phrased for THIS "
        "situation. The UI always adds a free-text field, so do NOT add an "
        '"other"/"свой вариант" option. You MAY add "Пока неизвестно" as the '
        "last option when not knowing is itself meaningful.\n"
        '- "multi": true only when several options can honestly apply at once; '
        "otherwise false.\n"
        '- "id": short snake_case slug, unique within the set.\n'
        f"- Write every question and option in this language: {language}. "
        "Keep them tight — no preamble, no numbering.\n"
        f"- Also classify the situation into one decision_type: "
        f"{', '.join(DECISION_TYPE_LABELS)}.\n\n"
        f"Situation:\n{query}\n\n"
        'Return JSON: {"decision_type": "<type>", "questions": '
        '[{"id": "...", "text": "...", "options": ["...", "..."], "multi": false}]}.'
    )
    try:
        text, _usage = await llm.complete(
            prompt=prompt,
            system="You are a precise management advisor. Return only valid JSON.",
            file_id="advisor-questions",
            agent_type="advisor",
            response_format="json",
            json_schema=_GENERATE_SCHEMA,
            schema_name="advisor_questions",
        )
        data = json.loads(text)
        dt = str(data.get("decision_type", "")).strip()
        dt = dt if dt in DECISION_TYPE_LABELS else "generic"
        questions = _normalize_questions(data.get("questions"))
        if questions:
            return {"decision_type": dt, "questions": questions}
    except Exception:  # noqa: BLE001
        pass
    # Fallback: classify + curated static set so the flow never breaks.
    dt = await classify_decision_type(llm, query)
    return {"decision_type": dt, "questions": questions_for(dt)}
