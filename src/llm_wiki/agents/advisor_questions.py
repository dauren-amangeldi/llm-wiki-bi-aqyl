"""Static clarifying-question sets for the AI advisor, keyed by decision type.

The advisor classifies the user's situation into one of these types, then shows
the type's fixed question set (multiple-choice, one free-text "своими словами"
option always allowed on the frontend). This is the "фикс-набор под тип решения"
the product chose over dynamically generated questions.
"""

from __future__ import annotations

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
    """Static question set for a type, falling back to the generic set."""
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
