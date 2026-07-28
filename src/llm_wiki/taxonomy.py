"""Fixed case-tag taxonomy (BI Group).

A closed set of ~30 business tags. They are:
  * auto-assigned when a case is created (an LLM picks the relevant ones from
    this list — the descriptions below are the classifier's guide),
  * editable by the user afterwards,
  * used to filter cases in the navigation.

Keeping the taxonomy fixed (vs free-form tags) avoids a zoo of synonyms and
makes filtering predictable. Edit this list to change the vocabulary.
"""

from __future__ import annotations

# (name, description) — description guides the LLM classifier, not shown as-is.
CASE_TAGS: tuple[tuple[str, str], ...] = (
    ("Качество", "Дефекты, стандарты отделки, контроль на объекте, выявление проблем"),
    ("Безопасность", "Техника безопасности на площадке, охрана труда, incidents, prevention"),
    ("Проекты", "Сроки, бюджет, ресурсы, координация подрядчиков, риски задержек"),
    ("Эффективность", "Оптимизация процессов, автоматизация на стройке, sunk time, затраты"),
    ("Стандартизация", "Унификация процессов, воспроизводимость, checklist, система контроля"),
    ("Сроки", "Планирование, отслеживание прогресса, причины задержек, управление рисками"),
    ("NPS", "NPS покупателей, передача ключей, обслуживание после продажи, жалобы"),
    ("Опыт", "От первого контакта до получения ключей, touchpoints, journey"),
    ("Лояльность", "Repeat purchases, lifetime value, loyalty программы, почему клиент уходит к конкурентам"),
    ("Ценообразование", "Расчёт стоимости, EVC, стратегия цены, позиционирование по цене, переговоры о цене"),
    ("Маркетинг", "Позиционирование, бренд-коммуникации, customer journey, омниканальность, Digital vs Offline"),
    ("Инновация", "Разработка нового продукта, услуги или бизнес-модели; вывод на рынок; преодоление барьеров"),
    ("Конкуренция", "Отличие от других застройщиков, уникальность предложения, positioning"),
    ("Рост", "Расширение, масштабирование, пересчёт ёмкости, стратегия роста, риски при росте"),
    ("Локализация", "Различия между регионами, адаптация к местным условиям, масштабирование по географиям"),
    ("Стратегия", "Долгосрочное позиционирование, выбор рынков (Ansoff), конкурентное преимущество"),
    ("Масштабирование", "Готовность системы расти, сохранение качества при росте, воспроизводимость модели"),
    ("Бренд", "Идентичность, репутация, resonance с аудиторией, возрождение бренда, ассоциации"),
    ("Доверие", "В отсутствующем рынке, сигналы надёжности, репутация, честность, transparent mechanics"),
    ("Лидерство", "Руководство людьми, принятие решений в кризис, видение, развитие команды, персональный стиль"),
    ("Культура", "Ценности компании, символы и нормы, как транслируются убеждения, культурные конфликты"),
    ("Ценности", "Принципы лидера/компании, во что люди верят, на что готовы идти на компромисс"),
    ("Команда", "Состав, динамика, эффективность, разнообразие, развитие способностей членов"),
    ("HR", "Найм, развитие персонала, retention, compensation, культурный fit, succession planning"),
    ("Управление", "Принятие решений, управленческие механики, процессы, контроль, делегирование"),
    ("Переговоры", "Торг с клиентом, с инвесторами, закрытие сделки, влияние и power dynamics"),
    ("Перемены", "Трансформация, организационные сдвиги, resistance to change, change management"),
    ("Трансформация", "Кардинальное переизобретение, поворот стратегии, спасение от упадка"),
    ("Аналитика", "Анализ данных, ML-модели, прогнозирование, insights, поведенческие паттерны, data-driven decisions"),
    ("Обучение", "Сохранение экспертизы, обучение команды, документирование процессов, передача опыта"),
    ("Финансы", "ROI, расчёт эффекта проекта, бюджетирование, экономия средств, обоснование инвестиций"),
    ("Принципы", "Ключевые принципы, фреймворки, алгоритмы действий, применимые в других контекстах; transferable insights"),
)

# Fast membership + preserved display order.
TAG_NAMES: frozenset[str] = frozenset(name for name, _ in CASE_TAGS)
TAG_ORDER: dict[str, int] = {name: i for i, (name, _) in enumerate(CASE_TAGS)}


def clean_tags(tags: object) -> list[str]:
    """Keep only real taxonomy tags — de-duplicated, in taxonomy order.

    Defensive: accepts any input (a client could send junk / unknown tags) and
    silently drops anything not in the taxonomy so the DB only ever holds valid
    tags.
    """
    if not isinstance(tags, list):
        return []
    seen: set[str] = set()
    for t in tags:
        if isinstance(t, str) and t in TAG_NAMES:
            seen.add(t)
    return sorted(seen, key=lambda t: TAG_ORDER[t])
