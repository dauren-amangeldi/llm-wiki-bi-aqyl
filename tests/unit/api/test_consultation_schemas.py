"""Schema-level tests for the AI-advisor consultation contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_wiki.api.schemas import (
    ClarificationQuestion,
    ClarificationRequiredResponse,
    ConsultationOutcomeRequest,
    ConsultationRespondRequest,
    ConsultationSnapshotUpdate,
    ConsultationStartRequest,
    DecisionBrief,
    DecisionBriefResponse,
    QuestionAnswer,
    UnderstandingSnapshot,
    UnderstandingSnapshotResponse,
)


def test_consultation_start_request_requires_query() -> None:
    with pytest.raises(ValidationError):
        ConsultationStartRequest(query="ab")  # too short, min_length=3


def test_clarification_required_response_shape() -> None:
    resp = ClarificationRequiredResponse(
        mode="clarification_required",
        session_id="advisor-session-1",
        decision_type="initiative_scaling",
        questions=[
            ClarificationQuestion(
                id="q1",
                text="Какой результат важнее?",
                why_it_matters="Определит приоритет",
                options=["Скорость", "Эффект"],
                allow_custom=True,
            )
        ],
        question_limit=5,
    )
    assert resp.mode == "clarification_required"
    assert resp.questions[0].id == "q1"


def test_respond_request_accepts_answers_and_skip_all() -> None:
    body = ConsultationRespondRequest(
        answers=[QuestionAnswer(question_id="q1", answer="Скорость", skipped=False)],
        give_advice_now=False,
    )
    assert body.answers[0].answer == "Скорость"

    skip_all = ConsultationRespondRequest(answers=[], give_advice_now=True)
    assert skip_all.give_advice_now is True


def test_understanding_snapshot_response_shape() -> None:
    snap = UnderstandingSnapshot(
        decision="Масштабировать ли пилот",
        desired_outcome="Сократить срок решений",
        horizon="Текущий год",
        constraints=["Команда ограничена"],
        stakeholders=["ИТ"],
        success_criteria=["Измеримое сокращение"],
        assumptions=["Пилот репрезентативен"],
    )
    resp = UnderstandingSnapshotResponse(mode="understanding_snapshot", session_id="s1", snapshot=snap)
    assert resp.snapshot.decision == "Масштабировать ли пилот"


def test_snapshot_update_is_partial() -> None:
    update = ConsultationSnapshotUpdate(decision="Новая формулировка")
    assert update.desired_outcome is None


def test_decision_brief_response_shape() -> None:
    brief = DecisionBrief(
        recommendation="Ограниченное масштабирование",
        why_now="Пилот дал сигнал",
        problem_frame="Выбор между масштабированием и проверкой",
        key_assumption="Эффект сохранится",
        rationale="Проверяет переносимость",
        alternatives=["Полное масштабирование"],
        risks=["Разная операционная модель"],
        first_step="Выбрать два подразделения",
        reconsider_if=["Эффект ниже 20%"],
        evidence_strength="medium",
        assumptions=["Пилот репрезентативен"],
        sources=["Отчёт по пилоту"],
    )
    resp = DecisionBriefResponse(mode="decision_brief", session_id="s1", brief=brief)
    assert resp.brief.evidence_strength == "medium"


def test_outcome_request_validates_enum() -> None:
    ok = ConsultationOutcomeRequest(outcome="decided")
    assert ok.outcome == "decided"
    with pytest.raises(ValidationError):
        ConsultationOutcomeRequest(outcome="not_a_real_outcome")
