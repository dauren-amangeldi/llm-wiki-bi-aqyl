from llm_wiki.agents.twin_citations import normalize_citations


def test_only_real_case_sources_are_clickable():
    content = normalize_citations({"text": "Вывод [[real#part|Название]], [[foreign]].", "cite": "[[real]]"}, {"real": "Документ"})
    assert content["citations"] == [{"anchor": "real", "title": "Документ"}]
    assert "[[real]]" in content["text"]
    assert "foreign" not in content["text"]
    assert not content["citation_unavailable"]


def test_legacy_placeholder_is_never_assigned_an_arbitrary_document():
    content = normalize_citations({"text": "Вывод. [Source Document]", "cite": "[Source Document]"}, {"a": "A", "b": "B"})
    assert content["text"] == "Вывод."
    assert content["citations"] == []
    assert content["citation_unavailable"]
    assert normalize_citations(content, {"a": "A"})["citation_unavailable"]


def test_saved_citations_survive_reopening_and_deleted_sources_disappear():
    saved = {"text": "Вывод", "citations": [{"anchor": "real", "title": "Old title"}]}
    assert normalize_citations(saved, {"real": "New title"})["citations"][0]["title"] == "New title"
    assert normalize_citations(saved, {})["citations"] == []
