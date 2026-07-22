from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.models.model_implementations.pylate_models import PylateSearchEncoder
from tests.mock_tasks import MockRetrievalTask


@pytest.mark.parametrize("model_name", ["colbert-ir/colbertv2.0"])
@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_colbert_model_e2e(task: AbsTask, model_name: str, tmp_path: Path):
    pytest.importorskip("pylate", reason="pylate not installed")
    task._eval_splits = ["test"]

    model = mteb.get_model(model_name)
    results = mteb.evaluate(model, task, cache=None)

    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.0


def test_pylate_rerank_selects_ragged_document_embeddings(
    monkeypatch: pytest.MonkeyPatch,
):
    document_embeddings = [[[1.0], [2.0]], [[3.0]]]
    captured: dict[str, object] = {}

    def rerank(**kwargs):
        captured.update(kwargs)
        return [[{"id": "doc-b", "score": 2.0}, {"id": "doc-a", "score": 1.0}]]

    pylate = ModuleType("pylate")
    pylate.rank = SimpleNamespace(rerank=rerank)
    monkeypatch.setitem(sys.modules, "pylate", pylate)

    class FakeSearchEncoder(PylateSearchEncoder):
        def encode(self, *_args, **_kwargs):
            return document_embeddings

    encoder = FakeSearchEncoder()
    encoder.task_corpus = [{"id": "doc-a"}, {"id": "doc-b"}]
    with patch(
        "mteb.models.model_implementations.pylate_models.create_dataloader",
        return_value=object(),
    ):
        result = encoder._pylate_rerank_documents(
            query_idx_to_id={0: "query"},
            query_embeddings=[[[4.0]]],
            top_ranked={"query": ["doc-b", "doc-a"]},
            top_k=2,
            task_metadata=SimpleNamespace(),
            hf_subset="default",
            hf_split="test",
            encode_kwargs={"batch_size": 2},
        )

    assert captured["documents_ids"] == [["doc-b", "doc-a"]]
    assert captured["documents_embeddings"] == [
        [document_embeddings[1], document_embeddings[0]]
    ]
    assert sorted(result["query"]) == [(1.0, "doc-a"), (2.0, "doc-b")]
