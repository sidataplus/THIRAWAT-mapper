import pytest

from thirawat_mapper_beta.models.rag_pipeline import RAGPipeline
from thirawat_mapper_beta.models.rag_prompt import RagCandidate


class DummyLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        return self._response


def test_rag_pipeline_accepts_structured_concept_ids_and_appends_remaining():
    pipeline = RAGPipeline(DummyLLM('{"concept_ids":[2,1]}'))
    cands = [RagCandidate(concept_id=1), RagCandidate(concept_id=2), RagCandidate(concept_id=3)]
    out = pipeline.rerank("q", cands)
    assert out.concept_ids == [2, 1, 3]
    assert len(out.scores) == 3


def test_rag_pipeline_accepts_json_code_fence():
    response = "```json\n{\"concept_ids\":[3,2]}\n```"
    pipeline = RAGPipeline(DummyLLM(response))
    cands = [RagCandidate(concept_id=1), RagCandidate(concept_id=2), RagCandidate(concept_id=3)]
    out = pipeline.rerank("q", cands)
    assert out.concept_ids == [3, 2, 1]


def test_rag_pipeline_raises_on_invalid_json():
    pipeline = RAGPipeline(DummyLLM("not json"))
    cands = [RagCandidate(concept_id=1), RagCandidate(concept_id=2)]
    with pytest.raises(ValueError):
        pipeline.rerank("q", cands)


def test_rag_pipeline_raises_when_no_valid_ids_returned():
    pipeline = RAGPipeline(DummyLLM('{"concept_ids":[999]}'))
    cands = [RagCandidate(concept_id=1), RagCandidate(concept_id=2)]
    with pytest.raises(ValueError):
        pipeline.rerank("q", cands)

