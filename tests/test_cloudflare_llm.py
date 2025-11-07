from thirawat_mapper_beta.models.rag_llm import (
    _build_plain_messages,
    _build_responses_messages,
    _extract_text_from_cf_payload,
)


def test_build_plain_messages_shape() -> None:
    messages = _build_plain_messages("system text", "user prompt")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "system text"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "user prompt"


def test_build_responses_messages_wraps_text() -> None:
    messages = _build_responses_messages("sys", "user")
    assert messages[0]["content"] == "sys"
    assert messages[1]["content"] == "user"


def test_extract_text_from_responses_payload() -> None:
    payload = {
        "result": {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Rank candidates: 1,2,3",
                        }
                    ]
                }
            ]
        }
    }
    assert _extract_text_from_cf_payload(payload) == "Rank candidates: 1,2,3"


def test_extract_text_from_run_payload() -> None:
    payload = {
        "result": {
            "response": "Preferred order is A, B, C.",
            "usage": {"prompt_tokens": 10},
        }
    }
    assert _extract_text_from_cf_payload(payload) == "Preferred order is A, B, C."
