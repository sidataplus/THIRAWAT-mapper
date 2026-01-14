from thirawat_mapper.utils import normalize_text_value


def test_normalize_text_value_lowercases_and_collapses_whitespace():
    assert normalize_text_value("  MixED\tCase  Text  ") == "mixed case text"
    assert normalize_text_value("multi\nline\rvalue") == "multi line value"
    assert normalize_text_value(123) == "123"
