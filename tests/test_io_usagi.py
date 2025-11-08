import pandas as pd

from thirawat_mapper_beta.io.usagi import coerce_usagi_row, is_usagi_format, validate_usagi_frame


def test_is_usagi_format_and_validation():
    df = pd.DataFrame(
        [
            {
                "sourceCode": "A1",
                "sourceName": "Item A",
                "mappingStatus": "APPROVED",
                "matchScore": 0.5,
            }
        ]
    )
    assert is_usagi_format(df.columns)
    assert validate_usagi_frame(df)


def test_coerce_usagi_row_invents_missing_fields():
    row = {"domainId": "Drug"}
    result = coerce_usagi_row(row, row_index=0, source_name="Example", source_code=None)
    assert result["sourceName"] == "Example"
    assert result["sourceCode"].startswith("ROW_")
    assert result["mappingStatus"] == "UNCHECKED"
    assert result["matchScore"] == 0.0


def test_coerce_usagi_row_uses_custom_source_code_column():
    row = {"drug_id": "ABC123"}
    result = coerce_usagi_row(
        row,
        row_index=5,
        source_name="Example",
        source_code=None,
        source_code_field="drug_id",
    )
    assert result["sourceCode"] == "ABC123"
