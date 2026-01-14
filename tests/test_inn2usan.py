from thirawat_mapper_beta.infer.conversion import (
    DEFAULT_INN_TO_USAN,
    MAPPER_EXTRA_INN_TO_USAN,
    convert_inn_ban_to_usan,
)


def test_inn2usan_default_mapping_matches_trainer_and_extras_are_opt_in():
    assert "glucose" not in DEFAULT_INN_TO_USAN
    assert convert_inn_ban_to_usan("glucose") == "glucose"

    merged = {**DEFAULT_INN_TO_USAN, **MAPPER_EXTRA_INN_TO_USAN}
    assert convert_inn_ban_to_usan("glucose", mapping=merged) == "dextrose"

