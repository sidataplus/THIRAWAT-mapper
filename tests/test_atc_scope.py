import pandas as pd

from thirawat_mapper_beta.infer.bulk import _apply_atc_scope
from thirawat_mapper_beta.infer.shared_filters import AtcScopeResolver


def test_apply_atc_scope_stable_sorts_matches_first():
    df = pd.DataFrame({"concept_id": [222, 333, 111]})
    out = _apply_atc_scope(df, {111})
    assert out["concept_id"].tolist() == [111, 222, 333]


def test_atc_scope_resolver_builds_allowlist_from_codes(monkeypatch, tmp_path):
    df_in = pd.DataFrame({"atc_codes": ["A01"], "atc_ids": [None]})

    df_atc_vocab = pd.DataFrame({"concept_id": [10], "concept_code": ["A01"]})
    df_desc = pd.DataFrame({"atc_id": [10, 10], "concept_id": [111, 222]})

    class DummyResult:
        def __init__(self, df):
            self._df = df

        def fetch_df(self):
            return self._df

    class DummyConn:
        def execute(self, sql):  # noqa: ANN001
            if "vocabulary_id = 'ATC'" in sql:
                return DummyResult(df_atc_vocab)
            if "FROM concept_ancestor" in sql:
                return DummyResult(df_desc)
            raise AssertionError(sql)

    import duckdb  # type: ignore

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: DummyConn())

    resolver = AtcScopeResolver(tmp_path / "vocab.duckdb")
    allowlist = resolver.build_allowlist(df_in, allowlist_max_ids=1000)
    assert allowlist[0] == {111, 222}


def test_atc_scope_resolver_respects_allowlist_max_ids(monkeypatch, tmp_path):
    df_in = pd.DataFrame({"atc_codes": ["A01"], "atc_ids": [None]})

    df_atc_vocab = pd.DataFrame({"concept_id": [10], "concept_code": ["A01"]})
    df_desc = pd.DataFrame({"atc_id": [10, 10], "concept_id": [111, 222]})

    class DummyResult:
        def __init__(self, df):
            self._df = df

        def fetch_df(self):
            return self._df

    class DummyConn:
        def execute(self, sql):  # noqa: ANN001
            if "vocabulary_id = 'ATC'" in sql:
                return DummyResult(df_atc_vocab)
            if "FROM concept_ancestor" in sql:
                return DummyResult(df_desc)
            raise AssertionError(sql)

    import duckdb  # type: ignore

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: DummyConn())

    resolver = AtcScopeResolver(tmp_path / "vocab.duckdb")
    allowlist = resolver.build_allowlist(df_in, allowlist_max_ids=1)
    assert allowlist == {}

