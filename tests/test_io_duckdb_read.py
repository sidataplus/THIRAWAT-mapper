from pathlib import Path

import duckdb

from thirawat_mapper.io.duckdb_read import read_concept_profiles


def _build_duckdb(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE concept (
            concept_id INTEGER,
            concept_name TEXT,
            domain_id TEXT,
            concept_class_id TEXT,
            standard_concept TEXT,
            invalid_reason TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE concept_profiles (
            concept_id INTEGER,
            profile_text TEXT,
            extra_col TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO concept VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "Name1", "Drug", "Clinical Drug", "S", None),
            (2, "Name2", "Drug", "Ingredient", "S", None),
            (3, "Name3", "Drug", "Clinical Drug", None, None),  # non standard
            (4, "Name4", "Drug", "Clinical Drug", "S", "D"),   # invalid
        ],
    )
    con.executemany(
        "INSERT INTO concept_profiles VALUES (?, ?, ?)",
        [
            (1, "Profile 1", "keep"),
            (2, "Profile 2", "keep"),
            (3, "Profile 3", "drop"),
            (4, "Profile 4", "drop"),
        ],
    )
    con.close()


def test_read_concept_profiles_filters_standard_and_invalid(tmp_path):
    db_path = tmp_path / "db.duckdb"
    _build_duckdb(db_path)

    df = read_concept_profiles(
        duckdb_path=db_path,
        profiles_table="concept_profiles",
        concepts_table="concept",
        extra_profile_columns=["extra_col"],
    )

    # Only concept_id 1 and 2 should remain (standard + valid)
    assert list(sorted(df["concept_id"].tolist())) == [1, 2]
    assert "extra_col" in df.columns


def test_read_concept_profiles_domain_and_class_filters(tmp_path):
    db_path = tmp_path / "db2.duckdb"
    _build_duckdb(db_path)

    df = read_concept_profiles(
        duckdb_path=db_path,
        profiles_table="concept_profiles",
        concepts_table="concept",
        domain_ids=["Drug"],
        concept_class_ids=["Clinical Drug"],
    )

    # Only concept_id 1 matches domain=Drug and class=Clinical Drug
    assert df.shape[0] == 1
    assert df["concept_id"].iat[0] == 1
