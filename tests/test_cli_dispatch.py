import pytest

from thirawat_mapper import cli


def test_build_parser_has_expected_metadata():
    parser = cli.build_parser()
    assert parser.prog == "thirawat"
    assert parser.description


def test_main_dispatches_index_build(monkeypatch):
    calls: list[list[str]] = []

    def _capture(argv):  # noqa: ANN001
        calls.append(list(argv))

    monkeypatch.setattr(cli, "_run_index_build", _capture)

    cli.main(
        [
            "index",
            "build",
            "--duckdb",
            "x.duckdb",
            "--profiles-table",
            "concept_profiles",
            "--out-db",
            "db",
            "--table",
            "concepts_drug",
        ]
    )

    assert calls == [
        [
            "--duckdb",
            "x.duckdb",
            "--profiles-table",
            "concept_profiles",
            "--out-db",
            "db",
            "--table",
            "concepts_drug",
        ]
    ]


def test_main_dispatches_infer_bulk(monkeypatch):
    calls: list[list[str]] = []

    def _capture(argv):  # noqa: ANN001
        calls.append(list(argv))

    monkeypatch.setattr(cli, "_run_infer_bulk", _capture)

    cli.main(
        [
            "infer",
            "bulk",
            "--db",
            "db",
            "--table",
            "concepts_drug",
            "--input",
            "input.csv",
            "--out",
            "runs/out",
        ]
    )

    assert calls == [["--db", "db", "--table", "concepts_drug", "--input", "input.csv", "--out", "runs/out"]]


def test_main_dispatches_infer_query(monkeypatch):
    calls: list[list[str]] = []

    def _capture(argv):  # noqa: ANN001
        calls.append(list(argv))

    monkeypatch.setattr(cli, "_run_infer_query", _capture)

    cli.main(
        [
            "infer",
            "query",
            "--db",
            "db",
            "--table",
            "concepts_drug",
            "--query",
            "amoxicillin",
        ]
    )

    assert calls == [["--db", "db", "--table", "concepts_drug", "--query", "amoxicillin"]]


@pytest.mark.parametrize(
    ("argv", "attr"),
    [
        (["index", "build", "-h"], "_run_index_build"),
        (["infer", "bulk", "-h"], "_run_infer_bulk"),
        (["infer", "query", "-h"], "_run_infer_query"),
    ],
)
def test_leaf_help_is_delegated_to_underlying_parser(monkeypatch, argv, attr):
    calls: list[list[str]] = []

    def _capture(args):  # noqa: ANN001
        calls.append(list(args))

    monkeypatch.setattr(cli, attr, _capture)
    cli.main(argv)
    assert calls == [["-h"]]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["index"],
        ["infer"],
        ["infer", "unknown"],
    ],
)
def test_main_rejects_invalid_command_shapes(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_version_flag_uses_resolved_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_version", lambda: "0.2.0")

    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0

    captured = capsys.readouterr()
    assert "thirawat 0.2.0" in captured.out
