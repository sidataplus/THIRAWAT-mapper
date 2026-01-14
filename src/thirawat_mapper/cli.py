"""Console entry point for the THIRAWAT mapper tooling."""

from __future__ import annotations

import argparse
from typing import Sequence

from thirawat_mapper import __version__


def _run_index_build(argv: Sequence[str]) -> None:
    from thirawat_mapper.index.build import main as index_build_main

    index_build_main(list(argv))


def _run_infer_bulk(argv: Sequence[str]) -> None:
    from thirawat_mapper.infer.bulk import main as infer_bulk_main

    infer_bulk_main(list(argv))


def _run_infer_query(argv: Sequence[str]) -> None:
    from thirawat_mapper.infer.query import main as infer_query_main

    infer_query_main(list(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thirawat",
        description="THIRAWAT mapper CLI (indexing + inference).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index operations")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser("build", help="Build a LanceDB index")
    index_build_parser.add_argument("args", nargs=argparse.REMAINDER)
    index_build_parser.set_defaults(func=lambda ns: _run_index_build(ns.args))

    infer_parser = subparsers.add_parser("infer", help="Inference operations")
    infer_subparsers = infer_parser.add_subparsers(dest="infer_command", required=True)
    infer_bulk_parser = infer_subparsers.add_parser("bulk", help="Run bulk inference")
    infer_bulk_parser.add_argument("args", nargs=argparse.REMAINDER)
    infer_bulk_parser.set_defaults(func=lambda ns: _run_infer_bulk(ns.args))

    infer_query_parser = infer_subparsers.add_parser("query", help="Interactive query (REPL)")
    infer_query_parser.add_argument("args", nargs=argparse.REMAINDER)
    infer_query_parser.set_defaults(func=lambda ns: _run_infer_query(ns.args))

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)
