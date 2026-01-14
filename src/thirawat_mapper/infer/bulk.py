"""Bulk inference CLI."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from thirawat_mapper.io import coerce_usagi_row, export_relabel_csv, is_usagi_format, validate_usagi_frame
from thirawat_mapper.models import (
    DEFAULT_RERANKER_ID,
    CloudflareConfig,
    CloudflareLLMClient,
    LlamaCppConfig,
    LlamaCppLLMClient,
    LlamaCppServerConfig,
    LlamaCppServerLLMClient,
    OllamaConfig,
    OllamaLLMClient,
    OpenRouterConfig,
    OpenRouterLLMClient,
    RAGPipeline,
    RAGPromptBuilder,
    SapBERTEmbedder,
    TemplatePromptBuilder,
    ThirawatReranker,
    to_candidates,
)
from thirawat_mapper.models.embedder import DEFAULT_MODEL_ID
from .conversion import DEFAULT_INN_TO_USAN, MAPPER_EXTRA_INN_TO_USAN, convert_inn_ban_to_usan
from .shared_filters import AtcScopeResolver, ConceptClassResolver, safe_int, to_exclusion_set
from .utils import (
    configure_torch_for_infer,
    minmax_normalize,
    resolve_device,
    rank_candidates,
    enrich_with_post_scores,
    should_apply_post,
    sanitize_query_text,
    normalize_strength_spacing,
)
from thirawat_mapper.utils import connect_table, normalize_text_value


DEFAULT_TOPK = 100
EVAL_K = (1, 2, 5, 10, 20, 50, 100)


def _load_manifest(db_path: str, table: str) -> Optional[dict]:
    manifest_path = Path(db_path) / f"{table}_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_encoder_config(args: argparse.Namespace, manifest: Optional[dict]) -> dict:
    manifest = manifest or {}
    model_id = getattr(args, "encoder_model_id", None) or manifest.get("model_id") or DEFAULT_MODEL_ID
    pooling = getattr(args, "encoder_pooling", None) or manifest.get("pooling") or "cls"
    max_length = getattr(args, "encoder_max_length", None)
    if max_length is None:
        max_length = manifest.get("max_length") or 128
    trust_remote_code = getattr(args, "encoder_trust_remote_code", None)
    if trust_remote_code is None:
        trust_remote_code = bool(manifest.get("trust_remote_code", False))
    return {
        "model_id": str(model_id),
        "pooling": str(pooling),
        "max_length": int(max_length),
        "trust_remote_code": bool(trust_remote_code),
    }


def _load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def _prepare_queries(df: pd.DataFrame, name_col: str, code_col: str | None) -> List[str]:
    queries: List[str] = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "") or "").strip()
        if not name:
            queries.append("")
            continue
        if code_col:
            code_val = row.get(code_col, None)
            code = str(code_val).strip() if code_val is not None and pd.notna(code_val) else ""
            if code:
                queries.append(f"{name} ({code})")
                continue
        queries.append(name)
    return queries


def _compute_metrics(
    rows: Iterable[Mapping[str, object]],
    topk: int,
) -> Dict[str, float]:
    rows_list = list(rows)
    hits = {k: 0 for k in EVAL_K}
    mrr = 0.0
    labeled = 0
    coverage = 0

    for row in rows_list:
        candidates = row.get("candidates")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            coverage += 1
        gold = row.get("gold_concept_id")
        if gold is None or (isinstance(gold, float) and pd.isna(gold)):
            continue
        if isinstance(gold, str) and not gold.strip():
            continue
        labeled += 1
        try:
            gold_int = int(gold)  # type: ignore[arg-type]
        except Exception:
            continue
        preds: List[int] = []
        if isinstance(candidates, pd.DataFrame):
            preds = [
                int(v)
                for v in candidates["concept_id"].head(topk).tolist()
                if pd.notna(v)
            ]
        rank = None
        for idx, cid in enumerate(preds, start=1):
            if cid == gold_int:
                rank = idx
                break
        for k in EVAL_K:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None and rank <= 100:
            mrr += 1.0 / rank

    metrics: Dict[str, float] = {
        "n_rows": float(len(rows_list)),
        "coverage": coverage / max(len(rows_list), 1),
    }
    if labeled:
        for k in EVAL_K:
            metrics[f"hit@{k}"] = hits[k] / labeled
        metrics["mrr@100"] = mrr / labeled
        metrics["n_labeled"] = labeled
    else:
        metrics["n_labeled"] = 0.0
    return metrics


def _build_rag_pipeline(
    args: argparse.Namespace,
) -> tuple[Optional[RAGPipeline], int, Optional[List[str]]]:
    provider = getattr(args, "rag_provider", None)
    if not provider:
        return None, 0, None
    provider = provider.lower()
    candidate_limit = int(getattr(args, "rag_candidate_limit", 0))
    if candidate_limit <= 0:
        raise ValueError("rag-candidate-limit must be > 0 when rag-provider is set.")
    candidate_limit = min(candidate_limit, int(getattr(args, "candidate_topk", candidate_limit)))
    if candidate_limit <= 0:
        raise ValueError("rag-candidate-limit must not exceed candidate-topk.")
    if getattr(args, "rag_template", None):
        prompt_builder = TemplatePromptBuilder(
            args.rag_template,
            candidate_field=args.rag_template_field,
            profile_char_limit=args.rag_profile_char_limit,
        )
    else:
        prompt_builder = RAGPromptBuilder(
            profile_char_limit=args.rag_profile_char_limit,
            include_retrieval_score=args.rag_include_retrieval_score,
            include_final_score=args.rag_include_final_score,
        )
    provider = provider.strip()
    default_model = args.rag_model or "openai/gpt-oss-20b"
    if provider == "cloudflare":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        if not account_id or not api_token:
            raise ValueError(
                "cloudflare provider requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables."
            )
        model_name = default_model
        if not model_name.startswith("@cf/"):
            model_name = f"@cf/{model_name}"
        use_responses_api = args.cloudflare_use_responses_api
        lowered = model_name.lower()
        if lowered.startswith("@cf/meta/") and use_responses_api:
            print("[info] Forcing Cloudflare run endpoint for meta models (disabling Responses API).")
            use_responses_api = False
        cfg = CloudflareConfig(
            account_id=account_id,
            api_token=api_token,
            model_name=model_name,
            base_url=args.cloudflare_base_url,
            use_responses_api=use_responses_api,
            reasoning_effort=args.gpt_reasoning_effort,
            reasoning_summary=args.cf_reasoning_summary,
        )
        llm_client = CloudflareLLMClient(cfg)
    elif provider == "ollama":
        model_name = args.ollama_model or default_model
        cfg = OllamaConfig(
            base_url=args.ollama_base_url,
            model=model_name,
            timeout=args.ollama_timeout,
            keep_alive=args.ollama_keep_alive,
        )
        llm_client = OllamaLLMClient(cfg)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("openrouter provider requires OPENROUTER_API_KEY to be set in the environment.")
        model_name = args.rag_model or "openrouter/polaris-alpha"
        cfg = OpenRouterConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=args.openrouter_base_url,
        )
        llm_client = OpenRouterLLMClient(cfg)
    elif provider == "llamacpp":
        base_url = getattr(args, "llamacpp_base_url", None)
        if base_url:
            cfg = LlamaCppServerConfig(
                base_url=base_url,
                timeout=args.llamacpp_timeout,
                model_name=default_model,
            )
            llm_client = LlamaCppServerLLMClient(cfg)
        else:
            model_path = args.llamacpp_model_path
            if not model_path:
                raise ValueError(
                    "llamacpp provider requires either --llamacpp-base-url (llama-server) or --llamacpp-model-path pointing to a GGUF file."
                )
            cfg = LlamaCppConfig(
                model_path=model_path,
                n_ctx=args.llamacpp_n_ctx,
                n_gpu_layers=args.llamacpp_n_gpu_layers,
                n_threads=args.llamacpp_n_threads,
                chat_format=args.llamacpp_chat_format,
                system_prompt=args.llamacpp_system_prompt,
            )
            llm_client = LlamaCppLLMClient(cfg)
    else:
        raise ValueError(f"Unsupported rag provider: {args.rag_provider}")
    pipeline = RAGPipeline(llm_client, prompt_builder=prompt_builder)
    raw_stops = list(getattr(args, "rag_stop_sequence", []) or [])
    stop_sequences = raw_stops or None
    return pipeline, candidate_limit, stop_sequences


def _apply_brand_penalty(
    query_text: str,
    df_candidates: pd.DataFrame,
    ordered_concept_ids: List[int],
) -> List[int]:
    return ordered_concept_ids


def _apply_atc_scope(df_candidates: pd.DataFrame, allowlist: set[int]) -> pd.DataFrame:
    """Stable-rerank candidates by whether they match the ATC allowlist."""

    if df_candidates is None or df_candidates.empty or not allowlist:
        return df_candidates
    df = df_candidates.copy()
    df["atc_match"] = df["concept_id"].apply(safe_int).isin(allowlist)
    return df.sort_values("atc_match", ascending=False, kind="mergesort").reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    table, vector_column = connect_table(args.db, args.table)
    manifest = _load_manifest(args.db, args.table)
    df = _load_input(Path(args.input))
    input_is_usagi = False
    if not df.empty and is_usagi_format(df.columns):
        try:
            input_is_usagi = validate_usagi_frame(df)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Input file failed Usagi validation: {exc}") from exc
    # Optional row limit for smoke/testing
    if getattr(args, "n_limit", 0):
        try:
            n = int(args.n_limit)
        except Exception:
            n = 0
        if n > 0:
            df = df.head(n)
    if df.empty:
        raise SystemExit("Input file contained no rows")

    queries = _prepare_queries(df, args.source_name_column, args.source_code_column)
    if getattr(args, "strip_non_latin", False) or getattr(args, "strip_chars", ""):
        queries_sanitized = [
            sanitize_query_text(q, strip_non_latin=bool(args.strip_non_latin), strip_chars=str(args.strip_chars or ""))
            for q in queries
        ]
    else:
        queries_sanitized = queries
    if getattr(args, "inn2usan", False):
        mapping = None
        if getattr(args, "inn2usan_extra", False):
            mapping = {**DEFAULT_INN_TO_USAN, **MAPPER_EXTRA_INN_TO_USAN}
        queries_sanitized = [convert_inn_ban_to_usan(q, mapping=mapping) for q in queries_sanitized]
    queries_sanitized = [normalize_strength_spacing(q) for q in queries_sanitized]
    queries_norm = [normalize_text_value(q) for q in queries_sanitized]

    device = resolve_device(args.device)
    configure_torch_for_infer(device)
    encoder_cfg = _resolve_encoder_config(args, manifest)
    embedder = SapBERTEmbedder(
        model_id=encoder_cfg["model_id"],
        device=device,
        batch_size=args.batch_size,
        max_length=encoder_cfg["max_length"],
        pooling=encoder_cfg["pooling"],
        trust_remote_code=encoder_cfg["trust_remote_code"],
    )
    vectors = embedder.encode(queries_norm)

    reranker_id = args.reranker_id or DEFAULT_RERANKER_ID
    reranker = ThirawatReranker(model_id=reranker_id, device=device, return_score="all")

    rag_pipeline: Optional[RAGPipeline] = None
    rag_candidate_limit = 0
    rag_stop_sequences: Optional[List[str]] = None
    if getattr(args, "rag_provider", None):
        try:
            rag_pipeline, rag_candidate_limit, rag_stop_sequences = _build_rag_pipeline(args)
        except Exception as exc:
            raise SystemExit(f"Failed to initialize RAG provider '{args.rag_provider}': {exc}") from exc
    rag_use_normalized_query = bool(getattr(args, "rag_use_normalized_query", True))
    rag_extra_context_column = getattr(args, "rag_extra_context_column", None)
    results: List[Dict[str, object]] = []
    rag_prompt_logs: List[Dict[str, object]] = []
    error_logs: List[Dict[str, object]] = []

    status_series = df[args.status_column] if args.status_column in df.columns else None
    duckdb_path: Optional[str] = None
    concepts_table: Optional[str] = None
    if manifest:
        duckdb_path = manifest.get("duckdb")
        concepts_table = manifest.get("concepts_table")

    exclude_concept_class_ids = to_exclusion_set(getattr(args, "exclude_concept_class_id", None))
    concept_class_resolver: Optional[ConceptClassResolver] = None
    if exclude_concept_class_ids:
        if duckdb_path and concepts_table:
            concept_class_resolver = ConceptClassResolver(Path(duckdb_path), str(concepts_table))
        else:
            print("[warn] Concept class exclusions unavailable (missing duckdb manifest info).")

    allowlist_by_row: dict[int, set[int]] = {}
    if getattr(args, "atc_scope", False):
        vocab_path = getattr(args, "vocab", None) or duckdb_path
        if not vocab_path:
            raise SystemExit("--atc-scope requires --vocab or an index manifest with a duckdb path.")
        if "atc_ids" not in df.columns and "atc_codes" not in df.columns:
            print("[warn] ATC scoping requested but input has no atc_ids/atc_codes columns; no ATC filters applied.")
        else:
            resolver = AtcScopeResolver(Path(str(vocab_path)))
            allowlist_by_row = resolver.build_allowlist(df, allowlist_max_ids=int(getattr(args, "allowlist_max_ids", 1000)))

    failures = 0
    for idx in tqdm(range(len(queries)), desc="Infer", unit="q"):
        query_text = queries[idx]
        query_text_norm = queries_norm[idx]
        row = df.iloc[idx]
        query_vec = vectors[idx]
        source_name_value = row.get(args.source_name_column, None)
        source_code_value = None
        if args.source_code_column:
            code_raw = row.get(args.source_code_column, None)
            if code_raw is not None and not (isinstance(code_raw, float) and pd.isna(code_raw)):
                code_text = str(code_raw).strip()
                if code_text:
                    source_code_value = code_text

        input_row = row.to_dict()
        usagi_row = coerce_usagi_row(
            input_row,
            row_index=idx,
            source_name=source_name_value,
            source_code=source_code_value,
            source_code_field=args.source_code_column,
        )

        builder = table.search(
            query_vec.astype(float).tolist(),
            vector_column_name=vector_column,
            query_type="vector",
        )
        if args.where:
            try:
                builder = builder.where(args.where)
            except Exception as _:
                pass
        try:
            retrieval_topk = max(args.candidate_topk, args.retrieval_topk)
            result_builder = builder.distance_type("cosine").limit(retrieval_topk)
            if not args.no_rerank:
                result_builder = result_builder.rerank(reranker=reranker, query_string=query_text_norm)
            arrow_table = result_builder.limit(args.candidate_topk).to_arrow()
            df_candidates = arrow_table.to_pandas()
        except Exception as exc:
            failures += 1
            print(f"[warn] search/rerank failed for row {idx}: {exc}")
            error_logs.append(
                {
                    "type": "search",
                    "index": idx,
                    "source_name": source_name_value,
                    "source_code": source_code_value,
                    "message": str(exc),
                }
            )
            df_candidates = pd.DataFrame(columns=["concept_id", "profile_text", "concept_name"])

        keep_cols = [
            col
            for col in [
                "concept_id",
                "concept_name",
                "domain_id",
                "concept_class_id",
                "vocabulary_id",
                "profile_text",
                "_relevance_score",
            ]
            if col in df_candidates.columns
        ]
        df_candidates = df_candidates.loc[:, keep_cols] if keep_cols else df_candidates

        if exclude_concept_class_ids:
            if "concept_class_id" not in df_candidates.columns and concept_class_resolver is not None:
                concept_ids_list = [cid for cid in df_candidates["concept_id"].apply(safe_int).tolist() if cid is not None]
                if concept_ids_list:
                    mapping = concept_class_resolver.lookup(concept_ids_list)
                    df_candidates["concept_class_id"] = [
                        mapping.get(safe_int(cid)) for cid in df_candidates["concept_id"].tolist()
                    ]
            if "concept_class_id" in df_candidates.columns:
                df_candidates = df_candidates[
                    ~df_candidates["concept_class_id"].astype(str).str.strip().isin(exclude_concept_class_ids)
                ].reset_index(drop=True)

        if not df_candidates.empty:
            if should_apply_post(str(args.post_mode), float(args.post_weight)):
                df_candidates = enrich_with_post_scores(
                    df_candidates,
                    query_text_norm,
                    post_strength_weight=float(args.post_strength_weight),
                    post_jaccard_weight=float(args.post_jaccard_weight),
                    post_brand_penalty=float(args.post_brand_penalty),
                    post_minmax=bool(args.post_minmax),
                    post_weight=float(args.post_weight),
                    prefer_brand=True,
                    post_mode=str(args.post_mode),
                    tiebreak_eps=float(args.tiebreak_eps),
                    tiebreak_topn=int(args.tiebreak_topn),
                    brand_strict=bool(args.brand_strict),
                )
            else:
                base_col = "_relevance_score" if "_relevance_score" in df_candidates.columns else "score"
                if base_col in df_candidates.columns:
                    df_candidates = df_candidates.sort_values(base_col, ascending=False, kind="mergesort").reset_index(drop=True)
                    df_candidates["final_score"] = pd.to_numeric(df_candidates[base_col], errors="coerce").fillna(0.0)
                else:
                    df_candidates["final_score"] = 0.0
            allowlist = allowlist_by_row.get(idx)
            if allowlist:
                df_candidates = _apply_atc_scope(df_candidates, allowlist)
        else:
            df_candidates = pd.DataFrame(columns=["concept_id", "concept_name", "profile_text", "final_score"])

        rag_prompt: Optional[str] = None
        rag_response: Optional[str] = None
        if rag_pipeline and not df_candidates.empty and rag_candidate_limit > 0:
            subset_df = df_candidates.head(rag_candidate_limit).copy()
            if not subset_df.empty:
                try:
                    rag_candidates = to_candidates(subset_df.to_dict("records"))
                    extra_context = None
                    if rag_extra_context_column and rag_extra_context_column in df.columns:
                        extra_value = row.get(rag_extra_context_column)
                        if isinstance(extra_value, str):
                            extra_context = extra_value.strip() or None
                        elif pd.notna(extra_value):
                            extra_context = str(extra_value)
                    rag_query_text = query_text_norm if rag_use_normalized_query else query_text
                    rag_result = rag_pipeline.rerank(
                        rag_query_text,
                        rag_candidates,
                        extra_context=extra_context,
                        stop_sequences=rag_stop_sequences,
                    )
                    adjusted_ids = _apply_brand_penalty(query_text, subset_df, list(rag_result.concept_ids))
                    score_lookup = {cid: score for cid, score in zip(rag_result.concept_ids, rag_result.scores)}
                    order_map = {cid: idx for idx, cid in enumerate(adjusted_ids)}
                    base_order = pd.Series(range(len(df_candidates)), index=df_candidates.index, dtype=float) + float(rag_candidate_limit)
                    mapped_order = df_candidates["concept_id"].apply(safe_int).map(order_map)
                    df_candidates["__rag_order"] = base_order
                    mask = mapped_order.notna()
                    df_candidates.loc[mask, "__rag_order"] = mapped_order[mask]
                    df_candidates = (
                        df_candidates.sort_values("__rag_order", kind="mergesort")
                        .reset_index(drop=True)
                        .drop(columns="__rag_order")
                    )
                    rank_map = {cid: idx + 1 for idx, cid in enumerate(adjusted_ids)}
                    score_map = score_lookup
                    concept_ids_norm = df_candidates["concept_id"].apply(safe_int)
                    df_candidates["rag_rank"] = concept_ids_norm.map(rank_map)
                    df_candidates["rag_score"] = concept_ids_norm.map(score_map)
                    rag_prompt = rag_result.prompt
                    rag_response = rag_result.response
                    rag_prompt_logs.append(
                        {
                            "index": idx,
                            "source_name": query_text,
                            "source_code": source_code_value,
                            "prompt": rag_prompt,
                            "response": rag_response,
                        }
                    )
                except Exception as exc:
                    print(f"[warn] RAG rerank failed for row {idx}: {exc}")
                    error_logs.append(
                        {
                            "type": "rag",
                            "index": idx,
                            "source_name": query_text,
                            "source_code": source_code_value,
                            "message": str(exc),
                        }
                    )

        gold = None
        if args.label_column and args.label_column in df.columns:
            value = row.get(args.label_column)
            if pd.notna(value) and str(value).strip():
                gold = value
        if gold is None and status_series is not None and args.status_column in df.columns:
            status_value = str(row.get(args.status_column, "") or "").strip()
            if status_value.upper() == args.approved_value.upper():
                fallback_col = args.label_column if args.label_column in df.columns else "conceptId"
                gold = row.get(fallback_col)

        record: Dict[str, object] = {
            "source_name": source_name_value,
            "source_code": source_code_value,
            "candidates": df_candidates.head(args.candidate_topk),
            "input_row": input_row,
            "usagi_row": usagi_row,
        }
        if gold is not None and gold != "":
            record["gold_concept_id"] = gold
        if rag_prompt:
            record["rag_prompt"] = rag_prompt
        if rag_response:
            record["rag_response"] = rag_response
        if rag_pipeline:
            record["rag_provider"] = args.rag_provider
            if getattr(args, "rag_model", None):
                record["rag_model"] = args.rag_model

        results.append(record)

    metrics = _compute_metrics(results, args.candidate_topk)
    if failures and metrics.get("coverage", 0.0) == 0.0:
        print(f"[error] All {len(queries)} queries failed during search/rerank; no outputs produced.")

    if rag_prompt_logs:
        prompts_path = Path(args.out) / "rag_prompts.md"
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with prompts_path.open("a", encoding="utf-8") as fh:
            for log in rag_prompt_logs:
                fh.write(f"## Query {log['index'] + 1}: {log['source_name']}\n\n")
                if log.get("source_code"):
                    fh.write(f"- Source code: {log['source_code']}\n\n")
                fh.write("### Prompt\n\n")
                fh.write("```text\n")
                fh.write(log.get("prompt", ""))
                fh.write("\n```\n\n")
                if log.get("response"):
                    fh.write("### Response\n\n")
                    fh.write("```text\n")
                    fh.write(log.get("response", ""))
                    fh.write("\n```\n\n")
            fh.write("\n")

    if error_logs:
        errors_path = Path(args.out) / "errors.log"
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "type", "source_name", "source_code", "message"])
            for err in error_logs:
                writer.writerow(
                    [
                        err.get("index"),
                        err.get("type", "error"),
                        err.get("source_name") or "",
                        err.get("source_code") or "",
                        err.get("message") or "",
                    ]
                )

    export_relabel_csv(
        results,
        args.out,
        topk=args.candidate_topk,
        metrics=metrics,
        preserve_input_order=not input_is_usagi,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bulk terminology inference")
    parser.add_argument("--db", required=True, help="Path to LanceDB directory")
    parser.add_argument("--table", required=True, help="LanceDB table name")
    parser.add_argument("--input", required=True, help="Input CSV/TSV/Parquet file")
    parser.add_argument("--out", required=True, help="Directory for outputs")
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit to the first N rows from input (0 = all)",
    )
    parser.add_argument("--candidate-topk", type=int, default=DEFAULT_TOPK, help="Candidates before rerank")
    parser.add_argument(
        "--retrieval-topk",
        type=int,
        default=400,
        help="Number of vector candidates fetched before rerank (>= candidate-topk).",
    )
    parser.add_argument(
        "--no-rerank",
        action=argparse.BooleanOptionalAction,
        help="Skip reranking and rely on vector similarity only.",
    )
    parser.add_argument("--device", default="cpu", help="Device: auto|cuda|mps|cpu (default: cpu for stability)")
    parser.add_argument(
        "--reranker-id",
        default=None,
        help="Model identifier or path for reranker (default: na399/THIRAWAT-reranker-beta; supports local directories)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument(
        "--encoder-model-id",
        default=None,
        help="Encoder model id for query embeddings (default: from index manifest or SapBERT).",
    )
    parser.add_argument(
        "--encoder-pooling",
        choices=["cls", "mean"],
        default=None,
        help="Encoder pooling for query embeddings (default: from index manifest).",
    )
    parser.add_argument(
        "--encoder-max-length",
        type=int,
        default=None,
        help="Encoder max token length for query embeddings (default: from index manifest).",
    )
    parser.add_argument(
        "--encoder-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="trust_remote_code for encoder model loading (default: from index manifest).",
    )
    parser.add_argument("--where", default=None, help="Optional LanceDB filter expression, e.g. vocabulary_id = 'RxNorm'")
    parser.add_argument(
        "--atc-scope",
        action="store_true",
        help="Boost candidates that match provided ATC ids/codes (requires atc_ids/atc_codes columns in input + a DuckDB vocab).",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="DuckDB vocabulary file (used for ATC scoping; defaults to index manifest duckdb when available).",
    )
    parser.add_argument(
        "--allowlist-max-ids",
        type=int,
        default=1000,
        help="Safety cap for per-row ATC allowlists (skip scoping if exceeded).",
    )
    # Post-scorer controls
    parser.add_argument("--post-weight", type=float, default=0.05, help="Weight for simple post-score in final blend (0.0 = ML only)")
    parser.add_argument(
        "--post-mode",
        choices=["blend", "tiebreak", "lex"],
        default="tiebreak",
        help="Post-score behavior: blend, tiebreak within near-ties, or lexicographic sort",
    )
    parser.add_argument("--tiebreak-eps", type=float, default=0.01, help="Score gap threshold for tie groups")
    parser.add_argument("--tiebreak-topn", type=int, default=50, help="Only tiebreak within the top-N candidates")
    parser.add_argument("--brand-strict", action="store_true", help="Drop brand-mismatched candidates for bracketed brand queries when possible")
    parser.add_argument("--post-strength-weight", type=float, default=0.6, help="Weight for strength feature within simple score")
    parser.add_argument("--post-jaccard-weight", type=float, default=0.4, help="Weight for jaccard feature within simple score")
    parser.add_argument(
        "--post-brand-penalty",
        type=float,
        default=0.3,
        help="Penalty weight applied when candidate brand conflicts with query text",
    )
    parser.add_argument("--post-minmax", action=argparse.BooleanOptionalAction, default=True, help="Enable per-query min-max normalization of simple features (default: enabled)")
    parser.add_argument("--source-name-column", default="sourceName", help="Column containing source names")
    parser.add_argument("--source-code-column", default="sourceCode", help="Column containing source codes")
    parser.add_argument(
        "--strip-non-latin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove non‑Latin characters from query before retrieval/rerank",
    )
    parser.add_argument(
        "--strip-chars",
        default="",
        help="Characters to remove from query before retrieval/rerank (e.g., '()[]{}').",
    )
    parser.add_argument(
        "--inn2usan",
        "--convert-inn-to-usan",
        dest="inn2usan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize INN/BAN drug names to USAN before retrieval (default: enabled).",
    )
    parser.add_argument(
        "--inn2usan-extra",
        action="store_true",
        help="Enable mapper-only extra INN→USAN aliases in addition to trainer parity mapping (default: off).",
    )
    parser.add_argument(
        "--label-column",
        default="conceptId",
        help="Column with gold concept IDs (optional)",
    )
    parser.add_argument(
        "--status-column",
        default="mappingStatus",
        help="Column containing Usagi mapping status (optional)",
    )
    parser.add_argument(
        "--approved-value",
        default="APPROVED",
        help="Value in status column treated as gold label",
    )
    parser.add_argument(
        "--exclude-concept-class-id",
        action="append",
        default=[],
        help="Concept class IDs to exclude from candidate results (comma-separated or repeat flag).",
    )
    rag_group = parser.add_argument_group("LLM RAG")
    rag_group.add_argument(
        "--rag-provider",
        choices=["cloudflare", "ollama", "openrouter", "llamacpp"],
        help="Enable LLM reranking with the specified provider.",
    )
    rag_group.add_argument(
        "--rag-model",
        default="openai/gpt-oss-20b",
        help="Model identifier or path for the selected provider (default: openai/gpt-oss-20b).",
    )
    rag_group.add_argument(
        "--rag-candidate-limit",
        type=int,
        default=50,
        help="Number of top candidates to send to the LLM (<= candidate-topk).",
    )
    rag_group.add_argument(
        "--rag-profile-char-limit",
        type=int,
        default=512,
        help="Character limit for candidate profile text in prompts.",
    )
    rag_group.add_argument(
        "--rag-include-retrieval-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include retrieval scores in the prompt (default: enabled).",
    )
    rag_group.add_argument(
        "--rag-include-final-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include blended final scores in the prompt (default: enabled).",
    )
    rag_group.add_argument(
        "--rag-template",
        default=None,
        help="Optional markdown template path for prompt rendering.",
    )
    rag_group.add_argument(
        "--rag-template-field",
        default="profile_text",
        help="Candidate field to use when rendering a template prompt.",
    )
    rag_group.add_argument(
        "--rag-extra-context-column",
        default=None,
        help="Optional column name to pass as additional context to the LLM.",
    )
    rag_group.add_argument(
        "--rag-stop-sequence",
        action="append",
        default=[],
        help="Stop sequence passed to the LLM (can be provided multiple times).",
    )
    rag_group.add_argument(
        "--rag-use-normalized-query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use normalized query text when constructing LLM prompts (default: enabled).",
    )

    cf_group = parser.add_argument_group("Cloudflare provider")
    cf_group.add_argument(
        "--cloudflare-base-url",
        default="https://api.cloudflare.com/client/v4",
        help="Base URL for Cloudflare Workers AI (default: https://api.cloudflare.com/client/v4).",
    )
    cf_group.add_argument(
        "--cloudflare-use-responses-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Workers AI Responses API (default: enabled).",
    )
    cf_group.add_argument(
        "--gpt-reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional reasoning effort hint for GPT-OSS models.",
    )
    cf_group.add_argument(
        "--cf-reasoning-summary",
        choices=["auto", "concise", "detailed"],
        default=None,
        help="Optional reasoning summary style for Cloudflare Responses API.",
    )

    ollama_group = parser.add_argument_group("Ollama provider")
    ollama_group.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama-compatible server (default: http://localhost:11434).",
    )
    ollama_group.add_argument(
        "--ollama-model",
        default=None,
        help="Model tag registered with the Ollama server (fallback for --rag-model).",
    )
    ollama_group.add_argument(
        "--ollama-timeout",
        type=int,
        default=240,
        help="Timeout for Ollama requests in seconds (default: 240).",
    )
    ollama_group.add_argument(
        "--ollama-keep-alive",
        default=None,
        help="Optional keep_alive value for the Ollama server (e.g., 5m).",
    )

    openrouter_group = parser.add_argument_group("OpenRouter provider")
    openrouter_group.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for the OpenRouter API (default: https://openrouter.ai/api/v1).",
    )

    llama_group = parser.add_argument_group("llamacpp provider")
    llama_group.add_argument(
        "--llamacpp-model-path",
        default=None,
        help="Path to a GGUF model file for llama-cpp-python (local bindings).",
    )
    llama_group.add_argument(
        "--llamacpp-base-url",
        default=None,
        help="Existing llama-server base URL (e.g., http://127.0.0.1:8080). Overrides --llamacpp-model-path when set.",
    )
    llama_group.add_argument(
        "--llamacpp-timeout",
        type=int,
        default=240,
        help="Timeout in seconds for llama-server HTTP requests (default: 240).",
    )
    llama_group.add_argument(
        "--llamacpp-n-ctx",
        type=int,
        default=8192,
        help="Context window for llama.cpp inference (default: 8192).",
    )
    llama_group.add_argument(
        "--llamacpp-n-gpu-layers",
        type=int,
        default=-1,
        help="Number of GPU layers for llama.cpp (default: -1, auto).",
    )
    llama_group.add_argument(
        "--llamacpp-n-threads",
        type=int,
        default=None,
        help="Number of CPU threads for llama.cpp (default: auto).",
    )
    llama_group.add_argument(
        "--llamacpp-chat-format",
        default=None,
        help="Optional chat template name for llama.cpp models.",
    )
    llama_group.add_argument(
        "--llamacpp-system-prompt",
        default=None,
        help="Optional system prompt prepended for llama.cpp completion.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
