from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.providers import ProviderBundle
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.runtime.artifacts import register_workspace_artifact
from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
from agentic_chatbot_next.sandbox.exceptions import SandboxUnavailableError
from agentic_chatbot_next.tools.calculator import calculator
from agentic_chatbot_next.tools.data_analyst_nlp import DataAnalystNlpRunner
from agentic_chatbot_next.tools.skills_search_tool import make_skills_search_tool

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
_WRITABLE_EXTENSIONS = {".xlsx", ".csv"}
_SUPPORTED_NLP_TASKS = {"sentiment", "categorize", "keywords", "summarize"}
_NLP_TASK_ALIASES = {
    "sentiment": "sentiment",
    "sentiment_analysis": "sentiment",
    "sentiment_labeling": "sentiment",
    "sentiment_labelling": "sentiment",
    "classify": "categorize",
    "classification": "categorize",
    "categorization": "categorize",
    "category": "categorize",
    "categorize": "categorize",
    "keyword": "keywords",
    "keywords": "keywords",
    "keyword_extraction": "keywords",
    "keyphrase": "keywords",
    "keyphrases": "keywords",
    "summary": "summarize",
    "summaries": "summarize",
    "summarize": "summarize",
    "summarization": "summarize",
}


@dataclass
class LoadedDataset:
    path: Path
    resolved_ref: str
    ext: str
    dataframe: Any
    sheet_name: str = ""
    sheet_names: List[str] | None = None


def make_data_analyst_tools(
    stores: KnowledgeStores,
    session: Any,
    *,
    settings: Settings,
    providers: ProviderBundle | None = None,
) -> List[Any]:
    nlp_runner = DataAnalystNlpRunner(settings, providers)

    def _first_workspace_dataset_name() -> str:
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return ""
        for filename in workspace.list_files():
            if Path(filename).suffix.lower() in _SUPPORTED_EXTENSIONS:
                return filename
        return ""

    def _first_loaded_dataset_ref() -> str:
        for key in sorted(session.scratchpad.keys()):
            if key.startswith("dataset_") and not key.endswith("_ext"):
                return key[len("dataset_") :]
        return ""

    def _slugify(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_") or "analysis"

    def _safe_json(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def _resolve_dataset_path(dataset_ref: str) -> tuple[Path | None, str]:
        doc = stores.doc_store.get_document(dataset_ref, tenant_id=session.tenant_id)
        if doc is not None:
            source_path = getattr(doc, "source_path", None) or getattr(doc, "file_path", None)
            if not source_path:
                source_uri = getattr(doc, "source_uri", "") or ""
                if source_uri.startswith("file://"):
                    source_path = source_uri[7:]
            if source_path:
                candidate = Path(str(source_path))
                if candidate.exists():
                    return candidate, dataset_ref

        workspace = getattr(session, "workspace", None)
        if workspace is not None:
            candidate_name = Path(str(dataset_ref)).name
            if workspace.exists(candidate_name):
                return workspace.root / candidate_name, candidate_name

        return None, dataset_ref

    def _ensure_workspace_copy(path: Path) -> None:
        workspace = getattr(session, "workspace", None)
        if workspace is None or path.parent == workspace.root:
            return
        try:
            workspace.copy_file(path)
        except Exception as exc:
            logger.warning("Could not copy %s into workspace: %s", path.name, exc)

    def _load_dataframe(path: Path, *, sheet_name: str = "") -> LoadedDataset:
        import pandas as pd

        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
            return LoadedDataset(path=path, resolved_ref=path.name, ext=ext, dataframe=df, sheet_name="", sheet_names=[])

        excel_file = pd.ExcelFile(path)
        sheet_names = [str(name) for name in excel_file.sheet_names]
        if not sheet_names:
            raise ValueError(f"Workbook {path.name!r} does not contain any sheets.")
        active_sheet = str(sheet_name or "").strip() or sheet_names[0]
        if active_sheet not in sheet_names:
            raise ValueError(f"Sheet '{active_sheet}' not found. Available sheets: {sheet_names}")
        df = pd.read_excel(excel_file, sheet_name=active_sheet)
        return LoadedDataset(
            path=path,
            resolved_ref=path.name,
            ext=ext,
            dataframe=df,
            sheet_name=active_sheet,
            sheet_names=sheet_names,
        )

    def _load_dataset_handle(doc_id: str = "", *, sheet_name: str = "") -> LoadedDataset:
        dataset_ref = str(doc_id or "").strip() or _first_loaded_dataset_ref() or _first_workspace_dataset_name()
        if not dataset_ref:
            raise ValueError("No dataset reference was provided and no uploaded workspace dataset is available.")
        path, resolved_ref = _resolve_dataset_path(dataset_ref)
        if path is None:
            raise FileNotFoundError(f"Dataset '{dataset_ref}' not found in the knowledge base or session workspace.")
        if not path.exists():
            raise FileNotFoundError(f"File not found on disk: {path}")
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'.")
        loaded = _load_dataframe(path, sheet_name=sheet_name)
        loaded.resolved_ref = resolved_ref
        session.scratchpad[f"dataset_{resolved_ref}"] = str(path)
        session.scratchpad[f"dataset_{resolved_ref}_ext"] = ext
        _ensure_workspace_copy(path)
        return loaded

    def _derive_output_destination(
        source_path: Path,
        *,
        task_slug: str,
        target_filename: str = "",
        needs_workbook: bool = False,
    ) -> tuple[Path, str]:
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            raise ValueError("No session workspace is available.")
        requested_name = Path(str(target_filename or "").strip()).name if str(target_filename or "").strip() else ""
        source_ext = source_path.suffix.lower()
        if requested_name:
            output_ext = Path(requested_name).suffix.lower()
            if output_ext not in _WRITABLE_EXTENSIONS:
                raise ValueError("Writable output files must end in .csv or .xlsx")
            if source_ext in {".xlsx", ".xls"} and output_ext != ".xlsx":
                raise ValueError("Excel sources must be written back as .xlsx files.")
            filename = requested_name
        else:
            if source_ext == ".csv" and not needs_workbook:
                output_ext = ".csv"
            else:
                output_ext = ".xlsx"
            filename = f"{source_path.stem}__analyst_{task_slug}{output_ext}"
        if filename == source_path.name:
            suffix = Path(filename).suffix
            filename = f"{Path(filename).stem}__copy{suffix}"
        return workspace.root / filename, filename

    def _write_mutated_output(
        source: LoadedDataset,
        *,
        dataframe: Any,
        target_filename: str = "",
        task_slug: str,
    ) -> str:
        import pandas as pd

        dest_path, dest_name = _derive_output_destination(
            source.path,
            task_slug=task_slug,
            target_filename=target_filename,
        )
        dest_ext = dest_path.suffix.lower()
        if source.ext == ".csv" and dest_ext == ".csv":
            dataframe.to_csv(dest_path, index=False)
            return dest_name

        if source.ext in {".xlsx", ".xls"}:
            if source.ext == ".xlsx" and dest_ext == ".xlsx":
                shutil.copy2(source.path, dest_path)
                with pd.ExcelWriter(dest_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    dataframe.to_excel(writer, sheet_name=source.sheet_name or "Sheet1", index=False)
                return dest_name

            excel_file = pd.ExcelFile(source.path)
            with pd.ExcelWriter(dest_path, engine="openpyxl") as writer:
                for name in excel_file.sheet_names:
                    frame = dataframe if name == (source.sheet_name or excel_file.sheet_names[0]) else pd.read_excel(excel_file, sheet_name=name)
                    frame.to_excel(writer, sheet_name=str(name), index=False)
            return dest_name

        with pd.ExcelWriter(dest_path, engine="openpyxl") as writer:
            dataframe.to_excel(writer, sheet_name="SourceData", index=False)
        return dest_name

    def _summarize_label_results(task: str, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if task not in {"sentiment", "categorize"}:
            return counts
        for row in rows:
            label = str(row.get("label") or "").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _normalize_nlp_task_name(task_name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(task_name or "sentiment").strip().lower()).strip("_")
        return _NLP_TASK_ALIASES.get(normalized, normalized)

    def _build_nlp_summary_text(
        *,
        task: str,
        doc_id: str,
        column: str,
        processed_rows: int,
        blank_rows: int,
        result_counts: Dict[str, int],
        written_file: str = "",
    ) -> str:
        dataset_label = str(doc_id or "the dataset")
        if task in {"sentiment", "categorize"} and result_counts:
            ordered_counts = ", ".join(f"{label}: {count}" for label, count in sorted(result_counts.items()))
            summary = (
                f"Processed {processed_rows} rows from '{column}' in {dataset_label}. "
                f"{task.title()} counts: {ordered_counts}."
            )
        else:
            summary = f"Processed {processed_rows} rows from '{column}' in {dataset_label} for {task}."
        if blank_rows:
            summary += f" Skipped {blank_rows} blank rows."
        if written_file:
            summary += f" Wrote the derived output to {written_file}."
        return summary

    def _default_nlp_output_columns(
        task: str,
        *,
        label_column: str = "",
        score_column: str = "",
    ) -> tuple[List[str], str, str]:
        if task == "sentiment":
            label_name = label_column.strip() or "sentiment_label"
            score_name = score_column.strip() or "sentiment_score"
            return [label_name, score_name], label_name, score_name
        if task == "categorize":
            label_name = label_column.strip() or "category_label"
            score_name = score_column.strip()
            columns = [label_name]
            if score_name:
                columns.append(score_name)
            return columns, label_name, score_name
        if task == "keywords":
            label_name = label_column.strip() or "keywords"
            return [label_name], label_name, ""
        label_name = label_column.strip() or "row_summary"
        return [label_name], label_name, ""

    def _coerce_nlp_cell_value(task: str, row: Dict[str, Any]) -> Any:
        if task in {"sentiment", "categorize"}:
            return str(row.get("label") or "")
        if task == "keywords":
            keywords = [str(value).strip() for value in list(row.get("keywords") or []) if str(value).strip()]
            return ", ".join(keywords)
        return str(row.get("summary") or "")

    def _build_nlp_preview_rows(
        *,
        source_column: str,
        task: str,
        expanded_rows: List[Dict[str, Any]],
        derived_columns: List[str],
    ) -> List[Dict[str, Any]]:
        preview_rows: List[Dict[str, Any]] = []
        for row in expanded_rows[:5]:
            preview: Dict[str, Any] = {"row_index": row.get("row_index"), source_column: row.get("source_text", "")}
            if task in {"sentiment", "categorize"}:
                preview[derived_columns[0]] = str(row.get("label") or "")
                if len(derived_columns) > 1:
                    preview[derived_columns[1]] = row.get("score")
            elif task == "keywords":
                preview[derived_columns[0]] = ", ".join(
                    str(value).strip() for value in list(row.get("keywords") or []) if str(value).strip()
                )
            else:
                preview[derived_columns[0]] = str(row.get("summary") or "")
            preview_rows.append(preview)
        return preview_rows

    @tool
    def load_dataset(doc_id: str = "", sheet_name: str = "") -> str:
        """Load a dataset (Excel or CSV) from the knowledge base or session workspace."""
        try:
            loaded = _load_dataset_handle(doc_id, sheet_name=sheet_name)
            df_head = loaded.dataframe.head(5)
            head_records = [{str(key): _safe_json(value) for key, value in record.items()} for record in df_head.to_dict(orient="records")]
            nrows, ncols = loaded.dataframe.shape
            return json.dumps(
                {
                    "file_path": str(loaded.path),
                    "doc_id": loaded.resolved_ref,
                    "sheet_name": loaded.sheet_name,
                    "sheet_names": list(loaded.sheet_names or []),
                    "columns": list(loaded.dataframe.columns.astype(str)),
                    "shape": [nrows, ncols],
                    "dtypes": {str(col): str(dtype) for col, dtype in loaded.dataframe.dtypes.items()},
                    "head": head_records,
                    "info_summary": f"{nrows:,} rows x {ncols} columns",
                }
            )
        except Exception as exc:
            logger.warning("load_dataset failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "", sheet_name: str = "") -> str:
        """Get detailed statistics for specific columns in a loaded dataset."""
        try:
            dataset_ref = str(doc_id or "").strip() or _first_loaded_dataset_ref() or _first_workspace_dataset_name()
            if not dataset_ref:
                return json.dumps({"error": "No dataset reference was provided and no loaded dataset is available."})
            path_str = session.scratchpad.get(f"dataset_{dataset_ref}")
            ext = ""
            resolved_ref = dataset_ref
            if path_str:
                path = Path(path_str)
                ext = str(session.scratchpad.get(f"dataset_{dataset_ref}_ext", path.suffix.lower()) or path.suffix.lower())
                loaded = _load_dataframe(path, sheet_name=sheet_name)
                loaded.resolved_ref = resolved_ref
                loaded.ext = ext
            elif not str(doc_id or "").strip() and dataset_ref == _first_workspace_dataset_name():
                loaded = _load_dataset_handle(dataset_ref, sheet_name=sheet_name)
            else:
                return json.dumps({"error": f"Dataset '{dataset_ref}' not loaded. Call load_dataset first."})
            df = loaded.dataframe
            if columns.strip():
                col_list = [column.strip() for column in columns.split(",") if column.strip()]
                missing = [column for column in col_list if column not in df.columns]
                if missing:
                    return json.dumps({"error": f"Columns not found: {missing}. Available: {list(df.columns.astype(str))}"})
                df = df[col_list]

            import pandas as pd

            stats: dict = {}
            for col in df.columns:
                series = df[col]
                null_count = int(series.isna().sum())
                total_count = len(series)
                unique_count = int(series.nunique(dropna=True))
                col_key = str(col)
                if pd.api.types.is_numeric_dtype(series):
                    desc = series.describe()
                    stats[col_key] = {
                        "dtype": str(series.dtype),
                        "count": total_count,
                        "nulls": null_count,
                        "unique": unique_count,
                        "mean": _safe_float(desc.get("mean")),
                        "std": _safe_float(desc.get("std")),
                        "min": _safe_float(desc.get("min")),
                        "p25": _safe_float(desc.get("25%")),
                        "p50": _safe_float(desc.get("50%")),
                        "p75": _safe_float(desc.get("75%")),
                        "max": _safe_float(desc.get("max")),
                    }
                else:
                    top_values = series.value_counts(dropna=True).head(5).to_dict()
                    stats[col_key] = {
                        "dtype": str(series.dtype),
                        "count": total_count,
                        "nulls": null_count,
                        "unique": unique_count,
                        "top_values": {str(key): int(value) for key, value in top_values.items()},
                    }
            payload = dict(stats)
            payload["_meta"] = {
                "doc_id": loaded.resolved_ref,
                "sheet_name": loaded.sheet_name,
                "sheet_names": list(loaded.sheet_names or []),
            }
            return json.dumps(payload)
        except Exception as exc:
            logger.warning("inspect_columns failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute Python code in a secure Docker sandbox to analyze data."""
        try:
            executor = DockerSandboxExecutor(
                image=settings.sandbox_docker_image,
                timeout_seconds=settings.sandbox_timeout_seconds,
                memory_limit=settings.sandbox_memory_limit,
            )
            workspace = getattr(session, "workspace", None)
            if workspace is not None:
                for raw_id in (doc_ids or "").split(","):
                    doc_id = raw_id.strip()
                    if not doc_id:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{doc_id}")
                    if path_str:
                        source = Path(path_str)
                        if not workspace.exists(source.name):
                            try:
                                workspace.copy_file(source)
                            except Exception as exc:
                                logger.warning("Could not copy %s to workspace: %s", source.name, exc)
                result = executor.execute(code=code, workspace_path=workspace.root)
            else:
                files: dict = {}
                for raw_id in (doc_ids or "").split(","):
                    doc_id = raw_id.strip()
                    if not doc_id:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{doc_id}")
                    if path_str:
                        host_path = Path(path_str)
                        files[f"/workspace/{host_path.name}"] = str(host_path)
                result = executor.execute(code=code, files=files or None)

            return json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.success,
                    "execution_time_seconds": round(result.execution_time_seconds, 3),
                    "truncated": result.truncated,
                }
            )
        except SandboxUnavailableError as exc:
            return json.dumps({"error": f"Docker sandbox is not available: {exc}", "success": False, "stdout": "", "stderr": ""})
        except Exception as exc:
            logger.warning("execute_code unexpected error: %s", exc)
            return json.dumps({"error": str(exc), "success": False, "stdout": "", "stderr": ""})

    @tool
    def run_nlp_column_task(
        doc_id: str = "",
        sheet_name: str = "",
        column: str = "",
        task: str = "sentiment",
        classification_rules: str = "",
        allowed_labels_csv: str = "",
        batch_size: int = 5,
        output_mode: str = "summary_only",
        target_filename: str = "",
        label_column: str = "",
        score_column: str = "",
        dataset: str = "",
        dataset_name: str = "",
    ) -> str:
        """Run a bounded LLM-powered NLP task over a text column."""
        try:
            resolved_doc_id = str(doc_id or dataset or dataset_name).strip()
            clean_task = _normalize_nlp_task_name(task)
            if clean_task not in _SUPPORTED_NLP_TASKS:
                return json.dumps({"error": f"Unsupported NLP task '{task}'."})
            clean_output_mode = str(output_mode or "summary_only").strip().lower()
            if clean_output_mode not in {"summary_only", "append_columns"}:
                return json.dumps({"error": f"Unsupported output_mode '{output_mode}'."})

            loaded = _load_dataset_handle(resolved_doc_id, sheet_name=sheet_name)
            df = loaded.dataframe.copy()
            if column not in df.columns:
                return json.dumps({"error": f"Column '{column}' not found. Available columns: {list(df.columns.astype(str))}"})

            allowed_labels = [item.strip() for item in str(allowed_labels_csv or "").split(",") if item.strip()]
            if clean_task == "sentiment" and not allowed_labels:
                allowed_labels = ["positive", "neutral", "negative"]

            non_empty_items: List[Dict[str, Any]] = []
            dedupe_index: Dict[str, Dict[str, Any]] = {}
            blank_rows = 0
            for row_index, value in df[column].items():
                if value is None:
                    blank_rows += 1
                    continue
                text = str(value).strip()
                if not text or text.lower() == "nan":
                    blank_rows += 1
                    continue
                existing = dedupe_index.get(text)
                if existing is None:
                    item = {
                        "item_id": f"row_{len(dedupe_index) + 1}",
                        "text": text,
                        "row_indices": [row_index],
                    }
                    dedupe_index[text] = item
                    non_empty_items.append(item)
                else:
                    existing["row_indices"].append(row_index)

            if not non_empty_items:
                return json.dumps({"error": f"Column '{column}' does not contain any non-empty text values."})

            effective_batch_size = max(1, int(batch_size or getattr(settings, "data_analyst_nlp_batch_size", 5)))
            nlp_result = nlp_runner.run_task(
                task=clean_task,
                classification_rules=classification_rules,
                allowed_labels=allowed_labels,
                items=non_empty_items,
                batch_size=effective_batch_size,
            )
            if not nlp_result.rows:
                return json.dumps({"error": "NLP task did not produce any successful rows.", "failed_batches": nlp_result.failed_batches})

            dedup_rows = {row["item_id"]: row for row in nlp_result.rows}
            expanded_rows: List[Dict[str, Any]] = []
            for item in non_empty_items:
                result = dedup_rows.get(item["item_id"])
                if result is None:
                    continue
                for row_index in item["row_indices"]:
                    expanded = {
                        "row_index": _safe_json(row_index),
                        "source_text": item["text"],
                        **result,
                    }
                    expanded_rows.append(expanded)

            summary_counts = _summarize_label_results(clean_task, nlp_result.rows)
            derived_columns, label_name, score_name = _default_nlp_output_columns(
                clean_task,
                label_column=label_column,
                score_column=score_column,
            )
            written_file = ""
            payload: Dict[str, Any] = {
                "task": clean_task,
                "doc_id": loaded.resolved_ref,
                "sheet_name": loaded.sheet_name,
                "column": column,
                "output_mode": clean_output_mode,
                "batch_size": effective_batch_size,
                "blank_rows": blank_rows,
                "unique_text_inputs": len(non_empty_items),
                "processed_rows": len(expanded_rows),
                "failed_batches": nlp_result.failed_batches,
                "model_name": nlp_result.model_name,
                "result_counts": summary_counts,
                "derived_columns": list(derived_columns),
                "preview_columns": ["row_index", column, *derived_columns],
            }

            if clean_output_mode == "append_columns":
                df[label_name] = ""
                if score_name:
                    df[score_name] = ""
                for item in non_empty_items:
                    result = dedup_rows.get(item["item_id"])
                    if result is None:
                        continue
                    for raw_index in item["row_indices"]:
                        df.at[raw_index, label_name] = _coerce_nlp_cell_value(clean_task, result)
                        if score_name:
                            df.at[raw_index, score_name] = result.get("score")
                written_file = _write_mutated_output(
                    loaded,
                    dataframe=df,
                    target_filename=target_filename,
                    task_slug=_slugify(clean_task),
                )
                session.scratchpad["last_output_file"] = written_file
                payload.update(
                    {
                        "written_file": written_file,
                        "label_column": label_name,
                        "score_column": score_name,
                        "next_action": f"Call return_file with filename='{written_file}' to publish the file to the user.",
                    }
                )

            payload["preview_rows"] = _build_nlp_preview_rows(
                source_column=column,
                task=clean_task,
                expanded_rows=expanded_rows,
                derived_columns=derived_columns,
            )
            payload["preview"] = list(payload["preview_rows"])
            payload["summary_text"] = _build_nlp_summary_text(
                task=clean_task,
                doc_id=loaded.resolved_ref,
                column=column,
                processed_rows=len(expanded_rows),
                blank_rows=blank_rows,
                result_counts=summary_counts,
                written_file=written_file,
            )

            return json.dumps(payload)
        except Exception as exc:
            logger.warning("run_nlp_column_task failed: %s", exc)
            return json.dumps({"error": str(exc)})

    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """Save an intermediate finding or plan to the scratchpad."""
        session.scratchpad[key] = value
        return json.dumps({"saved": key, "length": len(value)})

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a previously saved value from the scratchpad."""
        if key in session.scratchpad:
            return json.dumps({"key": key, "value": session.scratchpad[key]})
        available = [item for item in session.scratchpad.keys() if not item.startswith("dataset_")]
        return json.dumps({"error": f"Key '{key}' not found", "available_keys": available})

    @tool
    def scratchpad_list() -> str:
        """List all user-written keys currently in the scratchpad."""
        user_keys = [item for item in session.scratchpad.keys() if not item.startswith("dataset_")]
        return json.dumps({"keys": user_keys, "count": len(user_keys)})

    @tool
    def workspace_write(filename: str, content: str) -> str:
        """Write a text file to the persistent session workspace."""
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            dest = workspace.write_text(filename, content)
            return json.dumps({"written": dest.name, "size_bytes": dest.stat().st_size})
        except Exception as exc:
            logger.warning("workspace_write failed for %s: %s", filename, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def workspace_read(filename: str) -> str:
        """Read a file from the persistent session workspace."""
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            return workspace.read_text(filename)
        except FileNotFoundError:
            return json.dumps({"error": f"File '{filename}' not found.", "available_files": workspace.list_files()})
        except Exception as exc:
            logger.warning("workspace_read failed for %s: %s", filename, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def workspace_list() -> str:
        """List all files currently in the persistent session workspace."""
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            files = [item for item in workspace.list_files() if not item.startswith(".")]
            return json.dumps({"files": files, "count": len(files)})
        except Exception as exc:
            logger.warning("workspace_list failed: %s", exc)
            return json.dumps({"error": str(exc)})

    @tool
    def return_file(filename: str = "", label: str = "") -> str:
        """Publish a workspace file so the end user can download it."""
        selected = str(filename or "").strip() or str(session.scratchpad.get("last_output_file") or "").strip()
        if not selected:
            return json.dumps({"error": "No filename was provided and no prior output file is recorded in the scratchpad."})
        try:
            artifact = register_workspace_artifact(session, filename=selected, label=label)
            return json.dumps(artifact)
        except Exception as exc:
            logger.warning("return_file failed for %s: %s", selected, exc)
            return json.dumps({"error": str(exc)})

    skills_search = None
    try:
        skills_search = make_skills_search_tool(settings, stores=stores, session=session)
    except Exception as exc:
        logger.warning("Could not build search_skills tool: %s", exc)

    tools: List[Any] = [
        load_dataset,
        inspect_columns,
        execute_code,
        run_nlp_column_task,
        return_file,
        calculator,
        scratchpad_write,
        scratchpad_read,
        scratchpad_list,
        workspace_write,
        workspace_read,
        workspace_list,
    ]
    if skills_search is not None:
        tools.append(skills_search)
    return tools


def _safe_float(value: Any) -> Optional[float]:
    try:
        import math

        coerced = float(value)
        if math.isnan(coerced) or math.isinf(coerced):
            return None
        return round(coerced, 6)
    except (TypeError, ValueError):
        return None
