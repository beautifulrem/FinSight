from __future__ import annotations

import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - dependency installed via requirements.txt
    snapshot_download = None

from .models import DatasetSource, SyncResult

MAX_HTTP_DOWNLOAD_BYTES = 5 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024
SUBPROCESS_TIMEOUT_SECONDS = 900

_SCRIPT_BACKED_DOWNLOADS: dict[str, tuple[str, ...]] = {
    "msra_ner": (
        "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/MSRA/msra_train_bio.txt",
        "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/MSRA/msra_test_bio.txt",
    ),
    "peoples_daily_ner": (
        "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People's%20Daily/example.train",
        "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People's%20Daily/example.dev",
        "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People's%20Daily/example.test",
    ),
    "fiqa": (
        "https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/main/train.tsv?download=true",
        "https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/main/dev.tsv?download=true",
        "https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/main/test.tsv?download=true",
    ),
}


def sync_source(source: DatasetSource, raw_root: Path) -> SyncResult:
    target_dir = raw_root / source.source_id / source.version
    if target_dir.exists() and any(target_dir.iterdir()):
        if source.source_type == "huggingface_dataset":
            _materialize_script_backed_dataset(source.source_id, target_dir)
            if source.source_id == "dailydialog":
                _extract_zip_files(target_dir)
        if source.source_id == "smp2017":
            _extract_zip_files(target_dir)
        if source.source_id == "thucnews":
            _extract_7z_files(target_dir)
        return SyncResult(source_id=source.source_id, status="cached", output_dir=str(target_dir))

    if source.source_type == "local_seed":
        target_dir.mkdir(parents=True, exist_ok=True)
        return SyncResult(source_id=source.source_id, status="skipped", output_dir=str(target_dir))

    if source.source_type == "huggingface_dataset":
        download_huggingface_dataset(source.entrypoint, target_dir, allow_patterns=_allow_patterns_for_source(source.source_id))
        _materialize_script_backed_dataset(source.source_id, target_dir)
        if source.source_id == "dailydialog":
            _extract_zip_files(target_dir)
    elif source.source_type in ("github_repo", "github_release"):
        branch = source.version if source.version != "default" else "main"
        download_github_repo(source.entrypoint, target_dir, branch=branch)
        if source.source_id == "smp2017":
            _extract_zip_files(target_dir)
    elif source.source_type == "direct_http":
        filename = source.version if "." in source.version else "download.bin"
        output_path = download_http_file(source.entrypoint, target_dir, filename=filename)
        if output_path.suffix.lower() == ".7z":
            _extract_7z_files(target_dir)
    else:
        return SyncResult(source_id=source.source_id, status="unsupported")

    return SyncResult(source_id=source.source_id, status="downloaded", output_dir=str(target_dir))


def download_huggingface_dataset(
    repo_id: str,
    target_dir: Path,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required to download HuggingFace datasets")
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(target_dir),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    snapshot_download(**kwargs)
    return target_dir


def download_github_repo(repo_url: str, target_dir: Path, *, branch: str = "main") -> Path:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and (target_dir / ".git").exists():
        return target_dir
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(target_dir)],
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return target_dir


def download_http_file(
    url: str,
    target_dir: Path,
    filename: str = "download.bin",
    *,
    max_bytes: int = MAX_HTTP_DOWNLOAD_BYTES,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    if filename == "download.bin":
        candidate = Path(unquote(urlparse(url).path)).name
        if candidate:
            filename = candidate
    output_path = _safe_child_path(target_dir, filename)
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            content_length = int(declared_length)
        except ValueError as exc:
            raise ValueError(f"invalid Content-Length for {url}: {declared_length!r}") from exc
        if content_length > max_bytes:
            raise ValueError(f"download too large: {content_length} bytes exceeds limit {max_bytes}")
    tmp_path = output_path.with_name(f".{output_path.name}.part")
    bytes_written = 0
    try:
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise ValueError(f"download too large: exceeded limit {max_bytes} bytes")
                    handle.write(chunk)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        close = getattr(response, "close", None)
        if callable(close):
            close()
    return output_path


def _safe_child_path(root: Path, filename: str) -> Path:
    if not filename:
        raise ValueError("download filename must not be empty")
    name = Path(filename).name
    if name != filename or name in {".", ".."}:
        raise ValueError(f"unsafe download filename: {filename!r}")
    resolved_root = root.resolve()
    output_path = (root / name).resolve()
    if not output_path.is_relative_to(resolved_root):
        raise ValueError(f"unsafe download filename: {filename!r}")
    return output_path


def _materialize_script_backed_dataset(source_id: str, target_dir: Path) -> None:
    urls = _SCRIPT_BACKED_DOWNLOADS.get(source_id)
    if not urls:
        return
    qrels_dir = target_dir / "qrels" if source_id == "fiqa" else target_dir
    if source_id == "fiqa":
        existing_qrels = list(qrels_dir.glob("*.tsv"))
        if len(existing_qrels) >= len(urls):
            return
    else:
        existing_data_files = [path for path in target_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".tsv", ".json", ".jsonl", ".parquet"} and path.name != "README.md"]
        if existing_data_files:
            return
    for url in urls:
        download_http_file(url, qrels_dir)
    if source_id == "smp2017":
        _extract_zip_files(target_dir)


def _allow_patterns_for_source(source_id: str) -> list[str] | None:
    if source_id == "t2ranking":
        return ["README.md", "data/queries.*.tsv", "data/qrels*.tsv", "data/collection.tsv"]
    if source_id == "fiqa":
        return ["README.md", "queries/*.parquet", "corpus/*.parquet"]
    if source_id == "mxode_finance":
        return ["README.md", "single_turn/Finance-Economics/train.json"]
    if source_id == "baai_finance_instruction":
        return ["README.md", "industry_instruction_semantic_cluster_dedup_*_train.jsonl"]
    if source_id == "qrecc":
        return ["README.md", "qrecc_train.json", "qrecc_test.json"]
    if source_id == "naturalconv":
        return ["README.md", "dialog_release.json"]
    if source_id == "dailydialog":
        return ["README.md", "*.zip"]
    if source_id == "risawoz":
        return ["README.md", "dataset_infos.json", "train.json", "dev.json", "test.json"]
    if source_id == "fincprg":
        return ["README.md", "corpus.jsonl", "queries.jsonl", "qrels/*.tsv"]
    if source_id in {"fir_bench_reports", "fir_bench_announcements"}:
        return ["README.md", "data/*.parquet"]
    return None


def _safe_archive_target(extract_dir: Path, member_name: str) -> Path:
    normalized_name = member_name.replace("\\", "/")
    member_path = Path(normalized_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Blocked unsafe archive member path: {member_name!r}")
    resolved_root = extract_dir.resolve()
    target = (extract_dir / member_path).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"Blocked unsafe archive member path: {member_name!r}")
    return target


def _validate_archive_limits(file_count: int, total_size: int) -> None:
    if file_count > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"archive has too many members: {file_count} > {MAX_ARCHIVE_MEMBERS}")
    if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"archive expands to too many bytes: {total_size} > {MAX_ARCHIVE_UNCOMPRESSED_BYTES}"
        )


def _safe_zip_extractall(archive: zipfile.ZipFile, extract_dir: Path) -> None:
    """Extract a zip archive after validating paths, symlinks, count, and size."""
    members = archive.infolist()
    total_size = 0
    safe_targets: list[tuple[zipfile.ZipInfo, Path]] = []
    for member in members:
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"Blocked unsafe zip symlink member: {member.filename!r}")
        total_size += member.file_size
        safe_targets.append((member, _safe_archive_target(extract_dir, member.filename)))
    _validate_archive_limits(len(members), total_size)

    for member, target in safe_targets:
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _replace_with_validated_extract_dir(tmp_dir: Path, extract_dir: Path) -> None:
    _validate_extracted_tree(tmp_dir)
    tmp_dir.replace(extract_dir)


def _validate_extracted_tree(extract_dir: Path) -> None:
    resolved_root = extract_dir.resolve()
    file_count = 0
    total_size = 0
    for path in extract_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Blocked unsafe archive symlink: {path}")
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Blocked archive path escaping target directory: {path}")
        if path.is_file():
            file_count += 1
            total_size += path.stat().st_size
    _validate_archive_limits(file_count, total_size)


def _fresh_extract_dir(path: Path) -> Path:
    tmp_dir = path.with_name(f".{path.stem}.extracting")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    return tmp_dir


def _extract_zip_files(target_dir: Path) -> None:
    for path in target_dir.rglob("*.zip"):
        extract_dir = path.with_suffix("")
        if extract_dir.exists():
            continue
        tmp_dir = _fresh_extract_dir(path)
        tmp_dir.mkdir(parents=True)
        try:
            with zipfile.ZipFile(path) as archive:
                _safe_zip_extractall(archive, tmp_dir)
            _replace_with_validated_extract_dir(tmp_dir, extract_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


def _extract_7z_files(target_dir: Path) -> None:
    for path in target_dir.rglob("*.7z"):
        extract_dir = path.with_suffix("")
        if extract_dir.exists():
            continue
        tmp_dir = _fresh_extract_dir(path)
        tmp_dir.mkdir(parents=True)
        try:
            subprocess.run(
                ["unar", "-quiet", "-output-directory", str(tmp_dir), str(path)],
                check=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            _replace_with_validated_extract_dir(tmp_dir, extract_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


_download_huggingface = download_huggingface_dataset
_download_github_repo = download_github_repo
_download_http = download_http_file


def sync_all(raw_root: Path, sources: list[DatasetSource]) -> list[SyncResult]:
    results: list[SyncResult] = []
    for source in sources:
        try:
            result = sync_source(source, raw_root)
        except Exception as exc:  # pragma: no cover - integration guard
            result = SyncResult(
                source_id=source.source_id,
                status="failed",
                output_dir=str(raw_root / source.source_id / source.version),
                error=str(exc),
            )
        results.append(result)
    return results
