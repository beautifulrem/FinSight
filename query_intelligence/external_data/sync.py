from __future__ import annotations

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
    )
    return target_dir


def download_http_file(url: str, target_dir: Path, filename: str = "download.bin") -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if filename == "download.bin":
        candidate = Path(unquote(urlparse(url).path)).name
        if candidate:
            filename = candidate
    output_path = target_dir / filename
    output_path.write_bytes(response.content)
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


def _extract_zip_files(target_dir: Path) -> None:
    for path in target_dir.rglob("*.zip"):
        extract_dir = path.with_suffix("")
        if extract_dir.exists():
            continue
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)


def _extract_7z_files(target_dir: Path) -> None:
    for path in target_dir.rglob("*.7z"):
        extract_dir = path.with_suffix("")
        if extract_dir.exists():
            continue
        subprocess.run(["unar", "-quiet", "-output-directory", str(extract_dir), str(path)], check=True)


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
