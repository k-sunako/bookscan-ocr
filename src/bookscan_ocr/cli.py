"""Batch searchable-PDF creation through OCRmyPDF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "paths": {"input_dir": "", "output_dir": ""},
    "ocr": {
        "engine": "paddle",
        "language": "jpn",
        "use_gpu": True,
        "jobs": 1,
        "pdfa": True,
        "skip_text": True,
        "deskew": False,
        "rotate_pages": False,
    },
}


def load_config(path: Path) -> dict[str, Any]:
    """Load TOML settings, filling omitted values with safe defaults."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    config = {
        "paths": {**DEFAULTS["paths"], **raw.get("paths", {})},
        "ocr": {**DEFAULTS["ocr"], **raw.get("ocr", {})},
    }
    for key in ("input_dir", "output_dir"):
        value = str(config["paths"].get(key, "")).strip()
        if not value or value.startswith("/path/to/"):
            raise ValueError(f"[paths].{key} を config/local.toml で実際のパスに設定してください")
        config["paths"][key] = Path(os.path.expandvars(value)).expanduser()

    config["ocr"]["engine"] = str(config["ocr"]["engine"]).lower()
    if config["ocr"]["engine"] not in {"paddle", "tesseract"}:
        raise ValueError("[ocr].engine は paddle または tesseract を指定してください")
    config["ocr"]["language"] = str(config["ocr"]["language"]).strip()
    if not config["ocr"]["language"]:
        raise ValueError("[ocr].language を空にはできません")
    config["ocr"]["jobs"] = int(config["ocr"]["jobs"])
    if config["ocr"]["jobs"] < 1:
        raise ValueError("[ocr].jobs は1以上にしてください")
    if config["ocr"]["engine"] == "tesseract" and config["ocr"]["use_gpu"]:
        raise ValueError("Tesseractはこの構成ではGPU OCRをしません。--no-use-gpu を指定してください")
    return config


def build_command(
    *,
    executable: str,
    source: Path,
    destination: Path,
    ocr: dict[str, Any],
) -> list[str]:
    """Build an OCRmyPDF argv list without invoking a shell."""
    command = [executable, "--jobs", str(ocr["jobs"]), "--language", ocr["language"]]
    if ocr["skip_text"]:
        command.append("--skip-text")
    if ocr["deskew"]:
        command.append("--deskew")
    if ocr["rotate_pages"]:
        command.append("--rotate-pages")
    command.extend(["--output-type", "pdfa" if ocr["pdfa"] else "pdf"])

    if ocr["engine"] == "paddle":
        command.extend(["--plugin", "ocrmypdf_paddleocr"])
        if ocr["use_gpu"]:
            command.append("--paddle-use-gpu")

    command.extend([str(source), str(destination)])
    return command


def resolve_ocrmypdf() -> str | None:
    """Prefer OCRmyPDF installed beside the active virtualenv interpreter."""
    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.prefix) / "bin" / "ocrmypdf"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("ocrmypdf")


def validate_directories(input_dir: Path, output_dir: Path) -> None:
    """Refuse missing inputs and output trees nested inside the input tree."""
    if not input_dir.is_dir():
        raise ValueError(f"入力ディレクトリがありません: {input_dir}")
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    try:
        output_resolved.relative_to(input_resolved)
    except ValueError:
        return
    raise ValueError("出力ディレクトリを入力ディレクトリの中に置かないでください（再処理を防ぐため）")


def discover_pdfs(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).casefold(),
    )


def validate_output(path: Path) -> list[str]:
    """Run available structural/text checks; return non-fatal warnings."""
    warnings: list[str] = []
    qpdf = shutil.which("qpdf")
    if qpdf:
        result = subprocess.run([qpdf, "--check", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"qpdfによるPDF検査に失敗しました: {details}")

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        result = subprocess.run(
            [pdftotext, "-f", "1", "-l", "3", str(path), "-"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotextで出力を読めません: {result.stderr.strip()}")
        if not result.stdout.strip():
            warnings.append("先頭3ページからテキストを抽出できませんでした（白紙ページ等の可能性あり）")
    else:
        warnings.append("pdftotextが見つからないため、テキスト層の簡易検査を省略しました")
    if not qpdf:
        warnings.append("qpdfが見つからないため、構造検査を省略しました")
    return warnings


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookscan-ocr",
        description="OCRmyPDFを使って、ローカルのスキャンPDFを検索可能PDFへ一括変換します。",
    )
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    parser.add_argument("--input-dir", type=Path, help="設定ファイルの入力ディレクトリを上書き")
    parser.add_argument("--output-dir", type=Path, help="設定ファイルの出力ディレクトリを上書き")
    parser.add_argument("--engine", choices=("paddle", "tesseract"))
    parser.add_argument("--language", help="OCR言語コード（例: jpn、Tesseractではjpn+eng）")
    parser.add_argument("--jobs", type=int, help="OCRmyPDFの並列ジョブ数")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pdfa", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-text", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deskew", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rotate-pages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="実行せず、処理対象とコマンドだけ表示")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力PDFを再作成")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = config["paths"]
        ocr = config["ocr"]
        if args.input_dir:
            paths["input_dir"] = args.input_dir.expanduser()
        if args.output_dir:
            paths["output_dir"] = args.output_dir.expanduser()
        for arg_name, key in (
            ("engine", "engine"),
            ("language", "language"),
            ("jobs", "jobs"),
            ("use_gpu", "use_gpu"),
            ("pdfa", "pdfa"),
            ("skip_text", "skip_text"),
            ("deskew", "deskew"),
            ("rotate_pages", "rotate_pages"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                ocr[key] = value
        if ocr["engine"] == "tesseract" and ocr["use_gpu"]:
            raise ValueError("Tesseractはこの構成ではGPU OCRをしません。--no-use-gpu を指定してください")
        if ocr["jobs"] < 1:
            raise ValueError("--jobs は1以上にしてください")

        input_dir: Path = paths["input_dir"]
        output_dir: Path = paths["output_dir"]
        validate_directories(input_dir, output_dir)
        sources = discover_pdfs(input_dir)
        if not sources:
            print(f"PDFが見つかりません: {input_dir}")
            return 0

        resolved_executable = resolve_ocrmypdf()
        executable = resolved_executable or "ocrmypdf"
        if not args.dry_run and resolved_executable is None:
            raise RuntimeError(
                "ocrmypdfが現在の仮想環境またはPATHにありません。同じ環境へインストールしてください"
            )

        print(f"対象PDF: {len(sources)}件 / エンジン: {ocr['engine']} / 言語: {ocr['language']}")
        if ocr["engine"] == "paddle" and ocr["use_gpu"]:
            print("GPU: PaddleOCRプラグインへGPU利用を要求（実際のGPU認識は環境依存です）")

        failures = 0
        succeeded = 0
        skipped = 0
        for source in sources:
            relative = source.relative_to(input_dir)
            destination = output_dir / relative
            if destination.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip] 既存出力: {relative}")
                continue
            command = build_command(
                executable=executable,
                source=source,
                destination=destination,
                ocr=ocr,
            )
            if args.dry_run:
                import shlex

                print(f"[dry-run] {relative}\n  {shlex.join(command)}")
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=".partial.pdf", dir=destination.parent
            )
            os.close(fd)
            temporary = Path(temporary_name)
            temporary.unlink()
            command[-1] = str(temporary)
            print(f"[ocr] {relative}", flush=True)
            try:
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"ocrmypdfが終了コード{result.returncode}で失敗しました")
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("OCRmyPDFが空または未作成の出力を返しました")
                warnings = validate_output(temporary)
                os.replace(temporary, destination)
                append_jsonl(
                    output_dir / "ocr-manifest.jsonl",
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "source_relative": relative.as_posix(),
                        "output_relative": relative.as_posix(),
                        "engine": ocr["engine"],
                        "language": ocr["language"],
                        "gpu_requested": bool(ocr["use_gpu"]),
                        "output_bytes": destination.stat().st_size,
                    },
                )
                succeeded += 1
                print(f"[ok] {relative} -> {destination}")
                for warning in warnings:
                    print(f"[warn] {relative}: {warning}", file=sys.stderr)
            except Exception as exc:  # Continue the batch and report each failed file.
                failures += 1
                if temporary.exists():
                    temporary.unlink()
                append_jsonl(
                    output_dir / "ocr-errors.jsonl",
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "source_relative": relative.as_posix(),
                        "error": str(exc),
                    },
                )
                print(f"[error] {relative}: {exc}", file=sys.stderr)

        if args.dry_run:
            print("ドライラン完了。OCRは実行していません。")
            return 0
        print(f"完了: 成功 {succeeded}件 / スキップ {skipped}件 / 失敗 {failures}件")
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
