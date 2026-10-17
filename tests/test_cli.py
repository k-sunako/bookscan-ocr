from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bookscan_ocr.cli import (
    build_command,
    discover_pdfs,
    load_config,
    main,
    resolve_ocrmypdf,
    validate_directories,
)


class CliTests(unittest.TestCase):
    def test_paddle_command_asks_for_gpu_and_skips_text(self) -> None:
        command = build_command(
            executable="ocrmypdf",
            source=Path("in/book.pdf"),
            destination=Path("out/book.pdf"),
            ocr={
                "engine": "paddle",
                "language": "jpn",
                "use_gpu": True,
                "jobs": 1,
                "pdfa": True,
                "skip_text": True,
                "deskew": False,
                "rotate_pages": False,
            },
        )
        self.assertIn("--plugin", command)
        self.assertIn("ocrmypdf_paddleocr", command)
        self.assertIn("--paddle-use-gpu", command)
        self.assertIn("--skip-text", command)
        self.assertEqual(command[-2:], ["in/book.pdf", "out/book.pdf"])

    def test_tesseract_command_does_not_add_paddle_plugin(self) -> None:
        command = build_command(
            executable="ocrmypdf",
            source=Path("in/a.pdf"),
            destination=Path("out/a.pdf"),
            ocr={
                "engine": "tesseract",
                "language": "jpn+eng",
                "use_gpu": False,
                "jobs": 2,
                "pdfa": False,
                "skip_text": False,
                "deskew": True,
                "rotate_pages": True,
            },
        )
        self.assertNotIn("--plugin", command)
        self.assertNotIn("--paddle-use-gpu", command)
        self.assertIn("--deskew", command)
        self.assertIn("--rotate-pages", command)
        self.assertIn("--output-type", command)
        self.assertIn("pdf", command)

    def test_load_config_rejects_example_paths(self) -> None:
        example = Path(__file__).parents[1] / "config" / "config.example.toml"
        with self.assertRaisesRegex(ValueError, "実際のパス"):
            load_config(example)

    def test_output_inside_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "books"
            output_dir = input_dir / "searchable"
            input_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "入力ディレクトリの中"):
                validate_directories(input_dir, output_dir)

    def test_discover_pdfs_is_case_insensitive_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "B.PDF").write_bytes(b"")
            (root / "a.pdf").write_bytes(b"")
            (root / "note.txt").write_text("not a PDF")
            nested = root / "sub"
            nested.mkdir()
            (nested / "c.PdF").write_bytes(b"")
            found = discover_pdfs(root)
            self.assertEqual([path.name for path in found], ["a.pdf", "B.PDF", "c.PdF"])

    def test_resolve_prefers_ocrmypdf_in_active_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            executable = venv / "bin" / "ocrmypdf"
            executable.parent.mkdir()
            executable.write_text("#!/bin/sh\\nexit 0\\n", encoding="utf-8")
            executable.chmod(0o755)
            with (
                patch("bookscan_ocr.cli.sys.prefix", str(venv)),
                patch("bookscan_ocr.cli.sys.base_prefix", "/usr"),
                patch("bookscan_ocr.cli.shutil.which", return_value="/usr/bin/ocrmypdf"),
            ):
                self.assertEqual(resolve_ocrmypdf(), str(executable))

    def test_dry_run_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "sample.pdf").write_bytes(b"placeholder")
            config = root / "local.toml"
            config.write_text(
                f'''[paths]\ninput_dir = "{input_dir}"\noutput_dir = "{output_dir}"\n\n'''
                '[ocr]\nengine = "paddle"\nlanguage = "jpn"\nuse_gpu = true\n'
                'jobs = 1\npdfa = true\nskip_text = true\ndeskew = false\nrotate_pages = false\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["--config", str(config), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertIn("--paddle-use-gpu", stdout.getvalue())
            self.assertIn("sample.pdf", stdout.getvalue())
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
