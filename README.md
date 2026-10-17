# Bookscan OCR

`bookscan-ocr` は、ローカルに保存されたスキャンPDFを一括処理し、検索可能なPDFを生成するコマンドラインツールです。PDFの処理とテキスト層の生成には [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) を使用します。OCRエンジンは、GPUを利用するPaddleOCRプラグインと、CPUで動作するTesseractから選択できます。

## 主な機能

- 入力ディレクトリ以下のPDFを再帰的に検索して一括処理
- PaddleOCRプラグインによるGPU OCR、またはTesseractによるCPU OCR
- 元PDFを保持し、別の出力ディレクトリに結果を作成
- ドライラン、既存ファイルのスキップ、明示的な上書き
- `qpdf` と `pdftotext` が利用可能な場合の出力検査
- 成功記録とエラー記録をJSON Lines形式で出力

## 動作環境

- Linux x86_64
- Python 3.11以上（PaddleOCRを使用する場合は、互換性の観点からPython 3.12を推奨）
- [uv](https://docs.astral.sh/uv/) などのPython環境管理ツール
- Ghostscript、qpdf、Popplerのコマンドラインツール
- GPU OCRを使用する場合：対応するNVIDIA GPU・ドライバー、PaddlePaddle GPU版、PaddleOCR、OCRmyPDF-PaddleOCRプラグイン
- Tesseract OCRを使用する場合：Tesseract本体と言語データ

本リポジトリではManjaro Linux／Arch Linux系の手順を例示します。その他のOS・Linuxディストリビューションでは、必要なシステムパッケージを各環境の方法で導入してください。OCRmyPDFの依存関係については[公式インストールガイド](https://ocrmypdf.readthedocs.io/en/latest/installation.html)も参照してください。

## Manjaro／Arch Linuxでの準備

システムツールとTesseractの日本語・英語言語データをインストールします。

```bash
sudo pacman -S --needed ghostscript qpdf poppler tesseract \
  tesseract-data-jpn tesseract-data-eng
```

縦書き日本語の認識にTesseractを使う場合は、縦書き用データも追加します。

```bash
sudo pacman -S --needed tesseract-data-jpn_vert
```

このツールの出力検査に使用する `pdftotext`、`pdfinfo`、`pdftoppm` は、Arch系では `poppler` パッケージに含まれます。Tesseractの言語データは使用するOCR言語に対応したものを導入してください。

## インストール

プロジェクトのルートディレクトリでPython 3.12の仮想環境を作成し、プロジェクトをインストールします。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp config/config.example.toml config/local.toml
```

`config/local.toml` を編集し、入力ディレクトリと出力ディレクトリを実際のパスに設定してください。入力と出力は別のディレクトリにしてください。`config/local.toml` は `.gitignore` によりGit管理対象外です。

`bookscan-ocr` は、アクティブな仮想環境内の `bin/ocrmypdf` をシステムの `PATH` 上にある実行ファイルより優先します。OSコマンドであるGhostscript、qpdf、Poppler、Tesseractはシステム側にインストールしてください。

## GPU OCRのセットアップ（PaddleOCR）

以下は、NVIDIA GPUを利用するLinux環境向けの例です。ドライバーに対応するPaddlePaddle GPU wheelを選択してください。確認済み構成では、CUDA User-Mode Driver 13.3の環境でPaddlePaddle 3.3.0の`cu130` wheelを使用しました。ドライバーのCUDA表示とwheelのCUDA suffixを同じ値にする必要はありません。利用可能なwheelと対応条件は[PaddlePaddle公式インストールガイド](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html)で確認してください。

次の例は、Python 3.12とCUDA 13.0用のGPU wheelを使用します。

```bash
uv pip install --python .venv/bin/python \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
  paddlepaddle-gpu==3.3.0
uv pip install --python .venv/bin/python 'paddleocr==3.7.0'
```

続けて、OCRmyPDF-PaddleOCRプラグインを既知のコミットから導入します。

```bash
uv pip install --python .venv/bin/python --no-deps \
  'git+https://github.com/clefru/ocrmypdf-paddleocr.git@46fd027feb69de08ab831be27e337f97086ff2f0'
```

プラグインの依存メタデータはCPU版とGPU版を区別せず、`paddlepaddle`を要求します。一方、GPU版パッケージの名称は`paddlepaddle-gpu`です。そのため、GPU版を導入した環境にプラグインを通常の依存解決付きで追加すると、CPU版との衝突が起きる場合があります。上記のようにGPU版を先に導入し、プラグインを`--no-deps`で追加してください。`uv pip check`がこの依存名の不一致を報告する場合がありますが、解消のためにCPU版`paddlepaddle`を追加しないでください。

導入状態を確認します。

```bash
.venv/bin/python -c 'import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())'
.venv/bin/ocrmypdf --plugin ocrmypdf_paddleocr --help | grep -E 'paddle-use-gpu|paddle-engine'
```

GPU対応wheelの場合は`is_compiled_with_cuda()`が`True`となり、実際の推論時にはGPUデバイスが選択されます。OCRmyPDF-PaddleOCRプラグインはコミュニティ製のため、更新時にはPaddleOCR・PaddlePaddle・OCRmyPDFとの互換性を確認してください。

### 確認済み構成の例

以下の組み合わせで、GPU OCRによる検索可能PDFの生成を確認しています。異なるGPUやドライバーでは、PaddlePaddle公式ガイドに従ってwheelを選択してください。

- Manjaro Linux、x86_64
- NVIDIA GeForce RTX 5060 Ti（CUDA Capability 12.0）
- NVIDIA Driver/KMD 610.57.04、CUDA User-Mode Driver 13.3
- Python 3.12、PaddlePaddle GPU 3.3.0（`cu130`）、PaddleOCR 3.7.0
- OCRmyPDF 17.13.0、OCRmyPDF-PaddleOCRプラグイン `46fd027feb69de08ab831be27e337f97086ff2f0`

## 設定と実行

まず、ドライランで処理対象と実行コマンドを確認します。

```bash
.venv/bin/bookscan-ocr --config config/local.toml --dry-run
```

内容を確認した後、処理を実行します。

```bash
.venv/bin/bookscan-ocr --config config/local.toml
```

設定ファイルの値は、コマンドラインオプションで上書きできます。

```bash
# NVIDIA GPU + PaddleOCR
.venv/bin/bookscan-ocr --config config/local.toml \
  --engine paddle --use-gpu --language jpn

# CPU + Tesseract（日本語・英語）
.venv/bin/bookscan-ocr --config config/local.toml \
  --engine tesseract --no-use-gpu --language jpn+eng
```

PaddleOCRプラグインは、複数言語コードを指定しても先頭の言語のみを使用する仕様です。日本語OCRでは`jpn`を指定してください。Tesseractは`jpn+eng`のように複数言語を指定できます。OCRmyPDFの`--skip-text`を既定で有効にしているため、すでにテキスト層があるPDFページは処理をスキップします。

出力先に同名ファイルがある場合、既定ではスキップします。再処理する場合は`--overwrite`を指定してください。元PDFの保護のため、入力ディレクトリと出力ディレクトリは分け、出力先を入力ディレクトリの内部に配置しないでください。

## 出力と検査

- 出力PDFは既定でPDF/A形式です。`config/local.toml`またはCLIオプションで変更できます。
- `qpdf`が利用可能な場合、出力PDFの構造を検査します。
- `pdftotext`が利用可能な場合、先頭3ページからテキストが抽出できるか確認します。白紙ページなどによりテキストがない場合は警告として記録されます。
- 成功したPDFは一時ファイルから確定名へ移動します。成功記録は出力先の`ocr-manifest.jsonl`、失敗記録は`ocr-errors.jsonl`に追記されます。
- 記録には入力ファイルの相対パス等が含まれる場合があります。出力先にはGitリポジトリ外のディレクトリを指定してください。

## テスト

OCRエンジンや個人のPDFを使用しない単体テストは、次のコマンドで実行できます。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

GitHub ActionsでもPython 3.12を使用して同じ単体テストを実行します。ワークフローでは個人のPDFを利用しません。

## GitHubへ公開する前の確認

- `git status --ignored`でPDF、モデル重み、ログ、個人用設定ファイルが除外されていることを確認してください。
- 書籍PDF、OCRテキスト、個人情報を含むログ、モデル重みはコミットしないでください。
- OCRエンジン、プラグイン、モデルのライセンスと利用条件を確認してください。概要は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
- 本リポジトリのラッパーコードは[MIT License](LICENSE)で提供します。依存ソフトウェアのライセンスはそれぞれのプロジェクトに従います。
