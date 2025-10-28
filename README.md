# Excel RAG システム

このプロジェクトは、LangGraphとGoogle Geminiモデルを利用して、複数のExcelファイルの内容に関する質問に自然言語で回答するRAG（Retrieval-Augmented Generation）システムです。

## 概要

ローカルのディレクトリに保存されている複数のExcelファイルを対象に、自然言語で質問を投げかけると、システムが関連する情報をExcelファイルから検索し、的確な回答を生成します。

例えば、「営業部のメンバーを教えて」や「2024年のノートパソコンの総売上は？」といった質問に答えることができます。

## 主な特徴

- **Excelファイル横断検索:** `tantivy`を利用したインメモリの全文検索インデックスにより、複数のExcelファイル（シート含む）を高速に検索します。
- **RAGアーキテクチャ:** LangGraphフレームワークを利用して、検索（Retrieval）と生成（Generation）を組み合わせたエージェントを構築しています。
- **LLMによる高度な応答:** GoogleのGeminiモデルを利用して、検索結果に基づいた自然で分かりやすい回答を生成します。
- **ツールの活用:** `search_excel_files`（ファイル検索）と`retrieve_from_excel`（ファイル内容の詳細取得）という2つのツールをエージェントが自律的に使い分けます。

## 必要なもの

- Python 3.10以上
- [uv](https://github.com/astral-sh/uv) パッケージインストーラー
- Google AI Studioで取得した `GEMINI_API_KEY`

## インストール

1.  このリポジトリをクローンします。
    ```bash
    git clone <repository_url>
    cd excel-rag-system
    ```

2.  `uv` を使して仮想環境を作成し、依存関係をインストールします。
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
    *(もし `requirements.txt` がなければ、`pyproject.toml` から直接インストールします)*
    ```bash
    uv pip install .
    ```

3.  仮想環境を有効化します。
    ```bash
    source .venv/bin/activate
    ```

4.  プロジェクトのルートディレクトリに `.env` ファイルを作成し、お使いのGemini APIキーを記述します。
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    EXCEL_DIRECTORY="./excel_data"
    ```

## 実行方法

1.  **ダミーのExcelファイルを作成する**

    まず、サンプルとなるExcelファイルを `excel_data` ディレクトリに生成します。
    ```bash
    python create_dummy_excel.py
    ```
    これにより、`excel_data` ディレクトリ内に「売上データ.xlsx」や「社員データ.xlsx」などのファイルが作成されます。

2.  **メインプログラムを実行する**

    準備ができたら、メインのRAGシステムを起動します。
    ```bash
    python main.py
    ```

3.  **質問を入力する**

    プログラムが起動すると、インデックスが構築された後、質問の入力が求められます。コンソールに質問を入力してください。
    ```
    === Excel RAG System ===

    Building in-memory Tantivy index...
    Indexed: 在庫データ.xlsx
    Indexed: プロジェクトデータ.xlsx
    Indexed: 社員データ.xlsx
    Indexed: 売上データ.xlsx
    Index built successfully!

    質問を入力してください: 
    ```
