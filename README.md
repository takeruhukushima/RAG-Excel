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

main.py(tantivy)はノートパソコンとノートPCを正しく反映してない

(base) fukushimatakeru@Mac Excel % uv run main.py           
=== Excel RAG System ===

Building in-memory Tantivy index...
Indexed: 社員データ.xlsx
Indexed: プロジェクトデータ.xlsx
Indexed: 在庫データ.xlsx
Indexed: 売上データ.xlsx
Index built successfully!

質問を入力してください: ノートパソコンの料金を教えて

処理中...

--- [Debug] Calling model WITH tools (deciding action)...
--- [Debug] Re-constructing prompt to continue agent loop...

=== 回答 ===
ノートパソコンの単価は150,000円です。
(base) fukushimatakeru@Mac Excel % 


without_tantivy.pyはうまくいった。

(base) fukushimatakeru@Mac Excel % uv run without_tantivy.py
=== Excel RAG System (Chroma In-Memory Mode) ===

Building in-memory Chroma vector store...
  Chunking: 社員データ.xlsx
  Chunking: プロジェクトデータ.xlsx
  Chunking: 在庫データ.xlsx
  Chunking: 売上データ.xlsx
Converting 37 chunks to Documents...
Adding 37 documents to Chroma...
  Successfully added 37 documents!
In-memory Chroma vector store built successfully!

質問を入力してください: ノートパソコンの料金を教えて

処理中...

--- [Debug] Calling model WITH tools (deciding action)...
--- [Debug] Searching Chroma vector store for: ノートパソコン 料金
--- [Debug] Retrieving Top 3 MMR results (fetching 10)...
--- [Debug] Retrieved Docs: ---
  [Doc 1] Source: excel_data/社員データ.xlsx, Sheet: スキル管理
  Content:
EMP0016 高橋一郎 - 上級 初級 上級 中級
EMP0017 渡辺久美 上級 エキスパート 初級 初級 -
EMP0018 高橋一郎 上級 中級 中級 初級 -
EMP0019 中村さくら 初級 上級 - エキスパート 中級
EMP0020 鈴木花子 エキスパート 上級 エキスパート 上級 -
EMP0021 山本隆 エキスパート 中級 - エキスパート エキスパート
EMP0022 中村さくら - 初級 中級 エキスパート エキスパート
EMP0023 加藤恵 上級 中級 上級 初級 エキスパート
EMP0024 佐藤太郎 上級 エキスパート 中級 上級 -
EMP0025 佐藤太郎 エキスパート 上級 上級 中級 エキスパート
EMP0026 中村さくら エキスパート 初級 - 初級 -
EMP0027 田中美咲 初級 エキスパート 中級 中級 エキスパート
EMP0028 加藤恵 上級 初級 初級 中級 -
EMP0029 高橋一郎 上級 エキスパート 初級 - -
EMP0030 田中美咲 初級 - 上級 中級 上級
  ==================================================
  [Doc 2] Source: excel_data/在庫データ.xlsx, Sheet: 在庫一覧
  Content:
商品コード 商品名 カテゴリ 在庫数 最低在庫数 単価 仕入先
PRD00001 ヘッドセット 周辺機器 498 29 10653 F社
PRD00002 スマートフォン 電子機器 191 46 160705 D社
PRD00003 ノートPC 電子機器 467 17 50654 B社
PRD00004 スマートフォン 電子機器 366 12 132938 D社
PRD00005 LANケーブル アクセサリ 396 48 30001 H社
PRD00006 デスクトップPC 電子機器 381 33 116382 A社
PRD00007 ヘッドセット 周辺機器 75 12 142042 F社
PRD00008 Webカメラ 周辺機器 88 14 8993 G社
PRD00009 USBケーブル アクセサリ 371 38 36735 H社
PRD00010 マウス 周辺機器 307 44 139791 E社
PRD00011 ノートPC 電子機器 227 15 68659 B社
PRD00012 タブレット 電子機器 489 12 149003 C社
PRD00013 マウス 周辺機器 6
  ==================================================
  [Doc 3] Source: excel_data/売上データ.xlsx, Sheet: 2023年売上
  Content:
 高橋
2023-07-19 マウス 周辺機器 1 2500 2500 佐藤
2023-09-29 モニター 電子機器 2 35000 70000 佐藤
2023-03-05 モニター 電子機器 2 35000 70000 高橋
2023-06-16 外付けHDD 記憶装置 5 12000 60000 佐藤
  ==================================================
---------------------------------
--- [Debug] Re-constructing prompt to generate final answer...

=== 回答 ===
データソース2、在庫一覧、行番号4: 単価 50654円
データソース2、在庫一覧、行番号12: 単価 68659円