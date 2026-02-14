# DSPy デモ - PyCon mini Shizuoka

DSPy と MIPROv2 を使った LLM アプリケーションの自動最適化デモです。

## DSPy とは？

[DSPy](https://dspy.ai/) は Stanford NLP が開発した、LLM アプリケーションを**プログラミング的に**構築・最適化するためのフレームワークです。

- プロンプトを手書きする代わりに、**Signature**（入出力の宣言）を定義
- **MIPROv2** などのオプティマイザが、データに基づいて最適なプロンプトを自動発見
- 評価関数で**数値的に**改善を確認

## デモのタスク: HotPotQA（マルチホップ質問応答）

**マルチホップ QA** とは、複数の事実・ステップを組み合わせて推論する必要がある質問応答タスクです。

例:
- 「カルメンを作曲したのは誰？」→「その作曲家が生まれた国は？」

シンプルな `dspy.Predict` だけでは正答率が低く、MIPROv2 による最適化の効果が明確に出ます。

## セットアップ

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) がインストール済み
- OpenAI API キー

### 環境構築

```bash
# リポジトリをクローン
git clone https://github.com/sea-turt1e/dspy-demo.git
cd dspy-demo

# 依存関係をインストール（uv が自動で仮想環境を作成）
uv sync

# OpenAI API キーを設定
export OPENAI_API_KEY="sk-..."
```

## デモの実行

4つのスクリプトを順番に実行します。

### Step 1: DSPy の基本

```bash
uv run python 01_basics.py
```

DSPy の基本コンセプトを学びます:
- `dspy.LM()` で言語モデルを設定
- Signature（`"question -> answer"`）で入出力を宣言的に定義
- `dspy.Predict` と `dspy.ChainOfThought` の違い

### Step 2: データセットと評価（ベースライン測定）

```bash
uv run python 02_evaluate.py
```

HotPotQA（マルチホップ質問応答）データセットを使って、最適化前のベースライン正答率を測定します:
- `dspy.datasets.hotpotqa` で組み込みデータセットを読み込み
- `answer_exact_match` で回答の正確性を評価
- `dspy.Evaluate` でバッチ評価

### Step 3: MIPROv2 で自動最適化

```bash
uv run python 03_optimize.py
```

MIPROv2 オプティマイザを使って、プロンプトを自動最適化します:
- Bootstrap Few-Shot Examples → 命令文候補の生成 → ベイズ最適化
- 最適化前後の正答率を比較
- 結果を `optimized_hotpotqa.json` に保存

### Step 4: 最適化済みモデルで推論

```bash
uv run python 04_inference.py
```

保存した最適化済みプログラムを読み込み、新しいマルチホップ質問で推論します:
- `optimized_hotpotqa.json` から復元
- ベースラインとの比較

## プロジェクト構成

```
dspy-demo/
├── main.py                    # デモの案内（目次）
├── 01_basics.py               # Part 1: DSPy の基本
├── 02_evaluate.py             # Part 2: データセットと評価
├── 03_optimize.py             # Part 3: MIPROv2 で最適化
├── 04_inference.py            # Part 4: 推論デモ
├── optimized_hotpotqa.json    # 最適化結果（Part 3 実行後に生成）
├── examples/
│   └── gsm8k/                 # GSM8K（数学文章題）版のデモ
│       ├── 01_basics.py
│       ├── 02_evaluate.py
│       ├── 03_optimize.py
│       └── 04_inference.py
├── pyproject.toml             # プロジェクト設定
├── uv.lock                    # 依存関係ロックファイル
└── README.md
```

## 参考リンク

- [DSPy 公式ドキュメント](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [MIPROv2 論文](https://arxiv.org/abs/2406.11695)

## ライセンス

Apache License 2.0
