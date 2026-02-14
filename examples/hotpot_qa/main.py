"""
DSPy デモ - PyCon mini Shizuoka
================================
DSPy と MIPROv2 を使った LLM アプリケーションの自動最適化デモです。
タスク: HotPotQA（マルチホップ質問応答）

以下の順番で実行してください:
  1. uv run python 01_basics.py    - DSPy の基本（Signature, Predict, ChainOfThought）
  2. uv run python 02_evaluate.py  - データセットと評価（ベースライン測定）
  3. uv run python 03_optimize.py  - MIPROv2 による自動最適化
  4. uv run python 04_inference.py  - 最適化済みモデルで推論

事前準備:
  export OPENAI_API_KEY="sk-..."
"""


def main():
    print("=" * 60)
    print("DSPy デモ - PyCon mini Shizuoka")
    print("=" * 60)
    print()
    print("DSPy と MIPROv2 を使った LLM アプリケーションの自動最適化デモです。")
    print("タスク: HotPotQA（マルチホップ質問応答）")
    print()
    print("以下の順番でスクリプトを実行してください:")
    print()
    print("  Step 1: uv run python 01_basics.py")
    print("    → DSPy の基本を学ぶ（Signature, Predict, ChainOfThought）")
    print()
    print("  Step 2: uv run python 02_evaluate.py")
    print("    → HotPotQA データセットでベースラインの正答率を測定")
    print()
    print("  Step 3: uv run python 03_optimize.py")
    print("    → MIPROv2 でプロンプトを自動最適化し、正答率を比較")
    print()
    print("  Step 4: uv run python 04_inference.py")
    print("    → 最適化済みモデルを読み込んで推論")
    print()
    print("事前準備:")
    print('  export OPENAI_API_KEY="sk-..."')
    print()
    print("※ GSM8K（数学文章題）版のデモは examples/gsm8k/ にあります。")
    print()


if __name__ == "__main__":
    main()
