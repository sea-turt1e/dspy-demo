"""
Part 1: DSPy の基本
===================
DSPy の基本コンセプトを学びます。
- LM（言語モデル）の設定
- Signature（入出力の宣言的定義）
- Predict（基本的な予測モジュール）
- ChainOfThought（ステップバイステップ推論）

実行方法:
  uv run python 01_basics.py
"""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()


def main():
    # ============================================================
    # 1. 言語モデル（LM）の設定
    # ============================================================
    # DSPy は LiteLLM を内部で使用しており、様々なプロバイダーに対応しています。
    # 環境変数 OPENAI_API_KEY, OPENAI_MODEL を設定しておけば、dspy.LM() で自動的に OpenAI API に接続します。
    # .env.example をコピーして .env を作成し、API キー、モデル名を設定してください。
    print("=" * 60)
    print("Part 1: DSPy の基本")
    print("=" * 60)

    lm = dspy.LM(os.getenv("OPENAI_MODEL", "openai/gpt-5-nano"))
    dspy.configure(lm=lm)
    print(f"\n✅ 言語モデルを設定しました: {lm.model_name}")

    # ============================================================
    # 2. Signature（シグネチャ）- タスクの入出力を宣言的に定義
    # ============================================================
    # DSPy では「何をしたいか」をシグネチャで定義します。
    # プロンプトを手書きする必要はありません！
    #
    # 書式: "入力フィールド -> 出力フィールド"
    # 例:   "question -> answer"  … 質問を受け取り、回答を返す

    # ============================================================
    # 3. Predict - 最もシンプルなモジュール
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 dspy.Predict: シンプルな質問応答")
    print("-" * 60)

    # Predict はシグネチャをそのまま LLM に渡すだけのシンプルなモジュール
    predict = dspy.Predict("question -> answer")

    # 数学の問題を解いてみましょう
    question = "5個のりんごが入った箱が3箱あります。全部でりんごは何個ですか？"
    result = predict(question=question)

    print(f"\n📝 質問: {question}")
    print(f"💡 回答: {result.answer}")

    # ============================================================
    # 4. ChainOfThought - ステップバイステップで推論
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 dspy.ChainOfThought: ステップバイステップ推論")
    print("-" * 60)

    # ChainOfThought は「段階的に考えてから回答する」モジュール
    # 内部で自動的に reasoning（推論過程）フィールドが追加されます
    cot = dspy.ChainOfThought("question -> answer")

    # 少し複雑な数学の問題
    question = "太郎は1000円持っています。250円のノートを2冊、150円の鉛筆を3本買いました。残りはいくらですか？"
    result = cot(question=question)

    print(f"\n📝 質問: {question}")
    print(f"🤔 推論過程: {result.reasoning}")
    print(f"💡 回答: {result.answer}")

    # ============================================================
    # 5. プロンプトの中身を確認
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 プロンプトの中身を確認（inspect_history）")
    print("-" * 60)

    # DSPy が実際に LLM に送ったプロンプトを確認できます
    # これにより、DSPy がどのようにプロンプトを構成しているかがわかります
    print("\n最後のリクエストで使われたプロンプト:")
    lm.inspect_history(n=1)

    # ============================================================
    # まとめ
    # ============================================================
    print("\n" + "=" * 60)
    print("📌 Part 1 まとめ")
    print("=" * 60)
    print("""
DSPy の基本コンセプト:
  1. LM の設定      → dspy.LM() と dspy.configure() で言語モデルを設定
  2. Signature      → "question -> answer" のように入出力を宣言的に定義
  3. Predict        → シンプルに LLM を呼び出すモジュール
  4. ChainOfThought → 推論過程を含めてステップバイステップで回答

ポイント:
  - プロンプトを手書きする必要がない！
  - 「何をしたいか（What）」を定義するだけで、DSPy が最適なプロンプトを構成
  - 次の Part 2 では、データセットを使った評価（ベースライン測定）を行います

次のステップ:
  uv run python 02_evaluate.py
""")


if __name__ == "__main__":
    main()
