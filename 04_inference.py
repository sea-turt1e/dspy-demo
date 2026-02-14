"""
Part 4: 保存・読み込み・推論デモ
================================
Part 3 で保存した最適化済みプログラムを読み込み、
新しい問題で推論を行います。

これにより、一度最適化すれば何度でも再利用できることを示します。

実行方法:
  uv run python 04_inference.py

前提:
  Part 3 (03_optimize.py) を先に実行し、
  optimized_gsm8k.json が生成されていること。
"""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 60)
    print("Part 4: 保存・読み込み・推論デモ")
    print("=" * 60)

    # ============================================================
    # 1. 言語モデルの設定
    # ============================================================
    lm = dspy.LM(os.getenv("OPENAI_MODEL", "openai/gpt-5-nano"))
    dspy.configure(lm=lm)
    print(f"\n✅ 言語モデルを設定しました: {lm.model_name}")

    # ============================================================
    # 2. 最適化済みプログラムの読み込み
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 最適化済みプログラムを読み込み")
    print("-" * 60)

    save_path = "optimized_gsm8k.json"

    if not os.path.exists(save_path):
        print(f"\n❌ {save_path} が見つかりません。")
        print("   先に Part 3 を実行してください:")
        print("   uv run python 03_optimize.py")
        return

    # 最適化済みプログラムを読み込む
    # 1. まず同じ構造のプログラムを作成
    # 2. .load() で最適化済みのパラメータ（命令文、few-shot 例）を復元
    optimized_program = dspy.ChainOfThought("question -> answer")
    optimized_program.load(save_path)

    print(f"\n✅ {save_path} から最適化済みプログラムを読み込みました")

    # ============================================================
    # 3. 日本語の数学問題で推論
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 新しい問題で推論テスト")
    print("-" * 60)

    # 日本語の数学文章題を用意
    questions = [
        "太郎は500円持っています。150円のジュースを2本買いました。おつりはいくらですか？",
        "教室に男子が15人、女子が18人います。そのうち5人が帰りました。教室には何人残っていますか？",
        "花子は1日に3ページずつ本を読みます。この本は全部で42ページあります。読み終わるのに何日かかりますか？",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 50}")
        print(f"📝 問題 {i}: {question}")
        print(f"{'─' * 50}")

        result = optimized_program(question=question)

        print(f"🤔 推論過程: {result.reasoning}")
        print(f"💡 回答: {result.answer}")

    # ============================================================
    # 4. ベースライン（最適化なし）との比較
    # ============================================================
    print("\n" + "-" * 60)
    print("🔹 ベースラインとの比較（同じ問題で試す）")
    print("-" * 60)

    baseline = dspy.ChainOfThought("question -> answer")

    # 1つ目の問題で比較
    question = questions[0]
    print(f"\n📝 問題: {question}")

    baseline_result = baseline(question=question)
    optimized_result = optimized_program(question=question)

    print(f"\n  【ベースライン（最適化なし）】")
    print(f"    推論: {baseline_result.reasoning}")
    print(f"    回答: {baseline_result.answer}")

    print(f"\n  【最適化済み（MIPROv2）】")
    print(f"    推論: {optimized_result.reasoning}")
    print(f"    回答: {optimized_result.answer}")

    # ============================================================
    # まとめ
    # ============================================================
    print("\n" + "=" * 60)
    print("📌 全体のまとめ - DSPy × MIPROv2")
    print("=" * 60)
    print("""
DSPy の特徴:
  ✅ プロンプトを手書きしない
     → Signature で「何をしたいか」を宣言するだけ

  ✅ 自動最適化
     → MIPROv2 がデータに基づいて最適なプロンプトを発見

  ✅ 再現性・再利用性
     → 最適化結果を JSON に保存・読み込み可能

  ✅ 評価駆動の開発
     → dspy.Evaluate でベースラインと比較し、改善を数値で確認

従来のプロンプトエンジニアリング:
  😰 プロンプトを手動で試行錯誤
  😰 「なんとなく良くなった」主観的判断
  😰 モデル変更のたびにやり直し

DSPy のアプローチ:
  😊 データを用意 → 評価関数を定義 → 自動最適化
  😊 数値で改善を確認
  😊 モデル変更にも対応可能

詳しくは:
  - DSPy 公式ドキュメント: https://dspy.ai/
  - DSPy GitHub: https://github.com/stanfordnlp/dspy
""")


if __name__ == "__main__":
    main()
