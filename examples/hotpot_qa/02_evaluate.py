"""
Part 2: データセットと評価（ベースライン測定）
============================================
HotPotQA データセット（マルチホップ質問応答）を使って、
最適化前のベースライン正答率を測定します。

マルチホップ QA とは:
  「Aの監督は誰？」→「その人の出身地は？」のように、
  複数ステップの推論が必要な質問応答タスクです。

DSPy のポイント:
  - データセットが組み込みで用意されている
  - メトリクス（評価関数）も組み込みで用意されている
  - dspy.Evaluate で簡単に評価できる

実行方法:
  uv run python 02_evaluate.py
"""

import os

import dspy
from dotenv import load_dotenv
from dspy.datasets.hotpotqa import HotPotQA
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match

# override=True: .zshrc 等で設定済みの環境変数よりも .env の値を優先する
load_dotenv("../../.env", override=True)


def main():
    print("=" * 60)
    print("Part 2: データセットと評価（ベースライン測定）")
    print("=" * 60)

    # ============================================================
    # 1. 言語モデルの設定
    # ============================================================
    lm_model = os.getenv("OPENAI_MODEL", "openai/gpt-5-nano")
    lm = dspy.LM(lm_model)
    dspy.configure(lm=lm)
    print(f"\n✅ 言語モデルを設定しました: {lm_model}")

    # ============================================================
    # 2. HotPotQA データセットの読み込み
    # ============================================================
    # HotPotQA はマルチホップ質問応答のデータセットです。
    # 複数の情報を組み合わせて推論する必要があり、LLM にとって難易度が高いタスクです。
    # DSPy にはこのデータセットが組み込まれています。
    print("\n" + "-" * 60)
    print("🔹 HotPotQA データセットの読み込み")
    print("-" * 60)

    hotpotqa = HotPotQA(
        train_seed=1,       # 訓練データのシード（再現性のため）
        train_size=150,     # 訓練データ数（MIPROv2 の最適化に使用）
        eval_seed=2023,     # 評価データのシード
        dev_size=50,        # 評価データ数（デモ用に少なめ）
    )

    # .with_inputs("question") で入力フィールドを指定
    # trainset: 最適化に使うデータ
    # devset:   評価に使うデータ
    trainset = [x.with_inputs("question") for x in hotpotqa.train]
    devset = [x.with_inputs("question") for x in hotpotqa.dev]

    print(f"\n  訓練データ数: {len(trainset)}")
    print(f"  評価データ数: {len(devset)}")

    # ============================================================
    # 3. データの中身を確認
    # ============================================================
    # dspy.Example は DSPy のデータ型です。
    # HotPotQA の各データには question（質問）と answer（正解）が含まれます。
    print("\n" + "-" * 60)
    print("🔹 データの中身を確認")
    print("-" * 60)

    for i, example in enumerate(trainset[:3]):
        print(f"\n  --- 例 {i+1} ---")
        print(f"  質問: {example.question}")
        print(f"  正解: {example.answer}")

    # ============================================================
    # 4. ベースラインプログラムの定義
    # ============================================================
    # Predict を使ったシンプルなプログラムをベースラインとします。
    # マルチホップ推論が必要な問題には不十分なため、正答率は低くなるはずです。
    print("\n" + "-" * 60)
    print("🔹 ベースラインプログラム（最適化なし）")
    print("-" * 60)

    baseline = dspy.Predict("question -> answer")
    print("\n  プログラム: dspy.Predict('question -> answer')")
    print("  ※ 推論過程なし・命令文なし・few-shot 例なしの最もシンプルな状態")

    # 1件だけ試してみる
    example = devset[0]
    prediction = baseline(question=example.question)
    print(f"\n  質問: {example.question}")
    print(f"  予測: {prediction.answer}")
    print(f"  正解: {example.answer}")

    # ============================================================
    # 5. メトリクス（評価関数）の説明
    # ============================================================
    # answer_exact_match: 予測と正解が完全一致するかを判定
    # DSPy に組み込まれた評価関数です。
    print("\n" + "-" * 60)
    print("🔹 メトリクス（評価関数）")
    print("-" * 60)

    is_correct = answer_exact_match(example, prediction)
    print(f"\n  answer_exact_match の結果: {is_correct}")
    print("  （予測と正解の文字列を比較して正誤を判定します）")

    # ============================================================
    # 6. ベースラインの正答率を測定
    # ============================================================
    # dspy.Evaluate を使って、devset 全体での正答率を測定します。
    # これが最適化前の「ベースライン」になります。
    print("\n" + "-" * 60)
    print("🔹 ベースラインの正答率を測定中...")
    print("-" * 60)

    evaluator = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,           # 並列実行数（API 呼び出しを高速化）
        display_progress=True,   # プログレスバーを表示
        display_table=5,         # 結果の最初の5件をテーブル表示
    )

    # evaluator() は EvaluationResult オブジェクトを返すので .score で数値を取得
    baseline_result = evaluator(baseline)
    baseline_score = baseline_result.score

    print(f"\n📊 ベースライン正答率: {baseline_score:.1f}%")

    # ============================================================
    # まとめ
    # ============================================================
    print("\n" + "=" * 60)
    print("📌 Part 2 まとめ")
    print("=" * 60)
    print(f"""
ベースライン（最適化なし）の正答率: {baseline_score:.1f}%

マルチホップ QA は複数ステップの推論が必要なため、
シンプルな Predict だけでは正答率が低くなります。

DSPy の評価のポイント:
  1. HotPotQA データセット → 組み込みで利用可能
  2. answer_exact_match    → 組み込みの評価関数
  3. dspy.Evaluate         → 簡単にバッチ評価が可能

次の Part 3 では MIPROv2 を使ってこの正答率を自動的に改善します！

次のステップ:
  uv run python 03_optimize.py
""")


if __name__ == "__main__":
    main()
