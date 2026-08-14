import os
import sys
import glob
import json
import argparse

# Why add parent dir to sys.path?
# Guarantees that internal module imports succeed when running the script directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    parser = argparse.ArgumentParser(description="ベンチマーク結果JSONを元に、リファレンスデータ（正解データ）を3軸判定付きで更新します。")
    parser.add_argument("--benchmark-json", default=None, help="参照するベンチマーク結果JSONファイルパス（省略時はscratch/内の最新Gemini結果）")
    parser.add_argument("--base-timeline", default="tests/fixtures/reference_timeline.json", help="ベースとなるタイムラインJSONファイルパス")
    parser.add_argument("--output", default="tests/fixtures/reference_timeline.json", help="出力先リファレンスJSONファイルパス")
    args = parser.parse_args()

    # 1. Resolve benchmark result file
    benchmark_file = args.benchmark_json
    if not benchmark_file:
        scratch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch"))
        gemini_files = sorted(glob.glob(os.path.join(scratch_dir, "benchmark_results_gemini_*.json")))
        if not gemini_files:
            print("[ERROR] scratch/ 配下に Gemini ベンチマーク結果ファイル (benchmark_results_gemini_*.json) が見つかりません。")
            print("先に 'python tests/run_classifier_benchmark.py --classifier gemini' を実行してください。")
            sys.exit(1)
        benchmark_file = gemini_files[-1]

    if not os.path.exists(benchmark_file):
        print(f"[ERROR] 指定されたベンチマークファイルが見つかりません: {benchmark_file}")
        sys.exit(1)

    print(f"ベンチマーク結果を読み込み中: {benchmark_file}")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    details = bench_data.get("details", [])
    if not details:
        print("[ERROR] ベンチマーク結果内に details が存在しません。")
        sys.exit(1)

    # Build mapping by index
    index_map = {item["index"]: item for item in details if "index" in item}

    # 2. Load base timeline
    if not os.path.exists(args.base_timeline):
        print(f"[ERROR] ベースタイムラインが見つかりません: {args.base_timeline}")
        sys.exit(1)

    with open(args.base_timeline, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    # 3. Update timeline events with 3-axis predictions as new ground truth
    # Why store 3-axis predicted attributes as expected ground truth?
    # Contextual interpretations and 3-axis binary decisions produced by high-capacity LLM
    # serve as consistent, high-fidelity reference labels for future model benchmarking.
    updated_count = 0
    for idx, ev in enumerate(timeline):
        if ev.get("type") != "listener":
            continue

        pred_info = index_map.get(idx)
        if pred_info:
            ev["expected_category"] = pred_info.get("predicted_category", ev.get("expected_category", "other"))
            
            # Optional 3-axis & contextual reasoning metadata
            if "interpreted_comment" in pred_info and pred_info["interpreted_comment"]:
                ev["expected_interpreted_comment"] = pred_info["interpreted_comment"]
            if "is_subject_streamer" in pred_info and pred_info["is_subject_streamer"] is not None:
                ev["expected_is_subject_streamer"] = pred_info["is_subject_streamer"]
            if "is_topic_relevant" in pred_info and pred_info["is_topic_relevant"] is not None:
                ev["expected_is_topic_relevant"] = pred_info["is_topic_relevant"]
            if "is_future" in pred_info and pred_info["is_future"] is not None:
                ev["expected_is_future"] = pred_info["is_future"]
                
            updated_count += 1

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] リファレンスデータを更新しました: {args.output}")
    print(f"   更新されたリスナーコメント数: {updated_count} 件")

if __name__ == "__main__":
    main()
