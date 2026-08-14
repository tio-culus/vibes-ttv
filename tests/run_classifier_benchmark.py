import os
import sys
import json
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Any

# Why add parent dir to sys.path?
# Guarantees package references like vibes_ttv work correctly when running directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vibes_ttv.analyzers.comment_analyzer import CommentCategory
from vibes_ttv.analyzers.gemini_classifier import GeminiCommentClassifier
from vibes_ttv.analyzers.rule_based_classifier import RuleBasedCommentClassifier

def calculate_metrics(y_true: List[CommentCategory], y_pred: List[CommentCategory], categories: List[CommentCategory]):
    # Why calculate metrics manually instead of using scikit-learn?
    # Implementing the math (TP, FP, FN) in pure Python prevents adding complex C-extension dependencies 
    # like scikit-learn or numpy/pandas specific submodules, keeping the utility extremely lightweight 
    # and runnable in simple containerized or production-like environments.
    metrics = {}
    total_tp = 0
    total_samples = len(y_true)
    
    for cat in categories:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cat and p == cat)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p != cat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        expected_count = sum(1 for t in y_true if t == cat)
        predicted_count = sum(1 for p in y_pred if p == cat)
        
        metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "expected": expected_count,
            "predicted": predicted_count
        }
        total_tp += tp
        
    overall_accuracy = total_tp / total_samples if total_samples > 0 else 0.0
    return overall_accuracy, metrics

def calculate_binary_metrics(y_true: List[bool], y_pred: List[bool]):
    # Why calculate binary precision/recall/F1 for each 3-axis flag?
    # Breaking down accuracy into separate 3-axis metrics (subject, topic, future) highlights 
    # specific failure modes (e.g., topic drift vs future backseating false positives) across models.
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "expected_true": 0, "predicted_true": 0}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is True)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is True)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is False)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is False)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expected_true": sum(1 for t in y_true if t is True),
        "predicted_true": sum(1 for p in y_pred if p is True)
    }

def print_confusion_matrix(y_true: List[CommentCategory], y_pred: List[CommentCategory], categories: List[CommentCategory]):
    # Why print custom formatted grid instead of exporting CSV?
    # Visual console grids give immediate feedback on where model confusion lies (e.g. backseat vs advice)
    # without needing the developer to open external spreadsheet software.
    matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
        
    print("\n=== Confusion Matrix (Row: Expected, Col: Predicted) ===")
    
    # We truncate category labels to keep the grid aligned on small screens
    col_width = 12
    header = f"{'Expected \\ Pred':<18}" + "".join(f"{cat.value[:col_width]:>{col_width}}" for cat in categories)
    print(header)
    print("-" * len(header))
    
    for t_cat in categories:
        row = f"{t_cat.value:<18}"
        for p_cat in categories:
            val = matrix[t_cat][p_cat]
            row += f"{val:>{col_width}}"
        print(row)
    print("-" * len(header))

from datetime import datetime

def run_benchmark_for_classifier(classifier, timeline_data: List[Dict[str, Any]], output_path: str = None):
    print("=" * 80)
    print(f"Running Benchmark: {classifier.name}")
    print("=" * 80)

    # 1. Mask expected_category and 3-axis labels to simulate clean input
    timeline_masked = []
    y_true = []
    y_true_subject = []
    y_true_topic = []
    y_true_future = []
    has_3axis_reference = False
    listener_indices = []

    for idx, ev in enumerate(timeline_data):
        ev_copy = dict(ev)
        if ev_copy["type"] == "listener":
            raw_exp = ev_copy.pop("expected_category", "other")
            try:
                y_true.append(CommentCategory(raw_exp))
            except ValueError:
                y_true.append(CommentCategory.OTHER)
                
            # Extract 3-axis ground truth if present (or None if unlabelled)
            # Why append None instead of skipping?
            # Appending None preserves 1-to-1 index alignment with y_true and listener_indices,
            # allowing partial benchmark evaluation even if only a subset of comments have 3-axis ground truth.
            if "expected_is_subject_streamer" in ev_copy:
                has_3axis_reference = True
                y_true_subject.append(bool(ev_copy.pop("expected_is_subject_streamer")))
            else:
                y_true_subject.append(None)

            if "expected_is_topic_relevant" in ev_copy:
                y_true_topic.append(bool(ev_copy.pop("expected_is_topic_relevant")))
            else:
                y_true_topic.append(None)

            if "expected_is_future" in ev_copy:
                y_true_future.append(bool(ev_copy.pop("expected_is_future")))
            else:
                y_true_future.append(None)

            if "expected_interpreted_comment" in ev_copy:
                ev_copy.pop("expected_interpreted_comment")
                
            listener_indices.append(idx)
        timeline_masked.append(ev_copy)

    # 2. Run Classification with timer
    start_time = time.time()
    try:
        predicted_map = classifier.classify(timeline_masked)
    except Exception as e:
        print(f"[ERROR] Classifier execution failed: {e}")
        return
    elapsed_time = time.time() - start_time

    # 3. Align predictions with truth
    y_pred = []
    y_pred_subject = []
    y_pred_topic = []
    y_pred_future = []

    for idx in listener_indices:
        # Fallback to OTHER if comment was missed or unclassified
        pred = predicted_map.get(idx, CommentCategory.OTHER)
        y_pred.append(pred)

        ev_res = timeline_masked[idx]
        y_pred_subject.append(bool(ev_res.get("is_subject_streamer", False)))
        y_pred_topic.append(bool(ev_res.get("is_topic_relevant", True)))
        y_pred_future.append(bool(ev_res.get("is_future", False)))

    # 4. Compute Metrics
    categories = list(CommentCategory)
    accuracy, metrics = calculate_metrics(y_true, y_pred, categories)

    # 5. Output Results
    print(f"Results for {classifier.name}:")
    print(f"  Total Comments Evaluated: {len(y_true)}")
    print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"  Avg Time per Comment: {elapsed_time / len(y_true) if y_true else 0:.4f} seconds")
    print(f"  Overall Category Accuracy: {accuracy * 100:.2f}%\n")

    print(f"{'Category':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Expected Count':<16}{'Predicted Count'}")
    print("-" * 80)
    for cat in categories:
        m = metrics[cat]
        print(f"{cat.value:<15}{m['precision']*100:>10.2f}%{m['recall']*100:>10.2f}%{m['f1']*100:>10.2f}%{m['expected']:>14}{m['predicted']:>17}")
    print("-" * 80)

    print_confusion_matrix(y_true, y_pred, categories)
    print("\n")

    # 3-Axis Evaluation Reporting
    three_axis_metrics = {}
    if has_3axis_reference:
        sub_pairs = [(t, p) for t, p in zip(y_true_subject, y_pred_subject) if t is not None]
        top_pairs = [(t, p) for t, p in zip(y_true_topic, y_pred_topic) if t is not None]
        fut_pairs = [(t, p) for t, p in zip(y_true_future, y_pred_future) if t is not None]

        if sub_pairs and top_pairs and fut_pairs:
            sub_m = calculate_binary_metrics([p[0] for p in sub_pairs], [p[1] for p in sub_pairs])
            top_m = calculate_binary_metrics([p[0] for p in top_pairs], [p[1] for p in top_pairs])
            fut_m = calculate_binary_metrics([p[0] for p in fut_pairs], [p[1] for p in fut_pairs])
            three_axis_metrics = {
                "is_subject_streamer": sub_m,
                "is_topic_relevant": top_m,
                "is_future": fut_m
            }
            print("=== 3-Axis Binary Evaluation Performance (Ground Truth vs Prediction) ===")
            print(f"{'Axis':<28}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Exp (True)':<14}{'Pred (True)'}")
            print("-" * 100)
            print(f"{'Subject (Streamer)':<28}{sub_m['accuracy']*100:>10.2f}%{sub_m['precision']*100:>10.2f}%{sub_m['recall']*100:>10.2f}%{sub_m['f1']*100:>10.2f}%{sub_m['expected_true']:>12}{sub_m['predicted_true']:>15}")
            print(f"{'Topic Relevance':<28}{top_m['accuracy']*100:>10.2f}%{top_m['precision']*100:>10.2f}%{top_m['recall']*100:>10.2f}%{top_m['f1']*100:>10.2f}%{top_m['expected_true']:>12}{top_m['predicted_true']:>15}")
            print(f"{'Future Tense / Foresight':<28}{fut_m['accuracy']*100:>10.2f}%{fut_m['precision']*100:>10.2f}%{fut_m['recall']*100:>10.2f}%{fut_m['f1']*100:>10.2f}%{fut_m['expected_true']:>12}{fut_m['predicted_true']:>15}")
            print("-" * 100 + "\n")
    
    # Why display 3-axis reasoning samples?
    # Inspecting sample comments with their contextual interpretation and 3-axis flags 
    # gives direct qualitative visibility into whether the LLM's CoT reasoning is accurate.
    evaluated_samples = [
        ev for ev in timeline_masked 
        if ev.get("type") == "listener" and "interpreted_comment" in ev
    ]
    if evaluated_samples:
        print("=== Sample 3-Axis Binary Evaluations & Contextual Interpretations ===")
        for sample in evaluated_samples[:8]:
            print(f"[{sample.get('name', 'User')}]: \"{sample.get('text', '')}\"")
            print(f"  └ 補完解釈: {sample.get('interpreted_comment')}")
            print(f"  └ 3軸判定: 主語(配信者)={sample.get('is_subject_streamer')}, 話題関連={sample.get('is_topic_relevant')}, 未来={sample.get('is_future')}")
            print(f"  └ 判定カテゴリ: {predicted_map.get(timeline_masked.index(sample))}\n")
        print("=" * 80 + "\n")

    # 6. Save Benchmark Results to JSON file
    # Why auto-save results to a structured JSON file?
    # Saving evaluation metrics, confusion matrix, and individual comment predictions (along with 3-axis reasoning)
    # allows offline visualization, diffing between prompt versions, and tracking accuracy evolution over time.
    if output_path is None:
        tag = "gemini" if "Gemini" in classifier.name else "rule_based"
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        scratch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch"))
        os.makedirs(scratch_dir, exist_ok=True)
        output_path = os.path.join(scratch_dir, f"benchmark_results_{tag}_{timestamp_str}.json")

    # Prepare structured details
    details = []
    for l_idx, (idx, exp_cat, pred_cat) in enumerate(zip(listener_indices, y_true, y_pred)):
        ev = timeline_masked[idx]
        entry = {
            "index": idx,
            "offset_seconds": ev.get("offset_seconds"),
            "username": ev.get("name"),
            "message": ev.get("text"),
            "expected_category": exp_cat.value,
            "predicted_category": pred_cat.value,
            "interpreted_comment": ev.get("interpreted_comment", ""),
            "is_subject_streamer": ev.get("is_subject_streamer", None),
            "is_topic_relevant": ev.get("is_topic_relevant", None),
            "is_future": ev.get("is_future", None),
        }
        details.append(entry)

    # Compute serializable confusion matrix
    matrix_dict = defaultdict(dict)
    for t_cat in categories:
        for p_cat in categories:
            matrix_dict[t_cat.value][p_cat.value] = sum(1 for t, p in zip(y_true, y_pred) if t == t_cat and p == p_cat)

    result_payload = {
        "timestamp": datetime.now().isoformat(),
        "classifier_name": classifier.name,
        "total_comments_evaluated": len(y_true),
        "elapsed_time_seconds": elapsed_time,
        "avg_time_per_comment_seconds": elapsed_time / len(y_true) if y_true else 0,
        "overall_accuracy": accuracy,
        "metrics_per_category": {cat.value: metrics[cat] for cat in categories},
        "confusion_matrix": matrix_dict,
        "three_axis_metrics": three_axis_metrics,
        "details": details
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Benchmark result file saved to: {output_path}\n")
    except Exception as e:
        print(f"[WARNING] Failed to save benchmark results to {output_path}: {e}\n")


def main():
    # Why reconfigure stdout to UTF-8?
    # Windows environments default standard output to CP932/Shift-JIS.
    # Reconfiguring stdout to UTF-8 avoids UnicodeEncodeError when printing emojis or Japanese characters.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="コメント分類器の精度・処理時間ベンチマーク")
    parser.add_argument("--input", default="tests/fixtures/reference_timeline.json", help="テストデータJSON")
    parser.add_argument("--classifier", choices=["gemini", "rule-based", "all"], default="all", help="実行する分類器")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Geminiモデル名")
    parser.add_argument("--slice-size", type=int, default=100, help="スライス分割サイズ (Gemini用)")
    parser.add_argument("--api-key", default=None, help="Gemini API Key")
    # Why add --limit option?
    # Full VOD timelines can contain thousands of events. Providing a --limit flag allows rapid
    # iterative testing of prompts and schema changes on a subset of comments without incurring long waits or costs.
    parser.add_argument("--limit", type=int, default=None, help="評価するイベント数の上限（先頭N件）")
    # Why add --output option?
    # Specifying a custom output path enables CI/CD or benchmark suites to write reports to predetermined artifact locations.
    parser.add_argument("--output", default=None, help="ベンチマーク結果JSONの保存先パス（省略時はscratch/配下に自動生成）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Test data file not found: {args.input}")
        print("Please run 'python tests/create_reference_from_db.py' first to generate test data.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    if args.limit:
        timeline_data = timeline_data[:args.limit]

    # Check if timeline_data contains listener comments with expected_category
    has_listeners = any(ev["type"] == "listener" for ev in timeline_data)
    has_expected = any(ev["type"] == "listener" and "expected_category" in ev for ev in timeline_data)

    if not has_listeners:
        print("[ERROR] Input timeline contains no listener comments.")
        sys.exit(1)
    if not has_expected:
        print("[ERROR] Input timeline has no 'expected_category' labels for evaluation.")
        sys.exit(1)

    classifiers = []
    if args.classifier in ("rule-based", "all"):
        classifiers.append(RuleBasedCommentClassifier())
    if args.classifier in ("gemini", "all"):
        classifiers.append(GeminiCommentClassifier(
            api_key=args.api_key,
            model_name=args.model,
            slice_size=args.slice_size
        ))

    for clsf in classifiers:
        run_benchmark_for_classifier(clsf, timeline_data, output_path=args.output)

if __name__ == "__main__":
    main()
