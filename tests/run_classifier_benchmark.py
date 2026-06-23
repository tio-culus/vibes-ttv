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

def run_benchmark_for_classifier(classifier, timeline_data: List[Dict[str, Any]]):
    print("=" * 80)
    print(f"Running Benchmark: {classifier.name}")
    print("=" * 80)

    # 1. Mask expected_category to simulate clean input
    timeline_masked = []
    y_true = []
    listener_indices = []

    for idx, ev in enumerate(timeline_data):
        ev_copy = dict(ev)
        if ev_copy["type"] == "listener":
            raw_exp = ev_copy.pop("expected_category", "other")
            try:
                y_true.append(CommentCategory(raw_exp))
            except ValueError:
                y_true.append(CommentCategory.OTHER)
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
    for idx in listener_indices:
        # Fallback to OTHER if comment was missed or unclassified
        pred = predicted_map.get(idx, CommentCategory.OTHER)
        y_pred.append(pred)

    # 4. Compute Metrics
    categories = list(CommentCategory)
    accuracy, metrics = calculate_metrics(y_true, y_pred, categories)

    # 5. Output Results
    print(f"Results for {classifier.name}:")
    print(f"  Total Comments Evaluated: {len(y_true)}")
    print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"  Avg Time per Comment: {elapsed_time / len(y_true) if y_true else 0:.4f} seconds")
    print(f"  Overall Accuracy: {accuracy * 100:.2f}%\n")

    print(f"{'Category':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Expected Count':<16}{'Predicted Count'}")
    print("-" * 80)
    for cat in categories:
        m = metrics[cat]
        print(f"{cat.value:<15}{m['precision']*100:>10.2f}%{m['recall']*100:>10.2f}%{m['f1']*100:>10.2f}%{m['expected']:>14}{m['predicted']:>17}")
    print("-" * 80)

    print_confusion_matrix(y_true, y_pred, categories)
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="コメント分類器の精度・処理時間ベンチマーク")
    parser.add_argument("--input", default="tests/fixtures/reference_timeline.json", help="テストデータJSON")
    parser.add_argument("--classifier", choices=["gemini", "rule-based", "all"], default="all", help="実行する分類器")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Geminiモデル名")
    parser.add_argument("--slice-size", type=int, default=100, help="スライス分割サイズ (Gemini用)")
    parser.add_argument("--api-key", default=None, help="Gemini API Key")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Test data file not found: {args.input}")
        print("Please run 'python tests/create_reference_from_db.py' first to generate test data.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

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
        run_benchmark_for_classifier(clsf, timeline_data)

if __name__ == "__main__":
    main()
