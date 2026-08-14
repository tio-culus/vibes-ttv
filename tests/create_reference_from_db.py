import os
import sys
import json
import argparse

# Why add parent dir to sys.path?
# Adding the project root directory directly to sys.path guarantees that the script can resolve 
# internal imports (like vibes_ttv) when executed directly from the terminal.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vibes_ttv.database.db_manager import DBManager
from vibes_ttv.database.models import VOD

def main():
    parser = argparse.ArgumentParser(description="DBからテスト用の統合タイムラインデータを抽出してJSON保存します。")
    parser.add_argument("--db", default="sqlite:///vibes_ttv.db", help="DB接続URI")
    parser.add_argument("--limit", type=int, default=None, help="抽出するリスナーコメント数の上限（デフォルトは無制限＝全件）")
    parser.add_argument("--output", default="tests/fixtures/reference_timeline.json", help="出力先JSONファイルパス")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    db = DBManager(args.db)
    session = db.get_session()
    
    # Fetch first VOD that has an analyzed merged timeline
    # Why filter by merged_timeline_json.isnot(None)?
    # Unanalyzed VODs or raw downloads won't have classification context inside the DB.
    vod = session.query(VOD).filter(VOD.merged_timeline_json.isnot(None)).first()
    
    if not vod:
        # Why generate dummy data fallback?
        # If the local database is fresh, empty, or unanalyzed, running this script would crash 
        # or output nothing. Generating a dummy timeline containing all 8 CommentCategories 
        # allows running and validating the benchmark tool immediately without requiring pre-analysis.
        print("[WARNING] DBから分析済みのVOD（merged_timeline_json が存在するデータ）が見つかりませんでした。")
        print("代わりにすべてのカテゴリを含んだダミーデータを使用して reference_timeline.json を作成します。")
        
        dummy_timeline = [
            {"type": "streamer", "offset_seconds": 10.0, "name": "Streamer", "text": "こんにちは！今日は新作アクションゲームを遊んでいきます。"},
            {"type": "listener", "offset_seconds": 12.0, "name": "UserA", "text": "こんにちは！楽しみ！", "expected_category": "reaction"},
            {"type": "listener", "offset_seconds": 15.0, "name": "UserB", "text": "前作も面白かったし期待", "expected_category": "response"},
            {"type": "streamer", "offset_seconds": 20.0, "name": "Streamer", "text": "この武器、どうやって使うんだろう？"},
            {"type": "listener", "offset_seconds": 25.0, "name": "UserC", "text": "メニュー画面から装備して、R2ボタンで攻撃できるよ", "expected_category": "backseat"},
            {"type": "streamer", "offset_seconds": 30.0, "name": "Streamer", "text": "あ、R2ボタンですね。ありがとうございます！"},
            {"type": "listener", "offset_seconds": 32.0, "name": "UserA", "text": "お、できたできた", "expected_category": "reaction"},
            {"type": "listener", "offset_seconds": 35.0, "name": "UserD", "text": "ちなみにさっきのガチャでレア装備引けた？", "expected_category": "blogpost"},
            {"type": "listener", "offset_seconds": 40.0, "name": "UserE", "text": "この先ネタバレ注意。ボスが隠れてるよ", "expected_category": "backseat"},
            {"type": "streamer", "offset_seconds": 45.0, "name": "Streamer", "text": "えっ、ボスがいるんですか？気をつけます。"},
            {"type": "listener", "offset_seconds": 50.0, "name": "UserB", "text": "合ってる、その武器で大丈夫", "expected_category": "advice"},
            {"type": "listener", "offset_seconds": 55.0, "name": "UserC", "text": "〇〇さんは違う武器使ってたなー", "expected_category": "other"},
            {"type": "listener", "offset_seconds": 60.0, "name": "UserA", "text": "今日のご飯はカレーです", "expected_category": "blogpost"},
            {"type": "streamer", "offset_seconds": 65.0, "name": "Streamer", "text": "何でそんなこと言うの？"},
            {"type": "listener", "offset_seconds": 70.0, "name": "UserA", "text": "意味不明な発言ですみません", "expected_category": "other"}
        ]
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dummy_timeline, f, ensure_ascii=False, indent=2)
        print(f"ダミーデータを {args.output} に書き込みました。")
        db.remove_session()
        return

    try:
        timeline = json.loads(vod.merged_timeline_json)
    except Exception as e:
        print(f"[ERROR] VODの統合タイムラインJSONのパースに失敗しました: {e}")
        db.remove_session()
        return

    print(f"DBからVOD『{vod.title}』 (VOD ID: {vod.vod_id}) を取得しました。")
    print(f"タイムラインの総イベント数: {len(timeline)}")

    # Extract segment (all comments by default, or limited if args.limit is provided)
    extracted_events = []
    listener_count = 0
    
    for ev in timeline:
        # Why copy dict?
        # Modifying the direct objects from DB cache would alter session state.
        # Cloning the dictionary keeps database cache clean.
        ev_copy = dict(ev)
        if ev_copy["type"] == "listener":
            listener_count += 1
            # Relabel DB's auto-generated 'category' key to 'expected_category' for benchmarks
            cat = ev_copy.get("category", "other")
            ev_copy["expected_category"] = cat
            if "category" in ev_copy:
                del ev_copy["category"]

            # Why map 3-axis flags to expected_*?
            # Preserving 3-axis contextual classifications in exported references allows 
            # future benchmarks to evaluate accuracy on individual reasoning axes.
            if "interpreted_comment" in ev_copy:
                ev_copy["expected_interpreted_comment"] = ev_copy.pop("interpreted_comment")
            if "is_subject_streamer" in ev_copy:
                ev_copy["expected_is_subject_streamer"] = ev_copy.pop("is_subject_streamer")
            if "is_topic_relevant" in ev_copy:
                ev_copy["expected_is_topic_relevant"] = ev_copy.pop("is_topic_relevant")
            if "is_future" in ev_copy:
                ev_copy["expected_is_future"] = ev_copy.pop("is_future")
                
        extracted_events.append(ev_copy)
        
        if args.limit is not None and listener_count >= args.limit:
            break

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(extracted_events, f, ensure_ascii=False, indent=2)

    limit_str = "全件" if args.limit is None else f"{listener_count} 件"
    print(f"[{limit_str}] のタイムラインデータを {args.output} にエクスポートしました（内リスナーコメント: {listener_count}件）。")
    db.remove_session()

if __name__ == "__main__":
    main()
