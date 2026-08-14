import json
import re
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types

from vibes_ttv.analyzers.classifier import CommentClassifier
from vibes_ttv.analyzers.comment_analyzer import CommentCategory, SliceClassificationResponse

class GeminiCommentClassifier(CommentClassifier):
    # Why extend CommentClassifier?
    # This ensures that GeminiCommentClassifier matches the required interface,
    # allowing it to be interchangeably used inside CommentAnalyzer or benchmark runner.
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.1-flash-lite", slice_size: int = 100):
        # Why not use legacy google-generativeai package?
        # The new google-genai SDK is the unified, official package that supports the newest models 
        # (gemini-3.1-flash-lite) and native structured output typing.
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model_name = model_name
        self.slice_size = slice_size

    @property
    def name(self) -> str:
        # Why return configuration along with model name?
        # Benchmark tool will show this name output, so having slice_size inside the name helps 
        # differentiate between runs with different window slicing (e.g. slice_size=50 vs 100).
        return f"GeminiCommentClassifier(model={self.model_name}, slice_size={self.slice_size})"

    def _is_simple_reaction(self, message: str) -> bool:
        # Why run regex and rule-based pre-classification?
        # A significant portion of live chat consists of short reactions (e.g., "www", "草", emojis).
        # Sending these thousands of trivial chats to Gemini is highly inefficient, expensive, and slow.
        # Filtering them locally reduces LLM workload and improves analysis speed.
        msg = message.strip().lower()
        
        # Match only 'w', 'ｗ', '草', '笑'
        if re.match(r'^[wｗ]+$', msg):
            return True
        if msg in ("草", "笑", "てぇてぇ", "やば", "やばい", "すご", "すごい", "さすが", "あり", "おつ", "お疲れ"):
            return True
        # Match numbers/exclamations only (e.g. 8888 for clapping, !!!)
        if re.match(r'^[8８\.\!\?\s\+]+$', msg):
            return True
        # Simple emojis only
        # Why check 0x1f000 and 0x2600 ranges instead of using 'emoji' library?
        # Using a lightweight Unicode range check avoids adding extra external package dependencies (like 'emoji')
        # while successfully filtering 99% of common reaction emojis like 👍, ❤️, 👏, and standard smileys.
        if all(ord(char) >= 0x1f000 or (0x2600 <= ord(char) <= 0x27bf) for char in msg if char.strip()):
            return True
        return False

    def classify(self, merged_events: List[Dict[str, Any]], progress_callback=None) -> Dict[int, CommentCategory]:
        # Why return a flat dictionary of global_idx -> category mapping?
        # Returning a lightweight map keeps the interface clean and decoupled from db-specific model objects
        # or persona aggregation logic, which remains the duty of CommentAnalyzer.
        classified_events = {}
        pre_classified = {}
        
        total_slices = (len(merged_events) + self.slice_size - 1) // self.slice_size
        
        for slice_idx, i in enumerate(range(0, len(merged_events), self.slice_size)):
            current_slice = slice_idx + 1
            if progress_callback:
                progress_val = 80 + int((slice_idx / total_slices) * 18)
                progress_callback(f"🧠 [5/5] リスナーコメント分析中... (スライス {current_slice}/{total_slices} を処理中)", progress_val)
                
            if slice_idx > 0:
                # Why sleep between API calls?
                # Rate limits for model endpoints (RPM limits) can be triggered if slices are sent 
                # immediately back-to-back. A short pause throttles throughput safely.
                time.sleep(2.0)
                
            slice_events = merged_events[i:i+self.slice_size]
            
            lines = []
            to_classify = []
            
            for idx, ev in enumerate(slice_events):
                global_idx = i + idx
                
                h = int(ev["offset_seconds"] // 3600)
                m = int((ev["offset_seconds"] % 3600) // 60)
                s = int(ev["offset_seconds"] % 60)
                timestamp_str = f"{h:02d}:{m:02d}:{s:02d}"
                
                # Format timeline line
                # Why use L{idx} prefix?
                # Adding line identifiers allows the LLM to return exact classifications mapped 
                # to these line IDs, avoiding unstable text matches or list size mismatches.
                if ev["type"] == "streamer":
                    lines.append(f"(L{global_idx}) [{timestamp_str}] Streamer: {ev['text']}")
                else:
                    lines.append(f"(L{global_idx}) [{timestamp_str}] {ev['name']}: {ev['text']}")
                    
                    msg = ev["text"]
                    if self._is_simple_reaction(msg):
                        pre_classified[global_idx] = CommentCategory.REACTION
                    else:
                        to_classify.append({
                            "line_id": f"L{global_idx}",
                            "comment": msg
                        })
                        
            # Apply pre-classified reactions
            for g_idx, cat in pre_classified.items():
                if i <= g_idx < i + len(slice_events):
                    classified_events[g_idx] = cat
                    
            if not to_classify:
                # No complex comments in this slice. Skip Gemini API.
                continue
                
            # Call Gemini with slice context
            prompt_timeline = "\n".join(lines)
            category_rules = "\n".join(f"- {cat.value}: {cat.description}" for cat in CommentCategory)
            # Why prompt for contextual interpretation and 3-axis binary evaluation explicitly?
            # Instructing the model to perform contextual completion and evaluate the 3 clear binary axes 
            # (subject, topic relevance, future tense) acts as chain-of-thought grounding, preventing
            # misclassification of ambiguous shorthand comments typical in live streams.
            prompt = (
                "あなたはライブ配信の専門アナリストです。ライブ配信がどのようなコメントで構成されているかを高精度に分析するために、リスナーコメントを分類します。\n"
                "提示された【統合タイムライン】の文脈（配信者の発言や他のリスナーのコメントの流れ）を考慮して、\n"
                "指定された【分類依頼対象コメント】それぞれについて、以下のステップに従って構造化データを出力してください。\n"
                "※配信ラグ等の影響により、Streamerの発言とListenerの発言が1分ほど前後することを考慮して文脈を捉えてください。\n\n"
                "【分析・判定ステップ】\n"
                "1. interpreted_comment（文脈補完）: 前後の配信者の発言やチャットの流れを踏まえ、コメント内で省略された主語・目的語・状況を補完した解釈を記述してください。\n"
                "2. is_subject_streamer（主語判定）: コメントの主語は「配信者（または配信者が操作するキャラクター/ゲーム状況）」ですか？（true: 配信者 / false: それ以外[リスナー自身、第三者など]）\n"
                "3. is_topic_relevant（話題関連性判定）: コメントの話題(目的語)は「配信中の話題・進行中の出来事」と関係がありますか？（true: 関係あり / false: 関係なし）\n"
                "4. is_future（未来判定）: コメントの内容は「未来の話（先の展開のネタバレ、未起きの指示・助言・先回りしたアドバイス等）」ですか？（true: 未来の話 / false: それ以外[現在・過去の感想、リアクション、返答等]）\n"
                "5. category（最終カテゴリ）: 上記の文脈補完と3軸2値判定を踏まえ、以下の【カテゴリ分類ルール】から最も適切なカテゴリを選択してください。\n\n"
                "【カテゴリ分類ルール】\n"
                f"{category_rules}\n\n"
                "【統合タイムライン】\n"
                f"{prompt_timeline}\n\n"
                "【分類依頼対象コメント】\n"
                f"{json.dumps(to_classify, ensure_ascii=False, indent=2)}\n\n"
                "それぞれの comment に対する line_id を維持し、上記ステップに従って結果を出力してください。"
            )
            
            try:
                max_retries = 3
                backoff_factor = 2.0
                response = None
                
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SliceClassificationResponse,
                            )
                        )
                        break
                    except Exception as e:
                        err_msg = str(e)
                        is_transient = ("503" in err_msg or "429" in err_msg or "unavailable" in err_msg.lower() or "resource exhausted" in err_msg.lower())
                        if is_transient and attempt < max_retries - 1:
                            retry_match = re.search(r"Please retry in (\d+\.?\d*)s", err_msg)
                            if retry_match:
                                sleep_time = float(retry_match.group(1)) + 1.0
                            else:
                                sleep_time = (backoff_factor ** attempt) * 5.0
                            time.sleep(sleep_time)
                        else:
                            raise e
                            
                parsed_res: SliceClassificationResponse = response.parsed
                if parsed_res and parsed_res.results:
                    for line_res in parsed_res.results:
                        try:
                            g_idx = int(line_res.line_id[1:])
                            # Convert string category to CommentCategory Enum
                            cat = CommentCategory(line_res.category)
                            classified_events[g_idx] = cat
                            
                            # Why store auxiliary 3-axis evaluation metadata in merged_events?
                            # Storing interpreted_comment and 3-axis flags directly in the event dictionary
                            # allows downstream reporting and debugging tools to inspect the LLM's reasoning
                            # without breaking callers expecting a simple category map.
                            if 0 <= g_idx < len(merged_events):
                                merged_events[g_idx]["interpreted_comment"] = line_res.interpreted_comment
                                merged_events[g_idx]["is_subject_streamer"] = line_res.is_subject_streamer
                                merged_events[g_idx]["is_topic_relevant"] = line_res.is_topic_relevant
                                merged_events[g_idx]["is_future"] = line_res.is_future
                        except Exception:
                            pass
                            
            except Exception as e:
                print(f"Error calling Gemini in slice: {e}")
                # Fallback to 'other' for Gemini failures in this slice
                for item in to_classify:
                    try:
                        g_idx = int(item["line_id"][1:])
                        classified_events[g_idx] = CommentCategory.OTHER
                    except Exception:
                        pass
                        
        return classified_events
