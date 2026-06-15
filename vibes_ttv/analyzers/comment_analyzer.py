import json
import re
import time
from google import genai
from google.genai import types
from pydantic import BaseModel

# Why use Pydantic models for response_schema?
# Specifying response_schema using Pydantic ensures that Gemini returns a valid, typed JSON output.
# The SDK automatically parses this JSON into a Pydantic object accessible via `response.parsed`,
# eliminating fragile manual string parsing or json.loads logic.

class CommentClassification(BaseModel):
    message: str
    category: str # 'reaction', 'question', 'insight', 'instruction', 'other'

class ListenerClassification(BaseModel):
    username: str
    classifications: list[CommentClassification]

class BatchClassificationResponse(BaseModel):
    results: list[ListenerClassification]


class CommentAnalyzer:
    # Why not use legacy google-generativeai package?
    # The new google-genai SDK is the unified, official package that supports the newest models 
    # (gemini-3.5-flash) and native structured output typing.
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        
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

    def analyze_listeners(self, chat_data: list[dict], batch_size: int = 30, progress_callback=None) -> list[dict]:
        # Group chats by username
        # Why capture message and offset_seconds together?
        # Preserving the comment text and offset_seconds as dictionary items allows us to reconstruct 
        # the precise classification details timeline for the UI, keeping comments aligned with stream moments.
        user_chats = {}
        for chat in chat_data:
            username = chat["username"]
            if username not in user_chats:
                user_chats[username] = []
            user_chats[username].append({
                "message": chat["message"],
                "offset_seconds": chat["offset_seconds"]
            })
            
        final_results = []
        to_analyze = [] # List of users queued for Gemini analysis
        
        for username, chat_items in user_chats.items():
            pre_classified_details = []
            remaining_items = []
            
            for item in chat_items:
                msg = item["message"]
                offset = item["offset_seconds"]
                if self._is_simple_reaction(msg):
                    pre_classified_details.append({
                        "message": msg,
                        "offset_seconds": offset,
                        "category": "reaction"
                    })
                else:
                    remaining_items.append(item)
                    
            if not remaining_items:
                # All messages were simple reactions. Avoid calling Gemini entirely.
                final_results.append({
                    "username": username,
                    "total_comments": len(chat_items),
                    "reaction_comments_count": len(chat_items),
                    "question_comments_count": 0,
                    "insight_comments_count": 0,
                    "instruction_comments_count": 0,
                    "other_comments_count": 0,
                    "persona_type": "reaction",
                    "comment_details": pre_classified_details
                })
            else:
                # Queue for batch LLM analysis
                to_analyze.append({
                    "username": username,
                    "items": remaining_items,
                    "pre_classified": pre_classified_details,
                    "total_original": len(chat_items)
                })
                
        # Process in batches using Gemini API
        total_batches = (len(to_analyze) + batch_size - 1) // batch_size
        for idx, i in enumerate(range(0, len(to_analyze), batch_size)):
            current_batch = idx + 1
            if progress_callback:
                progress_val = 80 + int((idx / total_batches) * 18)
                progress_callback(f"🧠 [5/5] リスナーコメント分析中... (バッチ {current_batch}/{total_batches} を処理中)", progress_val)

            if i > 0:
                time.sleep(2.0)
            batch = to_analyze[i:i+batch_size]
            prompt_data = []
            for item in batch:
                prompt_data.append({
                    "username": item["username"],
                    "comments": [c["message"] for c in item["items"]]
                })
                
            prompt = (
                "以下のTwitchリスナーたちのコメント内容を分析し、それぞれのコメントが以下のどのカテゴリに属するかを分類してください。\n"
                "【カテゴリ分類ルール】\n"
                "- reaction: 「www」「やばい」「すごい」などの感想、相槌、感情表現、リアクション系コメント\n"
                "- question: 「今なんで〇〇したんですか？」「それ何ですか？」などの質問系コメント\n"
                "- insight: 「これは〇〇かもしれない」「おそらく〇〇だからこうなった」などの考察系コメント（比較的長文や論理的なもの）\n"
                "- instruction: 「〇〇しよう」「〇〇するのはどうですか？」などの指示、アドバイス、提案系コメント\n"
                "- other: 上記のいずれにも当てはまらない日常雑談やその他コメント\n\n"
                f"分析対象のデータ:\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
            )
            
            try:
                max_retries = 3
                backoff_factor = 2.0
                response = None
                
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model="gemini-3.1-flash-lite",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=BatchClassificationResponse,
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
                
                parsed_res: BatchClassificationResponse = response.parsed
                results_dict = {res.username: res for res in parsed_res.results}
                
                for item in batch:
                    username = item["username"]
                    res = results_dict.get(username)
                    
                    # Map the categorized responses by comment text
                    classified_map = {}
                    if res:
                        for c_res in res.classifications:
                            classified_map[c_res.message] = c_res.category
                            
                    gemini_classified_details = []
                    r_c = len(item["pre_classified"])
                    q_c, in_c, ins_c, o_c = 0, 0, 0, 0
                    
                    for c_item in item["items"]:
                        msg = c_item["message"]
                        offset = c_item["offset_seconds"]
                        
                        # Find classification or default to other
                        cat = classified_map.get(msg, "other")
                        if cat not in ("reaction", "question", "insight", "instruction", "other"):
                            cat = "other"
                            
                        if cat == "reaction":
                            r_c += 1
                        elif cat == "question":
                            q_c += 1
                        elif cat == "insight":
                            in_c += 1
                        elif cat == "instruction":
                            ins_c += 1
                        else:
                            o_c += 1
                            
                        gemini_classified_details.append({
                            "message": msg,
                            "offset_seconds": offset,
                            "category": cat
                        })
                        
                    # Combine pre-classified (regex) and gemini-classified comments
                    all_details = item["pre_classified"] + gemini_classified_details
                    # Why sort by offset_seconds?
                    # Stream comments are sent at different times. Sorting chronologically ensures 
                    # they display in the correct timeline order in the detailed UI view.
                    all_details.sort(key=lambda x: x["offset_seconds"])
                    
                    counts = {
                        "reaction": r_c,
                        "question": q_c,
                        "insight": in_c,
                        "instruction": ins_c,
                        "other": o_c
                    }
                    persona = max(counts, key=counts.get)
                    
                    final_results.append({
                        "username": username,
                        "total_comments": item["total_original"],
                        "reaction_comments_count": r_c,
                        "question_comments_count": q_c,
                        "insight_comments_count": in_c,
                        "instruction_comments_count": ins_c,
                        "other_comments_count": o_c,
                        "persona_type": persona,
                        "comment_details": all_details
                    })
            except Exception as e:
                print(f"Error calling Gemini in batch: {e}")
                for item in batch:
                    fallback_details = []
                    r_c = len(item["pre_classified"])
                    o_c = 0
                    for c_item in item["items"]:
                        fallback_details.append({
                            "message": c_item["message"],
                            "offset_seconds": c_item["offset_seconds"],
                            "category": "other"
                        })
                        o_c += 1
                        
                    all_details = item["pre_classified"] + fallback_details
                    all_details.sort(key=lambda x: x["offset_seconds"])
                    
                    final_results.append({
                        "username": username,
                        "total_comments": item["total_original"],
                        "reaction_comments_count": r_c,
                        "question_comments_count": 0,
                        "insight_comments_count": 0,
                        "instruction_comments_count": 0,
                        "other_comments_count": o_c,
                        "persona_type": "reaction" if r_c > 0 else "other",
                        "comment_details": all_details
                    })
                    
        return final_results
