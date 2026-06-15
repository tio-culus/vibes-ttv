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

class ListenerClassification(BaseModel):
    username: str
    reaction_count: int
    question_count: int
    insight_count: int
    instruction_count: int
    other_count: int

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
        user_chats = {}
        for chat in chat_data:
            username = chat["username"]
            if username not in user_chats:
                user_chats[username] = []
            user_chats[username].append(chat["message"])
            
        final_results = []
        to_analyze = [] # List of (username, comments) for Gemini
        
        for username, messages in user_chats.items():
            reaction_cnt = 0
            remaining_messages = []
            
            for msg in messages:
                if self._is_simple_reaction(msg):
                    reaction_cnt += 1
                else:
                    remaining_messages.append(msg)
                    
            if not remaining_messages:
                # All messages were simple reactions. Avoid calling Gemini entirely.
                final_results.append({
                    "username": username,
                    "total_comments": len(messages),
                    "reaction_comments_count": reaction_cnt,
                    "question_comments_count": 0,
                    "insight_comments_count": 0,
                    "instruction_comments_count": 0,
                    "other_comments_count": 0,
                    "persona_type": "reaction"
                })
            else:
                # Queue for batch LLM analysis
                to_analyze.append({
                    "username": username,
                    "messages": remaining_messages,
                    "pre_reaction": reaction_cnt,
                    "total_original": len(messages)
                })
                
        # Process in batches using Gemini API
        total_batches = (len(to_analyze) + batch_size - 1) // batch_size
        for idx, i in enumerate(range(0, len(to_analyze), batch_size)):
            current_batch = idx + 1
            if progress_callback:
                # Why report progress per batch?
                # Gemini batch processing takes several seconds per batch due to built-in RPD sleeping.
                # Keeping the user informed of the current batch index prevents them from thinking the UI hung.
                progress_val = 80 + int((idx / total_batches) * 18)
                progress_callback(f"🧠 [5/5] リスナーコメント分析中... (バッチ {current_batch}/{total_batches} を処理中)", progress_val)

            if i > 0:
                # Why sleep between batches?
                # To prevent triggering Requests Per Minute (RPM) limits on the Gemini API free tier.
                time.sleep(2.0)
            batch = to_analyze[i:i+batch_size]
            prompt_data = []
            for item in batch:
                prompt_data.append({
                    "username": item["username"],
                    "comments": item["messages"]
                })
                
            prompt = (
                "以下のTwitchリスナーたちのコメント内容を分析し、それぞれのコメントが以下のどのカテゴリに属するかを分類・カウントしてください。\n"
                "【カテゴリ分類ルール】\n"
                "- reaction: 「www」「やばい」「すごい」などの感想、相槌、感情表現、リアクション系コメント\n"
                "- question: 「今なんで〇〇したんですか？」「それ何ですか？」などの質問系コメント\n"
                "- insight: 「これは〇〇かもしれない」「おそらく〇〇だからこうなった」などの考察系コメント（比較的長文や論理的なもの）\n"
                "- instruction: 「〇〇しよう」「〇〇するのはどうですか？」などの指示、アドバイス、提案系コメント\n"
                "- other: 上記のいずれにも当てはまらない日常雑談やその他コメント\n\n"
                f"分析対象のデータ:\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
            )
            
            try:
                # Why implement exponential backoff retry for Gemini API?
                # Under high load, the Gemini API may throw transient 503 Service Unavailable 
                # or 429 rate limit errors. Retrying with increasing delays (exponential backoff)
                # gives the server time to recover while ensuring the batch data is successfully processed
                # instead of falling back to default categorization.
                max_retries = 3
                backoff_factor = 2.0
                response = None
                
                for attempt in range(max_retries):
                    try:
                        # Why gemini-3.1-flash-lite instead of gemini-3.5-flash?
                        # Because listener analysis is run in batches for each active chatter, it generates multiple API calls.
                        # Using gemini-3.1-flash-lite avoids hitting the strict 20 RPD free-tier limits of gemini-3.5-flash,
                        # and allows tracking rate usage on the Google AI Studio dashboard, unlike gemini-2.0-flash-lite.
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
                            # Why parse retry delay from error message?
                            # When Google API returns a 429 quota exhaustion, it often specifies the wait duration
                            # (e.g. "Please retry in 44s"). Parsing and respecting this duration prevents 
                            # consecutive retry attempts from failing immediately.
                            retry_match = re.search(r"Please retry in (\d+\.?\d*)s", err_msg)
                            if retry_match:
                                sleep_time = float(retry_match.group(1)) + 1.0  # Add 1s buffer
                                print(f"Gemini API rate limited. Sleeping for {sleep_time}s as requested by API...")
                            else:
                                sleep_time = (backoff_factor ** attempt) * 5.0  # Default to longer delay (5s, 10s...)
                                
                            print(f"Gemini API returned temporary error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                        else:
                            raise e
                
                # The response is parsed into the BatchClassificationResponse object automatically
                parsed_res: BatchClassificationResponse = response.parsed
                results_dict = {res.username: res for res in parsed_res.results}
                
                for item in batch:
                    username = item["username"]
                    res = results_dict.get(username)
                    
                    if res:
                        r_c = item["pre_reaction"] + res.reaction_count
                        q_c = res.question_count
                        in_c = res.insight_count
                        ins_c = res.instruction_count
                        o_c = res.other_count
                    else:
                        # Fallback if Gemini missed this user in the response list
                        r_c = item["pre_reaction"]
                        q_c, in_c, ins_c = 0, 0, 0
                        o_c = len(item["messages"])
                        
                    # Determine persona type by getting the category with the highest count
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
                        "persona_type": persona
                    })
            except Exception as e:
                # Why print and fallback instead of raising?
                # A single transient API error shouldn't stop the overall application flow.
                # We fall back to a safe default classification (other/reaction) for this batch.
                print(f"Error calling Gemini in batch: {e}")
                for item in batch:
                    final_results.append({
                        "username": item["username"],
                        "total_comments": item["total_original"],
                        "reaction_comments_count": item["pre_reaction"],
                        "question_comments_count": 0,
                        "insight_comments_count": 0,
                        "instruction_comments_count": 0,
                        "other_comments_count": len(item["messages"]),
                        "persona_type": "reaction" if item["pre_reaction"] > 0 else "other"
                    })
                    
        return final_results
