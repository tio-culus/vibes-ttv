import json
import re
import time
from enum import Enum
from google import genai
from google.genai import types
from pydantic import BaseModel

# Why use Pydantic models for response_schema?
# Specifying response_schema using Pydantic ensures that Gemini returns a valid, typed JSON output.
# The SDK automatically parses this JSON into a Pydantic object accessible via `response.parsed`,
# eliminating fragile manual string parsing or json.loads logic.

# Why not inherit from str as well (str, Enum)?
# Inheriting from str ensures the enum values act as native strings,
# allowing seamless JSON serialization and preserving backward compatibility with downstream DB/UI models.
class CommentCategory(str, Enum):
    REACTION = 'reaction'
    QUESTION = 'question'
    INSIGHT = 'insight'
    INSTRUCTION = 'instruction'
    OTHER = 'other'

    @property
    def description(self) -> str:
        # Why not embed descriptions directly inside a property?
        # Dynamic properties keep the Enum values simple strings for serialization,
        # while centralizing prompt instruction text to a single source of truth.
        descriptions = {
            CommentCategory.REACTION: "感想、相槌、笑い（www）、「やばい」「すごい」などの感情表現、簡単なツッコミ、または配信に対する単純な反応コメント",
            CommentCategory.QUESTION: "配信者への質問（「今何したの？」「何て言った？」「今なんで〇〇したんですか？」など）",
            CommentCategory.INSIGHT: "配信状況やゲーム内容に対する考察、状況の要約、論理的な指摘、または比較的長文の文脈を必要とするコメント",
            CommentCategory.INSTRUCTION: "配信者に対するアドバイス、提案、指示、指示厨的発言、プレイ方針の提示（「右に進もう」「〇〇を装備して」など）",
            CommentCategory.OTHER: "上記のいずれにも当てはまらない日常雑談やその他無関係なコメント",
        }
        return descriptions[self]

class CommentClassification(BaseModel):
    message: str
    category: CommentCategory

class ListenerClassification(BaseModel):
    username: str
    classifications: list[CommentClassification]

class BatchClassificationResponse(BaseModel):
    results: list[ListenerClassification]

# Why define slice schemas?
# Structured output from Gemini during timeline slice analysis maps classifications directly 
# to a timeline index (line_id), guaranteeing robust mapping results without textual mismatch.
class LineClassification(BaseModel):
    line_id: str
    category: CommentCategory

class SliceClassificationResponse(BaseModel):
    results: list[LineClassification]


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

    def analyze_listeners(self, chat_data: list[dict], merged_events: list[dict], batch_size: int = 30, progress_callback=None) -> list[dict]:
        # 1. Initialize result stores
        classified_events = {} # global_idx -> category
        pre_classified = {} # global_idx -> category
        
        # 2. Slice processing loops (100 events per slice)
        slice_size = 100
        total_slices = (len(merged_events) + slice_size - 1) // slice_size
        
        for slice_idx, i in enumerate(range(0, len(merged_events), slice_size)):
            current_slice = slice_idx + 1
            if progress_callback:
                progress_val = 80 + int((slice_idx / total_slices) * 18)
                progress_callback(f"🧠 [5/5] リスナーコメント分析中... (スライス {current_slice}/{total_slices} を処理中)", progress_val)
                
            if slice_idx > 0:
                time.sleep(2.0)
                
            slice_events = merged_events[i:i+slice_size]
            
            # Format timeline text for this slice
            lines = []
            to_classify = []
            
            for idx, ev in enumerate(slice_events):
                global_idx = i + idx
                
                # Format timestamp
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
                    
                    # Local pre-classification check
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
            prompt = (
                "あなたはTwitchのライブ配信のチャットモデレーター兼分析者です。\n"
                "提示された【統合タイムライン】の文脈（配信者の発言や他のリスナーのコメントの流れ）を考慮して、\n"
                "指定された【分類依頼対象コメント】が以下のどのカテゴリに属するかを分類してください。\n\n"
                "【カテゴリ分類ルール】\n"
                f"{category_rules}\n\n"
                "【統合タイムライン】\n"
                f"{prompt_timeline}\n\n"
                "【分類依頼対象コメント】\n"
                f"{json.dumps(to_classify, ensure_ascii=False, indent=2)}\n\n"
                "それぞれの comment に対する line_id を維持し、各コメントのカテゴリ分類を出力してください。"
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
                            classified_events[g_idx] = line_res.category
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
                        
        # 3. Aggregate classifications per listener
        user_stats = {} # username -> counts & details
        for global_idx, ev in enumerate(merged_events):
            if ev["type"] == "listener":
                username = ev["name"]
                msg = ev["text"]
                offset = ev["offset_seconds"]
                raw_cat = classified_events.get(global_idx, CommentCategory.OTHER)
                try:
                    cat = CommentCategory(raw_cat)
                except ValueError:
                    cat = CommentCategory.OTHER
                    
                if username not in user_stats:
                    # Why not hardcode key values?
                    # Generating the initial dictionary keys dynamically from CommentCategory Enum
                    # prevents missing keys if new categories are added to the system in the future.
                    user_stats[username] = {cat.value: 0 for cat in CommentCategory}
                    user_stats[username]["details"] = []
                    
                cat_str = cat.value
                user_stats[username][cat_str] += 1
                
                # Why assign category to the merged_events dict in place?
                # It updates the caller's list of events, allowing timeline text formatters 
                # or database serializers to access classifications without rebuilding maps.
                ev["category"] = cat_str
                
                user_stats[username]["details"].append({
                    "message": msg,
                    "offset_seconds": offset,
                    "category": cat_str
                })
                
        # 4. Generate final stats and persona type
        final_results = []
        for username, counts in user_stats.items():
            # Why not sum dynamically?
            # Summing values dynamically from the CommentCategory Enum prevents mathematical 
            # errors and ensures the total is accurate if categories are added or modified.
            total = sum(counts[cat.value] for cat in CommentCategory)
                     
            # Persona determination (highest count, fallback hierarchy)
            # Why sort CommentCategory alphabetically (via sorted)?
            # Using sorted() provides a simple, consistent, and automatic priority order (alphabetical)
            # for tie-breakers without requiring custom mapping logic or order tables.
            persona_candidates = [(cat.value, counts[cat.value]) for cat in sorted(CommentCategory)]
            persona_candidates.sort(key=lambda x: x[1], reverse=True)
            best_persona = persona_candidates[0][0]
            
            sorted_details = sorted(counts["details"], key=lambda x: x["offset_seconds"])
            
            # Why not hardcode count keys?
            # Generating category count keys dynamically (e.g. 'reaction_comments_count') from the Enum
            # eliminates duplicate string boilerplate and prevents omissions when schema changes occur.
            final_results.append({
                "username": username,
                "total_comments": total,
                "persona_type": best_persona,
                "comment_details": sorted_details,
                **{f"{cat.value}_comments_count": counts[cat.value] for cat in CommentCategory}
            })
            
        return final_results

