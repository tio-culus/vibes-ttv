import re
import time
from google import genai
from google.genai import types
from pydantic import BaseModel

# Why use Pydantic models for structured output?
# It guarantees that Gemini's topic timeline matches the database fields directly,
# eliminating the need for complex string parsing, regex matching, or error-prone validation.

class TopicItem(BaseModel):
    start_offset_seconds: int
    end_offset_seconds: int
    category: str  # 'game', 'daily_news', 'past_stream', 'other'
    description: str
    is_high_context: bool

class TopicAnalysisResponse(BaseModel):
    topics: list[TopicItem]


class TopicAnalyzer:
    # Why not use legacy google-generativeai package?
    # The new google-genai SDK is the unified, official package that supports the newest models 
    # (gemini-3.5-flash) and native structured output typing.
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        
    def analyze_topics(self, timeline_text: str) -> list[dict]:
        # Why not segment the timeline into chunks?
        # Gemini 3.5 Flash supports a 1M token window, meaning it can ingest the entire stream 
        # (transcription + chat) at once. This allows the model to maintain complete global awareness 
        # of the stream's timeline and detect continuous topics much better than chunked analysis,
        # preventing edge cases where a single topic is split across artificial chunk boundaries.
        
        prompt = (
            "あなたは Twitch 配信の分析AIです。提供された「配信者の発言とチャットを結合したタイムラインテキスト」を元に、"
            "配信者がいつ、どのような話題について話していたかをすべて検出し、話題のリストを作成してください。\n\n"
            "【話題カテゴリ分類基準】\n"
            "- game: ゲームのプレイ内容、システム、プレイングについての話\n"
            "- daily_news: 日常の出来事、雑談、最近のニュース、個人的な活動などの話\n"
            "- past_stream: 過去の配信、過去の出来事、過去のリスナーとの約束などの話\n"
            "- other: 上記のいずれにも当てはまらないもの\n\n"
            "【ハイコンテクスト判定基準 (is_high_context)】\n"
            "- is_high_context = True: 過去の配信での出来事や、特定の時事ネタ（インサイドジョーク、内輪ネタ等）について言及しているが、"
            "初めて見る視聴者に対する「説明や補足がない」場合。つまり、初見の視聴者が話の文脈を理解しづらい話題。\n"
            "- is_high_context = False: 会話の内容がゲームプレイそのものに関するもの、または日常の出来事について十分な補足説明を交えて話している場合。\n\n"
            "配信全体をカバーするように、話題の開始時間（秒）と終了時間（秒）を指定して、話題リストを生成してください。\n\n"
            f"分析対象のタイムラインデータ:\n{timeline_text}"
        )
        
        try:
            # Why implement exponential backoff retry for Gemini API?
            # Under high load, the Gemini API may throw transient 503 Service Unavailable 
            # or 429 rate limit errors. Retrying with increasing delays (exponential backoff)
            # gives the server time to recover while ensuring the analysis is successfully processed
            # instead of falling back to empty topics.
            max_retries = 3
            backoff_factor = 2.0
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=TopicAnalysisResponse,
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
            
            parsed_res: TopicAnalysisResponse = response.parsed
            topics_list = []
            for t in parsed_res.topics:
                # Ensure the category string matches one of the allowed types
                cat = t.category.lower()
                if cat not in ("game", "daily_news", "past_stream", "other"):
                    cat = "other"
                topics_list.append({
                    "start_offset_seconds": t.start_offset_seconds,
                    "end_offset_seconds": t.end_offset_seconds,
                    "category": cat,
                    "description": t.description,
                    "is_high_context": t.is_high_context
                })
            return topics_list
        except Exception as e:
            # Why print and fallback instead of raising?
            # Topic analysis failure shouldn't stop the overall application flow.
            # Returning an empty list lets the UI render the rest of the stream metrics.
            print(f"Error analyzing topics using Gemini: {e}")
            return []
