import re
from typing import List, Dict, Any
from vibes_ttv.analyzers.classifier import CommentClassifier
from vibes_ttv.analyzers.comment_analyzer import CommentCategory

class RuleBasedCommentClassifier(CommentClassifier):
    # Why create a rule-based classifier?
    # It serves as a zero-cost, instant-execution baseline to compare LLM performance against.
    # Evaluating a rule-based approach helps determine if the financial and latency costs 
    # of using Gemini are justified by the accuracy gains.

    @property
    def name(self) -> str:
        return "RuleBasedCommentClassifier"

    def _is_reaction(self, msg: str) -> bool:
        # A lightweight local check replicating the simple reaction filter.
        # Why not reuse GeminiCommentClassifier's helper?
        # Keeping RuleBasedCommentClassifier fully self-contained prevents strict import dependencies 
        # on the Gemini client libraries.
        msg_lower = msg.strip().lower()
        if re.match(r'^[wｗ]+$', msg_lower):
            return True
        if msg_lower in ("草", "笑", "てぇてぇ", "やば", "やばい", "すご", "すごい", "さすが", "あり", "おつ", "お疲れ"):
            return True
        if re.match(r'^[8８\.\!\?\s\+]+$', msg_lower):
            return True
        if all(ord(char) >= 0x1f000 or (0x2600 <= ord(char) <= 0x27bf) for char in msg_lower if char.strip()):
            return True
        return False

    def classify(self, merged_events: List[Dict[str, Any]], progress_callback=None) -> Dict[int, CommentCategory]:
        classified = {}
        total = len(merged_events)
        
        for idx, ev in enumerate(merged_events):
            if progress_callback and idx % 50 == 0:
                progress_callback(f"🏷️ [RuleBase] 分類中... ({idx}/{total})", int((idx / total) * 100))
                
            if ev["type"] != "listener":
                continue
                
            msg = ev["text"].strip()
            
            # Rule 1: Simple reactions
            if self._is_reaction(msg):
                classified[idx] = CommentCategory.REACTION
                continue
                
            # Rule 2: Backseat (unsolicited suggestions) keywords
            # Why check specific imperative/action keywords?
            # Commands like "go right", "should do", or action directions are strong indicators 
            # of backseat gaming without streamer prompting.
            if any(word in msg for word in ("右", "左", "上", "下", "したほうが", "すればいい", "行こう", "とれる", "取れる")):
                classified[idx] = CommentCategory.BACKSEAT
                
            # Rule 3: Cross chat (mentioning other streamers)
            # Why check honorific suffix "さん" or other streamer terms?
            # Viewer comments comparing with other players ("A-san was doing X") usually carry these terms.
            elif any(word in msg for word in ("さん", "他の方", "別の人", "枠")):
                classified[idx] = CommentCategory.CROSS_CHAT
                
            # Rule 4: Blogpost (self-talk, out of context)
            # Why check first-person pronouns or daily-life verbs?
            # Comments starting with "I did", "my case", or unrelated daily topics ("ate ramen") 
            # generally match self-talk.
            elif any(word in msg for word in ("自分", "私", "俺", "僕", "食べた", "寝た", "ガチャ", "爆死", "眠い")):
                classified[idx] = CommentCategory.BLOGPOST
                
            # Rule 5: Spoiler warnings
            elif any(word in msg for word in ("ネタバレ", "この先", "後で")):
                classified[idx] = CommentCategory.SPOILER
                
            # Rule 6: Advice (answers to streamer questions)
            # Why check agreement terms?
            # When streamers ask "is this correct?", viewers reply with simple confirmation like "correct" or "yes".
            elif any(word in msg for word in ("合ってる", "合ってます", "そうだよ", "そのとおり")):
                classified[idx] = CommentCategory.ADVICE
                
            # Rule 7: Responses (conversational questions or general discussion comments)
            elif "?" in msg or "？" in msg or "どうして" in msg or "なんですか" in msg:
                classified[idx] = CommentCategory.RESPONSE
                
            # Fallback
            else:
                classified[idx] = CommentCategory.OTHER
                
        return classified
