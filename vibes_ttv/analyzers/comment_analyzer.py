import json
import re
import time
from enum import Enum
from google import genai
from google.genai import types
from pydantic import BaseModel
from vibes_ttv.analyzers.classifier import (
    CommentClassifier,
    CommentCategory,
    CommentClassification,
    ListenerClassification,
    BatchClassificationResponse,
    LineClassification,
    SliceClassificationResponse,
)
from vibes_ttv.analyzers.gemini_classifier import GeminiCommentClassifier


class CommentAnalyzer:
    # Why not use legacy google-generativeai package?
    # The new google-genai SDK is the unified, official package that supports the newest models 
    # (gemini-3.5-flash) and native structured output typing.
    def __init__(self, classifier: CommentClassifier = None, api_key: str = None):
        # Why support injecting classifier?
        # Injecting a CommentClassifier implementation allows swappable backend strategies 
        # (e.g., using Gemini vs OpenAI vs local rules) without changing downstream database/UI code.
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = GeminiCommentClassifier(api_key=api_key)

    def _is_simple_reaction(self, message: str) -> bool:
        # Why delegate to classifier?
        # Preserves backward compatibility for external callers or unit tests that call this method directly.
        if hasattr(self.classifier, '_is_simple_reaction'):
            return self.classifier._is_simple_reaction(message)
        # Why fallback to regex?
        # If the injected classifier is not Gemini-based (e.g., RuleBasedCommentClassifier),
        # we replicate the baseline reaction check using simple checks to prevent AttributeError.
        msg = message.strip().lower()
        if re.match(r'^[wｗ]+$', msg):
            return True
        if msg in ("草", "笑", "てぇてぇ", "やば", "やばい", "すご", "すごい", "さすが", "あり", "おつ", "お疲れ"):
            return True
        return False

    def analyze_listeners(self, merged_events: list[dict], slice_size: int = 100, progress_callback=None) -> list[dict]:
        # Why override slice_size if it's GeminiCommentClassifier?
        # Keeps compatibility with older callers of analyze_listeners who configure slice_size on the fly.
        if hasattr(self.classifier, 'slice_size') and slice_size != 100:
            self.classifier.slice_size = slice_size

        # 1. Run classifier
        classified_events = self.classifier.classify(merged_events, progress_callback=progress_callback)
                        
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
            
            # Why not flatten category counts with suffix keys?
            # Keeping counts grouped under a single nested "category_counts" dictionary 
            # keeps the interface clean and aligns with the database serialization format.
            final_results.append({
                "username": username,
                "total_comments": total,
                "persona_type": best_persona,
                "comment_details": sorted_details,
                "category_counts": {cat.value: counts[cat.value] for cat in CommentCategory}
            })
            
        return final_results

