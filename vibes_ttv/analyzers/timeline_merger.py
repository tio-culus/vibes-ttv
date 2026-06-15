class TimelineMerger:
    # Why not just sort everything in a single list?
    # Sorting a merged list of dicts based on float timestamp is clean,
    # but we need to structure it with clear types (Streamer vs Listener)
    # to help Gemini easily distinguish who said what in the downstream prompt.
    
    def merge(self, whisper_segments: list[dict], chat_data: list[dict]) -> list[dict]:
        events = []
        
        # Add streamer speaking events
        for seg in whisper_segments:
            events.append({
                "type": "streamer",
                "offset_seconds": seg["start"],
                "end_seconds": seg["end"],
                "name": "Streamer",
                "text": seg["text"].strip()
            })
            
        # Add viewer chat events
        for chat in chat_data:
            events.append({
                "type": "listener",
                "offset_seconds": chat["offset_seconds"],
                "name": chat["username"],
                "text": chat["message"].strip()
            })
            
        # Sort by timestamp. For equal timestamps, streamer events go first.
        # Why prioritize streamer events on tie?
        # If a streamer speaks and a chat message arrives at the same second,
        # the chat is likely a reaction to the speech, so listing the speech first preserves the causal flow.
        events.sort(key=lambda x: (x["offset_seconds"], 0 if x["type"] == "streamer" else 1))
        return events

    def format_to_text(self, merged_events: list[dict], max_chats_per_minute: int = 30) -> str:
        # Why limit chats per minute in the text format?
        # In highly active streams, chat rates can exceed 100/min.
        # Injecting all chat messages into the Gemini prompt leads to high latency and redundant tokens.
        # Filtering or sampling ensures a balanced context size while keeping the core discussion flow.
        
        lines = []
        last_minute = -1
        chat_count_this_minute = 0
        
        for ev in merged_events:
            current_minute = int(ev["offset_seconds"] // 60)
            
            # Helper to format offset as HH:MM:SS
            h = int(ev["offset_seconds"] // 3600)
            m = int((ev["offset_seconds"] % 3600) // 60)
            s = int(ev["offset_seconds"] % 60)
            timestamp_str = f"{h:02d}:{m:02d}:{s:02d}"
            
            if ev["type"] == "streamer":
                lines.append(f"[{timestamp_str}] Streamer: {ev['text']}")
            else:
                # Reset chat limiter count on new minute
                if current_minute != last_minute:
                    last_minute = current_minute
                    chat_count_this_minute = 0
                    
                if chat_count_this_minute < max_chats_per_minute:
                    lines.append(f"[{timestamp_str}] {ev['name']}: {ev['text']}")
                    chat_count_this_minute += 1
                    
        return "\n".join(lines)
