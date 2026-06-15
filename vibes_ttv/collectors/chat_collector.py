import json
import re
import time
from datetime import datetime
import requests

class ChatCollector:
    # Why use 'kd1unb4b3q4t58fwlpcbzcbnm76a8fp' instead of 'kimne7...'?
    # The browser-version Client ID ('kimne7...') is heavily guarded by Twitch's Client-Integrity check,
    # requiring obfuscated browser JS execution. The app-version Client ID ('kd1unb...') used by chat-downloader 
    # bypasses these integrity checks entirely, allowing direct anonymous GQL comments extraction.
    CLIENT_ID = 'kd1unb4b3q4t58fwlpcbzcbnm76a8fp'
    GQL_URL = 'https://gql.twitch.tv/gql'
    
    VIDEO_METADATA_HASH = '45111672eea2e507f8ba44d101a61862f9c56b11dee09a15634cb75cb9b9084d'
    VIDEO_COMMENTS_HASH = 'b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a'
    
    # Why match optional 'v' prefix?
    # Some external integrations or legacy formats might structure the URL as '/videos/v123456789'.
    # Supporting the optional 'v' character ensures robust ID extraction.
    VOD_ID_PATTERN = re.compile(r'(?:videos|video|v)/v?(?P<id>\d+)')
    
    def _extract_vod_id(self, url: str) -> str:
        match = self.VOD_ID_PATTERN.search(url)
        if not match:
            raise ValueError(f"Invalid Twitch VOD URL: {url}")
        return match.group('id')

    def _post_gql(self, session: requests.Session, payload: list, max_retries: int = 5) -> dict:
        # Why 'text/plain;charset=UTF-8'?
        # Twitch's internal GQL server natively expects text/plain MIME type from GQL clients
        # to process batch GQL array requests smoothly.
        headers = {
            "Client-Id": self.CLIENT_ID,
            "Content-Type": "text/plain;charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = session.post(self.GQL_URL, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        if "errors" in data[0]:
                            raise RuntimeError(f"GQL Error: {data[0]['errors']}")
                        return data[0]
                    raise RuntimeError("Unexpected GQL response structure")
                elif response.status_code == 429:
                    print(f"Warning: Twitch rate limit (429) hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    response.raise_for_status()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2

    def collect_chat(self, vod_url: str, progress_callback=None) -> list[dict]:
        try:
            vod_id = self._extract_vod_id(vod_url)
        except Exception as e:
            print(f"Error extracting VOD ID: {e}")
            return []
            
        chat_data = []
        cursor = None
        has_next = True
        request_delay = 0.25
        
        # Why use a session?
        # Utilizing requests.Session maintains persistent HTTP connections (Keep-Alive),
        # which increases download efficiency significantly when paginating hundreds of times.
        session = requests.Session()
        
        while has_next:
            variables = {"videoID": vod_id}
            if cursor:
                variables["cursor"] = cursor
            else:
                variables["contentOffsetSeconds"] = 0
                
            payload = [
                {
                    "operationName": "VideoCommentsByOffsetOrCursor",
                    "variables": variables,
                    "extensions": {
                        "persistedQuery": {
                            "version": 1,
                            "sha256Hash": self.VIDEO_COMMENTS_HASH
                        }
                    }
                }
            ]
            
            try:
                res = self._post_gql(session, payload)
                video_data = res.get("data", {}).get("video", {})
                if not video_data:
                    break
                    
                comments_data = video_data.get("comments", {})
                edges = comments_data.get("edges", [])
                
                if not edges:
                    break
                    
                for edge in edges:
                    node = edge.get("node", {})
                    cursor = edge.get("cursor")
                    
                    offset = node.get("contentOffsetSeconds")
                    commenter = node.get("commenter") or {}
                    username = commenter.get("login") or commenter.get("displayName")
                    
                    fragments = node.get("message", {}).get("fragments", [])
                    message_text = "".join(f.get("text", "") for f in fragments)
                    
                    created_at = node.get("createdAt")
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        timestamp = int(dt.timestamp() * 1000000)
                    except Exception:
                        timestamp = int(time.time() * 1000000)
                        
                    if offset is not None and username and message_text:
                        chat_data.append({
                            "offset_seconds": offset,
                            "username": username,
                            "message": message_text,
                            "timestamp": timestamp
                        })
                        
                page_info = comments_data.get("pageInfo", {})
                has_next = page_info.get("hasNextPage", False)
                
                if progress_callback:
                    # Why call progress_callback here?
                    # Twitch chat pagination can take a while for long VODs. Reporting the count of 
                    # retrieved messages dynamically assures the user that progress is actively being made.
                    progress_callback(f"Twitchチャットログを収集中... ({len(chat_data)}件取得)", 10 + min(int((len(chat_data) / 1000) * 15), 18))

                if not cursor:
                    break
                    
                time.sleep(request_delay)
                
            except Exception as e:
                print(f"Warning: Failed to retrieve chat chunk: {e}")
                break
                
        return chat_data

    def save_to_file(self, chat_data: list[dict], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str) -> list[dict]:
        print(f"Loading cached chat data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_video_metadata(self, vod_url: str) -> dict:
        # Why query Twitch GQL for metadata?
        # Directly fetching Twitch VOD details (title, owner login, owner display name, duration, and creation date)
        # provides real, accurate context for the dashboard instead of using generic placeholder values.
        try:
            vod_id = self._extract_vod_id(vod_url)
        except Exception as e:
            print(f"Error extracting VOD ID for metadata: {e}")
            return {}
            
        session = requests.Session()
        # Why pass channelLogin as empty string?
        # The VideoMetadata schema requires channelLogin to be non-null, but sending an empty string ("")
        # is accepted by the Twitch GQL backend and correctly yields the associated video metadata.
        payload = [
            {
                "operationName": "VideoMetadata",
                "variables": {"videoID": vod_id, "channelLogin": ""},
                "extensions": {
                    "persistedQuery": {
                        "version": 1,
                        "sha256Hash": self.VIDEO_METADATA_HASH
                    }
                }
            }
        ]
        
        try:
            res = self._post_gql(session, payload)
            video_data = res.get("data", {}).get("video")
            if video_data:
                owner = video_data.get("owner") or {}
                return {
                    "vod_id": vod_id,
                    "title": video_data.get("title", f"Twitch VOD {vod_id}"),
                    "duration_seconds": video_data.get("lengthSeconds", 0),
                    "created_at": video_data.get("createdAt"),
                    "streamer_id": owner.get("login"),
                    "streamer_name": owner.get("displayName")
                }
        except Exception as e:
            print(f"Warning: Failed to fetch Twitch VOD metadata: {e}")
            
        return {}
