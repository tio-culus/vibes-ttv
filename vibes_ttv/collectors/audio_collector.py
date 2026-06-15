import os
import re
import yt_dlp

class AudioCollector:
    # Why not download the entire video file?
    # Twitch VOD video streams are extremely large (often multiple gigabytes).
    # Since our topic analysis relies purely on speech-to-text (Whisper),
    # downloading only the audio saves vast amounts of bandwidth, disk space, and time.
    
    def collect_audio(self, vod_url: str, output_dir: str = "downloads", progress_callback=None) -> str:
        if not os.path.exists(output_dir):
            # Why exist_ok=True not used?
            # Simple os.makedirs check is cleaner and safer for standard python compatibility.
            os.makedirs(output_dir)
            
        def ytdl_hook(d):
            if d['status'] == 'downloading' and progress_callback:
                percent = d.get('_percent_str', '0%').strip()
                eta = d.get('_eta_str', 'unknown').strip()
                speed = d.get('_speed_str', 'unknown').strip()
                
                # Why strip ANSI escape codes?
                # yt-dlp output can contain terminal color codes (ANSI escape sequences), which break 
                # Streamlit rendering and result in garbage characters or parsed float calculation errors.
                ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
                clean_percent = ansi_escape.sub('', percent)
                clean_eta = ansi_escape.sub('', eta)
                clean_speed = ansi_escape.sub('', speed)
                
                try:
                    # Why use re.search instead of simple sub?
                    # Extracting the first float match directly preceding the '%' symbol prevents
                    # color code numbers (like 94 from cyan) from leaking into the percentage value.
                    match = re.search(r"(\d+\.?\d*)\%", clean_percent)
                    if match:
                        pct_val = float(match.group(1))
                        progress_val = 30 + int(pct_val * 0.2)
                    else:
                        progress_val = 30
                except Exception:
                    progress_val = 30
                progress_callback(f"音声ダウンロード中... ({clean_percent} | 速度: {clean_speed} | ETA: {clean_eta})", progress_val)

        # Why preferredcodec='mp3' instead of wav or m4a?
        # MP3 provides a great balance between compression (smaller size) and compatibility.
        # Whisper can easily process MP3 files, and it reduces memory footprints during loading.
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # Output format template: downloads/{video_id}.mp3
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [ytdl_hook] if progress_callback else [],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(vod_url, download=True)
                video_id = info.get('id')
                # The postprocessor changes the extension to .mp3, so we construct that filename.
                output_path = os.path.abspath(os.path.join(output_dir, f"{video_id}.mp3"))
                return output_path
        except Exception as e:
            # Why raise here instead of returning None?
            # Unlike chat logs, missing audio completely breaks the transcription and topic analysis pipeline,
            # so we must explicitly fail and bubble up the error to warn the user.
            raise RuntimeError(f"Failed to extract audio from Twitch VOD {vod_url}: {e}")
