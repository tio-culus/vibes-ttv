# 分析実行時の経過時間および詳細進捗状況のリアルタイム表示計画書

分析処理中に「処理が停止しているのではないか」というユーザーの不安を解消するため、分析開始からの経過秒数および、処理のより詳細なステータス（ダウンロード進捗・速度・ETA、Gemini分析のバッチ進捗など）をStreamlit UI上にリアルタイム表示するための修正計画書です。

## ユーザーレビューが必要な箇所
> [!IMPORTANT]
> - **経過秒数および詳細進捗コールバックの導入**:
>   * `run_real_analysis` 内で `start_time = time.time()` を記録し、各処理フェーズで「⏱️ 経過時間: X秒」をプレフィックスとして表示し続けます。
>   * 各データ処理モジュールに `progress_callback` を渡し、進捗状況（チャット取得数、ダウンロード進捗率・速度・残り時間、Geminiの現在バッチ/総バッチ数）をリアルタイムにUIへ書き換えていきます。

## 提案される変更箇所

### 1. データ収集・解析モジュールの拡張（コールバック対応）

#### [MODIFY] [chat_collector.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/collectors/chat_collector.py)
*   `collect_chat` メソッドに `progress_callback=None` 引数を追加し、GQLページネーションループ毎に「チャットログを収集中... (X件取得)」という進捗を報告します。

#### [MODIFY] [audio_collector.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/collectors/audio_collector.py)
*   `collect_audio` メソッドに `progress_callback=None` 引数を追加します。
*   `yt-dlp` の `progress_hooks` を定義し、ダウンロード中に「音声ダウンロード中... (X% | 速度: Y | ETA: Z)」と詳細進捗を報告します。

#### [MODIFY] [comment_analyzer.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/analyzers/comment_analyzer.py)
*   `analyze_listeners` メソッドに `progress_callback=None` 引数を追加します。
*   バッチ処理ループ内で「コメント分析中... バッチ A/B を処理中」と進捗を報告します。

### 2. UI (Streamlit) の拡張

#### [MODIFY] [app.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/app.py)
*   `run_real_analysis` の中で `progress_callback(message, progress_val)` コールバック関数を定義し、各フェーズおよびモジュール呼び出し時にこれを渡してUI（`status_text`、`progress_bar`）を更新します。
*   Whisper文字起こし開始前には「文字起こし中... (数分かかる場合があります)」と明確な警告を出しつつ経過時間を更新します。

---

## 変更ファイルの詳細イメージ

### [audio_collector.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/collectors/audio_collector.py)
```python
    def collect_audio(self, vod_url: str, output_dir: str = "downloads", progress_callback=None) -> str:
        # ...
        def ytdl_hook(d):
            if d['status'] == 'downloading' and progress_callback:
                percent = d.get('_percent_str', '').strip()
                eta = d.get('_eta_str', '').strip()
                speed = d.get('_speed_str', '').strip()
                progress_callback(f"音声ダウンロード中... ({percent} | 速度: {speed} | ETA: {eta})", 40)

        ydl_opts = {
            # ...
            'progress_hooks': [ytdl_hook] if progress_callback else [],
        }
```

### [app.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/app.py)
```python
def run_real_analysis(...):
    status_text = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()

    def progress_callback(message: str, progress_val: int):
        elapsed = int(time.time() - start_time)
        status_text.text(f"⏱️ 経過時間: {elapsed}秒 | {message}")
        progress_bar.progress(progress_val)
```

---

## 検証計画

### 1. UIの動作確認
- 実際のVOD分析を実行し、進捗バー and ステータステキストが「経過秒数」「取得チャット数」「ダウンロード%・速度・ETA」「Geminiのバッチ進捗（A/B）」を含んで毎秒・毎ループ滑らかに更新されることを確認します。

### 2. 自動テストの実行
- 引数変更（コールバックの追加）によって既存のユニットテストや統合テストが影響を受けないことを検証します。
```bash
.\venv\Scripts\python -m pytest tests/test_vibes.py
```
