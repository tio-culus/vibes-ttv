# vibes-ttv | Twitch 配信分析ダッシュボード

Twitchの配信アーカイブ（VOD）からチャットログと配信音声を収集し、ローカルの Whisper（GPU加速）および Gemini API（gemini-3.5-flash）を組み合わせて、「視聴者の態度」と「配信中の話題」を高度に分析・可視化するツールです。

## 技術スタックと特徴
- **音声処理**: `yt-dlp` で VOD の音声（MP3）を抽出。
- **チャット取得**: `chat-downloader` によるチャットログ取得。
- **文字起こし**: `openai-whisper` を用いたローカルの GPU (GeForce RTX 4070 等) 加速による高精度文字起こし。
- **AI分析**: `google-genai` SDK を通じて `gemini-3.5-flash` を利用。Pydantic モデルを用いた安定した構造化出力 (Structured Outputs) により、コメントの内訳分類や話題タイムラインを抽出。
- **データ管理**: SQLAlchemy と SQLite を用いた堅牢なリレーショナルモデル。
- **UI**: Streamlit を採用し、ダークモードやカスタムCSSを適用したプレミアムなデザイン。

---

## 前提条件

1. **ffmpeg のインストール**
   音声の抽出と変換に `ffmpeg` を使用します。システムにインストールされ、PATHが通っている必要があります。
   *※環境確認でインストール済みであることを確認しています。*

2. **CUDA / PyTorch（GPU加速用）**
   RTX 4070 を使用して高速に文字起こしを行うには、仮想環境の PyTorch が CUDA を認識している必要があります。

3. **Gemini API キー**
   分析に Gemini API を使用するため、[Google AI Studio](https://aistudio.google.com/) から API キーを取得してください。

---

## 起動手順

1. **仮想環境の有効化**
   PowerShellで以下のコマンドを実行し、作成済みの仮想環境を有効にします。
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Streamlit の起動**
   以下のコマンドを実行し、ダッシュボードを起動します。自動的にブラウザでタブが開きます。
   ```powershell
   streamlit run vibes_ttv/app.py
   ```

---

## ダッシュボードの使い方

### A. デモ・モックデータ（推奨：即座に確認可能）
- APIキーの設定や、実際のVOD音声ダウンロード（通常数十分〜数時間かかります）を待たずに、本ツールの機能とプレミアムなUIデザインを即座に確認できます。
1. サイドバーから **「デモ・モックデータ」** を選択します。
2. **「デモデータを読み込む」** ボタンをクリックします。
3. 自動的にモックの分析結果（グラフ、タイムライン、ペルソナ円グラフなど）が読み込まれ、ダッシュボードが描画されます。

### B. 実際の VOD 分析
1. サイドバーから **「実際のVODを分析」** を選択します。
2. **Gemini API Key** を入力します（環境変数 `GEMINI_API_KEY` にセットして起動した場合は自動で入力されます）。
3. 分析したい **Twitch VODのURL** (例: `https://www.twitch.tv/videos/123456789`) と **平均同接数** を入力します。
4. **「分析を実行する」** ボタンをクリックします。
   - 音声抽出 → ローカルの Whisper による文字起こし → タイムラインマージ → Gemini API での分析 がパイプラインに沿って自動実行され、結果がデータベースに保存されます。

---

## 音声認識 (STT) ベンチマークの実行

各種STTエンジン（ローカル Whisper、Google Cloud STT、Gemini）の処理速度（実行時間）と文字起こし精度（文字一致度、文字数回収率など）を、客観的に比較・検証できるベンチマークスクリプトが用意されています。

### 1. 基準データ（リファレンス）の作成
まず、比較の基準となる高品質な文字起こしデータ（デフォルトでは Whisper (turbo) の出力）を生成します。

```powershell
# 仮想環境を有効化した状態で実行
.\venv\Scripts\python scratch/create_reference.py --audio-path "C:\Users\ratio\Downloads\1782089179_v2799586321.mp3"
```
*※ `tests/fixtures/reference_transcription.json` に基準データが保存されます。*

### 2. 各STTエンジンのベンチマーク測定

#### A. ローカル Whisper (turbo)
```powershell
.\venv\Scripts\python scratch/run_stt_benchmark.py --engine whisper
```

#### B. Google Cloud Speech-to-Text (5分モノラルチャンク分割方式)
```powershell
.\venv\Scripts\python scratch/run_stt_benchmark.py --engine google_stt --project-id "あなたのGCPプロジェクトID" --bucket-name "あなたのGCSバケット名"
```

#### C. Gemini (4分ステレオ・1分オーバーラップ/16並列方式)
```powershell
# APIキーを環境変数に設定して実行
$env:GEMINI_API_KEY="あなたのGeminiAPIキー"
.\venv\Scripts\python scratch/run_stt_benchmark.py --engine gemini --model-name gemini-3.1-flash-lite
```
*※ モデル名は `--model-name` オプションで `gemini-3.1-flash-lite` (標準・高コスパ) や `gemini-3.5-flash` を指定可能です。*

### 3. 過去の測定結果のオフライン比較
過去に実行されたベンチマーク結果のJSONファイルを指定し、APIを実行せずに再度リファレンスとの一致度や差分（diff）を表示できます。

```powershell
.\venv\Scripts\python scratch/run_stt_benchmark.py --compare-file scratch/benchmark_results_YYYYMMDD_HHMMSS.json
```

### 4. 実行結果の保存
`--compare-file` オプションなしで測定を実行すると、実行日時が付与された個別結果ファイル（例: `scratch/benchmark_results_20260622_174905.json`）が自動生成され、処理時間、一致スコア、すべてのパース済みセグメントなどが保存されます。

---

## コメント分類ベンチマークの実行

リスナーコメント分類器の精度（Accuracy, Precision, Recall, F1スコア）および処理速度を客観的に検証できるベンチマークツールが用意されています。

### 1. テストデータの生成（DBからエクスポート）
本番で分析済みの統合タイムラインデータを SQLite DB からエクスポートし、手動アノテーション用のテストデータを作成します。

```powershell
# データベースから分析済みのタイムラインデータをエクスポート
.\venv\Scripts\python tests/create_reference_from_db.py
```
*※ `tests/fixtures/reference_timeline.json` にエクスポートされます。必要に応じてJSONファイル内の `"expected_category"` （初期値はDB上での分類結果）を手動で修正・校正して正解ラベル（リファレンス）を確定させます。DBにまだ分析済みデータが無い場合は、自動的に全カテゴリを含んだダミーのタイムラインデータがフォールバック生成されます。*

### 2. ベンチマーク測定の実行

#### A. ルールベース分類器（ベースライン・ローカル動作）
APIキー不要で即座に動作し、LLM導入の費用対効果を評価するための基準（ベースライン）となります。
```powershell
.\venv\Scripts\python tests/run_classifier_benchmark.py --classifier rule-based
```

#### B. Gemini 分類器（LLM評価）
環境変数 `GEMINI_API_KEY` を設定するか、`--api-key` オプションでキーを指定して実行します。
```powershell
.\venv\Scripts\python tests/run_classifier_benchmark.py --classifier gemini --api-key "あなたのGeminiAPIキー"
```

#### C. 全ての分類器を一括実行して比較
```powershell
.\venv\Scripts\python tests/run_classifier_benchmark.py --classifier all --api-key "あなたのGeminiAPIキー"
```

#### D. 各種設定パラメータの指定
評価対象モデルや、一度に分類器へ渡すチャットの件数（スライスサイズ）を調整して、最適なモデル・分割パラメータを検証することができます。
```powershell
# モデル名とスライスサイズを指定して実行
.\venv\Scripts\python tests/run_classifier_benchmark.py --classifier gemini --model gemini-3.5-flash --slice-size 50 --api-key "あなたのGeminiAPIキー"
```

---

## テストの実行

単体テストを走らせる場合は、以下のコマンドを実行します。
```powershell
.\venv\Scripts\python -m pytest tests/
```
