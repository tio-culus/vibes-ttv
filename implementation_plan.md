# 話題分類割合グラフ追加、VOD時間ジャンプリンク、およびUIチカチカ防止実装計画書

ティオさんからご要望いただいた、以下の3点の実装計画書です：
1. 「🗣️ 配信中の話題分析」タブ内、話題の盛り上がりランキングの直上に「話題分類の割合」を示すプレミアムなドーナツチャートを表示。
2. 話題タイムラインの配信時間および分類ラベル（ならびにランキングカード内の時間枠）に、Twitch VOD の開始時刻へ直接ジャンプするURLリンクを埋め込み。
3. バックグラウンド分析実行中の1秒ごとのポーリング（再描画）に伴う、前回の分析データ表示エリアの不快なチカチカ（明滅）現象の防止。

## ユーザーレビューが必要な箇所
> [!IMPORTANT]
> - **話題分類比率の計算アルゴリズム**:
>   * 単純な「トピック件数」ではなく、各トピックの「実占有時間（終了秒数 - 開始秒数）」をカテゴリ別に合計して比率を算出します。これにより、配信全体の時間配分が正しく円グラフ（ドーナツチャート）に可視化されます。
> - **Twitch VOD開始時刻ジャンプリンクの仕様**:
>   * Twitchの標準的な時間パラメータ形式である `?t=XhYmZs` （時・分・秒）へ自動変換するヘルパー関数 `format_twitch_offset` を実装します。
>   * タイムラインおよびランキングカード内の配信時間枠をクリックすると、新規タブ（`_blank`）で該当時間の Twitch アーカイブが直接開くようにします。
> - **UIチカチカ防止のためのCSSハック**:
>   * Streamlitは再実行（`st.rerun`）中、画面上の古い要素を「stale（保留）」状態として認識し、自動的に半透明（`opacity: 0.4` 程度）にします。1秒おきのタイマー更新のたびにこの半透明化と再描画が繰り返されるため、画面全体がチカチカと明滅する不快な状態になっていました。
>   * カスタムCSSで `[data-stale="true"]` の不透明度を `0.7`（少し暗い状態）で固定し、かつアニメーション遷移（`transition: none !important;`）を適用することで、分析中である視覚効果（少し暗い表示）をキープしたまま、明滅だけを完全に抑え込みます。

---

## 提案される変更箇所

### 1. フロントエンドUIおよび時間変換ロジックの追加

#### [MODIFY] [app.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/app.py)
* `format_seconds` 関数の直下に、秒数を Twitch クエリパラメータ形式（例: `1h23m45s`）に変換する `format_twitch_offset` ヘルパー関数を新規追加します。
  ```python
  def format_twitch_offset(seconds: int) -> str:
      h = int(seconds // 3600)
      m = int((seconds % 3600) // 60)
      s = int(seconds % 60)
      parts = []
      if h > 0:
          parts.append(f"{h}h")
      if m > 0 or h > 0:
          parts.append(f"{m}m")
      parts.append(f"{s}s")
      return "".join(parts)
  ```
* 510行目の `<style>` タグ内に以下のCSSを追加して、毎秒の再描画による明滅（点滅）を解消します。
  ```css
  /* Prevent stale components from flickering during analysis runner reruns */
  [data-stale="true"] {
      opacity: 0.7 !important;
      transition: none !important;
  }
  ```
* 「🗣️ 配信中の話題分析」タブ（`tab_topics`）内で、トピックが存在する場合：
  * カテゴリごとの合計秒数を集計し、分単位に変換して pandas DataFrame にロードします。
  * Altair の `mark_arc(innerRadius=50, outerRadius=90)` を使用して、Twitchのパープルテーマ（`scale=alt.Scale(scheme="purples")`）に調和するスタイリッシュなドーナツチャートを描画します。
  * チャートの直下に `st.markdown("<br/>", unsafe_allow_html=True)` を挟み、ランキングカードとの間に程よい余白を作ります。
* ランキングカード（ベスト3 / ワースト3）の配信時間文字列部分をHTMLリンクにし、`https://www.twitch.tv/videos/{vod_id}?t={format_twitch_offset(start_offset)}` へのリンクを設定します。
* 話題タイムライン各行の `🕒 {time_range} | 分類: {cat_label}` の部分も同様にHTMLリンクにし、開始秒数の時間指定で Twitch アーカイブにジャンプできるように変更します。

---

## 検証計画

### 1. 自動テストの実行
UIのロジック追加やヘルパー関数追加により、既存のテストケースが壊れていないかを検証します。
```bash
.\venv\Scripts\python -m pytest tests/test_vibes.py
```

### 2. 手動動作確認
1. アプリを起動し、過去の分析結果をロードして「🗣️ 配信中の話題分析」タブを開きます。
2. 盛り上がりランキングの上に「📊 話題の分類比率」という美しいドーナツグラフが描画され、ツールチップで各カテゴリの占有時間が確認できることを確認します。
3. タイムライン行の配信時間リンク（紫色の下点線付き）およびランキングカード内の時間枠リンクをクリックした際に、TwitchのVODの該当開始時刻が別タブで正確に開くことを確認します。
4. 実際のVODに対して分析を実行し、進捗バーが表示されて経過秒数が1秒ごとにカウントアップされる間、メイン画面の過去の分析データが一瞬たりとも「明滅（チカチカ）」することなく、少し暗い状態で安定して表示され続けることを確認します。
