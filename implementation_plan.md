# 視聴者コメント分類詳細の確認機能の実装計画書

ティオさんからご要望いただいた「視聴者の態度の内訳が正しいか確認するため、どのコメントがどう分類されているかを確認する機能」を実装するための計画書です。

## ユーザーレビューが必要な箇所
> [!IMPORTANT]
> - **データベース拡張と自動マイグレーション**:
>   * `VODListenerStats` モデルに `comment_details_json` カラム（TEXT）を追加し、各コメントのメッセージテキスト、オフセット秒数、およびGeminiによる分類カテゴリ（`reaction`, `question`, `insight`, `instruction`, `other`）をJSONリストとして保存します。
>   * 起動時に `ALTER TABLE` を自動実行するマイグレーションロジックを `DBManager` に組み込みます。これにより、既存の `vibes_ttv.db` データベースを削除することなく、過去の分析データも維持したまま安全にアップグレードできます。
> - **UI（「👤 視聴者の態度分析」タブ）の拡張**:
>   * 態度分析タブに、特定のリスナーを選択できるセレクトボックス `st.selectbox` を配置します。
>   * リスナーを選択すると、そのリスナーの発言一覧が「配信時間」「コメント内容」「分類された態度（カラー付きバッジ）」と共に表形式で表示されます。
>   * 分類カテゴリでのフィルタリング機能（マルチセレクト）を実装し、「このユーザーの指示コメント（instruction）だけを確認する」といった絞り込みができるようにします。

---

## 提案される変更箇所

### 1. データベース層の拡張
#### [MODIFY] [models.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/database/models.py)
* `VODListenerStats` クラスに `comment_details_json = Column(String, nullable=True)` を追加します。

#### [MODIFY] [db_manager.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/database/db_manager.py)
* `create_tables` 内で `PRAGMA table_info` を使用し、`vod_listener_stats` に `comment_details_json` カラムが存在しない場合に `ALTER TABLE vod_listener_stats ADD COLUMN comment_details_json TEXT` を自動実行する軽量マイグレーション処理を追加します。

### 2. 解析モジュール
#### [MODIFY] [comment_analyzer.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/analyzers/comment_analyzer.py)
* Geminiに送信するレスポンススキーマ（Pydanticモデル）を拡張し、単なるカウント数だけでなく、個別コメントとカテゴリのペアリストを返してもらうようにします：
  ```python
  class CommentClassification(BaseModel):
      message: str
      category: str # 'reaction', 'question', 'insight', 'instruction', 'other'

  class ListenerClassification(BaseModel):
      username: str
      classifications: list[CommentClassification]
  ```
* `CommentAnalyzer.analyze_listeners` 内で、Geminiから得られた分類結果と、ルールベースでプレ分類された `reaction` コメント（「www」など）をマージし、各コメントのオフセット秒数を紐付けた `comment_details` リストを構築して `final_results` に含めるように修正します。

### 3. フロントエンドUIおよび実行制御
#### [MODIFY] [app.py](file:///c:/Users/ratio/src/github.com/tio-culus/vibes-ttv/vibes_ttv/app.py)
* 分析実行時に、新しく構築された個別分類データ `comment_details_json` を `VODListenerStats` のインスタンスに代入して保存するよう修正します。
* 「👤 視聴者の態度分析」タブの最下部に「💬 リスナー発言の分類詳細」セクションを追加します。
  * `st.selectbox` でリスナーを選択。
  * `st.multiselect` で態度カテゴリのフィルタリング（初期値は全選択）。
  * フィルタリングされたコメント一覧を、HTML/CSSを利用した綺麗なカラーバッジ（`reaction`: 紫, `question`: 青, `insight`: 黄色, `instruction`: 赤, `other`: 灰色）付きの表形式で描画します。

---

## 検証計画

### 1. 自動テストの実行
* レスポンススキーマ変更やカラム追加により、既存のテストケースが壊れていないか、またマイグレーションが正常に機能するかを検証します。
```powershell
.\venv\Scripts\python -m pytest tests/test_vibes.py
```

### 2. 手動動作確認
1. アプリを起動し、過去の分析データを開きます（このときマイグレーションが自動実行されることを確認）。
2. 新たに実際のVODに対して分析を実行し、完了後に「👤 視聴者の態度分析」タブを開きます。
3. リスナー詳細セクションで特定のリスナーを選択し、そのリスナーが送信したコメント一覧が表示されることを確認します。
4. コメントごとの分類カテゴリが、カラーバッジとして適切に色分けされ、かつマルチセレクトでの絞り込みが意図通りに機能することを確認します。
