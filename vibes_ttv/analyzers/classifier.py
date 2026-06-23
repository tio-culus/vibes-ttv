from abc import ABC, abstractmethod
from typing import List, Dict, Any
from enum import Enum
from pydantic import BaseModel

# Why inherit from str as well (str, Enum)?
# Inheriting from str ensures the enum values act as native strings,
# allowing seamless JSON serialization and preserving backward compatibility with downstream DB/UI models.
class CommentCategory(str, Enum):
    REACTION = 'reaction'
    RESPONSE = 'response'
    ADVICE = 'advice'
    BACKSEAT = 'backseat'
    CROSS_CHAT = 'cross-chat'
    BLOGPOST = 'blogpost'
    OTHER = 'other'

    @property
    def display_label(self) -> str:
        # Why not hardcode display labels in UI?
        # Centralizing display labels within the Enum guarantees UI label consistency 
        # and eliminates translation mapping tables scattered across the app modules.
        labels = {
            CommentCategory.REACTION: "リアクション",
            CommentCategory.RESPONSE: "レスポンス",
            CommentCategory.ADVICE: "アドバイス",
            CommentCategory.BACKSEAT: "指示・ネタバレ",
            CommentCategory.CROSS_CHAT: "鳩",
            CommentCategory.BLOGPOST: "自分語り",
            CommentCategory.OTHER: "その他",
        }
        return labels[self]

    @property
    def persona_label(self) -> str:
        # Defining persona labels directly in the Enum centralizes behavior classification terms.
        labels = {
            CommentCategory.REACTION: "リアクション",
            CommentCategory.RESPONSE: "レスポンス",
            CommentCategory.ADVICE: "アドバイス",
            CommentCategory.BACKSEAT: "指示・ネタバレ",
            CommentCategory.CROSS_CHAT: "鳩",
            CommentCategory.BLOGPOST: "自分語り",
            CommentCategory.OTHER: "その他",
        }
        return labels[self]

    @property
    def color_hex(self) -> str:
        # Defining base brand/theme color codes in the Enum allows synchronizing 
        # both graphs, badges, and any highlighting styles consistently.
        colors = {
            CommentCategory.REACTION: "#c084fc",      # Light purple
            CommentCategory.RESPONSE: "#60a5fa",      # Light blue
            CommentCategory.ADVICE: "#facc15",       # Light yellow
            CommentCategory.BACKSEAT: "#f87171",   # Light red
            CommentCategory.CROSS_CHAT: "#4ade80",      # Light green
            CommentCategory.BLOGPOST: "#fb923c",      # Light red
            CommentCategory.OTHER: "#9ca3af",         # Gray
        }
        return colors[self]

    @property
    def description(self) -> str:
        # Dynamic properties keep the Enum values simple strings for serialization,
        # while centralizing prompt instruction text to a single source of truth.
        descriptions = {
            CommentCategory.REACTION: "今の話題に対する感想。例、すごい、しらなかった、面白い、なるほど、いいね、今の何？",
            CommentCategory.RESPONSE: "今の話題に関連したコメント。例、それって本当はこうらしいよ、こういうものもあるんだって、自分の時はこうだった",
            CommentCategory.ADVICE: "ストリーマーから尋ねられたことに対しての助言や提案。例、「音量どうですか？」に対して「ゲーム音小さい」",
            CommentCategory.BACKSEAT: "ストリーマーから尋ねられていないのに、未来の展開や知らない知識に関するコメント・助言・提案。例、上からくるよ、その武器強いよ、右に行くと楽だよ、その武器は強化したほうが良いよ",            
            CommentCategory.CROSS_CHAT: "話題に上がっていない他の人物についてのコメント。例、 〇〇さんはこうしてたよ、〇〇さんが困っている",
            CommentCategory.BLOGPOST: "話題と関係ない自分についてのコメント。例、ガチャ爆死しました、ポンデリング食べました、今日風邪気味です",
            CommentCategory.OTHER: "上記に分類できないコメント。",
        }
        return descriptions[self]

# Classification Pydantic schemas
class CommentClassification(BaseModel):
    message: str
    category: CommentCategory

class ListenerClassification(BaseModel):
    username: str
    classifications: list[CommentClassification]

class BatchClassificationResponse(BaseModel):
    results: list[ListenerClassification]

class LineClassification(BaseModel):
    line_id: str
    category: CommentCategory

class SliceClassificationResponse(BaseModel):
    results: list[LineClassification]


class CommentClassifier(ABC):
    # Why use an abstract base class instead of a protocol or duck typing?
    # An ABC provides runtime checkability using isinstance() and enforces subclass compliance.
    
    @property
    @abstractmethod
    def name(self) -> str:
        """分類器の識別名および構成パラメーターを表す文字列を返します。"""
        pass

    @abstractmethod
    def classify(self, merged_events: List[Dict[str, Any]], progress_callback=None) -> Dict[int, CommentCategory]:
        """統合タイムラインイベントを受け取り、各コメントのインデックスから判定されたカテゴリへのマップを返します。

        Args:
            merged_events: 統合タイムラインのイベント（Streamer発言、Listener発言含む）のリスト。
            progress_callback: 進捗状況を通知するためのコールバック関数。

        Returns:
            キーが元の merged_events リストでのインデックス(int)、値が CommentCategory の辞書。
        """
        pass
