"""Conversation-turn representation, statistics, and JSON export utilities."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass(frozen=True)
class ConversationTurn:
    """Store one complete user/assistant exchange with developer diagnostics."""

    timestamp: str
    user: str
    answer: str
    matched_question: str | None
    similarity: float
    sentiment: str
    sentiment_score: float
    escalated: bool
    used_fallback: bool
    search_seconds: float
    sentiment_seconds: float


class ConversationHistory:
    """Collect turns and save them for later developer review."""

    def __init__(self):
        """Create an empty conversation history."""
        self.turns: list[ConversationTurn] = []

    def add_turn(self, **kwargs) -> ConversationTurn:
        """Create and append one timestamped conversation turn."""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            **kwargs,
        )
        self.turns.append(turn)
        return turn

    def statistics(self) -> dict:
        """Return turn count, averages, escalations, and fallback count."""
        if not self.turns:
            return {
                "count": 0,
                "average_similarity": 0.0,
                "average_sentiment_score": 0.0,
                "escalations": 0,
                "fallbacks": 0,
            }
        count = len(self.turns)
        return {
            "count": count,
            "average_similarity": sum(t.similarity for t in self.turns) / count,
            "average_sentiment_score": sum(t.sentiment_score for t in self.turns) / count,
            "escalations": sum(t.escalated for t in self.turns),
            "fallbacks": sum(t.used_fallback for t in self.turns),
        }

    def save_json(self, filepath) -> Path:
        """Save full turn details and summary statistics as UTF-8 JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "statistics": self.statistics(),
            "turns": [asdict(turn) for turn in self.turns],
        }
        filepath.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return filepath
