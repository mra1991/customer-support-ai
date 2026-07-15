"""
Sentiment-model loading, label formatting, and safe inference.

The original assignment used a Hugging Face pipeline and converted labels such
as POS, NEG, and NEU into clearer labels. This module keeps that behavior while
supporting selectable models and a signed meter value from -1 to +1.
"""

from dataclasses import dataclass
from time import perf_counter

from transformers import pipeline


SENTIMENT_MODELS = {
    "BERTweet (3 labels)": "finiteautomata/bertweet-base-sentiment-analysis",
    "DistilBERT SST-2": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter (3 labels)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}


@dataclass(frozen=True)
class SentimentResult:
    """Store one formatted sentiment result and its timing."""

    label: str
    score: float
    meter_value: float
    analysis_seconds: float


def format_sentiment_label(label: str) -> str:
    """Convert model-specific labels into POSITIVE, NEGATIVE, or NEUTRAL."""
    normalized = str(label).upper()
    if normalized in {"POS", "POSITIVE", "LABEL_2", "2"}:
        return "POSITIVE"
    if normalized in {"NEG", "NEGATIVE", "LABEL_0", "0"}:
        return "NEGATIVE"
    return "NEUTRAL"


class SentimentAnalyzer:
    """Wrap a Hugging Face sentiment pipeline with a neutral safe fallback."""

    def __init__(self, model_name: str):
        """Load the selected pretrained sentiment model."""
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, user_question: str) -> SentimentResult:
        """
        Detect whether the message is positive, neutral, or negative.

        The Hugging Face pipeline returns a list of dictionaries, so the first
        result is read and its ``label`` and ``score`` fields are extracted.
        """
        start = perf_counter()
        try:
            result = self.pipeline(user_question)[0]
            label = format_sentiment_label(result["label"])
            score = float(result["score"])
        except Exception:
            # Neutral is the safest default if sentiment inference fails.
            label = "NEUTRAL"
            score = 0.5

        meter_value = score if label == "POSITIVE" else -score if label == "NEGATIVE" else 0.0
        return SentimentResult(
            label=label,
            score=score,
            meter_value=meter_value,
            analysis_seconds=perf_counter() - start,
        )
