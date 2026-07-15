"""
Sentence embedding and semantic nearest-neighbor retrieval.

The original assignment used SentenceTransformer to convert stored questions
and the user's message into numerical vectors called embeddings. Cosine
similarity then identified the stored question with the closest meaning. This
module preserves that design and makes it reusable and serializable.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODELS = {
    "MiniLM (fast)": "all-MiniLM-L6-v2",
    "MiniLM L12 (balanced)": "all-MiniLM-L12-v2",
    "MPNet (accurate)": "all-mpnet-base-v2",
    "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
}


@dataclass(frozen=True)
class RetrievalResult:
    """Store one semantic knowledge-base search result."""

    answer: str
    matched_question: str | None
    similarity: float
    search_seconds: float
    used_fallback: bool


class SemanticRetriever:
    """Encode approved questions and retrieve the closest approved answer."""

    def __init__(
        self,
        model_name: str,
        questions: list[str],
        answers: list[str],
        question_embeddings=None,
    ):
        """Load the embedding model and prepare or restore question vectors."""
        if len(questions) != len(answers):
            raise ValueError("Question and answer counts must match.")
        if not questions:
            raise ValueError("Knowledge base cannot be empty.")

        self.model_name = model_name
        self.questions = list(questions)
        self.answers = list(answers)

        # Model files may download the first time and are cached afterward by
        # SentenceTransformers and Hugging Face.
        self.model = SentenceTransformer(model_name)

        if question_embeddings is None:
            self.question_embeddings = self.encode_questions(self.questions)
        else:
            embeddings = np.asarray(question_embeddings, dtype=np.float32)
            if embeddings.shape[0] != len(self.questions):
                raise ValueError("Saved embeddings do not match the knowledge base.")
            self.question_embeddings = embeddings

    def encode_questions(self, questions) -> np.ndarray:
        """Convert question strings into normalized semantic vectors."""
        embeddings = self.model.encode(
            list(questions),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def find_best_answer(
        self,
        user_question: str,
        similarity_threshold: float,
        sentiment: str | None = None,
    ) -> RetrievalResult:
        """
        Find the stored question semantically closest to the user's message.

        The model receives ``[user_question]`` because ``encode`` expects a list
        of strings. ``cosine_similarity`` returns a 2D array, so the code takes
        the first and only row, just as in the original assignment.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must lie between 0 and 1.")

        user_question = user_question.strip()
        if not user_question:
            raise ValueError("User question cannot be empty.")

        start = perf_counter()
        user_embedding = self.model.encode(
            [user_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = cosine_similarity(
            user_embedding,
            self.question_embeddings,
        )[0]

        best_match_index = int(np.argmax(similarities))
        best_score = float(np.max(similarities))
        elapsed = perf_counter() - start

        # A weak semantic match must not be presented as authoritative. This is
        # the main privacy and safety advantage of an approved-answer system.
        if best_score < similarity_threshold:
            if sentiment == "POSITIVE":
                fallback = "You're welcome! Is there anything else I can help you with?"
            else:
                fallback = (
                    "I couldn't find a confident answer in the approved support "
                    "knowledge base. Please contact a human support agent."
                )
            return RetrievalResult(
                answer=fallback,
                matched_question=None,
                similarity=best_score,
                search_seconds=elapsed,
                used_fallback=True,
            )

        return RetrievalResult(
            answer=self.answers[best_match_index],
            matched_question=self.questions[best_match_index],
            similarity=best_score,
            search_seconds=elapsed,
            used_fallback=False,
        )
