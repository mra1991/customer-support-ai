"""
Training-artifact serialization and automatic latest-artifact discovery.

This project does not fine-tune a neural network. Its training step loads and
cleans a knowledge base, loads selected pretrained models, and precomputes
embeddings for every approved question. Saving those embeddings avoids repeated
knowledge-base encoding on future runs.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np


ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class TrainingArtifact:
    """Store everything required to restore a trained support assistant."""

    filepath: Path
    questions: list[str]
    answers: list[str]
    embeddings: np.ndarray
    embedding_model: str
    sentiment_model: str
    similarity_threshold: float
    escalation_threshold: float
    knowledge_base_path: str
    created_at: str


def save_training_artifact(
    filepath,
    *,
    questions,
    answers,
    embeddings,
    embedding_model,
    sentiment_model,
    similarity_threshold,
    escalation_threshold,
    knowledge_base_path,
) -> Path:
    """Save cleaned Q&A data, embeddings, model names, and thresholds to NPZ."""
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".npz":
        filepath = filepath.with_suffix(".npz")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": ARTIFACT_VERSION,
        "embedding_model": embedding_model,
        "sentiment_model": sentiment_model,
        "similarity_threshold": float(similarity_threshold),
        "escalation_threshold": float(escalation_threshold),
        "knowledge_base_path": str(knowledge_base_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    np.savez_compressed(
        filepath,
        questions=np.asarray(list(questions), dtype=str),
        answers=np.asarray(list(answers), dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return filepath


def load_training_artifact(filepath) -> TrainingArtifact:
    """Load and validate a previously saved training artifact."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find artifact: {filepath}")

    try:
        with np.load(filepath, allow_pickle=False) as archive:
            questions = archive["questions"].astype(str).tolist()
            answers = archive["answers"].astype(str).tolist()
            embeddings = archive["embeddings"].astype(np.float32)
            metadata = json.loads(str(archive["metadata"]))
    except Exception as error:
        raise ValueError(f"Unable to load training artifact: {error}") from error

    if metadata.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported artifact version: {metadata.get('version')}")
    if len(questions) != len(answers):
        raise ValueError("Artifact question and answer counts do not match.")
    if embeddings.shape[0] != len(questions):
        raise ValueError("Artifact embeddings do not match question count.")

    return TrainingArtifact(
        filepath=filepath,
        questions=questions,
        answers=answers,
        embeddings=embeddings,
        embedding_model=metadata["embedding_model"],
        sentiment_model=metadata["sentiment_model"],
        similarity_threshold=float(metadata["similarity_threshold"]),
        escalation_threshold=float(metadata["escalation_threshold"]),
        knowledge_base_path=metadata["knowledge_base_path"],
        created_at=metadata["created_at"],
    )


def find_latest_artifact(folder) -> Path | None:
    """Return the most recently modified NPZ artifact, or None if none exist."""
    folder = Path(folder)
    if not folder.exists():
        return None
    artifacts = list(folder.glob("*.npz"))
    return max(artifacts, key=lambda path: path.stat().st_mtime) if artifacts else None
