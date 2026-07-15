"""
Knowledge-base CSV loading, validation, and saving utilities.

A knowledge base is stored as a CSV file with two required columns:
``question`` and ``answer``. The original assignment loaded these columns into
lists. This module preserves that behavior while adding validation, whitespace
cleanup, duplicate removal, and reusable save support.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"question", "answer"}


@dataclass(frozen=True)
class KnowledgeBase:
    """Store a validated question-and-answer knowledge base."""

    filepath: Path
    questions: list[str]
    answers: list[str]

    def __len__(self) -> int:
        """Return the number of question-and-answer pairs."""
        return len(self.questions)


def load_knowledge_base(filepath) -> KnowledgeBase:
    """
    Load and validate a question-and-answer CSV file.

    The function keeps the original assignment's safety checks: verify that the
    file exists, catch CSV reading errors, require the two expected columns, and
    remove rows with missing questions or answers. It also strips whitespace,
    removes empty rows, and drops duplicate pairs.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find knowledge base: {filepath}")

    try:
        data = pd.read_csv(filepath)
    except Exception as error:
        raise ValueError(f"Unable to read CSV file: {error}") from error

    if not REQUIRED_COLUMNS.issubset(data.columns):
        raise ValueError("CSV must contain 'question' and 'answer' columns.")

    data = data[["question", "answer"]].copy()
    data = data.dropna(subset=["question", "answer"])
    data["question"] = data["question"].astype(str).str.strip()
    data["answer"] = data["answer"].astype(str).str.strip()
    data = data[(data["question"] != "") & (data["answer"] != "")]
    data = data.drop_duplicates(subset=["question", "answer"])

    if data.empty:
        raise ValueError("Knowledge base contains no usable rows.")

    return KnowledgeBase(
        filepath=filepath,
        questions=data["question"].tolist(),
        answers=data["answer"].tolist(),
    )


def save_knowledge_base(filepath, questions, answers) -> Path:
    """Save matching question and answer sequences as a CSV file."""
    filepath = Path(filepath)
    if len(questions) != len(answers):
        raise ValueError("Question and answer counts must match.")

    rows = []
    for question, answer in zip(questions, answers):
        question = str(question).strip()
        answer = str(answer).strip()
        if question and answer:
            rows.append({"question": question, "answer": answer})

    if not rows:
        raise ValueError("There are no valid rows to save.")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(filepath, index=False)
    return filepath
