"""
Graphical interface for training, testing, and reviewing Customer Support AI.

The left side is the developer training area. It loads a CSV knowledge base,
selects models and thresholds, precomputes approved-question embeddings, saves
training artifacts, and restores previous training.

The right side contains a minimal Frontend customer chat and a Developer tab
showing the matched question, similarity score, sentiment score, escalation,
timing, statistics, and complete conversation log.
"""

from datetime import datetime
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from conversation import ConversationHistory
from embeddings import EMBEDDING_MODELS, SemanticRetriever
from knowledge_base import KnowledgeBase, load_knowledge_base
from sentiment import SENTIMENT_MODELS, SentimentAnalyzer
from training_artifact import (
    find_latest_artifact,
    load_training_artifact,
    save_training_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_bases"
TRAINING_DIR = PROJECT_ROOT / "trained_models"
HISTORY_DIR = PROJECT_ROOT / "conversation_history"


class CustomerSupportGUI:
    """Tkinter application for a safe retrieval-based website assistant."""

    def __init__(self, root):
        """Initialize training state, testing state, widgets, and auto-loading."""
        self.root = root
        self.root.title("Customer Support AI")
        self.root.state("zoomed")
        self.root.minsize(1180, 700)

        for folder in (KNOWLEDGE_BASE_DIR, TRAINING_DIR, HISTORY_DIR):
            folder.mkdir(parents=True, exist_ok=True)

        self.knowledge_base: KnowledgeBase | None = None
        self.knowledge_base_path: Path | None = None
        self.artifact_path: Path | None = None
        self.retriever: SemanticRetriever | None = None
        self.sentiment_analyzer: SentimentAnalyzer | None = None
        self.history = ConversationHistory()
        self.session_history_path = HISTORY_DIR / f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.is_busy = False

        self._configure_styles()
        self._build_layout()
        self._set_trained_state(False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Let the window appear before potentially large pretrained models load.
        self.root.after_idle(self._autoload_latest_training)

    def _configure_styles(self):
        """Configure reusable headings and diagnostic label styles."""
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Developer.TLabel", foreground="#304ffe")

    def _build_layout(self):
        """Create a fixed-width training panel and expandable testing panel."""
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.training_panel = ttk.LabelFrame(self.root, text="Training and Configuration")
        self.training_panel.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="ns")

        self.testing_panel = ttk.LabelFrame(self.root, text="Testing")
        self.testing_panel.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        self.testing_panel.columnconfigure(0, weight=1)
        self.testing_panel.rowconfigure(0, weight=1)

        self._build_training_panel()
        self._build_testing_panel()

    def _build_training_panel(self):
        """Create CSV, model, threshold, training, loading, and saving controls."""
        ttk.Label(self.training_panel, text="Knowledge Base", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(10, 4)
        )
        self.knowledge_base_var = tk.StringVar(value="No knowledge base selected")
        ttk.Label(self.training_panel, textvariable=self.knowledge_base_var, wraplength=360).grid(
            row=1, column=0, columnspan=2, padx=10, pady=4
        )

        self.browse_kb_button = ttk.Button(
            self.training_panel, text="Browse CSV", command=self.browse_knowledge_base
        )
        self.browse_kb_button.grid(row=2, column=0, padx=6, pady=6)
        self.load_artifact_button = ttk.Button(
            self.training_panel, text="Load Previous Training", command=self.browse_training_artifact
        )
        self.load_artifact_button.grid(row=2, column=1, padx=6, pady=6)

        ttk.Separator(self.training_panel).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=8
        )
        ttk.Label(self.training_panel, text="AI Models", style="Section.TLabel").grid(
            row=4, column=0, columnspan=2, pady=(4, 6)
        )

        ttk.Label(self.training_panel, text="Sentence Transformer").grid(
            row=5, column=0, sticky="w", padx=10
        )
        self.embedding_display_var = tk.StringVar(value="MiniLM (fast)")
        self.embedding_combo = ttk.Combobox(
            self.training_panel,
            textvariable=self.embedding_display_var,
            values=list(EMBEDDING_MODELS),
            state="readonly",
            width=28,
        )
        self.embedding_combo.grid(row=6, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")

        ttk.Label(self.training_panel, text="Sentiment Analyzer").grid(
            row=7, column=0, sticky="w", padx=10
        )
        self.sentiment_display_var = tk.StringVar(value="BERTweet (3 labels)")
        self.sentiment_combo = ttk.Combobox(
            self.training_panel,
            textvariable=self.sentiment_display_var,
            values=list(SENTIMENT_MODELS),
            state="readonly",
            width=28,
        )
        self.sentiment_combo.grid(row=8, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")

        ttk.Label(self.training_panel, text="Similarity Threshold").grid(
            row=9, column=0, columnspan=2
        )
        self.similarity_slider = tk.Scale(
            self.training_panel, from_=0.0, to=1.0, resolution=0.01,
            orient="horizontal", length=330
        )
        self.similarity_slider.set(0.30)
        self.similarity_slider.grid(row=10, column=0, columnspan=2, padx=10)

        ttk.Label(self.training_panel, text="Escalation Threshold").grid(
            row=11, column=0, columnspan=2, pady=(6, 0)
        )
        self.escalation_slider = tk.Scale(
            self.training_panel, from_=0.0, to=1.0, resolution=0.01,
            orient="horizontal", length=330
        )
        self.escalation_slider.set(0.90)
        self.escalation_slider.grid(row=12, column=0, columnspan=2, padx=10)

        self.train_button = ttk.Button(
            self.training_panel, text="Train / Build Embeddings", command=self.train_clicked
        )
        self.train_button.grid(row=13, column=0, columnspan=2, padx=10, pady=(12, 4), sticky="ew")
        self.save_training_button = ttk.Button(
            self.training_panel, text="Save Training", command=self.save_training_clicked
        )
        self.save_training_button.grid(row=14, column=0, columnspan=2, padx=10, pady=4, sticky="ew")

        self.training_status_var = tk.StringVar(
            value="Select a CSV knowledge base or load previous training."
        )
        ttk.Label(
            self.training_panel,
            textvariable=self.training_status_var,
            wraplength=360,
            justify="center",
        ).grid(row=15, column=0, columnspan=2, padx=10, pady=(8, 12))

    def _build_testing_panel(self):
        """Create separate customer-facing and developer-facing tabs."""
        notebook = ttk.Notebook(self.testing_panel)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.frontend_tab = ttk.Frame(notebook)
        self.developer_tab = ttk.Frame(notebook)
        notebook.add(self.frontend_tab, text="Frontend / Customer View")
        notebook.add(self.developer_tab, text="Backend / Developer View")
        self._build_frontend_tab()
        self._build_developer_tab()

    def _build_frontend_tab(self):
        """Create the minimal website chat visible to the end user."""
        self.frontend_tab.columnconfigure(0, weight=1)
        self.frontend_tab.rowconfigure(1, weight=1)
        ttk.Label(self.frontend_tab, text="Website Support Assistant", style="Title.TLabel").grid(
            row=0, column=0, padx=16, pady=(16, 8)
        )

        self.chat_text = tk.Text(
            self.frontend_tab, wrap="word", state="disabled",
            font=("Segoe UI", 11), padx=14, pady=14
        )
        self.chat_text.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")
        self.chat_text.tag_configure("user", foreground="#1565c0", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("assistant", foreground="#2e7d32", font=("Segoe UI", 11))
        self.chat_text.tag_configure("system", foreground="#666666", font=("Segoe UI", 10, "italic"))

        input_frame = ttk.Frame(self.frontend_tab)
        input_frame.grid(row=2, column=0, padx=16, pady=(8, 16), sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        self.message_entry = tk.Text(input_frame, height=3, wrap="word", font=("Segoe UI", 11))
        self.message_entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.message_entry.bind("<Control-Return>", lambda event: self.send_message())
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky="ns")
        ttk.Label(input_frame, text="Press Ctrl+Enter to send.").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )

    def _build_developer_tab(self):
        """Create latest-turn diagnostics, sentiment meter, stats, and full log."""
        self.developer_tab.columnconfigure(0, weight=1)
        self.developer_tab.rowconfigure(5, weight=1)

        metrics = ttk.LabelFrame(self.developer_tab, text="Latest Turn Diagnostics")
        metrics.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        metrics.columnconfigure(1, weight=1)

        self.matched_question_var = tk.StringVar(value="—")
        self.similarity_var = tk.StringVar(value="—")
        self.sentiment_var = tk.StringVar(value="—")
        self.escalation_var = tk.StringVar(value="—")
        self.timing_var = tk.StringVar(value="—")

        for row, (caption, variable) in enumerate([
            ("Matched question", self.matched_question_var),
            ("Similarity score", self.similarity_var),
            ("Sentiment", self.sentiment_var),
            ("Escalation", self.escalation_var),
            ("Timing", self.timing_var),
        ]):
            ttk.Label(metrics, text=f"{caption}:").grid(row=row, column=0, padx=8, pady=4, sticky="nw")
            ttk.Label(metrics, textvariable=variable, wraplength=760, style="Developer.TLabel").grid(
                row=row, column=1, padx=8, pady=4, sticky="w"
            )

        meter_frame = ttk.LabelFrame(self.developer_tab, text="Sentiment Meter")
        meter_frame.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        self.sentiment_canvas = tk.Canvas(meter_frame, height=70, highlightthickness=0)
        self.sentiment_canvas.pack(fill="x", padx=12, pady=8)
        self.sentiment_canvas.bind("<Configure>", lambda event: self._draw_sentiment_meter(0.0))

        self.statistics_var = tk.StringVar(
            value="Turns: 0 | Average similarity: 0.00 | Escalations: 0 | Fallbacks: 0"
        )
        ttk.Label(self.developer_tab, textvariable=self.statistics_var).grid(
            row=2, column=0, padx=16, pady=8, sticky="w"
        )

        history_actions = ttk.Frame(self.developer_tab)
        history_actions.grid(row=3, column=0, padx=16, pady=4, sticky="w")
        ttk.Button(history_actions, text="Save Conversation Now", command=self.save_history_now).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(history_actions, text="Clear Conversation", command=self.clear_conversation).grid(
            row=0, column=1
        )

        ttk.Label(self.developer_tab, text="Developer Conversation Log", style="Section.TLabel").grid(
            row=4, column=0, padx=16, pady=(8, 2), sticky="w"
        )
        self.developer_log = tk.Text(
            self.developer_tab, wrap="word", state="disabled", font=("Consolas", 10)
        )
        self.developer_log.grid(row=5, column=0, padx=16, pady=(4, 16), sticky="nsew")

    def _set_trained_state(self, trained: bool):
        """Enable chat and artifact saving only when both models are ready."""
        state = "normal" if trained else "disabled"
        self.send_button.config(state=state)
        self.save_training_button.config(state=state)

    def _set_busy(self, busy: bool, status: str | None = None):
        """Disable long-running controls while models load or messages process."""
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.train_button.config(state=state)
        self.browse_kb_button.config(state=state)
        self.load_artifact_button.config(state=state)
        if status is not None:
            self.training_status_var.set(status)

    def browse_knowledge_base(self):
        """Browse for and validate a customer-support CSV knowledge base."""
        filepath = filedialog.askopenfilename(
            initialdir=KNOWLEDGE_BASE_DIR,
            title="Select a customer-support knowledge base",
            filetypes=[("CSV Knowledge Base", "*.csv"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            kb = load_knowledge_base(filepath)
        except (OSError, ValueError) as error:
            messagebox.showerror("Knowledge Base Error", str(error))
            return
        self.knowledge_base = kb
        self.knowledge_base_path = Path(filepath)
        self.knowledge_base_var.set(f"{self.knowledge_base_path.name} ({len(kb)} approved Q&A pairs)")
        self.training_status_var.set("Knowledge base loaded. Click Train / Build Embeddings.")

    def train_clicked(self):
        """Load selected pretrained models and precompute question embeddings."""
        if self.knowledge_base is None:
            messagebox.showwarning("No Knowledge Base", "Select a CSV knowledge base before training.")
            return
        if self.is_busy:
            return
        self._set_busy(True, "Loading AI models and building embeddings. The first run may download model files...")
        self._set_trained_state(False)
        embedding_model = EMBEDDING_MODELS[self.embedding_display_var.get()]
        sentiment_model = SENTIMENT_MODELS[self.sentiment_display_var.get()]
        threading.Thread(
            target=self._train_worker,
            args=(embedding_model, sentiment_model),
            daemon=True,
        ).start()

    def _train_worker(self, embedding_model, sentiment_model):
        """Perform model loading outside Tkinter's event thread."""
        try:
            retriever = SemanticRetriever(
                embedding_model,
                self.knowledge_base.questions,
                self.knowledge_base.answers,
            )
            sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        except Exception as error:
            self.root.after(0, self._training_failed, str(error))
            return
        self.root.after(0, self._training_succeeded, retriever, sentiment_analyzer, None)

    def _training_succeeded(self, retriever, sentiment_analyzer, artifact_path):
        """Install ready model objects and update the interface."""
        self.retriever = retriever
        self.sentiment_analyzer = sentiment_analyzer
        self.artifact_path = artifact_path
        self._set_busy(False)
        self._set_trained_state(True)
        source = f" Loaded from {Path(artifact_path).name}." if artifact_path else ""
        self.training_status_var.set(f"Training ready with {len(retriever.questions)} questions.{source}")
        self._append_chat("system", "Support assistant is ready. Ask a question about the website.")

    def _training_failed(self, error_message):
        """Restore controls and display a training or model-loading error."""
        self._set_busy(False)
        self._set_trained_state(False)
        self.training_status_var.set("Training failed.")
        messagebox.showerror("Training Failed", error_message)

    def save_training_clicked(self):
        """Save current question embeddings, model names, and threshold settings."""
        if self.retriever is None or self.sentiment_analyzer is None:
            return
        filepath = filedialog.asksaveasfilename(
            initialdir=TRAINING_DIR,
            initialfile=f"support_training_{datetime.now():%Y%m%d_%H%M%S}.npz",
            title="Save trained support assistant",
            defaultextension=".npz",
            filetypes=[("Training Artifact", "*.npz"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            self.artifact_path = save_training_artifact(
                filepath,
                questions=self.retriever.questions,
                answers=self.retriever.answers,
                embeddings=self.retriever.question_embeddings,
                embedding_model=self.retriever.model_name,
                sentiment_model=self.sentiment_analyzer.model_name,
                similarity_threshold=self.similarity_slider.get(),
                escalation_threshold=self.escalation_slider.get(),
                knowledge_base_path=self.knowledge_base_path or "",
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Unable to Save Training", str(error))
            return
        self.training_status_var.set(f"Saved training artifact: {self.artifact_path.name}")

    def browse_training_artifact(self):
        """Browse for a previously saved NPZ training artifact."""
        filepath = filedialog.askopenfilename(
            initialdir=TRAINING_DIR,
            title="Load previous training",
            filetypes=[("Training Artifact", "*.npz"), ("All Files", "*.*")],
        )
        if filepath:
            self._load_artifact_async(Path(filepath))

    def _autoload_latest_training(self):
        """Automatically restore the newest saved artifact when available."""
        latest = find_latest_artifact(TRAINING_DIR)
        if latest is not None:
            self._load_artifact_async(latest, automatic=True)

    def _load_artifact_async(self, filepath, automatic=False):
        """Restore an artifact in a worker thread so the GUI stays responsive."""
        if self.is_busy:
            return
        prefix = "Automatically loading" if automatic else "Loading"
        self._set_busy(True, f"{prefix} training artifact: {filepath.name}...")
        self._set_trained_state(False)
        threading.Thread(target=self._artifact_worker, args=(filepath,), daemon=True).start()

    def _artifact_worker(self, filepath):
        """Load saved arrays and instantiate the selected pretrained models."""
        try:
            artifact = load_training_artifact(filepath)
            retriever = SemanticRetriever(
                artifact.embedding_model,
                artifact.questions,
                artifact.answers,
                question_embeddings=artifact.embeddings,
            )
            sentiment_analyzer = SentimentAnalyzer(artifact.sentiment_model)
        except Exception as error:
            self.root.after(0, self._training_failed, str(error))
            return
        self.root.after(0, self._artifact_succeeded, artifact, retriever, sentiment_analyzer)

    def _artifact_succeeded(self, artifact, retriever, sentiment_analyzer):
        """Apply saved thresholds, dropdown choices, data, and model objects."""
        self.knowledge_base = KnowledgeBase(
            filepath=Path(artifact.knowledge_base_path) if artifact.knowledge_base_path else Path("saved_artifact"),
            questions=artifact.questions,
            answers=artifact.answers,
        )
        self.knowledge_base_path = Path(artifact.knowledge_base_path) if artifact.knowledge_base_path else None
        self.knowledge_base_var.set(f"Saved artifact ({len(artifact.questions)} approved Q&A pairs)")
        self.similarity_slider.set(artifact.similarity_threshold)
        self.escalation_slider.set(artifact.escalation_threshold)
        self._select_display_for_model(self.embedding_display_var, EMBEDDING_MODELS, artifact.embedding_model)
        self._select_display_for_model(self.sentiment_display_var, SENTIMENT_MODELS, artifact.sentiment_model)
        self._training_succeeded(retriever, sentiment_analyzer, artifact.filepath)

    @staticmethod
    def _select_display_for_model(variable, mapping, model_name):
        """Choose the human-readable dropdown label for a saved model ID."""
        for display_name, identifier in mapping.items():
            if identifier == model_name:
                variable.set(display_name)
                break

    def send_message(self):
        """Read one customer message and process it in the background."""
        if self.retriever is None or self.sentiment_analyzer is None or self.is_busy:
            return
        user_question = self.message_entry.get("1.0", "end").strip()
        if not user_question:
            return
        self.message_entry.delete("1.0", "end")
        self._append_chat("user", f"You: {user_question}")
        self.send_button.config(state="disabled")
        self.is_busy = True
        threading.Thread(target=self._message_worker, args=(user_question,), daemon=True).start()

    def _message_worker(self, user_question):
        """Run sentiment analysis and semantic retrieval off the GUI thread."""
        try:
            sentiment = self.sentiment_analyzer.analyze(user_question)
            retrieval = self.retriever.find_best_answer(
                user_question,
                similarity_threshold=self.similarity_slider.get(),
                sentiment=sentiment.label,
            )
            escalated = (
                sentiment.label == "NEGATIVE"
                and sentiment.score > self.escalation_slider.get()
            )
        except Exception as error:
            self.root.after(0, self._message_failed, str(error))
            return
        self.root.after(0, self._message_succeeded, user_question, sentiment, retrieval, escalated)

    def _message_succeeded(self, user_question, sentiment, retrieval, escalated):
        """Update customer chat, developer diagnostics, and automatic history."""
        answer = retrieval.answer
        if escalated:
            answer += "\n\nI can also connect you with a human support agent."
        self._append_chat("assistant", f"Assistant: {answer}")

        turn = self.history.add_turn(
            user=user_question,
            answer=answer,
            matched_question=retrieval.matched_question,
            similarity=retrieval.similarity,
            sentiment=sentiment.label,
            sentiment_score=sentiment.score,
            escalated=escalated,
            used_fallback=retrieval.used_fallback,
            search_seconds=retrieval.search_seconds,
            sentiment_seconds=sentiment.analysis_seconds,
        )

        self.matched_question_var.set(retrieval.matched_question or "No confident match")
        self.similarity_var.set(f"{retrieval.similarity:.3f}")
        self.sentiment_var.set(f"{sentiment.label} ({sentiment.score:.3f})")
        self.escalation_var.set("Recommended" if escalated else "Not required")
        self.timing_var.set(
            f"Retrieval {retrieval.search_seconds * 1000:.1f} ms | "
            f"Sentiment {sentiment.analysis_seconds * 1000:.1f} ms"
        )
        self._draw_sentiment_meter(sentiment.meter_value)
        self._append_developer_turn(turn)
        self._update_statistics()

        # Persist after every turn so developer records survive unexpected exits.
        self.history.save_json(self.session_history_path)
        self.is_busy = False
        self.send_button.config(state="normal")
        self.message_entry.focus_set()

    def _message_failed(self, error_message):
        """Restore the send button and show an inference failure."""
        self.is_busy = False
        self.send_button.config(state="normal")
        messagebox.showerror("Message Processing Failed", error_message)

    def _append_chat(self, tag, text):
        """Append colored customer-facing text without exposing diagnostics."""
        self.chat_text.config(state="normal")
        self.chat_text.insert("end", text + "\n\n", tag)
        self.chat_text.config(state="disabled")
        self.chat_text.see("end")

    def _append_developer_turn(self, turn):
        """Append every stored metric for one turn to the backend log."""
        details = (
            f"[{turn.timestamp}]\n"
            f"USER: {turn.user}\n"
            f"MATCH: {turn.matched_question or 'None'}\n"
            f"SIMILARITY: {turn.similarity:.4f}\n"
            f"SENTIMENT: {turn.sentiment} ({turn.sentiment_score:.4f})\n"
            f"ESCALATED: {turn.escalated}\n"
            f"FALLBACK: {turn.used_fallback}\n"
            f"ANSWER: {turn.answer}\n"
            f"TIMING: retrieval={turn.search_seconds:.4f}s, sentiment={turn.sentiment_seconds:.4f}s\n"
            f"{'-' * 78}\n"
        )
        self.developer_log.config(state="normal")
        self.developer_log.insert("end", details)
        self.developer_log.config(state="disabled")
        self.developer_log.see("end")

    def _draw_sentiment_meter(self, value):
        """Draw a red-to-yellow-to-green meter and current sentiment marker."""
        canvas = self.sentiment_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        left, right, top, bottom = 24, width - 24, 22, 44
        steps = max(right - left, 1)

        for index in range(steps):
            normalized = index / max(steps - 1, 1)
            if normalized < 0.5:
                local = normalized / 0.5
                red, green, blue = 220, int(50 + local * 170), 50
            else:
                local = (normalized - 0.5) / 0.5
                red, green, blue = int(220 - local * 170), 220, 50
            canvas.create_line(
                left + index, top, left + index, bottom,
                fill=f"#{red:02x}{green:02x}{blue:02x}"
            )

        canvas.create_text(left, 58, text="Negative", anchor="w")
        canvas.create_text((left + right) / 2, 58, text="Neutral", anchor="center")
        canvas.create_text(right, 58, text="Positive", anchor="e")

        clamped = max(-1.0, min(1.0, float(value)))
        marker_x = left + ((clamped + 1.0) / 2.0) * (right - left)
        canvas.create_polygon(marker_x, top - 10, marker_x - 7, top - 1, marker_x + 7, top - 1, fill="#212121")
        canvas.create_line(marker_x, top - 1, marker_x, bottom + 3, fill="#212121", width=2)

    def _update_statistics(self):
        """Refresh aggregate conversation statistics in the developer view."""
        stats = self.history.statistics()
        self.statistics_var.set(
            f"Turns: {stats['count']} | Average similarity: {stats['average_similarity']:.2f} | "
            f"Escalations: {stats['escalations']} | Fallbacks: {stats['fallbacks']}"
        )

    def save_history_now(self):
        """Let the developer save the current JSON conversation explicitly."""
        if not self.history.turns:
            messagebox.showinfo("No Conversation", "There are no conversation turns to save yet.")
            return
        filepath = filedialog.asksaveasfilename(
            initialdir=HISTORY_DIR,
            initialfile=self.session_history_path.name,
            title="Save developer conversation history",
            defaultextension=".json",
            filetypes=[("JSON Conversation", "*.json"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            self.session_history_path = self.history.save_json(filepath)
        except OSError as error:
            messagebox.showerror("Unable to Save History", str(error))
            return
        messagebox.showinfo("History Saved", f"Conversation saved to:\n{self.session_history_path}")

    def clear_conversation(self):
        """Clear customer chat, developer diagnostics, and in-memory history."""
        if not messagebox.askyesno("Clear Conversation", "Clear the current conversation and developer history?"):
            return
        self.history = ConversationHistory()
        for widget in (self.chat_text, self.developer_log):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.config(state="disabled")
        for variable in (
            self.matched_question_var,
            self.similarity_var,
            self.sentiment_var,
            self.escalation_var,
            self.timing_var,
        ):
            variable.set("—")
        self._draw_sentiment_meter(0.0)
        self._update_statistics()
        self.session_history_path = HISTORY_DIR / f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json"

    def _on_close(self):
        """Automatically save any remaining developer history before exit."""
        if self.history.turns:
            try:
                self.history.save_json(self.session_history_path)
            except OSError:
                pass
        self.root.destroy()


def start_gui():
    """Create the Tkinter root window and start the event loop."""
    root = tk.Tk()
    CustomerSupportGUI(root)
    root.mainloop()
