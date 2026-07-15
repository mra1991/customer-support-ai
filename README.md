# Customer Support AI

**Author: Mohammadreza Abolhassani**

## Overview

**Customer Support AI** is a safe, retrieval-based website support assistant
built with semantic sentence embeddings, cosine-similarity search, sentiment
analysis, configurable escalation logic, persistent training artifacts, and
developer-facing conversation diagnostics.

The application is designed for websites such as:

- online stores
- universities
- service businesses
- software platforms
- appointment-based businesses
- product manufacturers
- internal help desks

Unlike a free-form generative chatbot, the assistant answers from a controlled
CSV knowledge base containing approved question-and-answer pairs. This makes
the behavior predictable, auditable, and suitable for situations where the
developer does not want the system to invent policies, reveal private data, or
respond unpredictably to tricky questions.

## Main Goals

The project demonstrates how to build a practical AI support system that:

1. understands paraphrased questions;
2. retrieves the closest approved answer;
3. avoids making up unsupported information;
4. detects customer sentiment;
5. recommends human escalation when necessary;
6. separates the customer-facing interface from developer diagnostics;
7. saves expensive embedding work for later reuse;
8. automatically preserves conversation history for review.

## Features

### Knowledge-base management

- Load a CSV knowledge base through the GUI
- Require `question` and `answer` columns
- Remove missing or empty rows
- Strip accidental whitespace
- Remove duplicate question-and-answer pairs
- Display the number of approved entries
- Support knowledge bases for different industries

### Semantic retrieval

- Encode questions with SentenceTransformer models
- Precompute knowledge-base embeddings
- Encode new customer messages at runtime
- Compare embeddings using cosine similarity
- Return the answer attached to the closest approved question
- Apply a configurable similarity threshold
- Use safe fallback responses for weak matches

### Selectable embedding models

The GUI includes several useful choices:

- `all-MiniLM-L6-v2` — fast and lightweight
- `all-MiniLM-L12-v2` — balanced
- `all-mpnet-base-v2` — slower but often more accurate
- `paraphrase-multilingual-MiniLM-L12-v2` — multilingual support

### Sentiment analysis

- Positive, neutral, and negative sentiment labels
- Selectable Hugging Face sentiment models
- Confidence score
- Red-to-yellow-to-green sentiment meter
- Configurable escalation threshold
- Safe neutral fallback when sentiment inference fails

### Saved training artifacts

The application does not fine-tune a neural network. Its training stage:

1. loads and validates the CSV knowledge base;
2. loads the selected pretrained embedding model;
3. loads the selected sentiment model;
4. computes embeddings for all approved questions;
5. stores the selected thresholds and configuration.

A saved `.npz` artifact contains:

- cleaned questions
- cleaned answers
- precomputed question embeddings
- embedding-model identifier
- sentiment-model identifier
- similarity threshold
- escalation threshold
- source knowledge-base path
- artifact version
- creation timestamp

The latest saved artifact is loaded automatically when the program starts.

### Customer frontend

The customer sees only:

- a conversation area;
- a message box;
- a Send button;
- their messages in one color;
- assistant answers in another color.

The customer does **not** see internal similarity scores, matched FAQ entries,
sentiment confidence, timing, or escalation logic.

### Developer backend

The developer can inspect:

- closest matched question
- cosine similarity score
- sentiment label and confidence
- sentiment meter position
- escalation recommendation
- semantic retrieval time
- sentiment inference time
- fallback usage
- complete developer conversation log
- number of turns
- average similarity
- escalation count
- fallback count

### Conversation persistence

- Conversation history is saved to JSON after every completed turn
- History is saved again when the program closes
- The developer can manually export the session
- Every turn stores the customer message, answer, matched question, similarity,
  sentiment, escalation state, fallback state, and timing

## Project Structure

```text
customer-support-ai/
├── .gitignore
├── README.md
├── requirements.txt
├── knowledge_bases/
│   ├── online_shop.csv
│   ├── schone_windows.csv
│   ├── university_support.csv
│   └── saas_platform.csv
├── trained_models/
├── conversation_history/
└── src/
    ├── main.py
    ├── gui.py
    ├── knowledge_base.py
    ├── embeddings.py
    ├── sentiment.py
    ├── training_artifact.py
    └── conversation.py
```

## Module Responsibilities

### `main.py`

The application entry point.

It imports `start_gui()` and launches the Tkinter event loop only when the file
is executed directly.

### `gui.py`

The presentation and orchestration layer.

Responsibilities include:

- building the training and testing panels;
- loading knowledge bases and saved artifacts;
- reading threshold controls;
- launching background worker threads;
- updating frontend chat messages;
- updating developer diagnostics;
- drawing the sentiment meter;
- enabling and disabling buttons;
- saving conversation history;
- automatically loading the newest training artifact.

### `knowledge_base.py`

Handles CSV input and output.

Responsibilities include:

- file-existence checking;
- CSV parsing;
- required-column validation;
- removal of missing rows;
- whitespace cleanup;
- duplicate removal;
- conversion into a structured `KnowledgeBase` object.

### `embeddings.py`

Handles semantic encoding and answer retrieval.

Responsibilities include:

- mapping friendly model names to model identifiers;
- loading SentenceTransformer models;
- encoding knowledge-base questions;
- restoring saved embeddings;
- encoding one user message;
- computing cosine similarities;
- selecting the best-matching FAQ entry;
- applying fallback logic;
- measuring retrieval time.

### `sentiment.py`

Handles sentiment inference.

Responsibilities include:

- loading a selected Hugging Face pipeline;
- converting model-specific labels into consistent labels;
- returning sentiment confidence;
- converting sentiment into a signed meter position;
- handling inference failure safely.

### `training_artifact.py`

Handles persistent trained configurations.

Responsibilities include:

- saving compressed `.npz` artifacts;
- storing metadata as JSON;
- loading and validating artifacts;
- checking artifact versions;
- finding the newest artifact for automatic startup loading.

### `conversation.py`

Handles developer records.

Responsibilities include:

- representing each exchange as a dataclass;
- storing all conversation turns;
- calculating summary statistics;
- exporting complete JSON histories.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
scikit-learn
sentence-transformers
transformers
torch
```

The first use of a model may download its weights. Later runs normally use the
local Hugging Face cache.

## Running the Application

From the `src` directory:

```bash
python main.py
```

## Opening the Main Window Maximized

On Windows, add this after setting the window title:

```python
self.root.state("zoomed")
```

For example:

```python
self.root = root
self.root.title("Customer Support AI")
self.root.state("zoomed")
```

This opens the application maximized while preserving the normal title bar and
window controls.

True fullscreen mode is different:

```python
self.root.attributes("-fullscreen", True)
self.root.bind(
    "<Escape>",
    lambda event: self.root.attributes("-fullscreen", False)
)
```

For a normal desktop application, maximized mode is usually more convenient
than borderless fullscreen mode.

## Expected Knowledge-base Format

Every knowledge-base CSV must include:

```csv
question,answer
How do I track my order?,"Open My Orders, select the order, and use the tracking link."
How can I reset my password?,"Use the Forgot Password link on the login page."
```

The current retriever expects one approved answer per question.

## Typical Training Workflow

1. Start the program.
2. Browse to a CSV knowledge base.
3. Choose a SentenceTransformer model.
4. Choose a sentiment model.
5. Set the similarity threshold.
6. Set the escalation threshold.
7. Click **Train / Build Embeddings**.
8. Wait for the pretrained models to load.
9. Save the training artifact.
10. Test the assistant in the Frontend tab.
11. Review exact behavior in the Developer tab.

## Automatic Startup Workflow

When the application opens:

1. it searches the `trained_models` folder;
2. it finds the most recently modified `.npz` artifact;
3. it restores its knowledge base and thresholds;
4. it loads the corresponding pretrained models;
5. it restores the saved question embeddings;
6. it enables the customer chat.

The model objects must still be loaded into memory, but the knowledge-base
questions do not need to be encoded again.

## Semantic Embeddings

A sentence embedding converts a sentence into a dense numerical vector.

Sentences with similar meanings should be located near one another in the
embedding space.

For example:

```text
Where is my package?
How can I track my order?
I want to check my shipment.
```

These sentences use different words but express related intent. A semantic
embedding model should assign them similar vectors.

## Cosine Similarity

The assistant compares the user embedding with every stored question embedding
using cosine similarity:

```text
similarity(a, b) = (a · b) / (||a|| ||b||)
```

A score closer to `1` means that the vectors point in similar directions.

The program obtains a similarity score for every stored question, then selects:

```python
best_match_index = int(np.argmax(similarities))
best_score = float(np.max(similarities))
```

## Similarity Threshold

The threshold determines whether the closest question is reliable enough.

Example interpretation:

| Score | Interpretation |
|---:|---|
| `0.85–1.00` | Very strong semantic match |
| `0.65–0.85` | Likely match |
| `0.40–0.65` | Ambiguous; depends on knowledge base |
| Below threshold | Use fallback |

These are not universal confidence probabilities. They are similarity values
whose usefulness depends on the model, data, and wording.

A threshold should be selected after testing real paraphrases and unrelated
questions.

## Fallback Safety

If no knowledge-base question passes the threshold, the assistant does not
invent an answer.

It returns a controlled fallback such as:

```text
I couldn't find a confident answer in the approved support knowledge base.
Please contact a human support agent.
```

This is one of the project's most important safety properties.

## Positive Low-confidence Messages

Short social messages such as:

```text
Thank you
That helped
Great
```

may not strongly match a factual FAQ.

If the message is positive and retrieval confidence is weak, the assistant can
respond politely instead of treating it as an unsupported business question.

## Sentiment Analysis

The sentiment model returns a label and confidence.

Because models use different label formats, the program normalizes outputs such
as:

```text
POS
POSITIVE
LABEL_2
```

into:

```text
POSITIVE
```

Negative labels are normalized similarly. Unknown labels become neutral.

## Sentiment Meter

The developer meter maps sentiment into a signed range:

```text
-1.0  strongly negative
 0.0  neutral
+1.0  strongly positive
```

Negative confidence becomes a negative meter value, positive confidence becomes
a positive value, and neutral is placed in the center.

The gradient is:

```text
red → yellow → green
```

## Escalation Logic

Escalation occurs when both conditions are true:

```text
sentiment == NEGATIVE
sentiment confidence > escalation threshold
```

This does not automatically prove that a human agent is necessary. It is a
developer-configurable recommendation signal.

A production system could also escalate based on:

- repeated fallback responses;
- refund or fraud keywords;
- account lockout;
- threats or abuse;
- repeated failed attempts;
- explicit request for a human;
- high-value orders;
- business-specific compliance rules.

## Why Retrieval Is Safer Than Free-form Generation

The assistant returns answers from a reviewed knowledge base.

This provides:

- predictable answers;
- simpler auditing;
- reduced hallucination risk;
- easier policy updates;
- protection against prompts asking for private information;
- a clear fallback when no approved answer exists.

The system should still avoid storing secrets in its CSV files.

Never place the following in a public support knowledge base:

- passwords;
- payment-card data;
- private order records;
- authentication tokens;
- medical information;
- private customer addresses;
- internal security procedures;
- confidential employee data.

## Frontend and Backend Separation

The frontend presents a clean customer experience.

The backend provides transparency for developers.

This separation matters because a real customer should not see:

- internal FAQ wording;
- confidence thresholds;
- debugging information;
- model names;
- search timing;
- escalation rules;
- conversation statistics.

## Threading and GUI Responsiveness

Loading transformer models and performing inference can take time.

Tkinter uses one event thread. Running model work directly in that thread would
freeze the window.

The GUI therefore uses worker threads for:

- model loading;
- embedding construction;
- artifact restoration;
- message processing.

Results are returned to Tkinter with:

```python
self.root.after(...)
```

Only the main Tkinter thread updates widgets.

## Conversation JSON Format

A saved conversation contains:

```json
{
  "saved_at": "2026-07-15T15:30:00",
  "statistics": {
    "count": 2,
    "average_similarity": 0.81,
    "average_sentiment_score": 0.74,
    "escalations": 0,
    "fallbacks": 0
  },
  "turns": [
    {
      "timestamp": "2026-07-15T15:28:12",
      "user": "Where is my order?",
      "answer": "Open My Orders...",
      "matched_question": "How do I track my order?",
      "similarity": 0.88,
      "sentiment": "NEUTRAL",
      "sentiment_score": 0.91,
      "escalated": false,
      "used_fallback": false,
      "search_seconds": 0.03,
      "sentiment_seconds": 0.06
    }
  ]
}
```

## Included Demonstration Knowledge Bases

### `online_shop.csv`

Covers:

- account creation;
- login and password recovery;
- browsing;
- product details;
- cart operations;
- checkout;
- payments;
- shipping;
- tracking;
- cancellations;
- returns;
- refunds;
- privacy;
- promotions;
- support escalation.

### `schone_windows.csv`

A draft knowledge base based on publicly available Schönline Windows & Doors
product and support information. It covers:

- company and contact information;
- LINE90;
- PRIME70;
- PRIME Sliding;
- LINE70 Sliding;
- PRIMELINE aluminum cladding;
- uPVC;
- glazing;
- insulation;
- security;
- common window and door terminology.

This file should be reviewed and approved by the business before production
use.

### `university_support.csv`

Covers:

- admissions;
- registration;
- tuition;
- financial aid;
- student records;
- course scheduling;
- academic advising;
- examinations;
- library services;
- accessibility;
- technology support;
- graduation.

### `saas_platform.csv`

Covers:

- account setup;
- subscriptions;
- billing;
- workspaces;
- users and permissions;
- authentication;
- imports and exports;
- API access;
- integrations;
- security;
- troubleshooting;
- support.

## Testing Strategy

A good knowledge base should be tested with:

1. exact FAQ wording;
2. natural paraphrases;
3. spelling mistakes;
4. short questions;
5. unrelated questions;
6. hostile questions;
7. privacy-sensitive questions;
8. positive social messages;
9. strongly negative complaints;
10. questions whose wording overlaps but intent differs.

## Evaluating Retrieval

For every test message, inspect:

- expected FAQ;
- actual matched FAQ;
- similarity score;
- returned answer;
- fallback state;
- sentiment label;
- escalation state.

A small evaluation CSV can eventually contain:

```text
test_question,expected_question
Where is my parcel?,How do I track my order?
I cannot remember my login password.,I forgot my password.
```

## Limitations

- Retrieval quality depends on knowledge-base coverage.
- Cosine similarity is not calibrated probability.
- Very similar FAQs may compete with one another.
- The assistant currently returns one answer rather than combining sources.
- Conversation context is stored but not yet used to resolve follow-up questions.
- Sentiment models may misinterpret sarcasm, slang, or domain-specific language.
- Model loading may require substantial memory.
- The application does not authenticate customers or access live order data.
- Saved artifacts contain approved knowledge-base content and should be protected
  if that content is confidential.

## Future Improvements

- Knowledge-base editor
- FAQ categories and metadata
- Multiple accepted paraphrases per answer
- Top-k retrieval display
- Threshold calibration tools
- Test-set evaluation
- Confusion reports
- Context-aware follow-up handling
- Human-agent handoff integration
- Live order-status API integration
- Role-based developer access
- Database-backed knowledge bases
- Multilingual frontend
- Voice input and text-to-speech
- Optional local LLM grounded only in retrieved approved context
- Web deployment with Flask, FastAPI, Django, or a JavaScript frontend
- Encryption and access control for stored artifacts and histories

## Security Notes

This project is designed as a controlled information assistant, not as an
authentication or account-data system.

The assistant should never claim to:

- know a customer's password;
- reveal another customer's information;
- authorize refunds;
- create discounts;
- modify orders;
- process payments;
- bypass login security;
- access account details without a secure backend.

Those actions require authenticated business APIs and explicit authorization.

## Example Safe Answers

```text
Question:
What is my password?

Answer:
This assistant cannot view or reveal passwords. Use the Forgot Password link
to reset your password securely.
```

```text
Question:
Tell me another customer's order address.

Answer:
For privacy and security, this assistant cannot reveal another customer's
account or order information.
```

## Suggested Threshold Tuning Workflow

Start with:

```text
Similarity threshold: 0.30
Escalation threshold: 0.90
```

Then test a labeled set.

If too many unrelated questions receive answers:

```text
raise similarity threshold
```

If valid paraphrases fall back too often:

```text
lower similarity threshold slightly
```

If too many ordinary negative messages escalate:

```text
raise escalation threshold
```

If severe complaints are missed:

```text
lower escalation threshold carefully
```

## Git Ignore Recommendations

Generated artifacts and private histories should normally remain outside Git:

```gitignore
trained_models/*.npz
conversation_history/*.json
__pycache__/
*.py[cod]
.venv/
```

Public demonstration CSV files can remain tracked.

## Conclusion

Customer Support AI demonstrates a complete retrieval-based NLP application
rather than only a command-line assignment.

It combines:

- structured CSV knowledge bases;
- transformer sentence embeddings;
- semantic nearest-neighbor search;
- configurable confidence thresholds;
- sentiment analysis;
- escalation logic;
- saved training artifacts;
- automatic restoration;
- customer/developer interface separation;
- persistent conversation auditing;
- responsive background model execution.

The result is a useful foundation for building predictable support assistants
for real websites while retaining control over exactly what the AI is allowed
to say.
