# ContextCoreEngine

A modular, Python-based engine for building context-aware conversational AI applications, supporting multiple context management strategies and LLM integrations.

## Features

- **Flexible Context Management:**  
  Choose from Sliding Window, Summarizing Window, Conversation Tree, Embedding Retrieval, and Hybrid modes for handling chat history and context.

- **Powerful LLM Integration:**  
  Uses Google's Gemini API for generating responses, with support for additional LLMs possible.

- **Semantic Search & Summarization:**  
  Embedding-based retrieval (`sentence-transformers`), HuggingFace `transformers` for dynamic summarization.

- **Interactive Streamlit UI:**  
  Chat with the bot, select modes, and adjust parameters in real-time.

## 🧠 Hybrid Context Handling (NEW!)

ContextCoreEngine now supports a **Hybrid Context Handling Architecture** that combines:
- **Sliding Window**: Keeps recent messages for recency.
- **Summarization**: Periodically summarizes older chunks for long-term memory.
- **Embedding Retrieval**: Fetches the most relevant historical messages by semantic similarity.
- **Conversation Tree**: Supports branching conversations or undo/redo.

When in Hybrid mode, the engine intelligently merges recency, summaries, and top-k relevant messages, providing the LLM with rich, non-redundant context.

**How it works:**
1. Retains the last N turns (Sliding Window).
2. Periodically summarizes older context (Summarization).
3. On every prompt, retrieves top-k semantically similar turns (Embedding Retrieval).
4. Optionally, reconstructs conversation paths for multi-branch dialogs (Tree).

This hybrid approach balances immediate relevance, long-term memory, and semantic focus for best-in-class conversational AI.

## Project Structure

```
main.py                        # Streamlit entry point
llm_handler.py                 # LLM and context handler orchestration
context_handler/
    sliding_window.py          # Sliding Window context logic
    summarizing_window.py      # Summarizing Window with HuggingFace
    conversation_tree_context.py # Tree-based context
    embedding_retrieval.py     # Embedding-based retrieval
    hybrid_context.py          # Hybrid context handler (NEW)
requirements.txt               # Python dependencies
.env.example                   # Example environment file
```

## Getting Started

### Prerequisites

- Python 3.8+
- API key for Google Gemini (or other LLM provider)

### Installation

```bash
git clone https://github.com/debashishbhagawati/ContextCoreEngine.git
cd ContextCoreEngine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # Add your API_KEY to .env
```

### Running the Chatbot

```bash
streamlit run main.py
```

### Environment Variables

- `API_KEY`: Your LLM provider API key (Google Gemini by default).

## Usage

- Launch the app and select your preferred context management mode in the sidebar.
- Adjust context length and summary settings as needed.
- Chat with the bot and observe context-aware responses.

## Extending

- To add new context strategies, implement a class in `context_handler/` and register it in `llm_handler.py`.
- To use a different LLM provider, abstract the LLM calls in `llm_handler.py`.

## TODO & Roadmap

- [ ] Pluggable LLM backend (OpenAI, Cohere, etc.)
- [ ] Persistent session storage
- [ ] REST API for backend
- [ ] Docker support
- [ ] Enhanced UI and analytics

## License

MIT License

---

**Author:** [debashishbhagawati](https://github.com/debashishbhagawati)