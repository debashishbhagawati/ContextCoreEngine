# ContextCoreEngine

_ContextCoreEngine_ is an advanced, modular, and extensible Python engine for **context-aware conversational AI**. Designed for next-generation chatbots and assistants, it offers a suite of context management strategies—individually and in combination—so your AI can remember, recall, summarize, and reason over long, complex conversations like never before.

---

## 🚀 Why ContextCoreEngine?

Modern conversational AI faces three big challenges:
- **Maintaining relevance:** Users expect bots to "remember" recent exchanges.
- **Handling long or complex conversations:** Bots need to retain crucial context without running into memory or token limits.
- **Providing intelligent responses:** The right context must be surfaced for the LLM to give smart, on-topic answers.

**ContextCoreEngine** solves these challenges by offering a range of memory architectures—each inspired by best practices in AI and cognitive science—culminating in a sophisticated hybrid system that combines their strengths.

---

## 🧠 Context Management Architectures

### 1. Sliding Window Context

- **What it does:**  
  Retains the last N user-bot exchanges.
- **When to use:**  
  When recency is most important for conversational coherence, and you want a simple, fast approach.
- **Strength:**  
  Fast, minimal memory, ensures latest messages are always included.

---

### 2. Summarizing Window Context

- **What it does:**  
  Periodically compresses older conversation turns into summaries using an LLM or summarizer model.
- **When to use:**  
  When you want to preserve long-term context without exceeding token/memory limits.
- **Strength:**  
  Retains key information from older turns while freeing up space for new conversation.

---

### 3. Embedding-Based (Semantic Retrieval) Context

- **What it does:**  
  Encodes all conversation turns into embeddings and, on each prompt, fetches the top-k past messages most semantically relevant to the current query.
- **When to use:**  
  When long-term, non-linear memory is crucial—such as technical support, tutoring, or any scenario where users may reference information from far earlier in the conversation.
- **Strength:**  
  Mimics human recall of relevant facts, regardless of recency.

---

### 4. Conversation Tree Context

- **What it does:**  
  Supports branching and forking dialogues, allowing multiple parallel threads or "undo/redo" conversational paths.
- **When to use:**  
  Ideal for multi-user chat, collaborative agents, or explorative conversations where users may want to revisit or fork from earlier points.
- **Strength:**  
  Enables powerful dialog control, history navigation, and advanced AI assistant behaviors.

---

### 5. Hybrid Context Handling (NEW!)

- **What it does:**  
  Combines Sliding Window, Summarizing, Embedding-Based, and Conversation Tree strategies into a single, unified memory system.
- **How it works:**  
  - Retains recent turns (recency)
  - Summarizes and compresses older context (summarization)
  - Fetches semantically relevant context from anywhere in the conversation (embedding retrieval)
  - Maintains branching/threaded memory (tree)
  - Deduplicates and merges all sources for a maximally informative LLM prompt
- **When to use:**  
  When you want the best of all worlds—continuity, long-term memory, semantic focus, and dialog navigation.
- **Strength:**  
  Delivers state-of-the-art conversational context, optimized for both coherence and depth.

---

## 🏗️ Project Structure

```
main.py                        # Streamlit entry point
llm_handler.py                 # LLM and context handler orchestration
context_handler/
    sliding_window.py          # Sliding Window context logic
    summarizing_window.py      # Summarizing Window with HuggingFace
    conversation_tree_context.py # Tree-based context
    embedding_retrieval.py     # Embedding-based retrieval
    hybrid_context.py          # Hybrid context handler (combines all)
requirements.txt               # Python dependencies
.env.example                   # Example environment file
```

---

## ⚡ How It Works

1. **User enters a message** in the Streamlit UI.
2. **LLM Handler** routes the message to the selected context handler (Sliding, Summarizing, Embedding, Tree, or Hybrid).
3. **Context Handler** assembles the best possible context for that mode.
4. **LLM (e.g., Gemini)** receives the prompt and context, generating a response.
5. **Response & context** are displayed and updated—ensuring future turns benefit from the full memory and intelligence of the system.

---

## 🗂️ Architecture Overview

```
User Input
    |
    v
Streamlit UI (User selects mode/parameters)
    |
    v
LLM Handler (llm_handler.py)
    |
    v
Context Handler
    |--- Sliding Window
    |--- Summarizing Window
    |--- Embedding Retrieval
    |--- Conversation Tree
    |--- Hybrid Context (merges all)
    |
    v
Context Assembly (merges recency, summary, semantic, tree)
    |
    v
LLM (Gemini, etc.)
    |
    v
Output to User (Streamlit UI)
    |
    v
Context Update (add user & bot turns to all context handlers)
```

---

## 🌟 Significance & Impact

- **Human-like contextual memory:**  
  Mimics how people recall not just the latest, but also the most important and relevant information.
- **Tame the “token limit” problem:**  
  Summarization and semantic retrieval keep the LLM focused and efficient, even in very long conversations.
- **Branching & collaboration:**  
  Conversation trees unlock new dialog capabilities for collaborative agents or complex customer support.
- **Plug & play for research and production:**  
  Modular design means you can swap out LLMs, context strategies, or frontends with ease.

---

## 🛠️ Getting Started

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

---

## 🕹️ Usage

- Launch the app and select your preferred context management mode in the sidebar.
- Adjust context length and summary settings as needed.
- Chat with the bot and observe context-aware responses.

---

## 🧩 Extending

- To add new context strategies, implement a class in `context_handler/` and register it in `llm_handler.py`.
- To use a different LLM provider, abstract the LLM calls in `llm_handler.py`.

---

## 🚧 Roadmap

- [ ] Pluggable LLM backend (OpenAI, Cohere, etc.)
- [ ] Persistent session storage
- [ ] REST API for backend
- [ ] Docker support
- [ ] Enhanced UI and analytics

---

## 📜 License

MIT License

---

**Author:** [debashishbhagawati](https://github.com/debashishbhagawati)
