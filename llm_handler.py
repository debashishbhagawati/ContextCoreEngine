import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from context_handler.sliding_window import SlidingWindowContext
from context_handler.summarizing_window import SummarizingWindowContext
from context_handler.conversation_tree_context import ConversationTreeContext
from context_handler.embedding_retrieval import EmbeddingRetrievalContext

load_dotenv()
api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_context_handler(mode, context_length=10, min_summary_length=10, max_summary_length=1000):
    if mode == "Sliding Window":
        return SlidingWindowContext(max_turns=context_length)
    elif mode == "Summarizing Window":
        return SummarizingWindowContext(max_turns=context_length, summary_min_length=min_summary_length, summary_max_length=max_summary_length)
    elif mode == "Conversation Tree":
        return ConversationTreeContext(depth=context_length)
    elif mode == "Embedding Retrieval":
        return EmbeddingRetrievalContext(max_recent_turns=context_length, top_k=3)
    return SlidingWindowContext(max_turns=context_length)

def get_response(prompt, history_context, parent_id=None):
    is_tree = isinstance(history_context, ConversationTreeContext)
    is_embedding = isinstance(history_context, EmbeddingRetrievalContext)

    if is_tree:
        history_context.add_turn({"role": "user", "parts": [prompt]}, parent_id=parent_id)
        chat_history = history_context.get_context(from_message_id=history_context.last_message_id)
        chat = model.start_chat(history=chat_history)
    elif is_embedding:
        history_context.add_turn({"role": "user", "parts": [prompt]})
        chat_history = history_context.get_context(prompt)
        chat = model.start_chat(history=chat_history)
    else:
        history_context.add_turn({"role": "user", "parts": [prompt]})
        chat = model.start_chat(history=history_context.get_context())

    response = chat.send_message(prompt)

    if is_tree:
        history_context.add_turn({"role": "model", "parts": [response.text]}, parent_id=history_context.last_message_id)
    else:
        history_context.add_turn({"role": "model", "parts": [response.text]})

    return response.text.strip()

def BotOperator():
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("💬 Context Core Engine")

    mode = st.sidebar.selectbox(
        "Choose Context Mode",
        ["Sliding Window", "Summarizing Window", "Conversation Tree", "Embedding Retrieval"]
    )

    context_length = st.sidebar.slider(
        "Set Context Length",
        min_value=2,
        max_value=128000,
        value=10,
        step=1
    )

    # Defaults
    summary_min = 10
    summary_max = 1000

    if mode == "Summarizing Window":
        summary_min, summary_max = st.sidebar.slider(
            "Set Summary Length Range (min to max)",
            min_value=1,
            max_value=30000,
            value=(10, 1000),
            step=1
        )

    if "context_mode" not in st.session_state or st.session_state.context_mode != mode:
        st.session_state.context_mode = mode
        st.session_state.history_context = get_context_handler(
            mode,
            context_length=context_length,
            min_summary_length=summary_min,
            max_summary_length=summary_max,
        )
        st.session_state.chat_log = []

    history_context = st.session_state.history_context

    parent_id = None
    parent_options = []
    parent_labels = []
    parent_map = {}

    with st.form(key="chat_form", clear_on_submit=True):
        show_parent_dropdown = (
            isinstance(history_context, ConversationTreeContext)
            and len(history_context.message_nodes) > 0
        )
        if show_parent_dropdown:
            parent_options = [
                (f"{mid}: {node.role}: {node.content[:40]}", mid)
                for mid, node in history_context.message_nodes.items()
            ]
            parent_options.sort(key=lambda x: int(x[1]) if str(x[1]).isdigit() else x[1])
            parent_labels = [label for label, _ in parent_options]
            parent_map = {label: mid for label, mid in parent_options}
            default_index = len(parent_labels) - 1 if parent_labels else 0

            selected_label = st.selectbox(
                "Select parent message to tag:",
                parent_labels,
                key="parent_selector",
                index=default_index
            )
            parent_id = parent_map[selected_label]
        elif isinstance(history_context, ConversationTreeContext):
            st.info("No messages yet. Your first message will start the conversation.")

        user_input = st.text_input("Your message", key="user_input")
        send_disabled = show_parent_dropdown and not parent_id
        send = st.form_submit_button("Send", disabled=send_disabled)

    if send and user_input.strip():
        with st.spinner("Thinking..."):
            if isinstance(history_context, ConversationTreeContext):
                response = get_response(user_input.strip(), history_context, parent_id=parent_id)
            else:
                response = get_response(user_input.strip(), history_context)
            st.session_state.chat_log.append({"role": "user", "text": user_input.strip()})
            st.session_state.chat_log.append({"role": "bot", "text": response})
        st.rerun()

    # Chat display
    st.markdown("""
        <style>
        .chat-bubble {
            max-width: 60%;
            padding: 1em;
            margin: 0.5em 0;
            border-radius: 1em;
            font-size: 1.1em;
            line-height: 1.5;
            word-break: break-word;
            color: #222 !important;
        }
        .user-bubble {
            background-color: #DCF8C6;
            margin-left: auto;
            margin-right: 0;
            text-align: right;
        }
        .bot-bubble {
            background-color: #F1F0F0;
            margin-right: auto;
            margin-left: 0;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### Conversation")
    for msg in reversed(st.session_state.chat_log):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble user-bubble">🧑‍💼 <b>You:</b> {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble bot-bubble">🤖 <b>Bot:</b> {msg["text"]}</div>',
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    BotOperator()