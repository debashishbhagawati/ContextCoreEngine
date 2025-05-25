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

def get_context_handler(mode):
    if mode == "Sliding Window":
        return SlidingWindowContext(max_turns=10)
    elif mode == "Summarizing Window":
        return SummarizingWindowContext(max_turns=10)
    elif mode == "Conversation Tree":
        return ConversationTreeContext(depth=10)
    elif mode == "Embedding Retrieval":
        return EmbeddingRetrievalContext(max_recent_turns=10, top_k=3)
    return SlidingWindowContext(max_turns=10)


# Update get_response to accept parent_id
def get_response(prompt, history_context, parent_id=None):
    is_tree = isinstance(history_context, ConversationTreeContext)
    is_embedding = isinstance(history_context, EmbeddingRetrievalContext)

    if is_tree:
        # Add user turn to the selected parent branch
        history_context.add_turn({"role": "user", "parts": [prompt]}, parent_id=parent_id)
        # Build context from the selected parent branch
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
        # Add model turn to the same branch
        history_context.add_turn({"role": "model", "parts": [response.text]}, parent_id=history_context.last_message_id)
    elif is_embedding:
        history_context.add_turn({"role": "model", "parts": [response.text]})
    else:
        history_context.add_turn({"role": "model", "parts": [response.text]})

    return response.text.strip()

def main():
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("💬 Chatbot with Custom Context Handling")

    mode = st.sidebar.selectbox("Choose Context Mode", [
        "Sliding Window", "Summarizing Window", "Conversation Tree", "Embedding Retrieval"
    ])

    if "context_mode" not in st.session_state or st.session_state.context_mode != mode:
        st.session_state.context_mode = mode
        st.session_state.history_context = get_context_handler(mode)
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
        parent_id = None

        if show_parent_dropdown:
            parent_options = [
                (f"{mid}: {node.role}: {node.content[:40]}", mid)
                for mid, node in history_context.message_nodes.items()
            ]
            parent_options.sort(key=lambda x: int(x[1]) if str(x[1]).isdigit() else x[1])
            parent_labels = [label for label, _ in parent_options]
            parent_map = {label: mid for label, mid in parent_options}

            # Default to the last message (most recent) as parent
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
        send_disabled = False
        if show_parent_dropdown and not parent_id:
            send_disabled = True
        send = st.form_submit_button("Send", disabled=send_disabled)

    # Process input and update chat log BEFORE displaying conversation
    if send and user_input.strip():
        with st.spinner("Thinking..."):
            if isinstance(history_context, ConversationTreeContext):
                response = get_response(user_input.strip(), history_context, parent_id=parent_id)
            else:
                response = get_response(user_input.strip(), history_context)
            st.session_state.chat_log.append({"role": "user", "text": user_input.strip()})
            st.session_state.chat_log.append({"role": "bot", "text": response})
        st.rerun()  # <-- Add this line to immediately update the parent dropdown

    # Display previous conversation with ChatGPT-like bubbles
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
    main()