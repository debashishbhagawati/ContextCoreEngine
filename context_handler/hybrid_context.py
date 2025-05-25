from context_handler.sliding_window import SlidingWindowContext
from context_handler.summarizing_window import SummarizingWindowContext
from context_handler.embedding_retrieval import EmbeddingRetrievalContext
from context_handler.conversation_tree_context import ConversationTreeContext

class HybridContextHandler:
    def __init__(self, max_recent_turns=4, top_k=3, summary_min_length=10, summary_max_length=100, tree_depth=5):
        self.sliding = SlidingWindowContext(max_turns=max_recent_turns)
        self.summarizer = SummarizingWindowContext(
            max_turns=max_recent_turns,
            summary_min_length=summary_min_length,
            summary_max_length=summary_max_length
        )
        self.embedding = EmbeddingRetrievalContext(
            max_recent_turns=max_recent_turns,
            top_k=top_k
        )
        self.tree = ConversationTreeContext(depth=tree_depth)

    def add_turn(self, turn, parent_id=None):
        self.sliding.add_turn(turn)
        self.summarizer.add_turn(turn)
        self.embedding.add_turn(turn)
        self.tree.add_turn(turn, parent_id=parent_id)

    def get_context(self, prompt=None, parent_id=None):
        # 1. Recency
        recency = self.sliding.get_context()
        # 2. Summary
        summary = self.summarizer.get_context()
        # 3. Relevant from embeddings
        relevant = self.embedding.get_context(query=prompt)
        # 4. Conversation tree path
        tree_context = self.tree.get_context(from_message_id=parent_id or self.tree.last_message_id)

        # Merge all while avoiding duplicates
        all_parts = recency + summary + relevant + tree_context
        seen = set()
        merged = []
        for ctx in all_parts:
            key = (ctx["role"], " ".join(ctx.get("parts", [])))
            if key not in seen:
                merged.append(ctx)
                seen.add(key)
        return merged

    # ✅ Expose tree attributes so the UI can access them
    @property
    def message_nodes(self):
        return self.tree.message_nodes

    @property
    def last_message_id(self):
        return self.tree.last_message_id