import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingRetrievalContext:
    def __init__(self, max_recent_turns=4, top_k=3, model_name='all-MiniLM-L6-v2'):
        self.turns = []  # List of dicts: {"role": ..., "content": ...}
        self.embeddings = []  # List of np.array
        self.embedding_model = SentenceTransformer(model_name)
        self.max_recent_turns = max_recent_turns
        self.top_k = top_k

    def add_turn(self, turn):
        # turn: dict like {"role": "user" or "model", "parts": [text]}
        content = turn.get("parts", [""])[0]
        role = turn.get("role", "user")
        self.turns.append({"role": role, "content": content})
        embedding = self.embedding_model.encode(content)
        self.embeddings.append(embedding)

    def get_context(self, query=None):
        """
        Returns a list of dicts for the LLM:
        - top_k semantically relevant past turns (excluding the query itself)
        - plus the most recent max_recent_turns
        """
        context_parts = []
        if not self.turns:
            return context_parts

        # If no query, just return the most recent turns
        if query is None:
            recent = self.turns[-self.max_recent_turns:]
            return [{"role": t["role"], "parts": [t["content"]]} for t in recent]

        # Embed the query
        query_embedding = self.embedding_model.encode(query)
        embeddings_np = np.array(self.embeddings)
        # Normalize for cosine similarity
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        norm_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        similarities = np.dot(norm_embeddings, norm_query)
        # Get top_k most similar (excluding the last turn if it's the query itself)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        relevant_turns = [self.turns[i] for i in top_indices]

        # Add most recent turns (avoid duplicates)
        recent = self.turns[-self.max_recent_turns:]
        seen = set()
        for t in relevant_turns + recent:
            key = (t["role"], t["content"])
            if key not in seen:
                context_parts.append({"role": t["role"], "parts": [t["content"]]})
                seen.add(key)
        return context_parts