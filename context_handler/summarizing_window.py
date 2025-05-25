# context_handler/smart_summary_window.py
from collections import deque
from transformers import pipeline

class SummarizingWindowContext:
    def __init__(self, max_turns=6, summary_min_length=1, summary_max_length=60):
        self.max_turns = max_turns
        self.summary_min_length = summary_min_length
        self.summary_max_length = summary_max_length

        self.detailed_window = deque(maxlen=max_turns)
        self.summaries = []
        self.pending_summary_batch = []

        self.summarizer = pipeline("summarization", model="t5-small")

    def add_turn(self, turn):
        self.detailed_window.append(turn)
        self.pending_summary_batch.append(turn)

        if len(self.pending_summary_batch) >= self.max_turns:
            self._summarize_pending()

    def _summarize_pending(self):
        text_to_summarize = "\n".join(
            [f"{turn['role']}: {' '.join(turn['parts'])}" for turn in self.pending_summary_batch]
        )
        input_text = "summarize: " + text_to_summarize
        summary = self.summarizer(
            input_text,
            min_length=self.summary_min_length,
            max_length=self.summary_max_length,
            do_sample=False,
            max_new_tokens=None
        )[0]['summary_text']
        self.summaries.append(summary)
        self.pending_summary_batch.clear()

    def get_context(self):
        context = []
        if self.summaries:
            context.append({"role": "model", "parts": [" ".join(self.summaries)]})
        context.extend(list(self.detailed_window))
        return context

    def _turn_to_text(self, turn):
        if isinstance(turn, dict):
            parts = turn.get('parts', [])
            return ' '.join(str(p) for p in parts)
        return str(turn)