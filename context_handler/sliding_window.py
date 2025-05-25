from collections import deque

class SlidingWindowContext:
    def __init__(self, max_turns=2):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add_turn(self, turn):
        """
        Add a new turn (dict or string) to the context.
        """
        self.history.append(turn)

    def get_context(self):
        """
        Return a list of the current context turns.
        """
        return list(self.history) 