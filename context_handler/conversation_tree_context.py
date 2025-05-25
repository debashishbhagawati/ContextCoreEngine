class MessageNode:
    def __init__(self, message_id, role, content, parent_id=None):
        self.id = message_id
        self.role = role
        self.content = content
        self.parent_id = parent_id
        self.children_ids = []

class ConversationTreeContext:
    def __init__(self, depth=20):
        self.message_nodes = {}
        self.current_message_id_counter = 0
        self.last_message_id = None
        self.depth = depth

    def add_turn(self, turn, parent_id=None):
        self.current_message_id_counter += 1
        role = turn.get("role", "user")
        content = turn.get("parts", [""])[0]
        if parent_id is None:
            parent_id = self.last_message_id

        node = MessageNode(self.current_message_id_counter, role, content, parent_id)
        self.message_nodes[self.current_message_id_counter] = node

        if parent_id and parent_id in self.message_nodes:
            self.message_nodes[parent_id].children_ids.append(node.id)

        self.last_message_id = node.id

    def get_context(self, from_message_id=None):
        context_parts = []
        curr_id = from_message_id if from_message_id is not None else self.last_message_id
        for _ in range(self.depth):
            if curr_id not in self.message_nodes:
                break
            node = self.message_nodes[curr_id]
            context_parts.append({"role": node.role, "parts": [node.content]})
            if node.parent_id is None:
                break
            curr_id = node.parent_id
        return list(reversed(context_parts))