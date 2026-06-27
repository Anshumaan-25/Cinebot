"""Memory manager: ties the store + extractor.

``observe`` is the write policy (record the turn, extract + store preferences);
``recall`` is the read policy (formatted memory context for prompt injection).
The LangGraph memory node (Part 6) calls these.
"""

from __future__ import annotations


class MemoryManager:
    def __init__(self, store, extractor=None) -> None:
        self.store = store
        self.extractor = extractor  # None -> skip preference extraction (e.g. no Groq key)

    def observe(self, user_id: str, user_message: str, assistant_reply: str | None = None) -> None:
        if user_message:
            self.store.add_interaction(user_id, "user", user_message)
            if self.extractor is not None:
                for preference in self.extractor.extract(user_message):
                    self.store.add_preference(user_id, preference)
        if assistant_reply:
            self.store.add_interaction(user_id, "assistant", assistant_reply)

    def recall(self, user_id: str) -> str:
        return self.store.recall(user_id)
