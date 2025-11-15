from typing import Optional
import uuid


class SessionManager:
    """Manage short-lived session ids stored in memory."""

    def __init__(self):
        self.sessions = []

    def resolve(self, session_id: Optional[str]) -> Optional[str]:
        """Return an existing session id or create a new one when none provided."""
        if session_id is None:
            new_id = uuid.uuid4().hex
            self.sessions.append(new_id)
            return new_id

        if session_id in self.sessions:
            return session_id

        return None