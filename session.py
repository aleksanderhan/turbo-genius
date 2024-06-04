
class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.title = None
        self.messages = [
            {"role": "system", "content": "Help the user with anything they want!"},
        ]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def get_messages(self):
        return self.messages
    
    def truncate_messages(self):
        self.messages = [self.messages[0]] + self.messages[3:]
    

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        return self.sessions[session_id]
    
    def get_session_list(self):
        return [{"id": session.session_id, "title": session.title} for session in self.sessions.values()]
    
    def get_new_session(self):
        session_id = len(self.sessions)
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
    
    def remove_session(self, session_id):
        del self.sessions[session_id]