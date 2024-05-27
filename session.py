


class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages