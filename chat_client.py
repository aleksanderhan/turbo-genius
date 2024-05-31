import argparse
import requests
import threading
import asyncio
import websockets
import time
import traceback
import re
import platform
import tkinter as tk
import tkinter.font as tkFont
from tkinter import scrolledtext
from tkinter import messagebox
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments.formatter import Formatter

class TkinterFormatter(Formatter):
    def __init__(self, text_widget, start_index, **options):
        super().__init__(**options)
        self.text_widget = text_widget
        self.start_index = start_index
        self.style = get_style_by_name(options.get('style', 'default'))
        self._configure_tags()

    def _configure_tags(self):
        for token, style in self.style:
            foreground = f'#{style["color"]}' if style['color'] else 'black'
            font_options = ('Courier', 10, 'normal')
            if style['bold']:
                font_options = ('Courier', 10, 'bold')
            if style['italic']:
                font_options = ('Courier', 10, 'italic')
            if style['bold'] and style['italic']:
                font_options = ('Courier', 10, 'bold italic')
            self.text_widget.tag_configure(str(token), foreground=foreground, font=font_options)

    def format(self, tokensource, outfile):
        current_index = self.start_index
        for ttype, value in tokensource:
            tag = str(ttype)
            self.text_widget.insert(current_index, value, tag)
            current_index = self.text_widget.index(f"{current_index} + {len(value)} chars")

def highlight_code(text_widget, code, start_index):
    lexer = PythonLexer()
    formatter = TkinterFormatter(text_widget, start_index)
    highlight(code, lexer, formatter)

async def stream_tokens(uri, prompt, chat_area, app):
    app.after(0, lambda: ChatApp.add_user_message(chat_area, prompt))

    response_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightgreen")
    response_text = tk.Text(response_frame, wrap='word', bg="lightgreen", font=("Arial", 10, "bold"), fg="black", padx=10, pady=5, height=1)
    response_text.pack(fill="both", expand=True)
    response_text.insert(tk.END, "")
    response_text.config(state='disabled')
    app.after(0, lambda: chat_area.window_create(tk.END, window=response_frame))
    app.after(0, lambda: chat_area.insert(tk.END, "\n\n"))  # Add space after the response

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(prompt)
            t0 = time.time()
            num_token = 0
            response_content = ""
            try:
                while True:
                    token = await websocket.recv()
                    response_content += token
                    app.after(0, lambda t=response_content: ChatApp.update_response_text(response_text, t, chat_area))
                    num_token += 1
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
            finally:
                dt = time.time() - t0
                stats = f"Time elapsed: {dt:.2f} seconds, Number of tokens/sec: {num_token/dt:.2f}, Number of tokens: {num_token}"
                app.after(0, lambda: ChatApp.add_system_message(chat_area, stats))
    except Exception as e:
        traceback.print_exc()
        app.after(0, lambda: ChatApp.add_system_message(chat_area, f"WebSocket connection failed: {e}"))

class ChatApp:
    def __init__(self, root, server, port):
        self.root = root
        self.server = server
        self.port = port

        self.root.title("Turbo-Genius chat interface")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)

        self.sidebar_frame = tk.Frame(root, bg="lightgray", width=400)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsw")
        self.sidebar_frame.grid_propagate(False)

        self.center_frame = tk.Frame(root)
        self.center_frame.grid(row=0, column=1, sticky="nsew")
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        self.chat_area = scrolledtext.ScrolledText(self.center_frame, width=80)
        self.chat_area.grid(row=0, column=0, sticky="nsew")
        self.chat_area.config(state='disabled')

        if platform.system() == "Windows":
            self.root.state('zoomed')
            self.chat_area.bind_all("<MouseWheel>", lambda event: self.on_mousewheel(event))
        elif platform.system() == "Linux":
            self.root.attributes('-zoomed', True)
            self.chat_area.bind_all("<Button-4>", lambda event: self.on_mousewheel(event))
            self.chat_area.bind_all("<Button-5>", lambda event: self.on_mousewheel(event))

        self.input_frame = tk.Frame(self.center_frame)
        self.input_frame.grid(row=1, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.message_entry = tk.Text(self.input_frame, height=3)
        self.message_entry.pack(side="left", fill="x", expand=True)
        self.message_entry.focus_set()

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="right")

        self.root.bind('<Return>', self.send_message)

        session_response = requests.get(f"http://{self.server}:{self.port}/session")
        self.session_id = session_response.json() # Active session

        sessions_response = requests.get(f"http://{self.server}:{self.port}/session-list")
        sessions = sessions_response.json()
        for i, session in enumerate(reversed(sessions)):
            session_title = session["title"]
            session_id = session["id"]
            label = tk.Label(self.sidebar_frame, text=session_title, bg="lightgray", font=("Arial", 12))
            label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")
            label.bind("<Button-1>", lambda event, session_id=session_id: self.load_chat(event, session_id))

    def load_chat(self, event, session_id):
        self.chat_area.config(state='normal')
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.config(state='disabled')
        session_response = requests.get(f"http://{self.server}:{self.port}/session/{session_id}")
        chat_data = session_response.json()
        for message in chat_data["messages"]:
            if message["role"] == "user":
                ChatApp.add_user_message(self.chat_area, message["content"])
            elif message["role"] == "assistant":
                print(message)
                ChatApp.add_assistant_message(self.chat_area, message["content"])
        self.session_id = session_id


    def send_message(self, event=None):
        prompt = self.message_entry.get("1.0", tk.END).strip()
        if prompt:
            self.message_entry.delete("1.0", tk.END)
            try:
                uri = f"ws://{self.server}:{self.port}/stream/{self.session_id}"
                asyncio.run_coroutine_threadsafe(stream_tokens(uri, prompt, self.chat_area, self.root), event_loop)
            except Exception as e:
                traceback.print_exc()
                self.add_system_message(self.chat_area, f"Failed to send message: {e}")

    @staticmethod
    def add_user_message(chat_area, text):
        chat_area.config(state='normal')
        chat_area.insert(tk.END, "\n\n")

        message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightblue")
        message_text = tk.Text(message_frame, wrap='word', bg="lightblue", font=("Arial", 10, "normal"), fg="black", padx=10, pady=5, height=1)
        message_text.insert(tk.END, text)
        message_text.config(state='disabled')
        message_text.pack(fill="both", expand=True)

        chat_area.window_create(tk.END, window=message_frame)
        chat_area.insert(tk.END, "\n\n")
        chat_area.config(state='disabled')

        ChatApp.update_text_widget_height(message_text, chat_area)

    @staticmethod
    def add_assistant_message(chat_area, text):
        chat_area.config(state='normal')
        chat_area.insert(tk.END, "\n\n")

        message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightgreen")
        message_text = tk.Text(message_frame, wrap='word', bg="lightgreen", font=("Arial", 10, "normal"), fg="black", padx=10, pady=5, height=1)

        # Regular expression to find code blocks
        pattern = r"```(?:python)?\n(.*?)\n```"
        parts = re.split(pattern, text, flags=re.DOTALL)

        for i, part in enumerate(parts):
            if i % 2 == 0:
                message_text.insert(tk.END, part)
            else:
                start_index = message_text.index(tk.END)
                highlight_code(message_text, part, start_index)

        message_text.config(state='disabled')
        message_text.pack(fill="both", expand=True)

        chat_area.window_create(tk.END, window=message_frame)
        chat_area.insert(tk.END, "\n\n")
        chat_area.config(state='disabled')

        ChatApp.update_text_widget_height(message_text, chat_area)


    @staticmethod
    def add_system_message(chat_area, text):
        chat_area.config(state='normal')
        chat_area.insert(tk.END, "\n\n")

        message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightgray")
        message_text = tk.Text(message_frame, wrap='word', bg="lightgray", font=("Arial", 10, "italic"), fg="black", padx=10, pady=5, height=1)
        message_text.insert(tk.END, text)
        message_text.config(state='disabled')
        message_text.pack(fill="both", expand=True)

        chat_area.window_create(tk.END, window=message_frame)
        chat_area.insert(tk.END, "\n\n")
        chat_area.config(state='disabled')

        ChatApp.update_text_widget_height(message_text, chat_area)

    @staticmethod
    def update_response_text(text_widget, text, chat_area):
        text_widget.config(state='normal')

        pattern = r"```(?:python)?\n(.*?)\n```"
        parts = re.split(pattern, text, flags=re.DOTALL)

        text_widget.delete("1.0", tk.END)

        for i, part in enumerate(parts):
            if i % 2 == 0:
                text_widget.insert(tk.END, part)
            else:
                start_index = text_widget.index(tk.END)
                highlight_code(text_widget, part, start_index)

        text_widget.config(state='disabled')
        num_lines = text_widget.count("1.0", tk.END, "displaylines")[0]
        text_widget.config(height=num_lines)
        chat_area.yview_moveto(1.0)

    @staticmethod
    def update_text_widget_height(text_widget, chat_area):
        font = tkFont.Font(font=text_widget.cget("font"))
        char_width = font.measure('0')
        chat_area_width = chat_area.winfo_width()
        max_chars_per_line = chat_area_width // char_width
        text_content = text_widget.get("1.0", "end-1c")
        num_lines = (len(text_content) + max_chars_per_line - 1) // max_chars_per_line
        text_widget.config(height=num_lines)
        chat_area.yview_moveto(1.0)

    def on_mousewheel(self, event):
        if platform.system() == "Windows":
            self.chat_area.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == "Linux":
            if event.num == 4:
                self.chat_area.yview_scroll(-1, "units")
            elif event.num == 5:
                self.chat_area.yview_scroll(1, "units")

def start_asyncio_loop():
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store', default="localhost")
    parser.add_argument('--port', action='store', default="8000")
    args = parser.parse_args()

    loop_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
    loop_thread.start()

    root = tk.Tk()
    app = ChatApp(root, args.server, args.port)
    root.mainloop()
