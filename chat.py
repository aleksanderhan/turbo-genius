import tkinter as tk
import argparse
import requests
import threading
import asyncio
import websockets
import time
import traceback
import re
import tkinter.font as tkFont
from tkinter import scrolledtext
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
    app.after(0, lambda: add_user_message(chat_area, prompt))

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
                    app.after(0, lambda t=response_content: update_response_text(response_text, t, chat_area))
                    num_token += 1
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
            finally:
                dt = time.time() - t0
                stats = f"Time elapsed: {dt:.2f} seconds, Number of tokens/sec: {num_token/dt:.2f}, Number of tokens: {num_token}"
                app.after(0, lambda: add_system_message(chat_area, stats))
    except Exception as e:
        traceback.print_exc()
        app.after(0, lambda: add_system_message(chat_area, f"WebSocket connection failed: {e}"))

def update_response_text(text_widget, text, chat_area):
    text_widget.config(state='normal')  # Ensure the widget is editable

    # Regular expression pattern to find code snippets between triple backticks
    pattern = r"```(?:python)?\n(.*?)\n```"
    parts = re.split(pattern, text, flags=re.DOTALL)
    
    text_widget.delete("1.0", tk.END)  # Clear existing text

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Non-code part
            text_widget.insert(tk.END, part)
        else:
            # Code part
            start_index = text_widget.index(tk.END)
            highlight_code(text_widget, part, start_index)
            #text_widget.insert(tk.END, "\n")  # Ensure there's a newline after the code block

    text_widget.config(state='disabled')  # Disable editing after insertion

    # Adjust the height based on content
    num_lines = text_widget.count("1.0", tk.END, "displaylines")[0]
    text_widget.config(height=num_lines)
    chat_area.yview_moveto(1.0)  # Ensure scrolling to the bottom

def add_user_message(chat_area, text):
    chat_area.config(state='normal')  # Enable the text widget to allow modifications
    chat_area.insert(tk.END, "\n\n") 

    message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightblue")
    message_text = tk.Text(message_frame, wrap='word', bg="lightblue", font=("Arial", 10, "normal"), fg="black", padx=10, pady=5, height=1)
    message_text.insert(tk.END, text)
    message_text.config(state='disabled')
    message_text.pack(fill="both", expand=True)

    chat_area.window_create(tk.END, window=message_frame)
    chat_area.insert(tk.END, "\n\n") 
    chat_area.config(state='disabled')

    # Dynamically adjust height
    update_text_widget_height(message_text, chat_area)

def add_system_message(chat_area, text):
    chat_area.config(state='normal')  # Enable the text widget to allow modifications
    chat_area.insert(tk.END, "\n\n") 

    message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightgray")
    message_text = tk.Text(message_frame, wrap='word', bg="lightgray", font=("Arial", 10, "italic"), fg="black", padx=10, pady=5, height=1)
    message_text.insert(tk.END, text)
    message_text.config(state='disabled')
    message_text.pack(fill="both", expand=True)
    
    chat_area.window_create(tk.END, window=message_frame)
    chat_area.insert(tk.END, "\n\n") 
    chat_area.config(state='disabled')

    # Dynamically adjust height
    update_text_widget_height(message_text, chat_area)

def update_text_widget_height(text_widget, chat_area):
    # Use the tkinter font module to measure the width of a single character '0'
    font = tkFont.Font(font=text_widget.cget("font"))
    char_width = font.measure('0')
    chat_area_width = chat_area.winfo_width()
    max_chars_per_line = chat_area_width // char_width
    text_content = text_widget.get("1.0", "end-1c")
    num_lines = (len(text_content) + max_chars_per_line - 1) // max_chars_per_line
    text_widget.config(height=num_lines)
    chat_area.yview_moveto(1.0)

def start_asyncio_loop():
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

def send_message(args, session_id):
    prompt = message_entry.get("1.0", tk.END).strip()
    if prompt:
        message_entry.delete("1.0", tk.END)
        try:
            uri = f"ws://{args.server}:{args.port}/stream/{session_id}"
            asyncio.run_coroutine_threadsafe(stream_tokens(uri, prompt, chat_area, app), event_loop)
        except Exception as e:
            traceback.print_exc()
            add_system_message(chat_area, f"Failed to send message: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store', default="localhost")
    parser.add_argument('--port', action='store', default="8000")
    args = parser.parse_args()

    # Setup the asyncio event loop in a separate thread
    loop_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
    loop_thread.start()

    # Tkinter setup
    app = tk.Tk()
    app.title("Turbo-Genius Chat Interface")

    session_response = requests.get(f"http://{args.server}:{args.port}/session")
    session_id = session_response.json()

    chat_area = scrolledtext.ScrolledText(app, height=20, width=75)
    chat_area.grid(row=1, column=1, columnspan=2, sticky='nsew')
    chat_area.config(state='disabled')

    # Input field and send button
    input_frame = tk.Frame(app)
    input_frame.grid(row=2, column=1, columnspan=2, sticky='ew')

    message_entry = tk.Text(input_frame, height=3)
    message_entry.grid(row=0, column=0, sticky='ew')
    message_entry.focus_set()

    send_button = tk.Button(input_frame, text="Send", command=lambda args=args, session_id=session_id: send_message(args, session_id))
    send_button.grid(row=0, column=1, sticky='ew')

    # Make the input field and send button expand with the window
    app.grid_columnconfigure(1, weight=1)
    app.grid_rowconfigure(1, weight=10)  # More weight for the chat area
    app.grid_rowconfigure(2, weight=1)  # Less weight for the input frame
    input_frame.grid_columnconfigure(0, weight=1)

    # Add padding columns
    app.grid_columnconfigure(0, weight=1)
    app.grid_columnconfigure(3, weight=1)

    app.bind('<Return>', lambda event, args=args, session_id=session_id: send_message(args, session_id))

    app.mainloop()
