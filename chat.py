import tkinter as tk
import asyncio
import websockets
import threading
import time
import requests
import traceback
import argparse
from tkinter import scrolledtext
import tkinter.font as tkfont

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
                    app.after(0, lambda t=response_content: update_response_text(response_text, t))
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

def update_response_text(text_widget, text):
    text_widget.config(state='normal')
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, text)
    text_widget.config(state='disabled')
    # Dynamically adjust height
    num_lines = text_widget.count("1.0", tk.END, "displaylines")[0]
    text_widget.config(height=num_lines)
    chat_area.yview_moveto(1.0)

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
    font = tkfont.Font(font=text_widget.cget("font"))
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
            add_system_message(chat_area, f"Failed to send message: {e}", "stats")

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
