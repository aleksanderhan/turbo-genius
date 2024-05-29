import tkinter as tk
import asyncio
import websockets
import threading
import time
import requests
import traceback
import argparse
from tkinter import scrolledtext

async def stream_tokens(uri, prompt, chat_area, app):
    app.after(0, lambda: update_chat_area(chat_area, f"You: {prompt}", "user"))
    response_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg="lightgreen")
    response_label = tk.Label(response_frame, anchor="w", justify="left", wraplength=chat_area.winfo_width() - 20, bg="lightgreen", font=("Arial", 10, "bold"), fg="black")
    response_label.pack(fill="both", expand=True)
    app.after(0, lambda: chat_area.window_create(tk.END, window=response_frame))
    app.after(0, lambda: chat_area.insert(tk.END, "\n\n"))  # Add space after the response

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(prompt)
            t0 = time.time()
            num_token = 0
            response_text = ""
            try:
                while True:
                    token = await websocket.recv()
                    if token == "[DONE]":
                        break
                    response_text += token
                    app.after(0, lambda t=response_text: update_response_label(response_label, t))
                    num_token += 1
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
            finally:
                dt = time.time() - t0
                stats = f"Time elapsed: {dt:.2f} seconds, Number of tokens/sec: {num_token/dt:.2f}, Number of tokens: {num_token}"
                app.after(0, lambda: update_chat_area(chat_area, stats, "stats"))
    except Exception as e:
        traceback.print_exc()
        app.after(0, lambda: update_chat_area(chat_area, f"WebSocket connection failed: {e}", "stats"))

def update_response_label(label, text):
    label.config(text=text, fg="black")
    chat_area.yview_moveto(1.0)

def update_chat_area(chat_area, text, msg_type):
    chat_area.config(state='normal')  # Enable the text widget to allow modifications
    chat_area.insert(tk.END, "\n\n") 
    if msg_type == "user":
        bg_color = "lightblue"
        font = ("Arial", 10, "normal")
    elif msg_type == "response":
        bg_color = "lightgreen"
        font = ("Arial", 10, "bold")
    elif msg_type == "stats":
        bg_color = "lightgray"
        font = ("Arial", 10, "italic")
    else:
        bg_color = "white"
        font = ("Arial", 10, "normal")

    message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid", bg=bg_color)
    message_label = tk.Label(message_frame, text=text, anchor="w", justify="left", wraplength=chat_area.winfo_width() - 20, bg=bg_color, font=font, fg="black")
    message_label.pack(fill="both", expand=True)

    chat_area.window_create(tk.END, window=message_frame)
    chat_area.insert(tk.END, "\n\n") 
    chat_area.config(state='disabled')
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
            update_chat_area(chat_area, f"Failed to send message: {e}", "stats")

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
