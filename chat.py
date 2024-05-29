import tkinter as tk
import asyncio
import websockets
import threading
import time
import requests
import traceback
from tkinter import scrolledtext

async def stream_tokens(uri, prompt, chat_area, app):
    app.after(0, lambda: update_chat_area(chat_area, f"You: {prompt}"))
    response_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid")
    response_label = tk.Label(response_frame, anchor="w", justify="left", wraplength=chat_area.winfo_width() - 20)
    response_label.pack(fill="both", expand=True)
    app.after(0, lambda: chat_area.window_create(tk.END, window=response_frame))
    
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
                app.after(0, lambda: update_chat_area(chat_area, stats))
    except Exception as e:
        print(f"WebSocket connection failed: {e}")
        app.after(0, lambda: update_chat_area(chat_area, f"WebSocket connection failed: {e}"))

def update_response_label(label, text):
    label.config(text=text)

def update_chat_area(chat_area, text):
    chat_area.config(state='normal')  # Enable the text widget to allow modifications
    chat_area.insert(tk.END, "\n")    # Insert a newline before the message card

    message_frame = tk.Frame(chat_area, padx=10, pady=5, bd=1, relief="solid")
    message_label = tk.Label(message_frame, text=text, anchor="w", justify="left", wraplength=chat_area.winfo_width() - 20)
    message_label.pack(fill="both", expand=True)
    
    chat_area.window_create(tk.END, window=message_frame)
    chat_area.insert(tk.END, "\n\n")  # Insert newlines after the message card

    chat_area.config(state='disabled')  # Disable the text widget to prevent user modifications
    chat_area.yview(tk.END)  # Scroll to the end



def start_asyncio_loop():
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

def send_message():
    prompt = message_entry.get("1.0", tk.END).strip()
    if prompt:
        message_entry.delete("1.0", tk.END)
        try:
            session_response = requests.get(f"http://192.168.1.13:8000/session")
            session_id = session_response.json()
            print(f"Received session ID: {session_id}")
            uri = f"ws://192.168.1.13:8000/stream/{session_id}"
            asyncio.run_coroutine_threadsafe(stream_tokens(uri, prompt, chat_area, app), event_loop)
        except Exception as e:
            traceback.print_exc()
            update_chat_area(chat_area, f"Failed to send message: {e}")


# Setup the asyncio event loop in a separate thread
loop_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
loop_thread.start()

# Tkinter setup
app = tk.Tk()
app.title("Turbo-Genius Chat Interface")

chat_area = scrolledtext.ScrolledText(app, height=20, width=75)
chat_area.grid(row=0, column=0, columnspan=2, sticky='nsew')
chat_area.config(state='disabled')

# Input field and send button
input_frame = tk.Frame(app)
input_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

message_entry = tk.Text(input_frame, height=3)
message_entry.grid(row=0, column=0, sticky='ew')
message_entry.focus_set()

send_button = tk.Button(input_frame, text="Send", command=send_message)
send_button.grid(row=0, column=1, sticky='ew')

# Make the input field and send button expand with the window
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(0, weight=1)
input_frame.grid_columnconfigure(0, weight=1)

app.bind('<Return>', lambda event: send_message())

app.mainloop()
