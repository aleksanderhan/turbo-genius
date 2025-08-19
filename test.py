import dearpygui.dearpygui as dpg

dpg.create_context()
messages = []

def add_message(role, text):
    messages.append((role, text))
    dpg.add_text(f"{role}: {text}", parent="chat_history")
    dpg.set_y_scroll("chat_history", dpg.get_y_scroll_max("chat_history"))

def send_message():
    user_input = dpg.get_value("input_box").strip()
    if not user_input:
        return
    add_message("You", user_input)
    dpg.set_value("input_box", "")
    add_message("Bot", f"Echo: {user_input}")

with dpg.window(tag="main_window", no_move=True, no_scrollbar=True):
    dpg.add_child_window(tag="chat_history", border=True)
    dpg.add_input_text(tag="input_box", on_enter=True, callback=lambda: send_message())
    dpg.add_button(tag="send_btn", label="Send", callback=lambda: send_message())

def resize_every_frame():
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()

    margin = 10
    btn_w = 80
    row_h = 30
    gap = 6

    # resize main window
    dpg.configure_item("main_window", pos=(0, 0), width=vp_w, height=vp_h)

    # position input row
    input_w = vp_w - (btn_w + gap + margin * 2)
    input_y = vp_h - margin - row_h
    dpg.configure_item("input_box", pos=(margin, input_y), width=input_w, height=row_h)
    dpg.configure_item("send_btn", pos=(margin + input_w + gap, input_y), width=btn_w, height=row_h)

    # position chat history
    chat_h = input_y - margin
    dpg.configure_item("chat_history", pos=(margin, margin), width=vp_w - margin * 2, height=chat_h)

    # call again next frame
    dpg.set_frame_callback(1, resize_every_frame)

dpg.create_viewport(title="Resizable Chat", width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()

resize_every_frame()
add_message("Bot", "Welcome! Resize the window to see me stretch.")

dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
