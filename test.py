import tkinter as tk
from pygments import lex, highlight
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments.formatter import Formatter
import re
import itertools

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

def display_text():
    text_widget.config(state='normal')  # Ensure the widget is editable
    text_widget.delete("1.0", tk.END)  # Clear existing text

    text = input_text.get("1.0", tk.END)
    code_snippets = re.findall(r"```\n(.*?)\n```", text, flags=re.DOTALL)
    parts = re.split(r"```\n.*?\n```", text)

    for part, code in itertools.zip_longest(parts, code_snippets):
        if part:
            text_widget.insert(tk.END, part)
        if code:
            start_index = text_widget.index(tk.END)
            text_widget.insert(tk.END, "\n")
            highlight_code(text_widget, code, start_index)
            text_widget.insert(tk.END, "\n")  # Add spacing after code

    text_widget.config(state='disabled')  # Disable editing after insertion

# Tkinter setup
app = tk.Tk()
app.title("Code Highlighter Example")

input_frame = tk.Frame(app)
input_frame.pack()

input_text = tk.Text(input_frame, height=10, width=50)
input_text.pack()

button = tk.Button(app, text="Highlight Code", command=display_text)
button.pack()

text_widget = tk.Text(app, height=20, width=75, state='disabled')
text_widget.pack()

app.mainloop()
