// === Globals ===
let currentAssistantMessageId = null;
let assistantMessageBuffer = '';

// === DOM Elements ===
const chat = document.getElementById('chat');
const input = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const newSessionButton = document.getElementById('new-session-button');
const sessionList = document.getElementById('session-list');
const sessionTitleBar = document.getElementById('current-session-title');

// === Event Listeners ===
sendButton.addEventListener('click', sendMessage);
input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});
newSessionButton.addEventListener('click', function() {
    clearChat();
    window.pywebview.api.reset_session();
    currentAssistantMessageId = null;
    updateSessionTitleBar("New Session");
});
window.addEventListener('click', function() {
    const dropdowns = document.querySelectorAll('.dropdown-content');
    dropdowns.forEach(dropdown => {
        if (dropdown.style.display === 'block') {
            dropdown.style.display = 'none';
        }
    });
});

// === Chat Functions ===
function sendMessage() {
    const message = input.value.trim();
    if (message) {
        window.pywebview.api.send_message(message);
        addMessage('user', message);
        addMessage('assistant', '', true);
        input.value = '';
    }
}

function addMessage(role, message, withSpinner = false) {
    let messageDiv = null;

    if (role === 'assistant' && currentAssistantMessageId) {
        messageDiv = document.getElementById(currentAssistantMessageId);
        if (messageDiv) {
            assistantMessageBuffer += message;
            messageDiv.querySelector('.message-content').innerHTML = parseMessageContent(assistantMessageBuffer);
            stopSpinner(currentAssistantMessageId);
        } else {
            currentAssistantMessageId = null;
            assistantMessageBuffer = '';
        }
    } else if (role === "user") {
        currentAssistantMessageId = null;
    }

    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role);
        if (role === 'assistant') {
            currentAssistantMessageId = `assistant-${Date.now()}`;
            messageDiv.id = currentAssistantMessageId;
            if (withSpinner) {
                messageDiv.appendChild(createSpinner());
            }
        }

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = role === 'assistant'
            ? parseMessageContent(message)
            : parseMessageContent(message);

        if (role === 'assistant') assistantMessageBuffer = message;

        messageDiv.appendChild(contentDiv);
        chat.appendChild(messageDiv);
    }

    chat.scrollTop = chat.scrollHeight;
    Prism.highlightAllUnder(chat);
    MathJax.typesetPromise([chat]);
}

function parseMessageContent(message) {
    const codeBlockRegex = /```(?:python)?\n(.*?)\n```/gs;
    return message.replace(codeBlockRegex, (_, code) => {
        const escaped = code.trim().replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return `<pre><code class="language-javascript">${escaped}</code></pre>`;
    });
}

function stopSpinner(id) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        const spinner = messageDiv.querySelector(".spinner");
        if (spinner) spinner.remove();
    }
}

function createSpinner() {
    const div = document.createElement('div');
    div.classList.add('spinner');
    return div;
}

function clearChat() {
    chat.innerHTML = '';
    currentAssistantMessageId = null;
}

// === Session Functions ===
function addSession(sessionId, sessionTitle) {
    const item = document.createElement('li');
    item.setAttribute('data-session-id', sessionId);

    const titleDiv = document.createElement('div');
    titleDiv.classList.add('session-title');
    titleDiv.textContent = sessionTitle;
    titleDiv.addEventListener('click', () => {
        clearChat();
        window.pywebview.api.load_session(sessionId);
        updateSessionTitleBar(sessionTitle);
    });

    const dropdownDiv = document.createElement('div');
    dropdownDiv.classList.add('dropdown');
    dropdownDiv.innerHTML = '&#x2630;';
    dropdownDiv.addEventListener('click', function(e) {
        e.stopPropagation();
        const dropdownContent = dropdownDiv.querySelector('.dropdown-content');
        dropdownContent.style.display = dropdownContent.style.display === 'block' ? 'none' : 'block';
    });

    const dropdownContent = document.createElement('div');
    dropdownContent.classList.add('dropdown-content');

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', () => {
        window.pywebview.api.delete_session(sessionId);
        item.remove();
        updateSessionTitleBar("Select a session");
    });

    const titleGenBtn = document.createElement('button');
    titleGenBtn.textContent = 'Generate title';
    titleGenBtn.addEventListener('click', () => {
        window.pywebview.api.generate_title(sessionId);
    });

    dropdownContent.appendChild(deleteBtn);
    dropdownContent.appendChild(titleGenBtn);
    dropdownDiv.appendChild(dropdownContent);

    item.appendChild(titleDiv);
    item.appendChild(dropdownDiv);
    sessionList.insertBefore(item, sessionList.firstChild);
}

function updateSessionTitle(sessionId, newTitle) {
    const items = sessionList.getElementsByTagName('li');
    for (let item of items) {
        if (item.getAttribute('data-session-id') === sessionId) {
            item.querySelector('.session-title').textContent = newTitle;
            updateSessionTitleBar(newTitle);
            break;
        }
    }
}

function updateSessionTitleBar(title) {
    sessionTitleBar.textContent = title;
}
