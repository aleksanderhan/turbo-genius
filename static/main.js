// === Globals ===
let currentAssistantMessageId = null;
let assistantMessageBuffer = '';
let websocket = null;
let currentSessionId = null;

// === DOM Elements ===
const chat = document.getElementById('chat');
const input = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const newSessionButton = document.getElementById('new-session-button');
const sessionList = document.getElementById('session-list');
const sessionTitleBar = document.getElementById('current-session-title');

// === WebSocket Setup ===
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function() {
        console.log('WebSocket connected');
        // Load existing sessions on connect
        sendWebSocketMessage({
            action: 'get_sessions'
        });
    };
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onclose = function() {
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 3 seconds
        setTimeout(initWebSocket, 3000);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        addMessage('system', 'Connection error. Attempting to reconnect...');
    };
}

function sendWebSocketMessage(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
    } else {
        console.error('WebSocket not connected');
        addMessage('system', 'Connection lost. Please refresh the page.');
    }
}

function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'message':
            addMessage(data.role, data.content);
            break;
            
        case 'session_created':
            currentSessionId = data.session_id;
            addSession(data.session_id, 'New session');
            updateSessionTitleBar('New session');
            break;
            
        case 'session_loaded':
            currentSessionId = data.data.id;
            // Load messages from session
            data.data.messages.forEach(msg => {
                if (msg.role === 'user' || msg.role === 'assistant') {
                    addMessage(msg.role, msg.content);
                }
            });
            break;
            
        case 'session_deleted':
            if (data.success) {
                // Remove from UI
                const sessionItem = document.querySelector(`[data-session-id="${data.session_id}"]`);
                if (sessionItem) {
                    sessionItem.remove();
                }
                // If this was the current session, clear it
                if (currentSessionId === data.session_id) {
                    clearChat();
                    currentSessionId = null;
                    updateSessionTitleBar("Select a session");
                }
            }
            break;
            
        case 'sessions_list':
            // Clear existing sessions and populate
            sessionList.innerHTML = '';
            data.sessions.forEach(session => {
                addSession(session.id, session.title);
            });
            break;
            
        case 'session_title_updated':
            updateSessionTitle(data.session_id, data.title);
            break;
            
        case 'error':
            addMessage('system', `Error: ${data.message}`);
            break;
    }
}

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
    currentSessionId = null;
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
        // Add user message to UI immediately
        addMessage('user', message);
        
        // Prepare assistant message placeholder
        addMessage('assistant', '', true);
        
        // Send via WebSocket
        sendWebSocketMessage({
            action: 'send_message',
            session_id: currentSessionId,
            message: message
        });
        
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
    assistantMessageBuffer = '';
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
        sendWebSocketMessage({
            action: 'load_session',
            session_id: sessionId
        });
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
        sendWebSocketMessage({
            action: 'delete_session',
            session_id: sessionId
        });
    });

    const titleGenBtn = document.createElement('button');
    titleGenBtn.textContent = 'Generate title';
    titleGenBtn.addEventListener('click', () => {
        sendWebSocketMessage({
            action: 'generate_title',
            session_id: sessionId
        });
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

// === Initialize ===
document.addEventListener('DOMContentLoaded', function() {
    initWebSocket();
});