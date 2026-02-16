// script.js – Professional chat with history sidebar, thinking indicator, delete & rename
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const themeToggle = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-icon");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.getElementById("new-chat-btn");

// Theme persistence
let isDark = localStorage.getItem("theme") !== "light";
document.body.classList.toggle("light", !isDark);
themeIcon.className = isDark ? "fas fa-moon" : "fas fa-sun";

themeToggle.addEventListener("click", () => {
  isDark = !isDark;
  document.body.classList.toggle("light", !isDark);
  themeIcon.className = isDark ? "fas fa-moon" : "fas fa-sun";
  localStorage.setItem("theme", isDark ? "dark" : "light");
});

// Current chat state
let currentChatId = null;
let chats = JSON.parse(localStorage.getItem("mediChat_chats") || "{}");

// Load sidebar (no auto-load of messages)
function loadChatList() {
  chatList.innerHTML = "";
  Object.keys(chats).sort((a,b) => b - a).forEach(id => {
    const chat = chats[id];
    const item = document.createElement("div");
    item.classList.add("chat-item");
    item.dataset.id = id;
    if (id === currentChatId) item.classList.add("active");

    const titleSpan = document.createElement("span");
    titleSpan.className = "title";
    titleSpan.textContent = chat.title || "New Chat";
    item.appendChild(titleSpan);

    const actions = document.createElement("div");
    actions.className = "actions";

    const renameBtn = document.createElement("button");
    renameBtn.className = "action-btn rename-btn";
    renameBtn.innerHTML = '<i class="fas fa-edit"></i>';
    renameBtn.title = "Rename chat";
    renameBtn.onclick = (e) => { e.stopPropagation(); renameChat(id); };

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "action-btn";
    deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
    deleteBtn.title = "Delete chat";
    deleteBtn.onclick = (e) => { e.stopPropagation(); deleteChat(id); };

    actions.appendChild(renameBtn);
    actions.appendChild(deleteBtn);
    item.appendChild(actions);

    item.addEventListener("click", () => loadChat(id));
    chatList.appendChild(item);
  });
}

// Load specific chat messages
function loadChat(id) {
  currentChatId = id;
  chatBox.innerHTML = "";
  const chat = chats[id];
  if (chat && chat.messages) {
    chat.messages.forEach(msg => addMessage(msg.role, msg.content, msg.time));
  }
  document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
  document.querySelector(`.chat-item[data-id="${id}"]`)?.classList.add("active");
}

// Add message + save to storage
function addMessage(role, content, time = new Date().toLocaleTimeString("en-US", {hour:"numeric", minute:"2-digit", hour12:true})) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", role);

  const bubble = document.createElement("div");
  bubble.innerHTML = content.replace(/\n/g, "<br>");
  msgDiv.appendChild(bubble);

  const ts = document.createElement("div");
  ts.className = "timestamp";
  ts.textContent = time;
  msgDiv.appendChild(ts);

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  // Save to current chat
  if (!chats[currentChatId]) {
    chats[currentChatId] = {
      title: content.slice(0, 40) + (content.length > 40 ? "..." : ""),
      messages: []
    };
  }
  chats[currentChatId].messages.push({ role, content, time });
  localStorage.setItem("mediChat_chats", JSON.stringify(chats));
  loadChatList();
}

// Show thinking indicator
function showThinking() {
  const thinking = document.createElement("div");
  thinking.className = "message assistant thinking";
  thinking.innerHTML = `Assistant is thinking<span class="dots">...</span>`;
  chatBox.appendChild(thinking);
  chatBox.scrollTop = chatBox.scrollHeight;
  return thinking;
}

function removeThinking(thinkingEl) {
  if (thinkingEl && thinkingEl.parentNode) {
    thinkingEl.remove();
  }
}

// Send message
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  addMessage("user", query);
  userInput.value = "";

  const thinkingEl = showThinking();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    removeThinking(thinkingEl);
    addMessage("assistant", data.content);
  } catch (err) {
    removeThinking(thinkingEl);
    addMessage("assistant", `<span style="color:#ff6b6b">Error: ${err.message}</span>`);
  }
}

// New chat
newChatBtn.addEventListener("click", () => {
  currentChatId = Date.now().toString();
  chatBox.innerHTML = "";
  loadChatList();
  userInput.focus();
});

// Rename chat
function renameChat(id) {
  const newTitle = prompt("Enter new chat title:", chats[id].title || "New Chat");
  if (newTitle && newTitle.trim()) {
    chats[id].title = newTitle.trim();
    localStorage.setItem("mediChat_chats", JSON.stringify(chats));
    loadChatList();
  }
}

// Delete chat
function deleteChat(id) {
  if (confirm("Delete this chat permanently?")) {
    delete chats[id];
    localStorage.setItem("mediChat_chats", JSON.stringify(chats));
    if (currentChatId === id) {
      currentChatId = null;
      chatBox.innerHTML = "";
    }
    loadChatList();
  }
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

loadChatList();
userInput.focus();