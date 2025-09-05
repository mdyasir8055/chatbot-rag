const topicForm = document.getElementById('topicForm');
const productInput = document.getElementById('product');
const pdfInput = document.getElementById('pdf');
const ocrModeSelect = document.getElementById('ocrMode');
const topicStatus = document.getElementById('topicStatus');
const historyEl = document.getElementById('history') || document.querySelector('.history-content') || document.querySelector('.history');
const clearHistoryBtn = document.getElementById('clearHistory');
const newChatBtn = document.getElementById('newChatBtn');
const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.getElementById('sunIcon');
const moonIcon = document.getElementById('moonIcon');

const compareToggle = document.getElementById('compareToggle');
const compareDot = document.getElementById('compareDot');
const compareText = document.getElementById('compareText');
const compareForm = document.getElementById('compareForm');
const pdfB = document.getElementById('pdfB');
const productB = document.getElementById('productB');
const compareStatus = document.getElementById('compareStatus');

const form = document.getElementById('composer');
const questionInput = document.getElementById('question');
const messagesEl = document.getElementById('messages');
const micBtn = document.getElementById('micBtn');
const ragToggle = document.getElementById('ragToggle');
const ragDot = document.getElementById('ragDot');
const ragText = document.getElementById('ragText');
const pdfNameEl = document.getElementById('pdfName');

let attachedPdfFile = null;
let attachedPdfBFile = null;
let topicSelected = false;
let comparePairReady = false;

// === Feature 2 globals (for Connect modal) ===
window.currentProductRef = '';
window.currentQuotedPrice = '';
window.currentContextSummary = '';
let lastUserQuestion = '';

// -------- RAG Toggle --------
function updateRagUI(enabled) {
  ragDot.style.background = enabled ? '#29d344' : '#e63946';
  ragText.textContent = enabled ? 'RAG: ON' : 'RAG: OFF';
}
async function sendRag(enabled) {
  try {
    const fd = new FormData();
    fd.append('enabled', String(enabled));
    const res = await fetch('/api/rag', { method: 'POST', body: fd });
    if (res.ok) {
      const data = await res.json();
      updateRagUI(!!data?.data?.rag_enabled);
    }
  } catch {}
}
ragToggle?.addEventListener('change', (e) => {
  const enabled = e.target.checked;
  sendRag(enabled);
  updateRagUI(enabled);
});
updateRagUI(true);

// -------- Comparison Mode Toggle --------
function updateCompareUI(on) {
  compareDot.style.background = on ? '#29d344' : '#e63946';
  compareText.textContent = on ? 'ON' : 'OFF';
  compareForm.hidden = !on;
}
compareToggle?.addEventListener('change', (e) => {
  const on = !!e.target.checked;
  updateCompareUI(on);
});
updateCompareUI(false);

// -------- Topic Setup --------
pdfInput.addEventListener('change', (e) => {
  attachedPdfFile = e.target.files[0] || null;
  pdfNameEl.textContent = attachedPdfFile ? `Selected: ${attachedPdfFile.name}` : '';
});

topicForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData();

  const raw = (productInput.value || '').trim();
  const ocrMode = ocrModeSelect.value || 'auto';

  if (attachedPdfFile) fd.append('pdf', attachedPdfFile);
  if (raw) {
    if (raw.startsWith('http')) fd.append('url', raw);
    else fd.append('product_name', raw);
  }
  fd.append('ocr_mode', ocrMode);

  if (!attachedPdfFile && !raw) {
    topicStatus.textContent = 'Please attach a PDF or provide a URL/Product name.';
    return;
  }

  if (!window.currentProductRef) {
    window.currentProductRef = raw || (attachedPdfFile ? attachedPdfFile.name : '') || '';
  }

  topicStatus.textContent = '⏳ Uploading/processing…';
  try {
    const res = await fetch('/api/init-topic', { method: 'POST', body: fd });
    if (!res.ok) {
      const detail = await res.text();
      topicStatus.textContent = `❌ Init failed (${res.status}): ${detail || 'no details'}`;
      return;
    }
    const data = await res.json();
    if (data.ok) {
      topicSelected = true;
      const meta = (data.data && data.data.meta) || {};
      const from =
        meta.source === 'pdf' ? ` (from PDF${meta.filename ? ': ' + meta.filename : ''})` :
        meta.source === 'url' ? ' (from URL)' : ' (from product name)';
      topicStatus.textContent = `✅ Topic: ${data.data.primary || 'Unknown'}${from}`;
      form.removeAttribute('disabled');
      addHistoryEntry(`Topic selected: ${data.data.primary || 'Unknown'}${from}`);

      if (data.data && data.data.primary) {
        window.currentProductRef = data.data.primary;
      }
    } else {
      topicStatus.textContent = '❌ Failed to initialize topic';
    }
  } catch (err) {
    topicStatus.textContent = '❌ Network error during init: ' + (err?.message || 'unknown');
  }
});

// -------- Comparison Pair Setup --------
pdfB.addEventListener('change', (e) => {
  attachedPdfBFile = e.target.files[0] || null;
});
compareForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!topicSelected) {
    compareStatus.textContent = 'Please confirm the main topic first.';
    return;
  }
  const fd = new FormData();
  const rawB = (productB.value || '').trim();
  if (attachedPdfFile) fd.append('a_pdf', attachedPdfFile);
  const rawA = (productInput.value || '').trim();
  if (rawA) {
    if (rawA.startsWith('http')) fd.append('a_url', rawA);
    else fd.append('a_name', rawA);
  }
  if (attachedPdfBFile) fd.append('b_pdf', attachedPdfBFile);
  if (rawB) {
    if (rawB.startsWith('http')) fd.append('b_url', rawB);
    else fd.append('b_name', rawB);
  }
  compareStatus.textContent = '⏳ Pairing topics…';
  try {
    const res = await fetch('/api/compare/init-topic', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      comparePairReady = true;
      compareStatus.textContent = '✅ Comparison pair set. Ask a question to compare.';
    } else {
      compareStatus.textContent = '❌ Could not set comparison pair.';
    }
  } catch (err) {
    compareStatus.textContent = '❌ Network error: ' + (err?.message || 'unknown');
  }
});

// -------- Ask --------
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = questionInput.value.trim();
  if (!text || !topicSelected) return;

  lastUserQuestion = text;
  window.currentContextSummary = text;

  addMessage('user', text);
  questionInput.value = '';

  const typingNode = showTyping();

  try {
    let url = '/api/ask';
    let fd = new FormData();
    fd.append('question', text);

    const compareMode = compareToggle.checked && comparePairReady;
    if (compareMode) url = '/api/compare/ask';

    const res = await fetch(url, { method: 'POST', body: fd });
    const data = await res.json();

    hideTyping(typingNode);

    if (!data.ok) {
      addMessage('ai', 'Error: ' + JSON.stringify(data));
      return;
    }

    if (compareMode) {
      // Render Comparison Mode payload
      const p = data.data;
      addMessage('ai', `Summary — A (Web):\n${p.A.summary}`, p.A.sources || []);
      addMessage('ai', `Summary — B (Web):\n${p.B.summary}`, p.B.sources || []);

      // Simple matrix render: first row only (kept compact)
      if (Array.isArray(p.matrix) && p.matrix.length) {
        const row = p.matrix[0];
        addMessage('ai', `Comparison Matrix → ${row.dimension}\nA: ${row.a_value}\nB: ${row.b_value}`);
      }
      addMessage('ai', `Recommendation:\n${p.recommendation || '-'}`);
      return;
    }

    // Standard mode: show Dual-Engine results
    const payload = data.data.raw || null;
    if (payload && payload.rag && payload.rag.status === 'known') {
      addMessage('ai', `RAG Answer (From your document):\n${payload.rag.summary}`);
      addMessage('ai', `Web Answer (CSE):\n${payload.cse.summary}`, payload.cse.sources || []);
      if (payload.fused?.summary) {
        addMessage('ai', `Fused Summary:\n${payload.fused.summary}`);
      }
      addMessage('ai', payload.final_answer, (payload.final_citations || []).map(c => c.ref));
    } else {
      // Web-only fallback
      const ans = data.data.answer || 'No answer returned.';
      const src = data.data.sources || [];
      addMessage('ai', ans, src);
    }

  } catch (err) {
    hideTyping(typingNode);
    addMessage('ai', 'Error contacting server: ' + (err?.message || 'unknown'));
  }
});

// -------- History --------
function addHistoryEntry(line) {
  if (!historyEl) {
    console.error('History element not found!');
    return;
  }
  
  const p = document.createElement('p');
  p.textContent = line;
  
  // Add to the top of the history for latest first
  if (historyEl.firstChild) {
    historyEl.insertBefore(p, historyEl.firstChild);
  } else {
    historyEl.appendChild(p);
  }
}
// New Chat button functionality
function startNewChat() {
  // Clear messages but keep history
  messagesEl.innerHTML = '';
  
  // Reset form states but keep topic selected
  questionInput.value = '';
  
  // Clear current conversation context
  window.currentQuotedPrice = '';
  window.currentContextSummary = '';
  
  // Add a visual indicator of new chat
  if (topicSelected) {
    addHistoryEntry(`New chat started - ${new Date().toLocaleTimeString()}`);
  }
  
  // Focus on input
  if (questionInput && !form.hasAttribute('disabled')) {
    questionInput.focus();
  }
}

// Clear History functionality (more comprehensive reset)
function clearAllData() {
  try { 
    fetch('/api/clear', { method: 'POST' }); 
  } catch {}
  
  messagesEl.innerHTML = '';
  historyEl.innerHTML = '';
  topicSelected = false;
  comparePairReady = false;
  document.getElementById('topicStatus').textContent = "No topic selected";
  form.setAttribute('disabled', true);
  attachedPdfFile = null; 
  attachedPdfBFile = null;
  
  if (pdfNameEl) pdfNameEl.textContent = '';
  if (pdfInput) pdfInput.value = '';
  if (pdfB) pdfB.value = '';
  
  productInput.value = ''; 
  productB.value = '';
  window.currentProductRef = '';
  window.currentQuotedPrice = '';
  window.currentContextSummary = '';
}

// Event listeners
newChatBtn?.addEventListener('click', startNewChat);
clearHistoryBtn.addEventListener('click', clearAllData);

// -------- Voice (unchanged from your previous version) --------
(() => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let recording = false;
  let mediaRecorder = null;
  let chunks = [];

  const micIdleSvg = `
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M12 14a4 4 0 0 0 4-4V7a4 4 0 1 0-8 0v3a4 4 0 0 0 4 4Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M19 10a7 7 0 1 1-14 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <path d="M12 17v3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    </svg>`;
  const micActiveSvg = `
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect x="7" y="7" width="10" height="10" rx="2"></rect>
    </svg>`;

  function setMic(active) {
    recording = active;
    micBtn.setAttribute('aria-pressed', String(active));
    micBtn.title = active ? 'Stop voice input' : 'Start voice input';
  }

  async function ensurePermission() {
    try { await navigator.mediaDevices.getUserMedia({ audio: true }); return true; }
    catch { addMessage('ai', 'Microphone permission denied.'); return false; }
  }

  function startWebSpeech() {
    if (!SR) return false;
    try {
      recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      let finalText = '';
      recognition.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const res = e.results[i];
          if (res.isFinal) finalText += res[0].transcript + ' ';
          else interim += res[0].transcript;
        }
        questionInput.value = (finalText + interim).trim();
      };
      recognition.onerror = (e) => { addMessage('ai', 'Speech recognition error: ' + (e.error || 'unknown')); stop(); };
      recognition.onend = () => {
        const text = (questionInput.value || '').trim();
        setMic(false);
        if (text) form.dispatchEvent(new Event('submit'));
      };

      recognition.start();
      setMic(true);
      return true;
    } catch { return false; }
  }

  async function startWhisper() {
    const ok = await ensurePermission();
    if (!ok) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    chunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      try {
        if (!chunks.length) { setMic(false); return; }
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'speech.webm');
        const res = await fetch('/api/stt', { method: 'POST', body: fd });
        const data = await res.json();
        setMic(false);
        if (data.status === 'ok' && data.data && data.data.text) {
          questionInput.value = data.data.text.trim();
          if (questionInput.value) form.dispatchEvent(new Event('submit'));
        } else {
          addMessage('ai', 'Could not transcribe audio.');
        }
      } catch (err) {
        setMic(false);
        addMessage('ai', 'STT error: ' + (err?.message || 'unknown'));
      }
    };

    mediaRecorder.start(250);
    setMic(true);
  }

  function stopWebSpeech() { if (recognition) { try { recognition.stop(); } catch {} recognition = null; } }
  function stopWhisper() { if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop(); mediaRecorder = null; }
  function stop() { stopWebSpeech(); stopWhisper(); setMic(false); }

  micBtn.addEventListener('click', async () => {
    if (!topicSelected) { addMessage('ai', 'Please confirm a topic first.'); return; }
    if (micBtn.getAttribute('aria-pressed') === 'true') { stop(); return; }
    const started = startWebSpeech();
    if (!started) await startWhisper();
  });
})();

// -------- Render --------
// Function to format text into bullet points
function formatToBulletPoints(text) {
  if (!text) return text;
  
  // First, preserve any existing line breaks
  let originalLines = text.split(/\n/);
  
  // If the text already has multiple lines, use them
  if (originalLines.length > 2) {
    return originalLines
      .map(line => line.trim())
      .filter(line => line.length > 5)
      .map(line => {
        // Clean up existing bullets
        line = line.replace(/^[\-\*•\d+\.\)\s]+/, '').trim();
        if (!line) return '';
        // Add proper punctuation
        if (!line.endsWith('.') && !line.endsWith('!') && !line.endsWith('?') && !line.endsWith(':')) {
          line += '.';
        }
        return `• ${line}`;
      })
      .filter(line => line.length > 2)
      .join('\n');
  }
  
  // Split by sentences for single paragraph text
  let lines = text.split(/\. (?=[A-Z])|; (?=[A-Z])/)
    .map(line => line.trim())
    .filter(line => line.length > 15);
  
  // If we still have only one line, return as is
  if (lines.length <= 1) return text;
  
  // Format as bullet points
  return lines.map((line, index) => {
    // Clean up the line
    line = line.replace(/^[\-\*•]\s*/, '').trim();
    
    // Add period if it's not the last line and doesn't have punctuation
    if (index === lines.length - 1) {
      // Last line might already have punctuation
      if (!line.endsWith('.') && !line.endsWith('!') && !line.endsWith('?')) {
        line += '.';
      }
    } else {
      // Middle lines should end with period
      if (!line.endsWith('.')) {
        line += '.';
      }
    }
    return `• ${line}`;
  }).join('\n');
}

function addMessage(role, text, sources=[]) {
  const tmpl = document.getElementById('msg-template');
  const node = tmpl.content.firstElementChild.cloneNode(true);
  node.classList.add(role);

  const avatar = node.querySelector('.avatar');
  if (role === 'ai') {
    avatar.innerHTML = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 2a1 1 0 110 2h-1v1.126A7.002 7.002 0 005 12v5a3 3 0 003 3h8a3 3 0 003-3v-5a7.002 7.002 0 00-6-6.874V4h-1a1 1 0 110-2h1zM8 12a2 2 0 114 0 2 2 0 01-4 0zm8 0a2 2 0 11-4 0 2 2 0 014 0zM7 17h10v1a1 1 0 01-1 1H8a1 1 0 01-1-1v-1z"/></svg>';
    // Format AI responses as bullet points
    text = formatToBulletPoints(text);
  } else {
    avatar.innerHTML = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 12c2.761 0 5-2.239 5-5s-2.239-5-5-5-5 2.239-5 5 2.239 5 5 5zm0 2c-4.418 0-8 2.239-8 5v1h16v-1c0-2.761-3.582-5-8-5z"/></svg>';
  }

  // Use innerHTML for bullet points to preserve formatting
  const contentEl = node.querySelector('.content');
  if (role === 'ai') {
    // Always use innerHTML for AI responses to handle formatting
    const formattedText = text
      .replace(/\n/g, '<br>')
      .replace(/•/g, '<span style="color: #4c8dff; font-weight: bold; margin-right: 6px;">•</span>');
    contentEl.innerHTML = formattedText;
  } else {
    contentEl.textContent = text;
  }

  const controls = node.querySelector('.controls');
  if (role === 'ai') {
    controls.hidden = false;
    controls.innerHTML = `
      <button class="ttsToggleBtn" title="Play speech" aria-label="Play speech">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <path d="M8 5v14l11-7-11-7Z"></path>
        </svg>
      </button>
    `;
    const toggleBtn = controls.querySelector('.ttsToggleBtn');
    let playing = false; let utterance = null;
    function setIcon(isPlaying) {
      toggleBtn.title = isPlaying ? 'Pause speech' : 'Play speech';
      toggleBtn.setAttribute('aria-label', toggleBtn.title);
      toggleBtn.innerHTML = isPlaying
        ? `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M7 6h4v12H7zM13 6h4v12h-4z"/></svg>`
        : `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M8 5v14l11-7-11-7Z"/></svg>`;
    }
    setIcon(false);
    function createUtterance() {
      const u = new SpeechSynthesisUtterance(text);
      u.onend = () => { playing = false; setIcon(false); };
      u.onerror = () => { playing = false; setIcon(false); };
      u.onpause = () => { playing = false; setIcon(false); };
      u.onresume = () => { playing = true; setIcon(true); };
      return u;
    }
    toggleBtn.addEventListener('click', () => {
      try {
        if (!playing) {
          if (speechSynthesis.paused && utterance) {
            speechSynthesis.resume(); playing = true; setIcon(true);
          } else {
            speechSynthesis.cancel();
            utterance = createUtterance();
            speechSynthesis.speak(utterance);
            playing = true; setIcon(true);
          }
        } else {
          if (speechSynthesis.speaking && !speechSynthesis.paused) {
            speechSynthesis.pause();
          }
        }
      } catch {}
    });

    // Try to harvest a price string for Connect modal
    const maybePrice = extractPriceFromText(text);
    if (maybePrice) window.currentQuotedPrice = maybePrice;
  } else {
    controls.hidden = true;
    controls.innerHTML = '';
  }

  // Sources
  if (sources && sources.length) {
    const det = node.querySelector('.sources');
    det.hidden = false;
    const ul = det.querySelector('ul');
    sources.forEach((s, i) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = s; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.textContent = `[${i+1}] ${s}`;
      li.appendChild(a);
      ul.appendChild(li);
    });
  }

  messagesEl.appendChild(node);
  const nearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
  if (nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function showTyping() {
  const tmpl = document.getElementById('msg-template');
  const node = tmpl.content.firstElementChild.cloneNode(true);
  node.classList.add('ai', 'typing');
  node.querySelector('.controls').hidden = true;
  node.querySelector('.sources').hidden = true;
  node.querySelector('.content').innerHTML = '<div class="dots" aria-label="AI is typing" role="status"><span></span><span></span><span></span></div>';
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}
function hideTyping(node) { if (node && node.parentNode) node.parentNode.removeChild(node); }

function extractPriceFromText(str='') {
  const m = (str || '').match(/(?:(?:₹|Rs\.?\s?|INR\s?)\s?\d[\d,]*(?:\.\d{1,2})?|(?:USD|\$)\s?\d[\d,]*(?:\.\d{1,2})?)/i);
  return m ? m[0] : '';
}

// -------- Theme Toggle --------
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcons(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeIcons(newTheme);
}

function updateThemeIcons(theme) {
  if (theme === 'dark') {
    sunIcon.style.display = 'none';
    moonIcon.style.display = 'block';
  } else {
    sunIcon.style.display = 'block';
    moonIcon.style.display = 'none';
  }
}

// Initialize theme on page load
initTheme();

// Event listener for theme toggle
themeToggle?.addEventListener('click', toggleTheme);
