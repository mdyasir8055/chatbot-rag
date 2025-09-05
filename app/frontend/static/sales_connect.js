// app/frontend/sales_connect.js
// Lightweight, framework-free modal to collect lead details,
// preview the manager email, and send via backend.
//
// How to use from your existing UI:
//   1) Include this script in your HTML after your main script.
//   2) Call: window.openSalesConnect(productRef, quotedPrice, contextSummary)
//      from a button like “Connect to Manager”.

(function () {
  const qs = (s, el = document) => el.querySelector(s);
  const qsa = (s, el = document) => Array.from(el.querySelectorAll(s));
  const ce = (tag, props = {}) => Object.assign(document.createElement(tag), props);

  // ---------------- UI: Modal shell ----------------
  const modal = ce('div', { id: 'sales-modal', className: 'sales-modal hidden' });
  const styles = `
    .sales-modal.hidden { display: none; }
    .sales-modal { 
      position: fixed; inset: 0; 
      background: rgba(0,0,0,.5); 
      backdrop-filter: blur(4px);
      z-index: 9999; 
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .sales-modal .panel { 
      width: min(600px, 90vw); 
      max-height: 85vh; 
      overflow: hidden;
      background: var(--panel, #ffffff); 
      color: var(--ink, #1e293b); 
      border-radius: 16px; 
      box-shadow: 0 25px 50px rgba(0,0,0,.15), 0 0 0 1px rgba(255,255,255,.05);
      font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      flex-direction: column;
    }
    
    .sales-modal header { 
      padding: 24px 24px 16px; 
      border-bottom: 1px solid var(--border, #e2e8f0); 
      font-weight: 700; 
      font-size: 20px; 
      display: flex; 
      justify-content: space-between; 
      align-items: center;
      background: var(--panel, #ffffff);
    }
    .sales-modal header button { 
      border: none; 
      background: var(--panel-2, #f1f5f9); 
      width: 32px;
      height: 32px;
      border-radius: 8px;
      font-size: 16px; 
      cursor: pointer; 
      color: var(--muted, #64748b);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }
    .sales-modal header button:hover {
      background: var(--border-hover, #cbd5e1);
      color: var(--ink, #1e293b);
    }
    
    .sales-modal .content { 
      padding: 16px 24px; 
      display: grid; 
      gap: 20px; 
      overflow-y: auto;
      flex: 1;
    }
    
    .sales-row { 
      display: grid; 
      grid-template-columns: 1fr; 
      gap: 8px; 
    }
    .sales-row label { 
      font-size: 14px; 
      color: var(--ink, #1e293b); 
      font-weight: 600;
      margin-bottom: 4px;
    }
    .sales-row input, .sales-row textarea, .sales-row select {
      padding: 12px 16px; 
      border: 1px solid var(--border, #e2e8f0); 
      border-radius: 8px; 
      width: 100%; 
      font-size: 15px;
      background: var(--bg-2, #ffffff);
      color: var(--ink, #1e293b);
      transition: all 0.2s ease;
      font-family: Inter, sans-serif;
    }
    .sales-row input:focus, .sales-row textarea:focus {
      outline: none;
      border-color: var(--accent, #3b82f6);
      box-shadow: 0 0 0 3px rgba(59,130,246,.1);
    }
    .sales-row textarea { 
      min-height: 100px; 
      resize: vertical;
      line-height: 1.5;
    }
    
    .sales-actions { 
      display: flex; 
      gap: 12px; 
      justify-content: flex-end; 
      padding: 16px 24px 24px;
      border-top: 1px solid var(--border, #e2e8f0);
      background: var(--panel, #ffffff);
    }
    
    .btn { 
      padding: 12px 20px; 
      border-radius: 8px; 
      border: 1px solid transparent; 
      cursor: pointer; 
      font-weight: 600;
      font-size: 14px;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn.primary { 
      background: var(--accent, #3b82f6); 
      color: white;
      box-shadow: 0 4px 6px rgba(59,130,246,.2);
    }
    .btn.primary:hover {
      background: var(--accent-2, #2563eb);
      transform: translateY(-1px);
      box-shadow: 0 6px 12px rgba(59,130,246,.3);
    }
    .btn.secondary { 
      background: var(--panel-2, #f1f5f9); 
      color: var(--ink, #1e293b); 
      border: 1px solid var(--border, #e2e8f0);
    }
    .btn.secondary:hover { 
      background: var(--border-hover, #cbd5e1); 
      transform: translateY(-1px);
    }
    .btn.ghost { 
      background: transparent; 
      color: var(--muted, #64748b);
    }
    .btn.ghost:hover {
      background: var(--panel-2, #f1f5f9);
      color: var(--ink, #1e293b);
    }
    .btn:disabled { 
      opacity: 0.5; 
      cursor: not-allowed;
      transform: none !important;
    }
    
    .note { 
      font-size: 13px; 
      color: var(--muted, #64748b); 
      line-height: 1.5;
    }
    .preview { 
      background: var(--panel-2, #f1f5f9); 
      color: var(--ink, #1e293b); 
      border: 1px solid var(--border, #e2e8f0);
      border-radius: 8px; 
      padding: 16px; 
      font-family: ui-monospace, SFMono-Regular, 'SF Mono', 'Cascadia Code', 'Roboto Mono', Consolas, monospace; 
      font-size: 13px; 
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    .row2 { 
      grid-column: 1 / -1; 
    }
    .pill { 
      display: inline-block; 
      padding: 4px 12px; 
      border-radius: 999px; 
      background: rgba(59,130,246,.1); 
      color: var(--accent, #3b82f6); 
      font-size: 12px;
      font-weight: 600;
    }
    
    /* Dark theme support */
    [data-theme="dark"] .sales-modal .panel {
      background: var(--panel);
      box-shadow: 0 25px 50px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.1);
    }
    
    /* Success message */
    .success-message {
      position: fixed;
      top: 20px;
      right: 20px;
      background: #10b981;
      color: white;
      padding: 16px 20px;
      border-radius: 8px;
      box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
      z-index: 10000;
      font-weight: 600;
      font-size: 14px;
      max-width: 350px;
      animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    
    /* Responsive */
    @media (max-width: 640px) {
      .sales-modal .panel {
        width: 95vw;
        max-height: 90vh;
      }
      .sales-modal .content {
        padding: 16px;
      }
      .sales-modal header {
        padding: 20px 16px 12px;
      }
      .sales-actions {
        padding: 12px 16px 20px;
        flex-direction: column-reverse;
      }
      .btn {
        width: 100%;
        justify-content: center;
      }
      .success-message {
        top: 10px;
        right: 10px;
        left: 10px;
        max-width: none;
      }
    }
  `;
  const styleTag = ce('style', { textContent: styles });

  const panel = ce('div', { className: 'panel' });
  const header = ce('header');
  const title = ce('div', { textContent: 'Connect with Store Manager' });
  const close = ce('button', { innerHTML: '✕', title: 'Close' });
  header.append(title, close);

  const content = ce('div', { className: 'content' });

  // Fields
  const inputProduct = inputRow('Product / Variant *', 'text', 'sales-product', 'e.g., ACME Laptop 14 (16GB/512GB)');
  const inputQuoted = inputRow('Quoted Price (optional)', 'text', 'sales-quoted', 'e.g., ₹59,990');
  const inputName = inputRow('Your Full Name *', 'text', 'sales-name', '');
  const inputEmail = inputRow('Email *', 'email', 'sales-email', '');
  const inputPhone = inputRow('Phone *', 'tel', 'sales-phone', '+91XXXXXXXXXX');
  const inputBest = inputRow('Best Time to Reach', 'text', 'sales-best', 'e.g., Today 4–7 PM IST');
  const inputSummary = textRow('Summary to Send *', 'sales-summary', 'Short context of your request (you can edit before sending)');
  
  // Add Generate Summary button - positioned right below the summary textarea
  const summaryButtonRow = ce('div', { className: 'sales-row row2', style: 'margin-top: 8px; margin-bottom: 8px;' });
  const summaryButtonWrap = ce('div', { style: 'display: flex; gap: 12px; align-items: center; flex-wrap: wrap;' });
  const btnGenerateSummary = ce('button', { 
    className: 'btn secondary', 
    type: 'button',
    textContent: '✨ Generate Chat Summary',
    style: 'font-size: 13px; padding: 8px 16px; white-space: nowrap;'
  });
  const summaryStatus = ce('span', { 
    className: 'note', 
    textContent: 'Click to auto-generate a summary from your chat history',
    style: 'flex: 1; min-width: 200px;'
  });
  summaryButtonWrap.append(btnGenerateSummary, summaryStatus);
  summaryButtonRow.append(summaryButtonWrap);

  const consentRowEl = ce('div', { className: 'sales-row row2' });
  const consentWrap = ce('div', { 
    style: 'display: flex; align-items: center; gap: 8px; margin-top: 4px;' 
  });
  const consent = ce('input', { id: 'sales-consent', type: 'checkbox' });
  const consentLabel = ce('label', { 
    htmlFor: 'sales-consent', 
    innerHTML: 'I agree to share my details with the store manager for follow-up about this product.',
    className: 'note',
    style: 'margin: 0; cursor: pointer; line-height: 1.4; font-weight: 400;'
  });
  consentWrap.append(consent, consentLabel);
  consentRowEl.append(consentWrap);

  // Preview area
  const previewWrap = ce('div', { className: 'sales-row row2' });
  const previewLabel = ce('label', { innerHTML: 'Email Preview' });
  const previewBox = ce('div');
  const previewNote = ce('div', { className: 'note', textContent: 'Click “Preview Email” to generate the message that will be sent.' });
  const previewPre = ce('pre', { className: 'preview', textContent: '' });
  previewBox.append(previewNote, previewPre);
  previewWrap.append(previewLabel, previewBox);

  // Actions
  const actions = ce('div', { className: 'sales-actions' });
  const btnPreview = ce('button', { className: 'btn secondary', textContent: 'Preview Email' });
  const btnSend = ce('button', { className: 'btn primary', textContent: 'Send Email', disabled: true });
  const btnCancel = ce('button', { className: 'btn ghost', textContent: 'Cancel' });
  actions.append(btnCancel, btnPreview, btnSend);

  content.append(
    inputProduct.row,
    inputQuoted.row,
    inputName.row,
    inputEmail.row,
    inputPhone.row,
    inputBest.row,
    inputSummary.row,
    summaryButtonRow,
    consentRowEl,
    previewWrap
  );

  panel.append(header, content, actions);
  modal.append(panel);
  document.head.appendChild(styleTag);
  document.body.appendChild(modal);

  // ---------------- Handlers ----------------
  close.addEventListener('click', hide);
  btnCancel.addEventListener('click', hide);
  
  // Generate Summary handler
  btnGenerateSummary.addEventListener('click', async () => {
    try {
      btnGenerateSummary.disabled = true;
      btnGenerateSummary.textContent = '⏳ Generating...';
      summaryStatus.textContent = 'AI is analyzing your chat history...';
      
      // Get all messages from the chat - improved selector
      const messages = document.querySelectorAll('.msg, .message');
      let chatHistory = '';
      
      console.log('Found messages:', messages.length); // Debug log
      
      messages.forEach((msg, index) => {
        const isUser = msg.classList.contains('user');
        const isAI = msg.classList.contains('ai');
        const content = msg.querySelector('.content')?.textContent || msg.querySelector('.bubble .content')?.textContent || msg.textContent || '';
        
        if (content.trim() && !msg.classList.contains('typing')) {
          const role = isUser ? 'User' : (isAI ? 'AI' : 'Message');
          chatHistory += `${role}: ${content.trim()}\n\n`;
        }
      });
      
      console.log('Chat history length:', chatHistory.length); // Debug log
      
      if (!chatHistory.trim()) {
        summaryStatus.textContent = 'No chat history found to summarize';
        btnGenerateSummary.disabled = false;
        btnGenerateSummary.textContent = '✨ Generate Chat Summary';
        return;
      }
      
      // Send to backend for summarization
      // First try to initialize a basic topic if none exists
      try {
        const initFd = new FormData();
        initFd.append('product_name', 'General Inquiry');
        await fetch('/api/init-topic', { method: 'POST', body: initFd });
      } catch (e) {
        // Ignore init errors, continue with summary
        console.log('Topic init skipped:', e.message);
      }
      
      const fd = new FormData();
      fd.append('question', `Please provide a concise business summary of this chat conversation for a sales manager. Focus on: 1) What product/topic was discussed, 2) Key questions asked, 3) Customer interest level, 4) Any specific requirements mentioned. Keep it under 150 words and professional.\n\nChat History:\n${chatHistory}`);
      
      const res = await fetch('/api/ask', { method: 'POST', body: fd });
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const data = await res.json();
      console.log('API response:', data); // Debug log
      
      if (data.ok && data.data) {
        let summary = data.data.answer || data.data.final_answer || 'Could not generate summary';
        // Clean up the summary (remove bullet points for cleaner text)
        summary = summary.replace(/[•\-\*]/g, '').replace(/\n\s*\n/g, '\n').trim();
        
        if (summary && summary !== 'Could not generate summary' && summary.length > 10) {
          inputSummary.textarea.value = summary;
          summaryStatus.textContent = 'Summary generated! You can edit it before sending.';
        } else {
          // Fallback to manual summary based on chat history
          const manualSummary = generateManualSummary(chatHistory);
          inputSummary.textarea.value = manualSummary;
          summaryStatus.textContent = 'Generated basic summary. Please edit as needed.';
        }
      } else {
        // Fallback to manual summary based on chat history
        const manualSummary = generateManualSummary(chatHistory);
        inputSummary.textarea.value = manualSummary;
        summaryStatus.textContent = 'Generated basic summary. Please edit as needed.';
      }
    } catch (error) {
      console.error('Summary generation failed:', error);
      summaryStatus.textContent = 'Failed to generate summary. Please write manually.';
    } finally {
      btnGenerateSummary.disabled = false;
      btnGenerateSummary.textContent = '✨ Generate Chat Summary';
    }
  });

  btnPreview.addEventListener('click', async () => {
    const payload = collect();
    const errors = validate(payload);
    if (errors.length) {
      alert('Please fix:\n• ' + errors.join('\n• '));
      return;
    }
    try {
      btnPreview.disabled = true;
      const fd = new FormData();
      fd.append('user_name', payload.user_name);
      fd.append('user_email', payload.user_email);
      fd.append('user_phone', payload.user_phone);
      fd.append('product_ref', payload.product_ref);
      fd.append('summary', payload.summary);
      if (payload.best_time) fd.append('best_time', payload.best_time);
      if (payload.quoted_price) fd.append('quoted_price', payload.quoted_price);

      const res = await fetch('/api/sales/connect/prepare', { method: 'POST', body: fd });
      const json = await res.json();
      if (!json.ok) throw new Error('Server error: ' + JSON.stringify(json));
      const { to, subject, body } = json.data;

      previewPre.textContent = (
        `To: ${to}\nSubject: ${subject}\n\n${body}`
      );
      btnSend.disabled = false;
    } catch (e) {
      alert('Preview failed: ' + e.message);
      btnSend.disabled = true;
    } finally {
      btnPreview.disabled = false;
    }
  });

  btnSend.addEventListener('click', async () => {
    try {
      btnSend.disabled = true;
      const text = previewPre.textContent || '';
      const subj = (text.match(/^Subject:\s*(.*)$/m) || [,''])[1];
      const body = text.split(/\n\n/).slice(1).join('\n\n') || '';
      if (!subj || !body) {
        alert('Please click “Preview Email” first.');
        btnSend.disabled = false;
        return;
      }
      const fd = new FormData();
      fd.append('subject', subj);
      fd.append('body', body);

      const res = await fetch('/api/sales/connect/send', { method: 'POST', body: fd });
      const json = await res.json();
      if (!json.ok) throw new Error('Server error: ' + JSON.stringify(json));
      const { status, info } = json.data;
      if (status === 'sent') {
        showSuccessMessage('Email sent successfully! The store manager will contact you shortly.');
        setTimeout(() => hide(), 3000); // Auto-close after 3 seconds
      } else {
        // DRY RUN (no SMTP) – show preview in a nicer way
        previewPre.textContent = info;
        showSuccessMessage('Email preview generated successfully! (SMTP not configured - this is a dry run)');
        setTimeout(() => hide(), 3000); // Auto-close after 3 seconds
      }
    } catch (e) {
      alert('Send failed: ' + e.message);
    } finally {
      btnSend.disabled = false;
    }
  });

  // ---------------- Public API ----------------
  function show() {
    modal.classList.remove('hidden');
  }
  function hide() {
    modal.classList.add('hidden');
    // reset state
    btnSend.disabled = true;
    previewPre.textContent = '';
  }

  function inputRow(labelText, type, id, placeholder = '') {
    const row = ce('div', { className: 'sales-row' });
    const label = ce('label', { htmlFor: id, textContent: labelText });
    const input = ce('input', { id, type, placeholder });
    row.append(label, input);
    return { row, input };
  }

  function textRow(labelText, id, placeholder = '') {
    const row = ce('div', { className: 'sales-row row2' });
    const label = ce('label', { htmlFor: id, textContent: labelText });
    const textarea = ce('textarea', { id, placeholder });
    row.append(label, textarea);
    return { row, textarea };
  }

  function collect() {
    return {
      product_ref: inputProduct.input.value.trim(),
      quoted_price: inputQuoted.input.value.trim(),
      user_name: inputName.input.value.trim(),
      user_email: inputEmail.input.value.trim(),
      user_phone: inputPhone.input.value.trim(),
      best_time: inputBest.input.value.trim(),
      summary: inputSummary.textarea.value.trim(),
      consent: consent.checked
    };
  }

  function validate(p) {
    const errors = [];
    if (!p.product_ref) errors.push('Product / Variant is required.');
    if (!p.user_name) errors.push('Your full name is required.');
    if (!p.user_email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(p.user_email)) errors.push('A valid email is required.');
    if (!p.user_phone || !/^\+?\d[\d\s\-()]{7,}$/.test(p.user_phone)) errors.push('A valid phone number is required.');
    if (!p.summary) errors.push('Summary is required.');
    if (!p.consent) errors.push('Consent is required.');
    return errors;
    }

  // Manual summary generation fallback
  function generateManualSummary(chatHistory) {
    if (!chatHistory || !chatHistory.trim()) {
      return 'Customer inquiry - please provide details about the conversation.';
    }
    
    const lines = chatHistory.split('\n').filter(line => line.trim());
    const userMessages = lines.filter(line => line.startsWith('User:')).map(line => line.replace('User: ', '').trim());
    const aiMessages = lines.filter(line => line.startsWith('AI:')).map(line => line.replace('AI: ', '').trim());
    
    // Extract potential product names from the conversation
    const productKeywords = ['iPhone', 'Samsung', 'Galaxy', 'MacBook', 'iPad', 'Tesla', 'PlayStation', 'Xbox', 'Nike', 'Adidas'];
    let detectedProduct = 'Product inquiry';
    
    for (const keyword of productKeywords) {
      if (chatHistory.toLowerCase().includes(keyword.toLowerCase())) {
        detectedProduct = keyword + ' inquiry';
        break;
      }
    }
    
    // Create a basic summary
    let summary = `${detectedProduct} - Customer engaged in conversation about product features and pricing.`;
    
    if (userMessages.length > 1) {
      summary += ` Customer asked ${userMessages.length} questions showing active interest.`;
    }
    
    // Look for price mentions
    if (chatHistory.includes('$') || chatHistory.includes('price') || chatHistory.includes('cost')) {
      summary += ' Pricing was discussed.';
    }
    
    // Look for comparison mentions
    if (chatHistory.includes('compare') || chatHistory.includes('vs') || chatHistory.includes('alternative')) {
      summary += ' Customer interested in product comparisons.';
    }
    
    return summary;
  }

  // Success message function
  function showSuccessMessage(message) {
    // Remove any existing success messages
    const existingMsg = document.querySelector('.success-message');
    if (existingMsg) {
      existingMsg.remove();
    }
    
    // Create and show new success message
    const successMsg = ce('div', {
      className: 'success-message',
      textContent: message
    });
    
    document.body.appendChild(successMsg);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
      if (successMsg && successMsg.parentNode) {
        successMsg.remove();
      }
    }, 3000);
  }

  // Expose an entry point for your existing UI
  window.openSalesConnect = function (productRef = '', quotedPrice = '', contextSummary = '') {
    inputProduct.input.value = productRef || '';
    inputQuoted.input.value = quotedPrice || '';
    inputSummary.textarea.value = contextSummary || '';
    show();
  };
})();
