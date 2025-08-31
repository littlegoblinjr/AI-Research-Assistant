import React, { useEffect, useMemo, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Mic,
  Paperclip,
  Wand2,
  Bot,
  User,
  Sparkles,
  Settings,
  Brain,
  Zap,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  ChevronDown,
} from "lucide-react";

// ORIGINAL LAYOUT RECREATED (no Tailwind):
// - Same structure and styling as your very first Tailwind version
// - Glassmorphism, gradient background, floating blobs, soft grid overlay
// - Sidebar: Research Agent tools, Model card (select + creativity), Quick prompts
// - Chat panel with avatars, typing dots, Enter-to-send
// - Pure React + CSS-in-document (style tag), no TypeScript annotations

// ===== API base (no CRA/Vite): set window.__API_BASE__ in your HTML or fallback =====
const API_BASE = (typeof window !== "undefined" && window.__API_BASE__) || "http://127.0.0.1:8000";


let userId = localStorage.getItem("user_id");
if (!userId) {
  userId = uuidv4();
  localStorage.setItem("user_id", userId);}


const MessageType = {
  user: "user",
  bot: "bot",
};

function TypingDots() {
  return (
    <div className="typing-dots">
      <span />
      <span />
      <span />
    </div>
  );
}

function GlassButton({ icon: Icon, children, onClick }) {
  return (
    <button onClick={onClick} className="btn-glass">
      {Icon ? <Icon className="icon" size={16} /> : null}
      <span className="btn-label">{children}</span>
    </button>
  );
}

function SuggestionChip({ icon: Icon, label, onClick }) {
  return (
    <button onClick={onClick} className="chip">
      {Icon ? <Icon className="icon" size={16} /> : null}
      <span>{label}</span>
    </button>
  );
}

function MessageBubble({ role, text }) {
  const isUser = role === MessageType.user;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ type: "spring", stiffness: 200, damping: 20 }}
      className={`row ${isUser ? "row-right" : "row-left"}`}
    >
      {!isUser && (
        <div className="avatar subtle-card">
          <Bot size={18} />
        </div>
      )}
      <div className={`bubble subtle-card ${isUser ? "bubble-user" : "bubble-bot"}`}>
        <p className="bubble-text">{text}</p>
      </div>
      {isUser && (
        <div className="avatar subtle-card">
          <User size={18} />
        </div>
      )}
    </motion.div>
  );
}

const initialMessages = [
  
];

export default function AestheticChatbotUI() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [model, setModel] = useState("gpt-research-pro");
  const [creativity, setCreativity] = useState(30);

  // NEW: hybrid-mode selector
  const [augment, setAugment] = useState("file_plus_index"); // "file_only" | "index_only" | "file_plus_index"

  // NEW: attachment for the next message
  const [attachedFile, setAttachedFile] = useState(null);
  const attachInputRef = useRef(null);

  // new hybrid-mode state
 

  const chatRef = useRef(null);
  const pdfInputRef = useRef(null);
  const imageInputRef = useRef(null);
  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages.length, thinking]);

  async function handleSend(text) {
  const content = (text ?? input).trim();
  if (!content && !attachedFile) return; // allow file-only send
  setInput("");

  const shownText = content || (attachedFile ? `(sent a PDF: ${attachedFile.name})` : "");
  const userMsg = { id: String(Date.now()), role: MessageType.user, text: shownText };
  setMessages((prev) => [...prev, userMsg]);
  setThinking(true);

  try {
    let res;
    if (attachedFile) {
      // multipart: message + augment + file (+ model, creativity)
      const form = new FormData();
      form.append("user_id", userId)
      form.append("message", content);
      form.append("model", model);
      form.append("creativity", String(creativity));
      form.append("augment", augment); // <-- NEW
      form.append("file", attachedFile);
      res = await fetch(`${API_BASE}/chat`, { method: "POST", body: form });
    } else {
      // json
      res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, model, creativity, augment, "user_id": userId }), // <-- NEW
      });
    }

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const botMsg = {
      id: String(Date.now() + 1),
      role: MessageType.bot,
      text: data.reply || "(No reply)",
    };
    setMessages((prev) => [...prev, botMsg]);
  } catch (err) {
    setMessages((prev) => [
      ...prev,
      { id: String(Date.now() + 1), role: MessageType.bot, text: `⚠️ Error: ${String(err)}` },
    ]);
  } finally {
    setThinking(false);
    setAttachedFile(null);
    if (attachInputRef.current) attachInputRef.current.value = "";
  }
}


  function triggerPick(kind) {
    const ref = kind === "pdf" ? pdfInputRef : imageInputRef;
    if (ref.current) ref.current.click();
  }

  async function onFilePicked(kind, file) {
    if (!file) return;
    setThinking(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const url = kind === "pdf" ? `${API_BASE}/ingest/pdf` : `${API_BASE}/ingest/image`;
      const res = await fetch(url, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { id: String(Date.now()), role: MessageType.bot, text: `✅ Ingested ${kind.toUpperCase()}: ${data.chunks} chunk(s) added.` },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { id: String(Date.now()), role: MessageType.bot, text: `⚠️ Ingest failed: ${String(e)}` },
      ]);
    } finally {
      setThinking(false);
    }
  }

  const suggestions = useMemo(
    () => [
      { icon: Sparkles, label: "Summarize this paper (PDF)" },
      { icon: Brain, label: "Compare 3 SOTA methods" },
      { icon: Zap, label: "Draft a literature review" },
      { icon: MessageSquare, label: "Explain like I’m 5" },
    ],
    []
  );

  return (
    <div className="app">
      {/* Gradient Background */}
      <div className="bg-gradient" />

      {/* Floating gradient blobs */}
      <div className="blob blob-a" />
      <div className="blob blob-b" />

      {/* Soft grid overlay */}
      <div className="grid-overlay" />

      {/* App Shell */}
      <div className="container">
        <header className="header">
          <div className="brand">
            <div className="logo subtle-card">
              <Wand2 className="icon" />
            </div>
            <div>
              <h1 className="title gradient-text">AI Research Assistant</h1>
              <p className="subtitle">Glassmorphic Chat Workspace</p>
            </div>
          </div>
          <div className="header-actions">
          {/* NEW: context/augment select */}
            <div className="select-wrap subtle-card" style={{ marginRight: 8 }}>
              <select
                value={augment}
                onChange={(e) => setAugment(e.target.value)}
                className="select"
              >
                <option value="file_plus_index">File + Index</option>
                
                <option value="index_only">Index Only</option>
              </select>
            </div>

          
        </div>

        </header>

        <main className="layout">
          {/* Sidebar */}
          <aside className="sidebar">
            <section className="card subtle-card p">
              <div className="agent">
                <div className="agent-badge">
                  <Bot className="icon" />
                </div>
                <div>
                  <h2 className="h2">Research Agent</h2>
                  <p className="muted">Reasoning • Retrieval • Synthesis</p>
                </div>
              </div>
              <div className="tool-grid">
                
                
                
                
              </div>

              {/* Hidden File Inputs */}
              <input
                ref={pdfInputRef}
                type="file"
                accept="application/pdf"
                style={{ display: "none" }}
                onChange={(e) => onFilePicked("pdf", e.target.files && e.target.files[0])}
              />
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => onFilePicked("image", e.target.files && e.target.files[0])}
              />
            </section>

            {/* Model Card */}
            <section className="card subtle-card p">
              <div className="row between">
                <h3 className="label">Model</h3>
                <ChevronDown className="icon muted" />
              </div>
              <div className="row mt">
                <div className="select-wrap subtle-card">
                  <select value={model} onChange={(e) => setModel(e.target.value)} className="select">
                    <option value="gpt-research-pro">gemma-3-4b-it</option>
                   
                  </select>
                </div>
              </div>
              <div className="mt">
                <label className="muted small">Creativity</label>
                <input type="range" min={0} max={100} value={creativity} onChange={(e) => setCreativity(Number(e.target.value))} className="range" />
              </div>
            </section>

            
          </aside>

          {/* Chat Panel */}
          <section className="chat card subtle-card p">
            {/* Chat stream */}
            <div ref={chatRef} className="stream custom-scroll">
              <div className="stack">
                <AnimatePresence>
                  {messages.map((m) => (
                    <MessageBubble key={m.id} role={m.role} text={m.text} />
                  ))}
                </AnimatePresence>
                {thinking && (
                  <div className="row row-left">
                    <div className="avatar subtle-card">
                      <Bot size={18} />
                    </div>
                    <div className="bubble subtle-card"><TypingDots /></div>
                  </div>
                )}
              </div>
            </div>

            {/* Composer */}
            <div className="composer">
              <div className="input subtle-card">
                {/* ATTACH (for next message) */}
                <button
                  className="icon-btn subtle-card"
                  title={attachedFile ? `Attached: ${attachedFile.name}` : "Attach PDF to message"}
                  onClick={() => attachInputRef.current && attachInputRef.current.click()}
                >
                  <Paperclip className="icon" />
                </button>
                <input
                  ref={attachInputRef}
                  type="file"
                  accept="application/pdf"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files && e.target.files[0];
                    if (f) setAttachedFile(f);
                  }}
                />

                <textarea
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask about papers, datasets, methods…"
                  className="textarea"
                />

                
              </div>

              <button onClick={() => handleSend()} className="send subtle-card" title="Send">
                <Send className="icon" />
              </button>
            </div>

            {/* tiny file chip under composer */}
            {attachedFile && (
              <div className="muted small" style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 8 }}>
                📎 {attachedFile.name}
                <button
                  className="chip"
                  onClick={() => setAttachedFile(null)}
                  style={{ padding: "2px 8px", fontSize: 12 }}
                  title="Remove attachment"
                >
                  ✕
                </button>
              </div>
            )}

            <div className="hints">
              <span className="inline"><Wand2 className="icon" /> Glass mode</span>
              <span className="dot">•</span>
              <span className="inline"><Zap className="icon" /> Press Enter to send</span>
            </div>
          </section>
        </main>
      </div>

      {/* Styles */}
      <style>{`
        :root{
          --bg-a:#0b1020; --bg-b:#120b2e; --bg-c:#051726;
          --fg:#ffffff; --muted:rgba(255,255,255,0.7);
          --glass: rgba(255,255,255,0.10);
          --glass-card: rgba(255,255,255,0.08);
          --glass-brd: rgba(255,255,255,0.2);
          --shadow: 0 20px 40px rgba(0,0,0,0.35);
          --radius-xl: 18px; --radius-2xl: 22px;
          --grad: linear-gradient(90deg, #a5b4fc, #f0abfc, #67e8f9);
        }
        *{box-sizing:border-box}
        html,body,#root{height:100%}
        body{margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; background:#000}
        .app{position:relative; min-height:100dvh; color:var(--fg); overflow:hidden}
        .bg-gradient{position:absolute; inset:0; background:linear-gradient(135deg, var(--bg-a), var(--bg-b) 40%, var(--bg-c)); z-index:-3}
        .blob{position:absolute; border-radius:9999px; filter:blur(60px); opacity:.28; z-index:-2}
        .blob-a{top:-180px; left:-180px; width:520px; height:520px; background:linear-gradient(135deg,#6366f1,#f472b6,#22d3ee)}
        .blob-b{right:-160px; bottom:-160px; width:460px; height:460px; background:linear-gradient(135deg,#22d3ee,#38bdf8,#6366f1)}
        .grid-overlay{position:absolute; inset:0; z-index:-1; opacity:.08; background-image:radial-gradient(circle at 1px 1px, rgba(255,255,255,0.8) 1px, transparent 1px); background-size:24px 24px}

        .container{max-width:1120px; margin:0 auto; padding:24px 24px 40px}
        .header{display:flex; align-items:center; justify-content:space-between}
        .brand{display:flex; gap:12px; align-items:center}
        .logo{width:36px; height:36px; display:grid; place-items:center; border-radius:14px}
        .gradient-text{background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent}
        .title{margin:0; font-size:22px; font-weight:700}
        .subtitle{margin:2px 0 0; font-size:12px; color:var(--muted)}
        .header-actions{display:flex; gap:8px}

        .layout{display:grid; grid-template-columns:1fr; gap:20px; margin-top:24px}
        @media (min-width:1024px){ .layout{grid-template-columns: 1fr 2fr;} }

        .card{border-radius:var(--radius-2xl); border:1px solid var(--glass-brd); box-shadow:var(--shadow)}
        .subtle-card{backdrop-filter: blur(18px); background: var(--glass-card)}
        .p{padding:20px}

        .agent{display:flex; gap:12px; align-items:center}
        .agent-badge{width:48px; height:48px; border-radius:16px; display:grid; place-items:center; background: linear-gradient(135deg, rgba(99,102,241,.5), rgba(244,114,182,.5), rgba(34,211,238,.5))}
        .h2{margin:0; font-size:16px}
        .muted{color:var(--muted)}
        .small{font-size:12px}
        .label{font-size:14px; font-weight:600}
        .mb{margin-bottom:10px}
        .mt{margin-top:12px}
        .row{display:flex; align-items:center; gap:8px}
        .between{justify-content:space-between}

        .tool-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:14px}

        .btn-glass{display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:12px; border:1px solid var(--glass-brd); background:rgba(255,255,255,0.06); color:#fff; cursor:pointer}
        .btn-glass:hover{background:rgba(255,255,255,0.12)}
        .btn-label{font-size:13px}

        .chip{display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; border:1px solid var(--glass-brd); background:rgba(255,255,255,0.06); font-size:13px; color:#fff; cursor:pointer}
        .chip:hover{background:rgba(255,255,255,0.12)}
        .chips{display:flex; flex-wrap:wrap; gap:8px}

        .sidebar{display:flex; flex-direction:column; gap:20px}

        .chat {
          display: flex;
          flex-direction: column;
          height: 80vh;
        }
        .stream {
          flex: 1;
          overflow-y: auto;
        }

        .stack{display:flex; flex-direction:column; gap:12px}

        .row-left{justify-content:flex-start}
        .row-right{justify-content:flex-end}
        .avatar{width:36px; height:36px; border-radius:12px; display:grid; place-items:center; margin:0 8px}
        .bubble{max-width:min(700px, 76%); border-radius:16px; padding:12px 14px; font-size:14px; line-height:1.5; border:1px solid var(--glass-brd)}
        .bubble-user{background:rgba(255,255,255,0.14)}
        .bubble-bot{background:rgba(255,255,255,0.10)}
        .bubble-text{margin:0; color:rgba(255,255,255,.95); white-space:pre-wrap}

        .composer{display:flex; align-items:flex-end; gap:10px; margin-top:12px}
        .input{flex:1; display:flex; align-items:center; gap:8px; padding:10px 12px; border-radius:16px; border:1px solid var(--glass-brd)}
        .textarea{flex:1; resize:none; background:transparent; outline:none; border:none; color:#fff; font-size:14px}
        .icon{display:inline-block}
        .icon-btn{padding:8px; border-radius:10px; border:1px solid var(--glass-brd); display:grid; place-items:center; cursor:pointer}
        .icon-btn:hover{background:rgba(255,255,255,0.12)}
        .send{height:44px; width:44px; border-radius:14px; border:1px solid var(--glass-brd); display:grid; place-items:center; cursor:pointer}
        .send:hover{background:rgba(255,255,255,0.12)}

        .hints{margin-top:8px; display:flex; align-items:center; gap:8px; color:rgba(255,255,255,0.7); font-size:12px}
        .inline{display:inline-flex; align-items:center; gap:6px}
        .dot{opacity:.6}

        .typing-dots{display:flex; gap:6px; align-items:center}
        .typing-dots span{width:8px; height:8px; border-radius:999px; background:rgba(255,255,255,0.7); display:block; animation: bounce 1s infinite}
        .typing-dots span:nth-child(1){animation-delay:-.2s}
        .typing-dots span:nth-child(2){animation-delay:-.1s}
        @keyframes bounce{0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-4px)}}

        /* Scrollbar */
        .custom-scroll::-webkit-scrollbar{width:10px}
        .custom-scroll::-webkit-scrollbar-thumb{background:linear-gradient(180deg, rgba(255,255,255,0.25), rgba(255,255,255,0.05)); border-radius:9999px; border:2px solid rgba(255,255,255,0.1)}
        .custom-scroll::-webkit-scrollbar-track{background:transparent}
      `}</style>
    </div>
  );
}
