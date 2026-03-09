import { useEffect, useRef, useState } from "react";
import { streamChat } from "./lib/api";
import { ChatLayout } from "./components/ChatLayout";
import { MessageList } from "./components/MessageList";
import { Composer } from "./components/Composer";
import type { ChatMessage, StreamEvent } from "./types/chat";

const USER_ID = "Me";
const STORAGE_MESSAGES_KEY = "agent.chat.messages.v1";
const STORAGE_SESSION_KEY = "agent.chat.session.v1";

function makeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getLocalStorage(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage;
}

function makeSessionId(): string {
  return `backend-session-${makeId()}`;
}

function readStoredSessionId(): string {
  const storage = getLocalStorage();
  if (!storage) {
    return makeSessionId();
  }

  const existing = storage.getItem(STORAGE_SESSION_KEY);
  return existing && existing.trim() ? existing : makeSessionId();
}

function writeStoredSessionId(sessionId: string): void {
  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.setItem(STORAGE_SESSION_KEY, sessionId);
}

function readStoredMessages(): ChatMessage[] {
  const storage = getLocalStorage();
  if (!storage) {
    return [];
  }

  const raw = storage.getItem(STORAGE_MESSAGES_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    const restored: ChatMessage[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") {
        continue;
      }

      const id = typeof item.id === "string" ? item.id : "";
      const role = item.role === "user" || item.role === "assistant" ? item.role : null;
      const text = typeof item.text === "string" ? item.text : "";

      if (id && role) {
        restored.push({ id, role, text });
      }
    }
    return restored;
  } catch {
    return [];
  }
}

function writeStoredMessages(messages: ChatMessage[]): void {
  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.setItem(STORAGE_MESSAGES_KEY, JSON.stringify(messages));
}

function clearStoredMessages(): void {
  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.removeItem(STORAGE_MESSAGES_KEY);
}

function readStringField(data: Record<string, unknown>, key: string): string {
  const value = data[key];
  return typeof value === "string" ? value : "";
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => readStoredMessages());
  const [draft, setDraft] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const sessionIdRef = useRef<string>(readStoredSessionId());

  useEffect(() => {
    writeStoredSessionId(sessionIdRef.current);
  }, []);

  useEffect(() => {
    writeStoredMessages(messages);
  }, [messages]);

  const appendAssistantText = (assistantId: string, chunk: string) => {
    if (!chunk) {
      return;
    }
    setMessages((prev) =>
      prev.map((m) => (m.id === assistantId ? { ...m, text: `${m.text}${chunk}` } : m)),
    );
  };

  const finalizeAssistant = (assistantId: string, finalText: string) => {
    if (!finalText) {
      return;
    }
    setMessages((prev) =>
      prev.map((m) => (m.id === assistantId ? { ...m, text: finalText } : m)),
    );
  };

  const setAssistantError = (assistantId: string, message: string) => {
    const fallback = message || "unknown streaming error";
    setMessages((prev) =>
      prev.map((m) => (m.id === assistantId ? { ...m, text: `[error] ${fallback}` } : m)),
    );
  };

  const handleSend = async () => {
    const message = draft.trim();
    if (!message || isGenerating) {
      return;
    }

    const userMsg: ChatMessage = { id: makeId(), role: "user", text: message };
    const assistantId = makeId();
    const assistantMsg: ChatMessage = { id: assistantId, role: "assistant", text: "" };

    setDraft("");
    setIsGenerating(true);
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      await streamChat(
        {
          user_id: USER_ID,
          session_id: sessionIdRef.current,
          message,
        },
        (event: StreamEvent) => {
          if (event.event === "delta") {
            appendAssistantText(assistantId, readStringField(event.data, "text"));
            return;
          }

          if (event.event === "end") {
            finalizeAssistant(assistantId, readStringField(event.data, "text"));
            return;
          }

          if (event.event === "error") {
            setAssistantError(assistantId, readStringField(event.data, "message"));
          }
        },
      );
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setAssistantError(assistantId, detail);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReset = () => {
    if (isGenerating) {
      return;
    }
    sessionIdRef.current = makeSessionId();
    writeStoredSessionId(sessionIdRef.current);
    clearStoredMessages();
    setMessages([]);
    setDraft("");
  };

  return (
    <ChatLayout
      isGenerating={isGenerating}
      onReset={handleReset}
      composer={
        <Composer value={draft} onChange={setDraft} onSend={handleSend} disabled={isGenerating} />
      }
    >
      <MessageList messages={messages} isGenerating={isGenerating} />
    </ChatLayout>
  );
}
