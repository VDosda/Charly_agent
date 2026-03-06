import type { ChatMessage } from "../types/chat";

type MessageBubbleProps = {
  message: ChatMessage;
};

export function MessageBubble({ message }: MessageBubbleProps) {
  const isAssistant = message.role === "assistant";
  const text = isAssistant && message.text.length === 0 ? " " : message.text;

  return (
    <article className={`message-bubble ${isAssistant ? "assistant" : "user"}`}>
      <header className="message-role">{isAssistant ? "assistant" : "you"}</header>
      <p className="message-text">{text}</p>
    </article>
  );
}
