import type { ChatMessage } from "../types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

type MessageBubbleProps = {
  message: ChatMessage;
};

export function MessageBubble({ message }: MessageBubbleProps) {
  const isAssistant = message.role === "assistant";
  const text = isAssistant && message.text.length === 0 ? " " : message.text;

  return (
    <article className={`message-bubble ${isAssistant ? "assistant" : "user"}`}>
      <header className="message-role">{isAssistant ? "assistant" : "you"}</header>
      {isAssistant ? (
        <div className="message-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
            {text}
          </ReactMarkdown>
        </div>
      ) : (
        <p className="message-text">{text}</p>
      )}
    </article>
  );
}
