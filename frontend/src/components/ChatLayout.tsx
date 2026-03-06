import type { ReactNode } from "react";

type ChatLayoutProps = {
  children: ReactNode;
  composer: ReactNode;
  isGenerating: boolean;
  onReset: () => void;
};

export function ChatLayout({ children, composer, isGenerating, onReset }: ChatLayoutProps) {
  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div>
          <h1>Agent Chat</h1>
          <p>{isGenerating ? "generation in progress..." : "ready"}</p>
        </div>
        <button className="reset-button" type="button" onClick={onReset}>
          Reset
        </button>
      </header>

      {children}
      {composer}
    </main>
  );
}
