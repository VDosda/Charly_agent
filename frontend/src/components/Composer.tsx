import { FormEvent } from "react";

type ComposerProps = {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
};

export function Composer({ value, onChange, onSend, disabled }: ComposerProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (disabled || !value.trim()) {
      return;
    }
    onSend();
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <input
        className="composer-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Ask your agent..."
        disabled={disabled}
      />
      <button className="composer-send" type="submit" disabled={disabled || !value.trim()}>
        Send
      </button>
    </form>
  );
}
