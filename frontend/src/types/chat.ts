export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
}

export interface ChatStreamRequest {
  message: string;
  user_id: string;
  session_id: string;
}

export type SSEEventName = "start" | "delta" | "end" | "error";

export type StreamEventData = {
  [key: string]: unknown;
};

export interface StreamEvent {
  event: SSEEventName;
  data: StreamEventData;
}
