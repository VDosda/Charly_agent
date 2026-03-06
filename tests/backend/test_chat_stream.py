from __future__ import annotations

import asyncio
import threading
import unittest

from agent.providers.llm.base import LLMResult
from agent.api.chat import chat_stream
from agent.api.dependencies import RuntimeContext
from agent.api.schemas import ChatStreamRequest


class FakeStreamingLLM:
    def supports_tools(self) -> bool:
        return False

    def generate(self, messages, tools=None, tool_choice=None) -> LLMResult:
        return LLMResult(text="bonjour", tool_calls=[])

    def stream_generate(self, messages, tools=None, tool_choice=None, on_delta=None) -> LLMResult:
        if on_delta:
            on_delta("bon")
            on_delta("jour")
        return LLMResult(text="bonjour", tool_calls=[])


class FakeRuntime:
    def __init__(self, should_fail: bool = False) -> None:
        self.llm = FakeStreamingLLM()
        self.should_fail = should_fail

    def handle_message(self, user_id: str, session_id: str, message: str) -> str:
        if self.should_fail:
            raise RuntimeError("boom")
        result = self.llm.generate(
            messages=[{"role": "user", "content": message}],
            tools=None,
            tool_choice="none",
        )
        return result.text


async def _collect_stream_body(response) -> str:
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(chunk)
    return "".join(chunks)


class ChatStreamTests(unittest.TestCase):
    def test_stream_endpoint_emits_start_delta_end(self):
        context = RuntimeContext(runtime=FakeRuntime(), lock=threading.Lock())
        payload = ChatStreamRequest(user_id="u-1", session_id="s-1", message="salut")

        response = asyncio.run(chat_stream(payload=payload, context=context))
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers.get("X-Accel-Buffering"), "no")

        raw = asyncio.run(_collect_stream_body(response))
        blocks = [b for b in raw.split("\n\n") if b.strip()]

        self.assertGreaterEqual(len(blocks), 4)
        self.assertEqual(blocks[0], 'event: start\ndata: {"user_id":"u-1","session_id":"s-1"}')
        self.assertEqual(blocks[1], 'event: delta\ndata: {"text":"bon"}')
        self.assertEqual(blocks[2], 'event: delta\ndata: {"text":"jour"}')
        self.assertEqual(blocks[-1], 'event: end\ndata: {"text":"bonjour"}')

    def test_stream_endpoint_emits_error_on_failure(self):
        context = RuntimeContext(runtime=FakeRuntime(should_fail=True), lock=threading.Lock())
        payload = ChatStreamRequest(user_id="u-2", session_id="s-2", message="salut")

        response = asyncio.run(chat_stream(payload=payload, context=context))
        raw = asyncio.run(_collect_stream_body(response))

        self.assertIn('event: start\ndata: {"user_id":"u-2","session_id":"s-2"}\n\n', raw)
        self.assertIn('event: error\ndata: {"message":"boom"}\n\n', raw)


if __name__ == "__main__":
    unittest.main()
