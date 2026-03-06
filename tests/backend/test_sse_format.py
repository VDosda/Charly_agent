from __future__ import annotations

import unittest

from agent.api.sse import encode_sse_event


class SSEFormatTests(unittest.TestCase):
    def test_delta_event_is_strict(self):
        payload = encode_sse_event("delta", {"text": "hello"})
        self.assertEqual(payload, 'event: delta\ndata: {"text":"hello"}\n\n')

    def test_start_event_uses_json_data(self):
        payload = encode_sse_event("start", {"user_id": "u1", "session_id": "s1"})
        self.assertEqual(
            payload,
            'event: start\ndata: {"user_id":"u1","session_id":"s1"}\n\n',
        )

    def test_invalid_event_name_raises(self):
        with self.assertRaises(ValueError):
            encode_sse_event("ping", {"ok": True})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
