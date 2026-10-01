"""Unit tests for the explainer's local logic — no network, no key needed.

Run: python3 test_explainer.py
"""

import json
import unittest
from unittest import mock

import explainer


EXAMPLE_REPORT = {
    "baseFrame": 2,
    "frameCount": 6,
    "people": [
        {
            "person": "personA",
            "baseFrame": 2,
            "donorFrame": 2,
            "swapped": False,
            "baseScore": 0.62,
            "donorScore": 0.62,
            "level": "none",
            "checksPassed": False,
            "checkMetrics": {"seamRatio": 0.4},
            "identityDistance": None,
            "fallbackReason": "blink at every candidate frame",
        },
        {
            "person": "personB",
            "baseFrame": 2,
            "donorFrame": 4,
            "swapped": True,
            "baseScore": 0.41,
            "donorScore": 0.88,
            "level": "A",
            "checksPassed": True,
            "checkMetrics": {"seamRatio": 1.1},
            "identityDistance": 0.021,
            "fallbackReason": None,
        },
    ],
    # Bulk fields that must NOT be sent: per-frame score tables etc.
    "scores": {"personA": [0.1, 0.2, 0.62], "personB": [0.3, 0.2, 0.41]},
    "candidates": {"personA": [0, 1], "personB": [4, 5]},
}


class PayloadTest(unittest.TestCase):
    def test_only_anonymous_numbers_are_sent(self):
        payload = explainer.build_payload(EXAMPLE_REPORT)
        self.assertEqual(payload["frameCount"], 6)
        self.assertEqual(payload["baseFrame"], 2)
        self.assertEqual(len(payload["people"]), 2)
        swapped = payload["people"][1]
        self.assertEqual(swapped["label"], "personB")
        self.assertEqual(swapped["donorFrame"], 4)
        self.assertTrue(swapped["swapped"])
        # The bulk per-frame tables stay on the device.
        serialized = json.dumps(payload)
        self.assertNotIn("candidates", serialized)
        self.assertNotIn("0.1,0.2,0.62", serialized)

    def test_tolerates_missing_fields(self):
        payload = explainer.build_payload({"people": [{}], "baseFrame": 0})
        self.assertIsNone(payload["frameCount"])
        self.assertEqual(payload["people"][0]["label"], None)


class MessageTest(unittest.TestCase):
    def test_two_messages_and_marker_contract(self):
        messages = explainer.build_messages(explainer.build_payload(EXAMPLE_REPORT))
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("base frame 2", messages[1]["content"])
        self.assertIn("WHY:", messages[1]["content"])
        self.assertIn("CAPTION:", messages[1]["content"])

    def test_frame_count_optional(self):
        payload = explainer.build_payload({"people": [], "baseFrame": 1})
        messages = explainer.build_messages(payload)
        self.assertIn("several frames", messages[1]["content"])


class ParseTest(unittest.TestCase):
    def test_plain_reply(self):
        why, caption = explainer.parse_reply(
            "WHY: Everyone is smiling with open eyes.\nCAPTION: perfect group photo")
        self.assertEqual(why, "Everyone is smiling with open eyes.")
        self.assertEqual(caption, "perfect group photo")

    def test_reasoning_preamble_is_ignored(self):
        reply = ("Let me think about this report. Person B was swapped from frame 4 "
                 "because their score improved.\nWHY: B was lifted from a better moment; "
                 "A stayed as shot because every candidate caught a blink.\n"
                 "CAPTION: everyone's best moment, one photo")
        why, caption = explainer.parse_reply(reply)
        self.assertIn("lifted from a better moment", why)
        self.assertNotIn("Let me think", why)
        self.assertEqual(caption, "everyone's best moment, one photo")

    def test_case_insensitive_markers(self):
        why, caption = explainer.parse_reply("why: x\ncaption: y")
        self.assertEqual((why, caption), ("x", "y"))

    def test_missing_markers_raise(self):
        with self.assertRaises(SystemExit):
            explainer.parse_reply("Here is a friendly paragraph with no markers.")


class WireTest(unittest.TestCase):
    def test_call_sends_bearer_and_openai_shape(self):
        captured = {}

        class FakeResponse(mock.MagicMock):
            def __enter__(self):
                return self

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "WHY: w\nCAPTION: c"}}]}
                ).encode()

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode())
            return FakeResponse()

        with mock.patch.object(explainer.urllib.request, "urlopen", fake_urlopen):
            why, caption = explainer.parse_reply(
                explainer.call_nemotron(
                    explainer.build_messages(
                        explainer.build_payload(EXAMPLE_REPORT)),
                    "key-123",
                    explainer.DEFAULT_MODEL))
        self.assertEqual(captured["url"], explainer.ENDPOINT)
        self.assertEqual(captured["auth"], "Bearer key-123")
        self.assertEqual(captured["body"]["model"], explainer.DEFAULT_MODEL)
        self.assertEqual(captured["body"]["max_tokens"], 400)
        self.assertEqual(caption, "c")


if __name__ == "__main__":
    unittest.main(verbosity=2)
