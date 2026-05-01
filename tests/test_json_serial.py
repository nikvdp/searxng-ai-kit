import json
import unittest

from searxng_cli import json_serial


class AnswerLike:
    def as_dict(self):
        return {
            "answer": "OpenAI answer text",
            "engine": "duckduckgo",
        }


class JsonSerialTests(unittest.TestCase):
    def test_serializes_searxng_result_objects_with_as_dict(self):
        payload = {"answers": [AnswerLike()]}

        encoded = json.dumps(payload, default=json_serial)

        self.assertEqual(
            json.loads(encoded),
            {
                "answers": [
                    {
                        "answer": "OpenAI answer text",
                        "engine": "duckduckgo",
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
