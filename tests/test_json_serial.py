import json
import unittest

from searxng_cli import make_json_safe, json_serial


class AnswerLike:
    def as_dict(self):
        return {
            "answer": "OpenAI answer text",
            "engine": "duckduckgo",
        }


class ResultLike:
    def __init__(self):
        self.fields = {
            "title": "OpenAI",
            "url": "https://openai.com/",
            "content": "AI research and products",
            "engine": "duckduckgo",
            "parsed_url": object(),
            "engines": {"duckduckgo", "brave"},
        }

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, key):
        return self.fields[key]

    def as_dict(self):
        return self.fields


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

    def test_make_json_safe_converts_typed_result_rows(self):
        result = ResultLike()

        safe_value = make_json_safe(result)

        self.assertEqual(
            safe_value,
            {
                "title": "OpenAI",
                "url": "https://openai.com/",
                "content": "AI research and products",
                "engine": "duckduckgo",
                "engines": ["brave", "duckduckgo"],
            },
        )

    def test_json_serial_converts_typed_result_rows(self):
        payload = {"results": [ResultLike()]}

        encoded = json.dumps(payload, default=json_serial)

        self.assertEqual(
            json.loads(encoded),
            {
                "results": [
                    {
                        "title": "OpenAI",
                        "url": "https://openai.com/",
                        "content": "AI research and products",
                        "engine": "duckduckgo",
                        "engines": ["brave", "duckduckgo"],
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
