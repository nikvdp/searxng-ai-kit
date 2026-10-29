import unittest
from unittest.mock import patch

import httpx

from searxng_cli import run_http_server


class HttpServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_routes_receive_request_objects(self):
        async def exercise_routes(server):
            transport = httpx.ASGITransport(app=server.config.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1:8000"
            ) as client:
                response = await client.get("/")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["name"], "searxng-cli")

                response = await client.post(
                    "/", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["result"]["serverInfo"]["name"], "searxng-cli")

                response = await client.post("/register", json={"client_name": "test"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["token_endpoint_auth_method"], "none")

                response = await client.get("/", headers={"Origin": "http://untrusted.example"})
                self.assertEqual(response.status_code, 403, response.text)

        with patch("uvicorn.Server.serve", exercise_routes):
            await run_http_server(None, "127.0.0.1", 8000)


if __name__ == "__main__":
    unittest.main()
