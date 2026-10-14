"""CPU-only tests for per-request Responses generation URL selection."""

import ast
import logging
import unittest
from pathlib import Path
from urllib.parse import urlsplit

source = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/entrypoints/openai/pd_responses.py"
)
tree = ast.parse(source.read_text())
nodes = [
    n
    for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in ("is_http_router_origin", "select_generation_url"))
    or (
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "GENERATION_URL_HEADER" for t in n.targets)
    )
]
scope = dict(urlsplit=urlsplit, logger=logging.getLogger("pd_responses_test"))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
is_http_router_origin = scope["is_http_router_origin"]
select_generation_url = scope["select_generation_url"]

CONFIGURED = "http://router-a:9000"


class FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class GenerationUrlTest(unittest.TestCase):
    def test_origin_validation(self):
        for good in ("http://router:9000", "https://r.example/", "http://[fdaa::1]:9000"):
            self.assertTrue(is_http_router_origin(good), good)
        for bad in (
            "ftp://router",
            "router:9000",
            "http://",
            "http://user:pw@router",
            "http://router/generate",
            "http://router?x=1",
            "http://router#frag",
        ):
            self.assertFalse(is_http_router_origin(bad), bad)

    def test_header_overrides_configured_url(self):
        request = FakeRequest({"x-smg-generation-url": "http://router-b:9001"})
        self.assertEqual(
            select_generation_url(CONFIGURED, request), "http://router-b:9001"
        )

    def test_absent_header_or_request_falls_back(self):
        self.assertEqual(select_generation_url(CONFIGURED, FakeRequest({})), CONFIGURED)
        self.assertEqual(select_generation_url(CONFIGURED, None), CONFIGURED)

    def test_invalid_header_falls_back_with_warning(self):
        for bad in ("http://router/generate", "file:///etc/passwd", "http://u:p@r"):
            with self.assertLogs("pd_responses_test", level="WARNING"):
                self.assertEqual(
                    select_generation_url(
                        CONFIGURED, FakeRequest({"x-smg-generation-url": bad})
                    ),
                    CONFIGURED,
                )

    def test_header_ignored_when_not_a_pd_frontend(self):
        request = FakeRequest({"x-smg-generation-url": "http://router-b:9001"})
        self.assertIsNone(select_generation_url(None, request))
        self.assertEqual(select_generation_url("", request), "")


if __name__ == "__main__":
    unittest.main()
