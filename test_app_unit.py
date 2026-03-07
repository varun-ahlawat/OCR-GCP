"""
Unit tests for app.py with mocked vLLM backend.
Run: python -m pytest test_app_unit.py -v
"""

import base64
import io
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient

import app as app_module
from app import app, get_ocr_prompt, parse_model_output, image_to_base64, process_pages_parallel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png_bytes(width=10, height=10, color="red"):
    """Create a small in-memory PNG and return its bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


MOCK_RAW_OUTPUT = (
    "---\n"
    "primary_language: en\n"
    "is_table: false\n"
    "---\n"
    "Hello, world!"
)


# Mock check_vllm_ready for all tests that hit OCR endpoints (middleware check)
_mock_vllm_ready = patch.object(
    app_module, "check_vllm_ready", new_callable=AsyncMock, return_value=True
)


@pytest.fixture()
def client():
    """TestClient with vLLM health check mocked as ready."""
    with _mock_vllm_ready:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# 1. Helper function tests (no mocking needed)
# ---------------------------------------------------------------------------

class TestGetOcrPrompt:
    def test_returns_nonempty_string(self):
        prompt = get_ocr_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_expected_keywords(self):
        prompt = get_ocr_prompt()
        assert "document" in prompt.lower()
        assert "markdown" in prompt.lower()


class TestParseModelOutput:
    def test_plain_text(self):
        result = parse_model_output("Just some plain text")
        assert result["text"] == "Just some plain text"
        assert result["metadata"] == {}

    def test_yaml_frontmatter(self):
        raw = "---\nprimary_language: en\nis_table: false\n---\nBody text here"
        result = parse_model_output(raw)
        assert result["text"] == "Body text here"
        assert result["metadata"]["primary_language"] == "en"
        assert result["metadata"]["is_table"] == "false"

    def test_markdown_fence_wrapper(self):
        raw = "```markdown\n---\nlang: fr\n---\nBonjour\n```"
        result = parse_model_output(raw)
        assert result["text"] == "Bonjour"
        assert result["metadata"]["lang"] == "fr"

    def test_empty_string(self):
        result = parse_model_output("")
        assert result["text"] == ""
        assert result["metadata"] == {}

    def test_frontmatter_only(self):
        raw = "---\nkey: val\n---\n"
        result = parse_model_output(raw)
        assert result["metadata"]["key"] == "val"
        assert result["text"] == ""


class TestImageToBase64:
    def test_small_image(self):
        img = Image.new("RGB", (10, 10), color="blue")
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"\x89PNG"

    def test_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (5, 5), color=(255, 0, 0, 128))
        b64 = image_to_base64(img)
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"\x89PNG"

    def test_large_image_resized(self):
        img = Image.new("RGB", (3000, 2000), color="green")
        b64 = image_to_base64(img, target_longest_dim=1288)
        result_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert max(result_img.size) <= 1288

    def test_small_image_not_resized(self):
        img = Image.new("RGB", (100, 50), color="white")
        b64 = image_to_base64(img, target_longest_dim=1288)
        result_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert result_img.size == (100, 50)


# ---------------------------------------------------------------------------
# 2. API endpoint tests (mocked vLLM)
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    @patch.object(app_module, "check_vllm_ready", new_callable=AsyncMock, return_value=True)
    def test_root_returns_status(self, mock_ready, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["backend"] == "vllm"
        assert "olmOCR" in data["model"]


class TestHealthEndpoint:
    @patch.object(app_module, "check_vllm_ready", new_callable=AsyncMock, return_value=True)
    def test_healthy_when_vllm_ready(self, mock_ready, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    @patch.object(app_module, "check_vllm_ready", new_callable=AsyncMock, return_value=False)
    def test_unhealthy_when_vllm_not_ready(self, mock_ready):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/health")
        assert resp.status_code == 503
        assert "not ready" in resp.json()["detail"]


class TestOcrEndpoint:
    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    def test_image_upload(self, mock_inf, client):
        png_bytes = _make_png_bytes()
        resp = client.post("/ocr", files={"file": ("test.png", png_bytes, "image/png")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["text"] == "Hello, world!"
        assert data["metadata"]["primary_language"] == "en"
        assert isinstance(data["image_base64"], str)
        assert len(data["image_base64"]) > 0
        mock_inf.assert_called_once()

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    def test_jpeg_upload(self, mock_inf, client):
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = client.post("/ocr", files={"file": ("photo.jpg", buf.read(), "image/jpeg")})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_unsupported_file_type(self, client):
        resp = client.post("/ocr", files={"file": ("data.csv", b"a,b,c", "text/csv")})
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert ".csv" in data["message"]

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_pdf_via_ocr_endpoint(self, mock_render, mock_inf, client):
        resp = client.post("/ocr", files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "image_base64" in data
        mock_render.assert_called_once()

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, side_effect=RuntimeError("boom"))
    def test_inference_error_returns_500(self, mock_inf, client):
        png_bytes = _make_png_bytes()
        resp = client.post("/ocr", files={"file": ("test.png", png_bytes, "image/png")})
        assert resp.status_code == 500
        assert resp.json()["success"] is False


class TestOcrPdfEndpoint:
    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_single_page(self, mock_render, mock_inf, client):
        resp = client.post(
            "/ocr/pdf?pages=1",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 1
        assert data["results"][0]["success"] is True
        assert data["results"][0]["text"] == "Hello, world!"
        assert data["results"][0]["image_base64"] == "AAAA"

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_multiple_pages(self, mock_render, mock_inf, client):
        resp = client.post(
            "/ocr/pdf?pages=1,2,3",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        data = resp.json()
        assert data["total_pages"] == 3
        assert len(data["results"]) == 3
        assert all(r["success"] for r in data["results"])

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_page_range(self, mock_render, mock_inf, client):
        resp = client.post(
            "/ocr/pdf?pages=2-4",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        data = resp.json()
        assert data["total_pages"] == 3
        pages = [r["page"] for r in data["results"]]
        assert pages == [2, 3, 4]

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, side_effect=RuntimeError("page error"))
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_page_level_error(self, mock_render, mock_inf, client):
        resp = client.post(
            "/ocr/pdf?pages=1",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        data = resp.json()
        assert data["results"][0]["success"] is False
        assert "page error" in data["results"][0]["error"]


class TestBatchEndpoint:
    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    def test_multiple_images(self, mock_inf, client):
        files = [
            ("files", ("a.png", _make_png_bytes(color="red"), "image/png")),
            ("files", ("b.png", _make_png_bytes(color="blue"), "image/png")),
        ]
        resp = client.post("/ocr/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert all(r["success"] for r in data["results"])
        assert data["results"][0]["filename"] == "a.png"
        assert data["results"][1]["filename"] == "b.png"

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_mixed_image_and_pdf(self, mock_render, mock_inf, client):
        files = [
            ("files", ("img.png", _make_png_bytes(), "image/png")),
            ("files", ("doc.pdf", b"%PDF-fake", "application/pdf")),
        ]
        resp = client.post("/ocr/batch", files=files)
        data = resp.json()
        assert len(data["results"]) == 2
        assert all(r["success"] for r in data["results"])

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, side_effect=[MOCK_RAW_OUTPUT, RuntimeError("fail")])
    def test_partial_failure(self, mock_inf, client):
        files = [
            ("files", ("ok.png", _make_png_bytes(), "image/png")),
            ("files", ("bad.png", _make_png_bytes(), "image/png")),
        ]
        resp = client.post("/ocr/batch", files=files)
        data = resp.json()
        assert data["results"][0]["success"] is True
        assert data["results"][1]["success"] is False


# ---------------------------------------------------------------------------
# 3. Parallel processing tests
# ---------------------------------------------------------------------------

class TestParallelPdfProcessing:
    """Test that parallel page processing returns correct, ordered results."""

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_parallel_pages_returns_ordered(self, mock_render, mock_inf, client):
        """5 pages processed in parallel should return in page order."""
        resp = client.post(
            "/ocr/pdf?pages=1-5",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        data = resp.json()
        assert data["total_pages"] == 5
        assert len(data["results"]) == 5
        pages = [r["page"] for r in data["results"]]
        assert pages == [1, 2, 3, 4, 5]
        assert all(r["success"] for r in data["results"])
        assert all(r["text"] == "Hello, world!" for r in data["results"])

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_parallel_all_pages_get_inference(self, mock_render, mock_inf, client):
        """Each page should trigger one render and one inference call."""
        resp = client.post(
            "/ocr/pdf?pages=1-3",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        assert resp.status_code == 200
        assert mock_render.call_count == 3
        assert mock_inf.call_count == 3

    @patch.object(
        app_module, "run_inference", new_callable=AsyncMock,
        side_effect=[MOCK_RAW_OUTPUT, RuntimeError("page 2 failed"), MOCK_RAW_OUTPUT],
    )
    @patch.object(app_module, "render_pdf_page_to_base64", return_value="AAAA")
    def test_parallel_partial_failure(self, mock_render, mock_inf, client):
        """One page failing shouldn't affect other pages."""
        resp = client.post(
            "/ocr/pdf?pages=1-3",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        data = resp.json()
        assert data["total_pages"] == 3
        assert data["results"][0]["success"] is True
        assert data["results"][1]["success"] is False
        assert "page 2 failed" in data["results"][1]["error"]
        assert data["results"][2]["success"] is True


class TestParallelBatchProcessing:
    """Test that batch endpoint processes files concurrently."""

    @patch.object(app_module, "run_inference", new_callable=AsyncMock, return_value=MOCK_RAW_OUTPUT)
    def test_batch_parallel_multiple_images(self, mock_inf, client):
        """Multiple images should be processed concurrently."""
        files = [
            ("files", ("a.png", _make_png_bytes(color="red"), "image/png")),
            ("files", ("b.png", _make_png_bytes(color="blue"), "image/png")),
            ("files", ("c.png", _make_png_bytes(color="green"), "image/png")),
        ]
        resp = client.post("/ocr/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert all(r["success"] for r in data["results"])
        filenames = [r["filename"] for r in data["results"]]
        assert filenames == ["a.png", "b.png", "c.png"]
        assert mock_inf.call_count == 3
