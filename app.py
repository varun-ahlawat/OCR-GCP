"""
OlmOCR API Server
Provides OCR capabilities using the allenai/olmOCR-2-7B-1025-FP8 model
via a persistent vLLM server for fast inference.
"""

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import httpx
import io
import re
import logging
import traceback
import base64
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# vLLM server config
VLLM_BASE_URL = "http://localhost:8001/v1"

# Initialize FastAPI app
app = FastAPI(
    title="OlmOCR API",
    description="OCR API powered by allenai/olmOCR-2-7B-1025-FP8 via vLLM",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CONCURRENT_PAGES = 4  # Match vLLM --max-num-seqs

# Thread pool for CPU-bound PDF rendering
_render_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PAGES)


@app.middleware("http")
async def check_vllm_middleware(request: Request, call_next):
    """Return 503 on OCR endpoints if vLLM is down."""
    if request.url.path not in ["/health", "/", "/docs", "/openapi.json"]:
        if not await check_vllm_ready():
            return JSONResponse(
                status_code=503,
                content={"detail": "vLLM server unavailable"},
            )
    return await call_next(request)


def get_ocr_prompt():
    """Get the official olmOCR v2 prompt (build_no_anchoring_v4_yaml_prompt).
    Uses the olmocr package if available, otherwise uses the exact prompt text
    from olmocr/prompts/prompts.py so no dependency on olmocr is needed."""
    try:
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt  # type: ignore[import-not-found]
        return build_no_anchoring_v4_yaml_prompt()
    except ImportError:
        # Exact prompt from olmocr v0.4.25 - olmocr/prompts/prompts.py
        return (
            "Attached is one page of a document that you must process. "
            "Just return the plain text representation of this document as if you were reading it naturally. "
            "Convert equations to LateX and tables to HTML.\n"
            "If there are any figures or charts, label them with the following markdown syntax "
            "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
            "Return your output as markdown, with a front matter section on top specifying values for "
            "the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
        )


def render_pdf_page_to_base64(pdf_bytes, page_num=1, target_longest_dim=1288):
    """Render a PDF page to base64 PNG. Uses olmocr if available, otherwise falls back to pdf2image."""
    try:
        from olmocr.data.renderpdf import render_pdf_to_base64png  # type: ignore[import-not-found]
        # Write PDF bytes to a temp file since render_pdf_to_base64png expects a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            return render_pdf_to_base64png(tmp_path, page_num, target_longest_image_dim=target_longest_dim)
        finally:
            os.unlink(tmp_path)
    except ImportError:
        logger.info("olmocr.data.renderpdf not available, using pdf2image fallback")
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=200)
        if not images:
            raise ValueError(f"Could not render PDF page {page_num}")
        img = images[0]
        # Resize to target dimension
        max_dim = max(img.size)
        if max_dim > target_longest_dim:
            scale = target_longest_dim / max_dim
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def image_to_base64(image: Image.Image, target_longest_dim=1288) -> str:
    """Convert a PIL Image to base64 PNG, resizing if needed."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    max_dim = max(image.size)
    if max_dim > target_longest_dim:
        scale = target_longest_dim / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def parse_model_output(raw_text: str) -> dict:
    """Parse the model output, stripping YAML frontmatter and markdown fences."""
    text = raw_text.strip()

    # Remove outer markdown code fence if present (```markdown ... ```)
    fence_match = re.match(r"^```(?:markdown)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    metadata = {}
    body = text

    # Extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        body = fm_match.group(2).strip()
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()

    return {"text": body, "metadata": metadata}


async def check_vllm_ready() -> bool:
    """Check if the vLLM server is ready by querying /v1/models."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_BASE_URL}/models", timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


async def run_inference(image_base64: str) -> str:
    """Send a chat completion request to the vLLM server and return raw text output."""
    prompt = get_ocr_prompt()

    payload = {
        "model": "olmocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.8,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def render_all_pages(pdf_bytes: bytes, page_nums: list[int]) -> dict[int, str]:
    """Render multiple PDF pages to base64 in parallel using threads (CPU-bound)."""
    loop = asyncio.get_event_loop()
    tasks = {
        page_num: loop.run_in_executor(
            _render_pool, render_pdf_page_to_base64, pdf_bytes, page_num
        )
        for page_num in page_nums
    }
    results = {}
    for page_num, task in tasks.items():
        results[page_num] = await task
    return results


async def process_pages_parallel(pdf_bytes: bytes, page_nums: list[int]) -> list[dict]:
    """Process multiple PDF pages with parallel rendering and concurrent inference."""
    sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

    async def _process_one(page_num: int, image_base64: str) -> dict:
        async with sem:
            try:
                raw_output = await run_inference(image_base64)
                parsed = parse_model_output(raw_output)
                return {
                    "page": page_num,
                    "text": parsed["text"],
                    "metadata": parsed["metadata"],
                    "image_base64": image_base64,
                    "success": True,
                }
            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                return {
                    "page": page_num,
                    "text": "",
                    "image_base64": image_base64,
                    "success": False,
                    "error": str(e),
                }

    # Phase 1: Render all pages in parallel (CPU-bound, threaded)
    logger.info(f"Rendering {len(page_nums)} pages in parallel...")
    rendered = await render_all_pages(pdf_bytes, page_nums)

    # Phase 2: Run inference concurrently (GPU, semaphore-limited)
    logger.info(f"Running inference on {len(page_nums)} pages (max {MAX_CONCURRENT_PAGES} concurrent)...")
    inference_tasks = [
        _process_one(page_num, rendered[page_num]) for page_num in page_nums
    ]
    results = await asyncio.gather(*inference_tasks)

    # Sort by page number to preserve order
    return sorted(results, key=lambda r: r["page"])


@app.get("/")
async def root():
    """Root endpoint - service info"""
    ready = await check_vllm_ready()
    return {
        "status": "running",
        "model": "allenai/olmOCR-2-7B-1025-FP8",
        "backend": "vllm",
        "model_loaded": ready,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ready = await check_vllm_ready()
    if not ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "vLLM server not ready"},
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "backend": "vllm",
    }


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on an uploaded image or PDF.

    Accepts: JPEG, PNG, BMP, TIFF, WebP images and PDF files.
    For PDFs, processes the first page. Use /ocr/pdf for multi-page.

    Returns JSON with extracted text.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes")

        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"text": "", "success": False, "message": "File too large (max 50MB)"},
            )

        ext = os.path.splitext(file.filename or "")[1].lower()

        # Determine if it's a PDF or image
        if ext in PDF_EXTENSIONS or (file.content_type and "pdf" in file.content_type):
            logger.info("Processing as PDF (page 1)...")
            image_base64 = render_pdf_page_to_base64(contents, page_num=1)
        elif ext in IMAGE_EXTENSIONS or not ext:
            logger.info("Processing as image...")
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            image_base64 = image_to_base64(image)
        else:
            return JSONResponse(
                status_code=400,
                content={"text": "", "success": False, "message": f"Unsupported file type: {ext}"},
            )

        logger.info("Running inference...")
        raw_output = await run_inference(image_base64)
        parsed = parse_model_output(raw_output)

        logger.info(f"OCR complete, extracted {len(parsed['text'])} chars")

        return {
            "text": parsed["text"],
            "metadata": parsed["metadata"],
            "image_base64": image_base64,
            "success": True,
            "message": "OCR completed successfully",
        }

    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"text": "", "success": False, "message": f"OCR failed: {str(e)}"},
        )


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...), pages: str = "1"):
    """
    Perform OCR on a PDF file, processing specified pages.

    Args:
        file: PDF file
        pages: Comma-separated page numbers or range (e.g. "1,2,3" or "1-5"). Default: "1"

    Returns JSON with per-page OCR results.
    """
    try:
        logger.info(f"Received PDF: {file.filename}, pages={pages}")
        contents = await file.read()

        # Parse page numbers
        page_nums = []
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                page_nums.extend(range(int(start), int(end) + 1))
            else:
                page_nums.append(int(part))

        results = await process_pages_parallel(contents, page_nums)

        return {"results": results, "total_pages": len(page_nums)}

    except Exception as e:
        logger.error(f"PDF OCR error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"text": "", "success": False, "message": f"PDF OCR failed: {str(e)}"},
        )


@app.post("/ocr/batch")
async def batch_ocr(files: list[UploadFile] = File(...)):
    """
    Perform OCR on multiple files (images or PDFs, first page only).

    Returns list of OCR results.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

    async def _process_file(f_contents: bytes, filename: str, ext: str, content_type: str | None):
        async with sem:
            try:
                if ext in PDF_EXTENSIONS or (content_type and "pdf" in content_type):
                    loop = asyncio.get_event_loop()
                    image_base64 = await loop.run_in_executor(
                        _render_pool, render_pdf_page_to_base64, f_contents, 1
                    )
                else:
                    image = Image.open(io.BytesIO(f_contents))
                    image_base64 = image_to_base64(image)

                raw_output = await run_inference(image_base64)
                parsed = parse_model_output(raw_output)

                return {
                    "filename": filename,
                    "text": parsed["text"],
                    "metadata": parsed["metadata"],
                    "image_base64": image_base64,
                    "success": True,
                }
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                return {
                    "filename": filename,
                    "text": "",
                    "success": False,
                    "error": str(e),
                }

    # Read all files first, then process concurrently
    file_data = []
    for file in files:
        contents = await file.read()
        ext = os.path.splitext(file.filename or "")[1].lower()
        file_data.append((contents, file.filename, ext, file.content_type))

    tasks = [_process_file(c, fn, ext, ct) for c, fn, ext, ct in file_data]
    results = await asyncio.gather(*tasks)

    return {"results": list(results)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
