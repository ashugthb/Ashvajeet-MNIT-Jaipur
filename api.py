import os
import sys
import json
import time
import tempfile
import requests
import fitz  # PyMuPDF
import PIL.Image
import logging
import mimetypes
import asyncio
import gc
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Connection pooling for faster downloads
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({'User-Agent': 'BillExtractor/1.0'})

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BillExtractorAPI")

app = FastAPI(title="Bill Extraction API", description="API for extracting line items from medical bills")

# Global lock to ensure sequential processing (prevents memory overflow)
request_lock = asyncio.Lock()

# CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "Unknown"
    logger.info(f"🔔 INCOMING REQUEST: {request.method} {request.url.path} from {client_ip}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"📤 RESPONSE: {response.status_code} (took {process_time:.2f}s)")
    return response

@app.get("/")
def root():
    logger.info("💓 Health check")
    return {"status": "online", "message": "Bill Extractor API is Ready"}

# Configuration
API_KEY_NAME = "X-API-Key"
API_KEY_VALUE = os.getenv("SERVICE_API_KEY", "secret-key") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY not found. Please set it in .env")

# Initialize Gemini Client
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    logger.info("✅ Gemini Client Initialized Successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini Client: {e}")
    client = None

# --- PYDANTIC MODELS (EXACT FORMAT FROM PROBLEM STATEMENT) ---
class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageItem(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]

class ExtractedData(BaseModel):
    pagewise_line_items: List[PageItem]
    total_item_count: int

class ApiResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractedData] = None
    message: Optional[str] = None

class ApiRequest(BaseModel):
    document: Optional[str] = None
    url: Optional[str] = None
    file_url: Optional[str] = None
    link: Optional[str] = None
    image: Optional[str] = None

    class Config:
        extra = "allow"

async def get_document_url(request: Request, body: ApiRequest) -> str:
    """Helper to extract URL from various possible JSON keys."""
    # 1. Check Pydantic fields
    if body.document: return body.document
    if body.url: return body.url
    if body.file_url: return body.file_url
    if body.link: return body.link
    if body.image: return body.image
    
    # 2. Check raw dict for other keys
    try:
        raw_body = await request.json()
        for key, value in raw_body.items():
            if isinstance(value, str) and (value.startswith("http") or value.startswith("www")):
                logger.info(f"🔍 Found URL in key '{key}': {value}")
                return value
    except:
        pass
        
    raise HTTPException(status_code=400, detail="Could not find a valid URL in the request body. Please provide 'document', 'url', or 'file_url'.")

import psutil

def log_memory(stage: str):
    """Log current memory usage for monitoring."""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"🧠 MEMORY [{stage}]: {mem_mb:.2f} MB")
    except:
        pass

def load_page_image(file_path: str, page_index: int) -> Optional[PIL.Image.Image]:
    """
    Load a single page as PIL Image with memory management.
    """
    doc = None
    pix = None
    page = None
    
    try:
        doc = fitz.open(file_path)
        if page_index >= len(doc):
            doc.close()
            return None
            
        page = doc.load_page(page_index)
        
        # Calculate zoom for max 1024px dimension (higher = better accuracy)
        rect = page.rect
        max_dim = max(rect.width, rect.height)
        zoom = min(1.0, 1024 / max_dim) if max_dim > 1024 else 1.0
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Create PIL image from raw bytes (no intermediate numpy)
        pil_img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # CRITICAL: Explicit cleanup in correct order
        pix = None  # Release pixmap reference
        page = None  # Release page reference
        doc.close()
        doc = None
        
        return pil_img
        
    except fitz.FileDataError:
        # Not a PDF - try as image
        if doc:
            try: doc.close()
            except: pass
        
        if page_index == 0:
            try:
                img = PIL.Image.open(file_path).convert("RGB")
                img.thumbnail((1024, 1024))  # Higher res for accuracy
                return img
            except:
                return None
        return None
        
    except Exception as e:
        logger.error(f"load_page_image error: {e}")
        if doc:
            try: doc.close()
            except: pass
        return None

# Extraction schema for structured output
RECEIPT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "page_type": {"type": "STRING", "enum": ["Bill Detail", "Final Bill", "Pharmacy"], "description": "Type of the page content"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "item_name": {"type": "STRING", "description": "Description of the item"},
                    "quantity": {"type": "NUMBER", "description": "Quantity of the item"},
                    "item_rate": {"type": "NUMBER", "description": "Unit price of the item"},
                    "total_price": {"type": "NUMBER", "description": "Total price for this line item (Net Amount)"}
                },
                "required": ["item_name", "total_price"]
            }
        }
    },
    "required": ["page_type", "items"]
}

FORENSIC_PROMPT = """
### ROLE: Principal Forensic Document Examiner
### TASK: Precision Receipt Extraction

You are analyzing a financial document page. Extract **only the distinct line items** that contribute to the final total.

### RULES:
1.  **Filter "Structural" Rows**: IGNORE subtotals, headers, and payment records.
2.  **Handle "Combined" Columns**: Split "Rate x Qty" into `item_rate` and `quantity`.
3.  **Gross vs Net**: If "Gross" and "Net" columns exist, `total_price` MUST be the **Net** amount.
4.  **Repetitive Items**: EXTRACT EVERY SINGLE ROW. Do not group them.
5.  **Handwritten**: Extract items from BOTH printed and handwritten sections.
6.  **Page Type**: Classify the page as "Bill Detail", "Final Bill", or "Pharmacy".

### OUTPUT FORMAT (JSON):
{
  "page_type": "Bill Detail | Final Bill | Pharmacy",
  "items": [
    {
      "item_name": "string",
      "quantity": number,
      "item_rate": number,
      "total_price": number
    }
  ]
}
"""

# Helper functions
async def verify_api_key(x_api_key: str = Header(None)):
    # API key verification (currently disabled)
    return x_api_key

def download_file(url: str) -> str:
    """Download file using connection pooling."""
    try:
        logger.info(f"⬇️ Downloading: {url}")
        # Use pooled session + shorter timeout (fail fast)
        response = HTTP_SESSION.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Quick extension detection
        content_type = response.headers.get('content-type', '')
        ext = '.pdf' if 'pdf' in content_type else mimetypes.guess_extension(content_type) or '.pdf'
        
        # Larger chunks = fewer syscalls = faster
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):  # 64KB chunks
                f.write(chunk)
        logger.info(f"✅ Downloaded: {path}")
        return path
    except requests.exceptions.Timeout:
        logger.error(f"❌ Download TIMEOUT (15s)")
        raise HTTPException(status_code=504, detail="Document download timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

def clean_json_text(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def process_page_chunk_sync(chunk_idx: int, page_indices: List[int], temp_file_path: str) -> dict:
    """
    Process a chunk of pages with memory management.
    """
    log_memory(f"Start Chunk {chunk_idx}")
    logger.info(f"🚀 Chunk {chunk_idx}: Pages {page_indices}")
    
    chunk_images = []
    valid_indices = []
    
    # 1. Load images with tracking for cleanup
    for page_idx in page_indices:
        try:
            pil_img = load_page_image(temp_file_path, page_idx)
            if pil_img is not None:
                chunk_images.append(pil_img)
                valid_indices.append(page_idx)
        except Exception as e:
            logger.error(f"❌ Page {page_idx} load failed: {e}")

    log_memory(f"Loaded Chunk {chunk_idx}")

    if not chunk_images:
        return {"items": [], "tokens": (0,0,0), "success": False}

    # Batch schema and prompt for multi-page extraction
    BATCH_SCHEMA = {"type": "ARRAY", "items": RECEIPT_SCHEMA}
    BATCH_PROMPT = f"""You are extracting line items from {len(chunk_images)} medical bill pages.

RULES:
1. Extract EVERY line item row - do NOT skip or merge rows
2. item_name: exact text from description column
3. total_price: use NET amount (not gross). If only one amount shown, use that.
4. quantity: default 1.0 if not shown
5. item_rate: unit price. If not shown, use total_price.
6. IGNORE: headers, subtotals, grand totals, payment rows
7. page_type: "Bill Detail" for itemized, "Final Bill" for summary, "Pharmacy" for medicines

Return JSON array with one object per page, in order."""

    # Model failover - try primary first, then backup
    response = None
    MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
    
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[BATCH_PROMPT] + chunk_images,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=BATCH_SCHEMA
                )
            )
            if response and response.text:
                logger.info(f"✅ Chunk {chunk_idx} → {model_name}")
                break
        except Exception as e:
            logger.warning(f"⚡ {model_name} failed → next: {str(e)[:40]}")
            continue
    
    # Close all PIL images to free memory
    for img in chunk_images:
        try:
            img.close()
        except:
            pass
    chunk_images.clear()
    del chunk_images
    gc.collect()
    log_memory(f"After API Chunk {chunk_idx}")

    if not response:
        return {"items": [], "tokens": (0,0,0), "success": False}

    # Parse response and extract tokens
    tokens = (0, 0, 0)
    if response.usage_metadata:
        tokens = (
            response.usage_metadata.total_token_count,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count
        )

    extracted_pages = []
    try:
        text = clean_json_text(response.text)
        data_list = json.loads(text)
        
        for i, page_data in enumerate(data_list):
            if i >= len(valid_indices): 
                break
            
            real_page_idx = valid_indices[i]
            bill_items = []
            
            for item in page_data.get("items", []):
                qty = float(item.get("quantity", 1.0) or 1.0)
                rate = float(item.get("item_rate", 0.0) or 0.0)
                amt = float(item.get("total_price", 0.0) or 0.0)
                if amt == 0 and qty > 0 and rate > 0: 
                    amt = qty * rate
                
                bill_items.append(BillItem(
                    item_name=item.get("item_name", "Unknown"),
                    item_amount=amt,
                    item_rate=rate,
                    item_quantity=qty
                ))
            
            extracted_pages.append(PageItem(
                page_no=str(real_page_idx + 1),
                page_type=page_data.get("page_type", "Bill Detail"),
                bill_items=bill_items
            ))
            
    except Exception as e:
        logger.error(f"❌ Chunk {chunk_idx} parse error: {e}")
        return {"items": [], "tokens": tokens, "success": False}
    
    # Clear response to free memory
    del response
    
    return {"items": extracted_pages, "tokens": tokens, "success": True}

# Main extraction endpoints

async def process_bill_extraction(document_url: str) -> ApiResponse:
    """Core logic for extracting bill data from a document URL."""
    
    # Acquire lock to prevent concurrent processing (memory safety)
    if request_lock.locked():
        logger.warning("⚠️ Another request is currently processing. This request is queued...")
    
    async with request_lock:
        logger.info(f"🚀 Processing document: {document_url}")
        
        if not client:
            logger.critical("LLM Client is not initialized.")
            raise HTTPException(status_code=500, detail="LLM Client not initialized")

        temp_file_path = None
        try:
            # 1. Download File
            temp_file_path = download_file(document_url)
            
            # 2. Determine Page Count (with proper cleanup)
            num_pages = 1
            doc = None
            try:
                doc = fitz.open(temp_file_path)
                num_pages = len(doc)
                doc.close()
                doc = None
                logger.info(f"📄 PDF with {num_pages} pages")
            except:
                if doc:
                    try: doc.close()
                    except: pass
                logger.info("🖼️ Single Image")
                num_pages = 1

            pagewise_items = []
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0

            # CHUNK_SIZE=4 pages per API call
            CHUNK_SIZE = 4
            chunks = [list(range(i, min(i + CHUNK_SIZE, num_pages))) for i in range(0, num_pages, CHUNK_SIZE)]
            logger.info(f"⚡ {num_pages} pages → {len(chunks)} chunks")

            # Process SEQUENTIALLY with aggressive memory cleanup
            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"⏳ Chunk {chunk_idx + 1}/{len(chunks)}: Pages {chunk}")
                
                result = await asyncio.to_thread(process_page_chunk_sync, chunk_idx, chunk, temp_file_path)
                
                if result["success"]:
                    pagewise_items.extend(result["items"])
                    t_total, t_in, t_out = result["tokens"]
                    total_tokens += t_total
                    input_tokens += t_in
                    output_tokens += t_out
                
                # AGGRESSIVE cleanup after each chunk
                del result
                gc.collect()
                log_memory(f"After Chunk {chunk_idx + 1}")
            
            # Sort by page number
            pagewise_items.sort(key=lambda x: int(x.page_no))
            total_items_count = sum(len(p.bill_items) for p in pagewise_items)

            logger.info(f"🏁 Done. {total_items_count} items extracted")
            return ApiResponse(
                is_success=True,
                token_usage=TokenUsage(
                    total_tokens=total_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                ),
                data=ExtractedData(
                    pagewise_line_items=pagewise_items,
                    total_item_count=total_items_count
                )
            )

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.critical(f"🔥 CRITICAL SERVER ERROR: {e}")
            return ApiResponse(
                is_success=False,
                message=f"Internal Server Error: {str(e)}"
            )
        finally:
            # Cleanup
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"🧹 Cleaned up temp file: {temp_file_path}")
                except:
                    pass

@app.post("/extract-bill-data", response_model=ApiResponse)
async def extract_bill_data_post(request: Request, body: ApiRequest): 
    url = await get_document_url(request, body)
    return await process_bill_extraction(url)

@app.get("/extract-bill-data", response_model=ApiResponse)
async def extract_bill_data_get(document: Optional[str] = None, url: Optional[str] = None):
    target_url = document or url
    if not target_url:
         raise HTTPException(status_code=400, detail="Missing 'document' or 'url' query parameter.")
    return await process_bill_extraction(target_url)

# Additional endpoint aliases for compatibility

@app.post("/", response_model=ApiResponse)
async def extract_bill_data_root(request: Request, body: ApiRequest):
    """Handle POST to root URL."""
    logger.info("🛡️ POST /")
    url = await get_document_url(request, body)
    return await process_bill_extraction(url)

@app.post("/extract-bill-data/", response_model=ApiResponse)
async def extract_bill_data_trailing_slash(request: Request, body: ApiRequest):
    """Handle trailing slash variant."""
    logger.info("🛡️ POST /extract-bill-data/")
    url = await get_document_url(request, body)
    return await process_bill_extraction(url)

@app.post("/extract-bill-data/extract-bill-data", response_model=ApiResponse)
async def extract_bill_data_double(request: Request, body: ApiRequest):
    """Handle double path variant."""
    logger.info("🛡️ POST /extract-bill-data/extract-bill-data")
    url = await get_document_url(request, body)
    return await process_bill_extraction(url)

@app.post("/api/v1/hackrx/run", response_model=ApiResponse)
async def extract_bill_data_api(request: Request, body: ApiRequest):
    """Handle API path variant."""
    logger.info("🛡️ POST /api/v1/hackrx/run")
    url = await get_document_url(request, body)
    return await process_bill_extraction(url)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🔥 UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"is_success": False, "message": f"Internal error: {str(exc)[:100]}"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300,
        limit_concurrency=1,
        backlog=10,
    )
