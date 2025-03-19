from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path
import logging
import asyncio
import os
from dotenv import load_dotenv

# Import the crawler crew
from crawl_agent import CrawlCrew, CrawlOutput

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class CrawlRequest(BaseModel):
    url: str
    prompt: str
    max_sitemap_urls: Optional[int] = 50
    max_crawl_urls: Optional[int] = 5
    model: Optional[str] = None

class CrawlResponse(BaseModel):
    crawl_id: Optional[str] = None
    result: str
    urls_found: List[Dict[str, Any]] = []
    process_logs: List[str] = []

# Initialize FastAPI app
app = FastAPI(title="Agentic Crawl API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if they exist
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
else:
    static_path.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Store crawl sessions
crawl_sessions = {}

# Custom logging handler to capture logs
class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        return self.logs.copy()

    def clear(self):
        self.logs = []

@app.get("/")
async def read_root():
    """Serve the main interface"""
    if (static_path / "index.html").exists():
        return FileResponse(str(static_path / "index.html"))
    else:
        return {"message": "Welcome to Agentic Crawl API", "status": "API is running"}

@app.post("/start_crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    """Start a new crawl session"""
    try:
        # Set up log capture
        log_capture = LogCapture()
        log_capture.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(log_capture)

        # Generate a unique ID for this crawl session
        crawl_id = str(uuid.uuid4())
        
        # Define a progress callback function
        def progress_callback(step):
            message = f"Progress update: {step}"
            logger.info(message)
            log_capture.logs.append(message)
        
        # Get model from request or environment
        model_name = request.model or os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        
        # Create the crawler instance
        crawler = CrawlCrew(model_name=model_name)
        
        # Start the crawl
        result = await crawler.crawl(
            base_url=request.url,
            user_prompt=request.prompt,
            max_sitemap_urls=request.max_sitemap_urls,
            max_crawl_urls=request.max_crawl_urls,
            progress_callback=progress_callback
        )
        
        # Get captured logs
        process_logs = log_capture.get_logs()
        
        # Clean up
        logging.getLogger().removeHandler(log_capture)
        
        logger.info(f"Start crawl - Result type: {type(result)}")
        
        # Create a new session with crawl results
        new_session = {
            'url': request.url,
            'prompt': request.prompt,
            'result': result.content,
            'urls_found': result.urls,
            'process_logs': process_logs
        }
        
        # Store the session
        crawl_sessions[crawl_id] = new_session

        return CrawlResponse(
            crawl_id=crawl_id,
            result=result.content,
            urls_found=result.urls,
            process_logs=process_logs
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in start_crawl: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/{crawl_id}", response_model=CrawlResponse)
async def get_crawl(crawl_id: str):
    """Get the results of a crawl session"""
    try:
        if crawl_id not in crawl_sessions:
            raise HTTPException(status_code=404, detail="Crawl session not found")
        
        session = crawl_sessions[crawl_id]
        
        return CrawlResponse(
            crawl_id=crawl_id,
            result=session['result'],
            urls_found=session.get('urls_found', []),
            process_logs=session.get('process_logs', [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Define a function to run the FastAPI app
def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    run() 