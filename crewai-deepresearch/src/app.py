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
from deep_research_agent import DeepResearchCrew, DeepResearchOutput

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class ResearchRequest(BaseModel):
    topic: str
    depth: Optional[int] = 1
    breadth: Optional[int] = 1

class FeedbackRequest(BaseModel):
    research_id: str
    feedback: str

class ResearchResponse(BaseModel):
    research_id: Optional[str] = None
    result: str
    sources: List[Dict[str, Any]] = []
    learnings: List[str] = []
    directions: List[str] = []
    process_logs: List[str] = []

# Initialize FastAPI app
app = FastAPI(title="Deep Research API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Store research sessions
research_sessions = {}

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

# Initialize the research crew
crew = DeepResearchCrew()

@app.get("/")
async def read_root():
    """Serve the main chat interface"""
    return FileResponse(str(static_path / "index.html"))

@app.post("/start_research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """Start a new research session"""
    try:
        # Set up log capture
        log_capture = LogCapture()
        log_capture.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(log_capture)

        # Generate a unique ID for this research session
        research_id = str(uuid.uuid4())
        
        # Define a progress callback function
        def progress_callback(step):
            logger.info(f"Progress update: {step}")
        
        # Start the research
        result = await crew.crew(
            progress_callback=progress_callback,
            topic=request.topic,
            depth=request.depth,
            breadth=request.breadth
        )
        
        # Get captured logs
        process_logs = log_capture.get_logs()
        
        # Clean up
        logging.getLogger().removeHandler(log_capture)
        
        logger.info(f"Start research - Result type: {type(result)}")
        
        # Create a new session dictionary with proper structure
        new_session = {
            'topic': request.topic,
            'current_result': result.content,
            'sources': result.sources,
            'learnings': result.learnings,
            'directions': result.directions,
            'process_logs': process_logs
        }
        
        # Store the session
        research_sessions[research_id] = new_session

        return ResearchResponse(
            research_id=research_id,
            result=result.content,
            sources=result.sources,
            learnings=result.learnings,
            directions=result.directions,
            process_logs=process_logs
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in start_research: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/{research_id}", response_model=ResearchResponse)
async def get_research(research_id: str):
    """Get the results of a research session"""
    try:
        if research_id not in research_sessions:
            raise HTTPException(status_code=404, detail="Research session not found")
        
        session = research_sessions[research_id]
        
        return ResearchResponse(
            research_id=research_id,
            result=session['current_result'],
            sources=session.get('sources', []),
            learnings=session.get('learnings', []),
            directions=session.get('directions', []),
            process_logs=session.get('process_logs', [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Define a function to run the FastAPI app
def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    run()