from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from pathlib import Path
import logging
from deep_research_agent import DeepResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class ResearchRequest(BaseModel):
    topic: str
    max_iterations: Optional[int] = 3
    queries_per_iteration: Optional[int] = 2

class FeedbackRequest(BaseModel):
    research_id: str
    feedback: str

class ResearchResponse(BaseModel):
    research_id: Optional[str] = None
    result: str
    needs_feedback: bool
    process_logs: List[str]

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

# Initialize the research agent
agent = DeepResearchAgent()

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
        
        # Start the research
        result = agent.research(request.topic)
        
        # Get captured logs
        process_logs = log_capture.get_logs()
        
        # Clean up
        logging.getLogger().removeHandler(log_capture)
        
        logger.info(f"Start research - Result type: {type(result)}")
        logger.info(f"Start research - Result: {result}")
        
        # Create a new session dictionary with proper structure
        new_session = {
            'topic': str(request.topic),  # Ensure string type
            'current_result': str(result),  # Ensure string type
            'iteration': int(1),  # Ensure integer type
            'process_logs': list(process_logs) if process_logs else []  # Ensure list type
        }
        
        logger.info(f"New session type: {type(new_session)}")
        logger.info(f"New session state: {new_session}")
        
        # Store the session
        research_sessions[research_id] = new_session
        
        # Verify the stored session
        logger.info(f"Stored session type: {type(research_sessions[research_id])}")
        logger.info(f"Stored session state: {research_sessions[research_id]}")

        return ResearchResponse(
            research_id=research_id,
            result=result,
            needs_feedback=True,
            process_logs=process_logs
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"Error in start_research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/provide_feedback", response_model=ResearchResponse)
async def provide_feedback(request: FeedbackRequest):
    """Handle feedback for a research session"""
    try:
        session = await get_and_validate_session(request.research_id)
        result, process_logs = await execute_research(session, request.feedback)
        updated_session = await update_session(session, result, process_logs, request.research_id)
        
        needs_more_feedback = bool(request.feedback.strip())
        return ResearchResponse(
            result=result,
            needs_feedback=needs_more_feedback,
            process_logs=process_logs
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in provide_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_and_validate_session(research_id: str) -> dict:
    """Get and validate the research session"""
    try:
        # Get the session
        session = research_sessions.get(research_id)
        logger.info(f"Raw session type: {type(session)}")
        logger.info(f"Raw session content: {session}")
        
        if not session:
            raise HTTPException(status_code=404, detail="Invalid research ID")
        
        # If session is a string, try to convert it back to a dictionary
        if isinstance(session, str):
            logger.error(f"Session is a string, attempting to fix")
            # Create a new dictionary session
            fixed_session = {
                'topic': str(session),  # Use the string as the topic
                'current_result': str(''),
                'iteration': int(1),
                'process_logs': list([])
            }
            research_sessions[research_id] = fixed_session
            logger.info(f"Fixed session type: {type(fixed_session)}")
            logger.info(f"Fixed session content: {fixed_session}")
            return fixed_session
        
        # If session is not a dictionary, create a new one
        if not isinstance(session, dict):
            logger.error(f"Invalid session type: {type(session)}")
            new_session = {
                'topic': str(''),
                'current_result': str(''),
                'iteration': int(1),
                'process_logs': list([])
            }
            research_sessions[research_id] = new_session
            raise HTTPException(status_code=500, detail="Session was corrupted, please start a new research")
        
        # Ensure all required fields exist with correct types
        validated_session = {
            'topic': str(session.get('topic', '')),
            'current_result': str(session.get('current_result', '')),
            'iteration': int(session.get('iteration', 1)),
            'process_logs': list(session.get('process_logs', []))
        }
        
        logger.info(f"Validated session type: {type(validated_session)}")
        logger.info(f"Validated session content: {validated_session}")
        return validated_session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to validate session")

async def execute_research(session: dict, feedback: str) -> tuple[str, list]:
    """Execute research with feedback"""
    try:
        # Validate session type
        if not isinstance(session, dict):
            logger.error(f"Invalid session type in execute_research: {type(session)}")
            raise HTTPException(status_code=500, detail="Invalid session type in research execution")

        # Extract and validate topic
        topic = str(session.get('topic', ''))
        if not topic:
            logger.error("Missing topic in session")
            raise HTTPException(status_code=500, detail="Missing topic in session")

        logger.info(f"Executing research with topic: {topic}")
        logger.info(f"Session state before research: {session}")

        log_capture = LogCapture()
        log_capture.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(log_capture)

        try:
            if feedback and feedback.strip():
                logger.info("Providing human input to agent")
                agent.provide_human_input(feedback)

            result = agent.research(
                topic=topic,  # Use extracted topic
                feedback=str(feedback) if feedback else ''  # Ensure feedback is string
            )
            logger.info(f"Research result type: {type(result)}")
            logger.info(f"Research result content: {result}")

            process_logs = log_capture.get_logs()
            return str(result), list(process_logs)  # Ensure return types
        finally:
            logging.getLogger().removeHandler(log_capture)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"Error executing research: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute research")

async def update_session(session: dict, result: str, process_logs: list, research_id: str) -> dict:
    """Update the research session"""
    try:
        if not isinstance(session, dict):
            session = {
                'topic': str(''),
                'current_result': str(''),
                'iteration': int(1),
                'process_logs': list([])
            }
            research_sessions[research_id] = session
            raise HTTPException(status_code=500, detail="Session was corrupted during processing")

        logger.info(f"Before update - Session type: {type(session)}")
        logger.info(f"Before update - Session state: {session}")
        
        updated_session = {
            'topic': str(session.get('topic', '')),
            'current_result': str(result),
            'iteration': int(session.get('iteration', 0)) + 1,
            'process_logs': list(process_logs)
        }
        
        logger.info(f"Updated session type: {type(updated_session)}")
        logger.info(f"Updated session state: {updated_session}")
        
        research_sessions[research_id] = updated_session
        
        logger.info(f"After update - Session in store type: {type(research_sessions[research_id])}")
        logger.info(f"After update - Session in store: {research_sessions[research_id]}")
        
        return updated_session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update session data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)