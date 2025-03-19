#!/usr/bin/env python
"""
Run script for the Agentic Crawl API.
This script starts the FastAPI server for the web crawling application.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import the app
from app import run

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("agentic_crawl")
    
    # Log startup information
    logger.info("Starting Agentic Crawl API")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run the application
    run() 