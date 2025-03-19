import logging
import asyncio
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import the original CrawlCrew
from crew import CrawlCrew

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CrawlOutput:
    """Data class to represent crawl results"""
    content: str
    urls: List[Dict[str, Any]]
    
class CrawlCrewHandler:
    """Handler class that wraps the CrawlCrew to be used with FastAPI"""
    
    def __init__(self):
        """Initialize the handler"""
        self.crawlers = {}
    
    async def crawl(
        self,
        base_url: str,
        user_prompt: str,
        max_sitemap_urls: int = 50,
        max_crawl_urls: int = 5,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> CrawlOutput:
        """
        Execute the crawl operation asynchronously
        
        Args:
            base_url: The URL to crawl
            user_prompt: What information to look for
            max_sitemap_urls: Maximum number of URLs to process from sitemap
            max_crawl_urls: Maximum number of URLs to crawl for content
            model_name: LLM model to use
            progress_callback: Function to call with progress updates
            
        Returns:
            CrawlOutput object containing results
        """
        logger.info(f"Starting crawl of {base_url} with prompt: {user_prompt}")
        
        # Create a new crawler instance with specified model
        crawler = CrawlCrew(model_name=model_name)
        
        # Set URL limits based on parameters
        crawler.max_sitemap_urls = max_sitemap_urls
        crawler.max_crawl_urls = max_crawl_urls
        
        # Format the task descriptions with user inputs
        self._format_task_descriptions(crawler, base_url, user_prompt)
        
        # Create instructions for the manager
        manager_instructions = f"""
        Your team needs to extract information from {base_url} based on this user prompt:
        
        "{user_prompt}"
        
        Coordinate the following tasks:
        1. Extract URLs from the website's sitemap
        2. Filter URLs based on keywords from the user prompt
        3. Crawl filtered URLs to extract content
        4. Format the extracted information into a clear, well-structured report
        
        Ensure the original user prompt is maintained throughout the process to keep all specialists aligned.
        """
        
        # Set up initial input 
        initial_input = {
            "base_url": base_url, 
            "user_prompt": user_prompt,
            "manager_instructions": manager_instructions
        }
        
        # Run the crew in a thread to avoid blocking
        result = await self._run_crew_in_thread(crawler, initial_input, progress_callback)
        
        # Log the result type
        logger.info(f"Crew result type after _run_crew_in_thread: {type(result)}")
        
        # Make sure result is a string
        if not isinstance(result, str):
            if hasattr(result, '__str__'):
                result = str(result)
            else:
                result = f"Unknown result format: {type(result)}"
        
        # Extract URLs from the crew result
        urls_found = self._extract_urls_from_result(result)
        
        # Return the formatted output
        return CrawlOutput(
            content=result,
            urls=urls_found
        )
    
    def _format_task_descriptions(self, crawler, base_url, user_prompt):
        """Format the task descriptions with user inputs."""
        # Format sitemap task description
        crawler.sitemap_task.description = crawler.sitemap_task.description.format(base_url=base_url)
        
        # Format content extraction task description
        crawler.content_task.description = crawler.content_task.description.format(user_prompt=user_prompt)
        
        # Format results formatting task description
        crawler.format_task.description = crawler.format_task.description.format(user_prompt=user_prompt)
    
    async def _run_crew_in_thread(self, crawler, initial_input, progress_callback=None):
        """Run the crew in a separate thread to avoid blocking the event loop"""
        
        def run_crew():
            try:
                if progress_callback:
                    progress_callback("Starting crew execution")
                
                result = crawler.crew().kickoff(inputs=initial_input)
                
                if progress_callback:
                    progress_callback("Crew execution completed")
                
                logger.info(f"Crew result type: {type(result)}")
                
                # Convert CrewOutput to string if needed
                if hasattr(result, '__str__'):
                    return str(result)
                return result
            except Exception as e:
                logger.error(f"Error during crawl: {e}", exc_info=True)
                raise
        
        # Run the crew in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_crew)
    
    def _extract_urls_from_result(self, result):
        """Extract URLs information from the crew result"""
        # Check if result is CrewOutput object and get its string representation
        if hasattr(result, '__str__'):
            result_text = str(result)
        else:
            result_text = result
            
        # This is a simple implementation that could be expanded
        # Currently just extracts URLs mentioned in the result
        import re
        
        urls = []
        # Simple regex to find URLs in the text
        url_pattern = r'https?://[^\s)"]+'
        
        for match in re.finditer(url_pattern, result_text):
            url = match.group(0)
            if url not in [u.get('url') for u in urls]:
                urls.append({
                    'url': url,
                    'source': True  # Marking as a source URL
                })
        
        return urls 