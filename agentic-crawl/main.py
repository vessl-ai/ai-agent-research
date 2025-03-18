from crew import CrawlCrew
import argparse
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Web crawler that extracts information based on user prompt')
    parser.add_argument('--url', type=str, required=True, help='Base URL to crawl')
    parser.add_argument('--prompt', type=str, required=True, help='What information to look for')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str, default=os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash'), 
                        help='Model to use (e.g., "gemini-2.0-flash", "gemini-1.5-flash", etc.)')
    parser.add_argument('--max-sitemap-urls', type=int, default=50, 
                        help='Maximum number of URLs to process from sitemap (default: 50)')
    parser.add_argument('--max-crawl-urls', type=int, default=5, 
                        help='Maximum number of URLs to crawl for content (default: 5)')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create the crew instance
    crawler = CrawlCrew(model_name=args.model)
    
    # Set URL limits based on command line arguments
    crawler.max_sitemap_urls = args.max_sitemap_urls
    crawler.max_crawl_urls = args.max_crawl_urls
    
    logger.info(f"Starting crawl of {args.url} with prompt: {args.prompt} using model: {args.model}")
    logger.info(f"URL limits: max_sitemap_urls={args.max_sitemap_urls}, max_crawl_urls={args.max_crawl_urls}")
    
    # Format the task descriptions with user inputs
    format_task_descriptions(crawler, args.url, args.prompt)
    
    # Create instructions for the manager
    manager_instructions = f"""
    Your team needs to extract information from {args.url} based on this user prompt:
    
    "{args.prompt}"
    
    Coordinate the following tasks:
    1. Extract URLs from the website's sitemap
    2. Filter URLs based on keywords from the user prompt
    3. Crawl filtered URLs to extract content
    4. Format the extracted information into a clear, well-structured report
    
    Ensure the original user prompt is maintained throughout the process to keep all specialists aligned.
    """
    
    # Set up initial input 
    initial_input = {
        "base_url": args.url, 
        "user_prompt": args.prompt,
        "manager_instructions": manager_instructions
    }
    logger.info(f"Initial input: {initial_input}")
    
    # Run the crew
    try: 
        result = crawler.crew().kickoff(inputs=initial_input)
        print(result)
    except Exception as e:
        logger.error(f"Error during crawl: {e}", exc_info=True)
        raise Exception(f"Error: {e}")

def format_task_descriptions(crawler, base_url, user_prompt):
    """Format the task descriptions with user inputs."""
    
    # Format sitemap task description
    crawler.sitemap_task.description = crawler.sitemap_task.description.format(base_url=base_url)
    
    # Format content extraction task description
    crawler.content_task.description = crawler.content_task.description.format(user_prompt=user_prompt)
    
    # Format results formatting task description
    crawler.format_task.description = crawler.format_task.description.format(user_prompt=user_prompt)

if __name__ == "__main__":
    run()