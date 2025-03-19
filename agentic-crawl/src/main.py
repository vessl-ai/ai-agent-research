#!/usr/bin/env python
"""
Command-line interface for Agentic Crawl.
This script provides a command-line interface to the Agentic Crawl functionality.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import our crawler
from crawl_agent import CrawlCrew

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agentic_crawl")

async def run_async():
    """Run the crawler asynchronously"""
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
    
    # Create the crawler instance
    crawler = CrawlCrew(model_name=args.model)
    
    logger.info(f"Starting crawl of {args.url} with prompt: {args.prompt} using model: {args.model}")
    logger.info(f"URL limits: max_sitemap_urls={args.max_sitemap_urls}, max_crawl_urls={args.max_crawl_urls}")
    
    # Progress callback for CLI
    def progress_callback(message):
        print(f"\n--- Progress: {message} ---\n")
    
    # Run the crawl
    try: 
        result = await crawler.crawl(
            base_url=args.url,
            user_prompt=args.prompt,
            max_sitemap_urls=args.max_sitemap_urls,
            max_crawl_urls=args.max_crawl_urls,
            progress_callback=progress_callback
        )
        print("\n\nRESULTS:\n\n")
        print(result.content)
    except Exception as e:
        logger.error(f"Error during crawl: {e}", exc_info=True)
        raise Exception(f"Error: {e}")

def run():
    """Run the command-line interface"""
    asyncio.run(run_async())

if __name__ == "__main__":
    run() 