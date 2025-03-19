from crewai.tools import BaseTool
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio
import requests
from xml.etree import ElementTree
import re
import json
from urllib.parse import urljoin
import logging

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SitemapToolInput(BaseModel):
    """Input schema for SitemapTool."""
    base_url: str = Field(..., description="Base URL of the website to crawl.")

class SitemapTool(BaseTool):
    name: str = "Sitemap Extractor"
    description: str = (
        "This tool discovers and extracts URLs from a website's sitemap."
    )
    args_schema: Type[BaseModel] = SitemapToolInput
    
    def _run(self, base_url: str) -> str:
        # Try common sitemap locations
        sitemap_locations = [
            "sitemap.xml",
            "sitemap_index.xml",
            "sitemap/sitemap.xml",
            "sitemaps/sitemap.xml",
            "wp-sitemap.xml",
            "sitemap/index.xml"
        ]
        
        discovered_urls = []
        sitemap_url = None
        
        # First try robots.txt to find sitemap
        try:
            robots_url = urljoin(base_url, "robots.txt")
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        logger.info(f"Found sitemap URL in robots.txt: {sitemap_url}")
                        discovered_urls = self._extract_urls_from_sitemap(sitemap_url)
                        if discovered_urls:
                            break
        except Exception as e:
            logger.warning(f"Error checking robots.txt: {e}")
        
        # If no sitemap found in robots.txt, try common locations
        if not discovered_urls:
            for location in sitemap_locations:
                try:
                    sitemap_url = urljoin(base_url, location)
                    logger.info(f"Trying sitemap at: {sitemap_url}")
                    discovered_urls = self._extract_urls_from_sitemap(sitemap_url)
                    if discovered_urls:
                        break
                except Exception as e:
                    logger.warning(f"Error checking {sitemap_url}: {e}")
        
        if not discovered_urls:
            # Fallback to basic crawling to find links
            logger.info("No sitemap found, using basic crawling to find links")
            discovered_urls = self._basic_crawl(base_url)
            
        result = {
            "sitemap_url": sitemap_url,
            "urls_found": len(discovered_urls),
            "urls": discovered_urls,  # Limit to first 100 URLs to prevent overwhelming
        }
            
        return json.dumps(result)
    
    def _extract_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        """Extract URLs from a sitemap XML file."""
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            
            # Check if this is a sitemap index
            if "<sitemapindex" in response.text:
                return self._handle_sitemap_index(response)
            
            # Parse the XML
            root = ElementTree.fromstring(response.content)
            
            # Extract all URLs from the sitemap
            # The namespace is usually defined in the root element
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
            return urls
        except Exception as e:
            logger.warning(f"Error fetching sitemap: {e}")
            return []
    
    def _handle_sitemap_index(self, response: requests.Response) -> List[str]:
        """Handle sitemap index files that contain links to other sitemaps."""
        try:
            root = ElementTree.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Get all sitemap URLs from the index
            sitemap_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
            all_urls = []
            for sitemap_url in sitemap_urls[:5]:  # Limit to first 5 sitemaps to avoid too much processing
                urls = self._extract_urls_from_sitemap(sitemap_url)
                all_urls.extend(urls)
                
            return all_urls
        except Exception as e:
            logger.warning(f"Error handling sitemap index: {e}")
            return []
    
    def _basic_crawl(self, url: str) -> List[str]:
        """Basic crawling to find links when no sitemap is available."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Simple regex to find links
            urls = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
            
            # Filter and clean URLs
            filtered_urls = []
            base_domain = re.match(r'https?://([^/]+)', url).group(1)
            
            for found_url in urls:
                # Make absolute URL if relative
                if found_url.startswith('/'):
                    found_url = urljoin(url, found_url)
                
                # Keep only URLs from the same domain
                if base_domain in found_url and found_url.startswith('http'):
                    filtered_urls.append(found_url)
            
            return list(set(filtered_urls))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Error in basic crawling: {e}")
            return []


class WebCrawlerToolInput(BaseModel):
    """Input schema for WebCrawlerTool."""
    urls: List[str] = Field(..., description="List of URLs to crawl.")
    user_prompt: str = Field(..., description="User prompt defining what information to extract.")

class WebCrawlerTool(BaseTool):
    name: str = "Web Content Crawler"
    description: str = (
        "This tool crawls URLs and extracts content according to user requirements."
    )
    args_schema: Type[BaseModel] = WebCrawlerToolInput
    
    def _run(self, urls: List[str], user_prompt: str) -> str:
        # Run the crawler on each URL
        results = asyncio.run(self._crawl_parallel(urls, user_prompt))
        
        return json.dumps(results)
    
    async def _crawl_parallel(self, urls: List[str], user_prompt: str, max_concurrent: int = 3):
        """Crawl multiple URLs in parallel and extract information based on user prompt."""
        logger.info(f"Crawling {len(urls)} URLs in parallel")
        
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        
        # Create crawler instance
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        
        try:
            all_results = []
            
            for i in range(0, len(urls), max_concurrent):
                batch = urls[i:i+max_concurrent]
                tasks = []
                
                for j, url in enumerate(batch):
                    session_id = f"session_{i+j}"
                    task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                    tasks.append(task)
                
                # Gather results
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Error crawling {url}: {result}")
                        all_results.append({
                            "url": url,
                            "success": False,
                            "error": str(result)
                        })
                    elif result.success:
                        # Process the content according to user prompt
                        structured_data = self._extract_structured_data(result.markdown.raw_markdown, user_prompt)
                        
                        all_results.append({
                            "url": url,
                            "success": True,
                            "title": result.title if hasattr(result, 'title') else "Unknown",
                            "structured_data": structured_data
                        })
                    else:
                        logger.warning(f"Unsuccessful crawl of {url}")
                        all_results.append({
                            "url": url,
                            "success": False,
                            "error": "Crawl failed"
                        })
            
            return all_results
        
        finally:
            logger.info("Closing crawler...")
            await crawler.close()
    
    def _extract_structured_data(self, content: str, user_prompt: str) -> Dict[str, Any]:
        """Extract structured data from content based on user prompt."""
        # Extract key information categories from the user prompt
        categories = self._analyze_prompt_for_categories(user_prompt)
        
        # Create a structured data object with extracted information
        structured_data = {}
        
        for category in categories:
            # Find paragraphs most relevant to this category
            relevant_paragraphs = self._find_relevant_paragraphs(content, category)
            structured_data[category] = relevant_paragraphs
        
        return structured_data
    
    def _analyze_prompt_for_categories(self, prompt: str) -> List[str]:
        """Analyze the user prompt to determine what categories of information to extract."""
        # Look for phrases like "find X" or "extract X" or "information about X"
        categories = []
        
        # Common patterns in prompts
        patterns = [
            r'find (?:information about|details on|data on) ([^\.,]+)',
            r'extract (?:information about|details on|data on) ([^\.,]+)',
            r'information (?:about|on|regarding) ([^\.,]+)',
            r'details (?:about|on|regarding) ([^\.,]+)',
            r'data (?:about|on|regarding) ([^\.,]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, prompt.lower())
            categories.extend(matches)
        
        # If no categories found through patterns, use keyword extraction
        if not categories:
            words = re.findall(r'\w+', prompt.lower())
            # Filter out common words
            stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'for', 'to', 'with', 'about', 
                         'find', 'extract', 'information', 'details', 'data']
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            categories = list(set(keywords))[:3]  # Use top 3 unique keywords
        
        # If still no categories, use generic ones
        if not categories:
            categories = ["overview", "main_content", "key_information"]
        
        return categories
    
    def _find_relevant_paragraphs(self, content: str, category: str) -> str:
        """Find paragraphs in content most relevant to a category."""
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Score paragraphs by relevance to category
        scored_paragraphs = []
        for p in paragraphs:
            # Basic relevance scoring
            score = 0
            # Direct mention of category
            if category.lower() in p.lower():
                score += 10
            # Keyword frequency
            score += p.lower().count(category.lower())
            # Length preference (avoid very short paragraphs)
            if len(p.split()) >= 10:
                score += 2
                
            scored_paragraphs.append((score, p))
        
        # Sort by score and take top 2
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        best_paragraphs = [p for _, p in scored_paragraphs[:2]]
        
        # Join the best paragraphs
        return '\n\n'.join(best_paragraphs)
