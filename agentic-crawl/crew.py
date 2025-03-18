from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv
from tools.custom_tool import SitemapTool, WebCrawlerTool, KeywordFilterTool
import yaml
import json
import logging
import os
import re

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

class CrawlCrew:
    def __init__(self, model_name=None):
        # Load config files
        with open('config/agents.yaml', 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open('config/tasks.yaml', 'r') as f:
            self.tasks_config = yaml.safe_load(f)
        
        # Store user prompt
        self.user_prompt = ""
        self.base_url = ""
        
        # Set up LLM - Use specified model or the default Gemini 2.0 Flash
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        logger.info(f"Using model: {self.model_name}")
        
        self.llm = LLM(
            model=f"gemini/{self.model_name}",
            provider="google",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create agents
        self.sitemap_agent = self._create_sitemap_agent()
        self.content_crawler_agent = self._create_content_crawler_agent()
        self.formatter_agent = self._create_formatter_agent()
        
        # Create tasks
        self.sitemap_task = self._create_sitemap_extraction_task()
        self.content_task = self._create_content_extraction_task()
        self.format_task = self._create_format_results_task()
        
        # Create crew
        self._crew = self._create_crew()
    
    def _create_sitemap_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['sitemap_agent']['role'],
            goal=self.agents_config['sitemap_agent']['goal'],
            backstory=self.agents_config['sitemap_agent']['backstory'],
            verbose=True,
            tools=[SitemapTool()],
            llm=self.llm,
        )
    
    def _create_content_crawler_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['content_crawler_agent']['role'],
            goal=self.agents_config['content_crawler_agent']['goal'],
            backstory=self.agents_config['content_crawler_agent']['backstory'],
            verbose=True,
            tools=[WebCrawlerTool()],
            llm=self.llm,
        )
    
    def _create_formatter_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['formatter_agent']['role'],
            goal=self.agents_config['formatter_agent']['goal'],
            backstory=self.agents_config['formatter_agent']['backstory'],
            verbose=True,
            llm=self.llm,
        )
    
    def _create_sitemap_extraction_task(self) -> Task:
        """Create the sitemap extraction task."""
        return Task(
            description=self.tasks_config['sitemap_extraction_task']['description'],
            expected_output=self.tasks_config['sitemap_extraction_task']['expected_output'],
            agent=self.sitemap_agent,
            input_functions=[self._sitemap_task_input],
            output_functions=[self._sitemap_task_output],
            human_input=False
        )
    
    def _create_content_extraction_task(self) -> Task:
        """Create the content extraction task."""
        return Task(
            description=self.tasks_config['content_extraction_task']['description'],
            expected_output=self.tasks_config['content_extraction_task']['expected_output'],
            agent=self.content_crawler_agent,
            context=[self.sitemap_task],
            input_functions=[self._content_task_input],
            output_functions=[self._content_task_output],
            human_input=False
        )
    
    def _create_format_results_task(self) -> Task:
        """Create the format results task."""
        return Task(
            description=self.tasks_config['format_results_task']['description'],
            expected_output=self.tasks_config['format_results_task']['expected_output'],
            agent=self.formatter_agent,
            context=[self.content_task],
            input_functions=[self._format_task_input],
            human_input=False
        )
    
    def _create_crew(self) -> Crew:
        """Create a sequential crew."""
        return Crew(
            agents=[
                self.sitemap_agent,
                self.content_crawler_agent,
                self.formatter_agent
            ],
            tasks=[
                self.sitemap_task,
                self.content_task,
                self.format_task
            ],
            process=Process.sequential,  # Changed to sequential process
            verbose=True,
            llm=self.llm,
        )
    
    def crew(self) -> Crew:
        logger.info(f"Initializing crew with base_url: {self.base_url} and user_prompt: {self.user_prompt}")
        return self._crew
    
    # Helper method to extract keywords from a prompt
    def _extract_keywords(self, prompt: str) -> list:
        """Extract keywords from a user prompt."""
        # Remove common stop words
        stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'for', 'to', 'with', 'about']
        
        # Tokenize and filter words
        words = re.findall(r'\w+', prompt.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Return unique keywords
        return list(set(keywords))
    
    # Input/output handlers for tasks
    def _sitemap_task_input(self, task_input):
        """Get the base URL from the initial input."""
        logger.info(f"Sitemap task received input type: {type(task_input)}")
        logger.info(f"Sitemap task received input content: {task_input}")
        
        # Default prompt in case it's not provided
        default_prompt = "Extract information"
        
        if isinstance(task_input, dict):
            # Store the user prompt and base_url for later tasks
            if 'user_prompt' in task_input:
                self.user_prompt = task_input['user_prompt']
                logger.info(f"Storing user_prompt: '{self.user_prompt}'")
            else:
                self.user_prompt = default_prompt
                logger.warning(f"No user_prompt found in input! Using default: '{default_prompt}'")
            
            if 'base_url' in task_input:
                self.base_url = task_input['base_url']
                logger.info(f"Sitemap task using base_url: {self.base_url}")
                # Return both values to ensure the original prompt is preserved
                return {
                    'base_url': self.base_url,
                    'user_prompt': self.user_prompt
                }
        
        logger.warning(f"Sitemap task received unexpected input format: {task_input}")
        return task_input
    
    def _sitemap_task_output(self, output):
        """Process the output from the sitemap task including keyword filtering."""
        logger.info(f"Processing sitemap task output")
        logger.info(f"Current user_prompt value: '{self.user_prompt}'")
        
        # Default prompt in case self.user_prompt is empty
        default_prompt = "Extract information"
        
        # Ensure we have a user_prompt to pass along
        if not self.user_prompt:
            self.user_prompt = default_prompt
            logger.warning(f"user_prompt was empty in sitemap output, using default: '{default_prompt}'")
        
        try:
            # Try to parse as JSON
            sitemap_output = json.loads(output) if isinstance(output, str) else output
            
            # Handle the case where output might already be processed
            if isinstance(sitemap_output, dict) and 'urls' in sitemap_output:
                all_urls = sitemap_output.get('urls', [])
            else:
                all_urls = sitemap_output.get('urls', []) if isinstance(sitemap_output, dict) else []
                
            logger.info(f"Successfully processed sitemap output with {len(all_urls)} URLs")
            
            # Extract keywords from user prompt for filtering
            keywords = self._extract_keywords(self.user_prompt)
            logger.info(f"Filtering URLs using keywords extracted from prompt: {keywords}")
            
            # Filter URLs based on keywords from the user prompt
            filtered_urls = []
            for url in all_urls:
                matches = []
                for keyword in keywords:
                    if keyword.lower() in url.lower():
                        matches.append(keyword)
                
                if matches:
                    filtered_urls.append({
                        "url": url,
                        "matched_keywords": matches,
                        "relevance_score": len(matches)
                    })
            
            # Sort by relevance score (number of keyword matches)
            filtered_urls.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"Found {len(filtered_urls)} URLs matching keywords")
            
            # Return the filtered URLs with user_prompt
            result = {
                'urls': filtered_urls,
                'user_prompt': self.user_prompt
            }
            return result
        except Exception as e:
            logger.warning(f"Error processing sitemap task output: {e}")
            # If not JSON, assume it's a text output with a list of URLs
            try:
                lines = output.strip().split('\n') if isinstance(output, str) else []
            except:
                lines = []
                
            urls = []
            for line in lines:
                if isinstance(line, str) and line.startswith('http'):
                    urls.append(line.strip())
            
            logger.info(f"Extracted {len(urls)} URLs from non-JSON output")
            
            # Extract keywords from user prompt for filtering
            keywords = self._extract_keywords(self.user_prompt)
            logger.info(f"Filtering URLs using keywords extracted from prompt: {keywords}")
            
            # Filter URLs based on keywords from the user prompt
            filtered_urls = []
            for url in urls:
                matches = []
                for keyword in keywords:
                    if keyword.lower() in url.lower():
                        matches.append(keyword)
                
                if matches:
                    filtered_urls.append({
                        "url": url,
                        "matched_keywords": matches,
                        "relevance_score": len(matches)
                    })
            
            # Sort by relevance score (number of keyword matches)
            filtered_urls.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"Found {len(filtered_urls)} URLs matching keywords")
            
            # Return the filtered URLs with user_prompt
            result = {
                'urls': filtered_urls,
                'user_prompt': self.user_prompt
            }
            logger.info(f"Returning sitemap task output with {len(filtered_urls)} filtered URLs and user_prompt: '{result['user_prompt']}'")
            return result
    
    def _content_task_input(self, task_input):
        """Process input for the content extraction task."""
        logger.info(f"Content task received input type: {type(task_input)}")
        logger.info(f"Content task received input content: {task_input}")
        logger.info(f"Current stored user_prompt: '{self.user_prompt}'")
        
        # Handle different input types
        if isinstance(task_input, dict):
            # Handle dictionary input
            if 'urls' in task_input:
                urls = task_input.get('urls', [])
            else:
                # If we don't have urls in the dict, use it as is (might be a dict with a single url)
                urls = [task_input]
                logger.warning(f"No 'urls' key found in dict input, using whole dict as URL")
                
            # Update user prompt if it's available in the input
            if 'user_prompt' in task_input:
                # Only update if we don't already have one
                if not self.user_prompt:
                    self.user_prompt = task_input['user_prompt']
                    logger.info(f"Updated user_prompt from task input: '{self.user_prompt}'")
        elif isinstance(task_input, list):
            # Handle list input - assume it's a list of URLs
            urls = task_input
            logger.info(f"Received list input, assuming it's a list of URLs: {len(urls)} URLs")
        elif isinstance(task_input, str):
            # Handle string input - assume it's a single URL
            urls = [task_input]
            logger.info(f"Received string input, assuming it's a single URL: {task_input}")
        else:
            logger.warning(f"Unexpected input type: {type(task_input)}, using empty URL list")
            urls = []
        
        # Default extract prompt - fallback in case self.user_prompt is empty
        default_prompt = "Extract information"        

        # Always ensure we return a dictionary with both required parameters
        result = {
            'urls': urls,
            'user_prompt': self.user_prompt or default_prompt
        }
        
        logger.info(f"Content task using {len(urls)} URLs and user_prompt: '{result['user_prompt']}'")
        logger.info(f"Returning content task input: {result}")
        return result
    
    def _content_task_output(self, output):
        """Process the output from the content extraction task."""
        logger.info(f"Processing content task output")
        
        # Make sure to include the user prompt in the output
        try:
            if isinstance(output, dict) and 'user_prompt' not in output:
                output['user_prompt'] = self.user_prompt
                logger.info(f"Added user_prompt to content task output: '{self.user_prompt}'")
            elif isinstance(output, str):
                try:
                    # Try to parse as JSON
                    result = json.loads(output)
                    if 'user_prompt' not in result:
                        result['user_prompt'] = self.user_prompt
                        output = json.dumps(result)
                        logger.info(f"Added user_prompt to content task output (JSON): '{self.user_prompt}'")
                except:
                    # Not JSON, leave as is
                    pass
        except Exception as e:
            logger.warning(f"Error adding user_prompt to content task output: {e}")
            
        return output
    
    def _format_task_input(self, task_input):
        """Process input for the format results task."""
        logger.info(f"Format task received input type: {type(task_input)}")
        logger.info(f"Format task received input content: {task_input}")
        logger.info(f"Current stored user_prompt: '{self.user_prompt}'")
        
        # If task_input is a dict, make sure it has user_prompt
        if isinstance(task_input, dict) and 'user_prompt' not in task_input:
            task_input['user_prompt'] = self.user_prompt
            logger.info(f"Added user_prompt to format task input: '{self.user_prompt}'")
        elif isinstance(task_input, str):
            try:
                # Try to parse as JSON
                result = json.loads(task_input)
                if 'user_prompt' not in result:
                    result['user_prompt'] = self.user_prompt
                    task_input = result
                    logger.info(f"Added user_prompt to format task input (JSON): '{self.user_prompt}'")
            except:
                # Not JSON, wrap in a dict with user_prompt
                task_input = {
                    'content': task_input,
                    'user_prompt': self.user_prompt
                }
                logger.info(f"Wrapped string input with user_prompt for format task: '{self.user_prompt}'")
            
        return task_input
        