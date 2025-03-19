import logging
import os
import re
import json
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.crews.crew_output import CrewOutput
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from tools.custom_tool import SitemapTool, WebCrawlerTool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agentic_crawl")

def step_callback(step: Dict[str, Any]) -> None:
    """Simple callback function for monitoring agent steps."""
    if isinstance(step, dict):
        if "agent" in step and "agent_name" in step["agent"]:
            agent_name = step["agent"]["agent_name"]
            logger.info(f"Agent '{agent_name}' step: {step.get('type', 'unknown')}")
        elif "type" in step:
            logger.info(f"Step type: {step['type']}")

class CrawlOutput(BaseModel):
    """Output from the crawl process."""
    content: str
    urls: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_crew_output(cls, crew_output: Any) -> "CrawlOutput":
        """Create a CrawlOutput from a CrewOutput object."""
        if isinstance(crew_output, CrewOutput):
            # Log the structure of the CrewOutput object to understand available attributes
            cls._log_crew_output_structure(crew_output)
            
            # Try various ways to extract content
            content = cls._extract_content_from_crew_output(crew_output)
        elif hasattr(crew_output, '__str__'):
            content = str(crew_output)
        elif isinstance(crew_output, dict):
            content = json.dumps(crew_output)
        else:
            content = str(crew_output)

        # Extract URLs from content
        urls = []
        url_pattern = r'https?://[^\s)"]+'
        for match in re.finditer(url_pattern, content):
            url = match.group(0)
            if url not in [u.get('url') for u in urls]:
                urls.append({
                    'url': url,
                    'source': True
                })
        
        return cls(
            content=content,
            urls=urls
        )
        
    @classmethod
    def _log_crew_output_structure(cls, crew_output: Any) -> None:
        """Log the structure of a CrewOutput object to understand its attributes."""
        try:
            # Log the crew_output type
            logger.info(f"CrewOutput type: {type(crew_output)}")
            
            # Log available attributes and their types
            for attr_name in dir(crew_output):
                # Skip private/dunder attributes
                if attr_name.startswith('_'):
                    continue
                    
                try:
                    attr_value = getattr(crew_output, attr_name)
                    logger.info(f"CrewOutput.{attr_name} type: {type(attr_value)}")
                    
                    # If it's a list or dict, log more details
                    if isinstance(attr_value, list) and attr_value:
                        logger.info(f"CrewOutput.{attr_name} is a list with {len(attr_value)} items")
                        if len(attr_value) > 0:
                            logger.info(f"First item type: {type(attr_value[0])}")
                    elif isinstance(attr_value, dict) and attr_value:
                        logger.info(f"CrewOutput.{attr_name} is a dict with keys: {list(attr_value.keys())}")
                except Exception as e:
                    logger.warning(f"Error accessing CrewOutput.{attr_name}: {e}")
        except Exception as e:
            logger.warning(f"Error inspecting CrewOutput structure: {e}")
    
    @classmethod
    def _extract_content_from_crew_output(cls, crew_output: Any) -> str:
        """Extract textual content from a CrewOutput object using multiple strategies."""
        # Strategy 1: Check for tasks_output attribute
        if hasattr(crew_output, 'tasks_output') and crew_output.tasks_output:
            # Get the last task output if it's a list
            if isinstance(crew_output.tasks_output, list) and crew_output.tasks_output:
                return str(crew_output.tasks_output[-1])
            return str(crew_output.tasks_output)
            
        # Strategy 2: Check for output attribute
        if hasattr(crew_output, 'output') and crew_output.output:
            return str(crew_output.output)
            
        # Strategy 3: Check for result attribute
        if hasattr(crew_output, 'result') and crew_output.result:
            return str(crew_output.result)
            
        # Strategy 4: If crews, agents, tasks available, try to get the last task result
        if hasattr(crew_output, 'crews') and crew_output.crews:
            try:
                # Get the first crew's final task output
                for crew in crew_output.crews:
                    if hasattr(crew, 'tasks') and crew.tasks:
                        last_task = crew.tasks[-1]
                        if hasattr(last_task, 'output'):
                            return str(last_task.output)
            except (IndexError, AttributeError) as e:
                logger.warning(f"Error getting output from tasks: {e}")
        
        # Fallback: Full string representation
        return str(crew_output)

class CrawlCrew:
    """
    Agentic web crawler that extracts information based on user prompts.
    This crew follows a structured process:
    1. Extract URLs from the website's sitemap
    2. Filter URLs based on keywords from the user prompt
    3. Crawl filtered URLs to extract content
    4. Format the extracted information into a clear report
    """
    
    def __init__(self, model_name=None):
        # Load config files from the project's config directory
        config_dir = Path(__file__).parent.parent / 'config'
        
        with open(config_dir / 'agents.yaml', 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open(config_dir / 'tasks.yaml', 'r') as f:
            self.tasks_config = yaml.safe_load(f)
        
        # Store user prompt and base URL
        self.user_prompt = ""
        self.base_url = ""
        
        # Set URL limits for crawling
        self.max_sitemap_urls = 50
        self.max_crawl_urls = 5
        
        # Set up LLM - Use specified model or the default from environment
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        logger.info(f"Using model: {self.model_name}")
        
        self.llm = LLM(
            model=f"gemini/{self.model_name}",
            provider="google",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create agents and tasks
        self._setup_agents_and_tasks()
    
    def _setup_agents_and_tasks(self):
        """Create all agents and tasks for the crew."""
        # Create agents
        self.sitemap_agent = self._create_sitemap_agent()
        self.content_crawler_agent = self._create_content_crawler_agent()
        self.formatter_agent = self._create_formatter_agent()
        
        # Create tasks
        self.sitemap_task = self._create_sitemap_extraction_task()
        self.content_task = self._create_content_extraction_task()
        self.format_task = self._create_format_results_task()
    
    def _create_sitemap_agent(self) -> Agent:
        """Create the sitemap extraction agent."""
        return Agent(
            role=self.agents_config['sitemap_agent']['role'],
            goal=self.agents_config['sitemap_agent']['goal'],
            backstory=self.agents_config['sitemap_agent']['backstory'],
            verbose=True,
            tools=[SitemapTool()],
            llm=self.llm,
        )
    
    def _create_content_crawler_agent(self) -> Agent:
        """Create the content crawler agent."""
        return Agent(
            role=self.agents_config['content_crawler_agent']['role'],
            goal=self.agents_config['content_crawler_agent']['goal'],
            backstory=self.agents_config['content_crawler_agent']['backstory'],
            verbose=True,
            tools=[WebCrawlerTool()],
            llm=self.llm,
        )
    
    def _create_formatter_agent(self) -> Agent:
        """Create the formatter agent."""
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
            process=Process.sequential,
            verbose=True,
            llm=self.llm,
        )
    
    def format_task_descriptions(self, base_url, user_prompt):
        """Format the task descriptions with user inputs."""
        # Format sitemap task description
        self.sitemap_task.description = self.tasks_config['sitemap_extraction_task']['description'].format(base_url=base_url)
        
        # Format content extraction task description
        self.content_task.description = self.tasks_config['content_extraction_task']['description'].format(user_prompt=user_prompt)
        
        # Format results formatting task description
        self.format_task.description = self.tasks_config['format_results_task']['description'].format(user_prompt=user_prompt)
    
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
        
        try:
            # If a new model is specified, update the LLM
            if model_name and model_name != self.model_name:
                logger.info(f"Updating model from {self.model_name} to {model_name}")
                self.model_name = model_name
                self.llm = LLM(
                    model=f"gemini/{self.model_name}",
                    provider="google",
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
                # Need to recreate agents and tasks with the new LLM
                self._setup_agents_and_tasks()
            
            # Update limits based on parameters
            self.max_sitemap_urls = max_sitemap_urls
            self.max_crawl_urls = max_crawl_urls
            
            # Format the task descriptions with user inputs
            self.format_task_descriptions(base_url, user_prompt)
            
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
            
            # Set up progress callback
            if progress_callback:
                progress_callback("Starting crew execution")
            
            # Create and run the crew
            crew = self._create_crew()
            
            try:
                # Run the crew (without callback as it's not supported in kickoff_async)
                result = await crew.kickoff_async(inputs=initial_input)
                
                if progress_callback:
                    progress_callback("Crew execution completed")
                
                # Create CrawlOutput from result - catch any errors in conversion
                try:
                    output = CrawlOutput.from_crew_output(result)
                    logger.info("Successfully created CrawlOutput from CrewOutput")
                    return output
                except Exception as conv_error:
                    logger.error(f"Error converting CrewOutput to CrawlOutput: {conv_error}")
                    if progress_callback:
                        progress_callback(f"Error processing results: {str(conv_error)}")
                    
                    # Fallback to simple string conversion
                    return CrawlOutput(
                        content=str(result),
                        urls=[]
                    )
                
            except Exception as crew_error:
                logger.error(f"Error during crew execution: {crew_error}", exc_info=True)
                if progress_callback:
                    progress_callback(f"Error during execution: {str(crew_error)}")
                
                # Return an error output
                return CrawlOutput(
                    content=f"Error during execution: {str(crew_error)}",
                    urls=[]
                )
                
        except Exception as e:
            logger.error(f"Error setting up crawl: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Error setting up crawl: {str(e)}")
            
            # Return an error output
            return CrawlOutput(
                content=f"Error setting up crawl: {str(e)}",
                urls=[]
            ) 