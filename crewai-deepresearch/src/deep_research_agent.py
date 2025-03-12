import logging
from typing import Any, Callable, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai.agents.crew_agent_executor import ToolResult
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.crews.crew_output import CrewOutput
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FirecrawlScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deep_research")


# Custom step callback for monitoring agent progress
def step_callback(step: Dict[str, Any]) -> None:
    """Callback function for monitoring agent steps."""
    # Safely get values with defaults
    agent_name = step.get("agent_name", "Unknown") if isinstance(step, dict) else "Unknown"
    
    # Handle different step formats
    if isinstance(step, dict):
        action = step.get("action", {})
        if isinstance(action, dict):
            action_type = action.get("type", "unknown")

            if action_type == "tool":
                tool_name = action.get("tool", "unknown_tool")
                logger.info(f"Agent {agent_name} using tool: {tool_name}")
            elif action_type == "message":
                logger.info(f"Agent {agent_name} thinking...")

            # Log token usage if available
            if "tokens" in step:
                tokens = step.get("tokens", {})
                logger.info(
                    f"Token usage - Prompt: {tokens.get('prompt', 0)}, Completion: {tokens.get('completion', 0)}"
                )
    else:
        # For non-dict step objects, just log the type
        logger.info(f"Step callback received non-dict object of type: {type(step)}")


class DeepResearchOutput(BaseModel):
    """Output from the Deep Research agent."""

    name: str
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    learnings: List[str] = Field(default_factory=list)
    directions: List[str] = Field(default_factory=list)

    @classmethod
    def from_crew_output(cls, crew_output: Any) -> "DeepResearchOutput":
        """Convert CrewAI output to DeepResearchOutput."""
        if isinstance(crew_output, CrewOutput):
            crew_output_dict = crew_output.to_dict()
            return cls(
                name=str(crew_output_dict.get("name", "Research Report")),
                content=str(crew_output_dict.get("content", "")),
                sources=crew_output_dict.get("sources", []),
                learnings=crew_output_dict.get("learnings", []),
                directions=crew_output_dict.get("directions", []),
            )
        elif isinstance(crew_output, dict):
            return cls(
                name=str(crew_output.get("name", "Research Report")),
                content=str(crew_output.get("content", "")),
                sources=crew_output.get("sources", []),
                learnings=crew_output.get("learnings", []),
                directions=crew_output.get("directions", []),
            )
        elif hasattr(crew_output, "raw_output"):
            return cls(
                name="Research Report",
                content=str(crew_output.raw_output),
                sources=[],
                learnings=[],
                directions=[],
            )
        elif hasattr(crew_output, "return_values"):
            # Handle AgentFinish objects
            return_values = crew_output.return_values
            return cls(
                name="Research Report",
                content=str(return_values.get("output", "")),
                sources=return_values.get("sources", []),
                learnings=return_values.get("learnings", []),
                directions=return_values.get("directions", []),
            )
        else:
            # Fallback for any other type
            logger.warning(f"Unexpected output type: {type(crew_output)}")
            try:
                # Try to convert to string
                content = str(crew_output)
                return cls(
                    name="Research Report",
                    content=content,
                    sources=[],
                    learnings=[],
                    directions=[],
                )
            except Exception as e:
                logger.error(f"Error converting output to string: {str(e)}")
                return cls(
                    name="Research Report (Error)",
                    content=f"An error occurred while processing the research output: {str(e)}",
                    sources=[],
                    learnings=[],
                    directions=[],
                )


class ProgressCallback:
    """Callback handler for tracking Deep Research progress."""

    def __init__(self, custom_callback=None):
        """Initialize the progress callback handler."""
        self.current_agent = None
        self.current_task = None
        self.steps = {
            "planning": {"status": "pending", "progress": 0},
            "research": {"status": "pending", "progress": 0},
            "analysis": {"status": "pending", "progress": 0},
            "writing": {"status": "pending", "progress": 0},
            "review": {"status": "pending", "progress": 0},
        }
        self.custom_callback = custom_callback

    def __call__(self, output: Any):
        """Process the callback output."""
        # Update current agent and task based on output
        if isinstance(output, AgentAction):
            agent_name = self._get_agent_name(output)
            if agent_name:
                self.current_agent = agent_name
                self._update_step_status(agent_name, "in_progress")

            print(
                f"[{self.current_agent or 'Unknown Agent'}] Action: {output.tool} - {output.tool_input[:100]}..."
            )

        elif isinstance(output, AgentFinish):
            if self.current_agent:
                self._update_step_status(self.current_agent, "completed", 100)
                # Safely access return_values
                return_values = getattr(output, "return_values", {})
                output_text = return_values.get("output", "") if isinstance(return_values, dict) else str(return_values)
                print(f"[{self.current_agent}] Completed task: {output_text[:100]}...")

        elif isinstance(output, ToolResult):
            print(
                f"[{self.current_agent or 'Unknown Agent'}] Tool Result: {output.tool} (truncated)"
            )

        else:
            print(f"[Process] Unknown output type: {type(output)}")

        # Call custom callback if provided
        if self.custom_callback:
            self.custom_callback(output, self.get_progress())

    def _get_agent_name(self, output: AgentAction) -> Optional[str]:
        """Extract the agent name from the output."""
        agent_mapping = {
            "planner": ["planning", "plan", "outline"],
            "researcher": ["research", "search", "find", "gather"],
            "analyst": ["analysis", "analyze", "synthesize"],
            "writer": ["writing", "write", "draft", "compose"],
            "reviewer": ["review", "refine", "edit", "improve"],
        }

        # Try to determine the agent from the tool input
        tool_input = output.tool_input.lower() if hasattr(output, "tool_input") else ""

        for agent, keywords in agent_mapping.items():
            if any(keyword in tool_input for keyword in keywords):
                return agent

        return (
            self.current_agent
        )  # Return current agent if we can't determine a new one

    def _update_step_status(self, agent_name: str, status: str, progress: int = 50):
        """Update the status of a step."""
        step_mapping = {
            "planner": "planning",
            "researcher": "research",
            "analyst": "analysis",
            "writer": "writing",
            "reviewer": "review",
        }

        step = step_mapping.get(agent_name)
        if step and step in self.steps:
            self.steps[step]["status"] = status
            self.steps[step]["progress"] = progress

    def get_progress(self) -> Dict[str, Any]:
        """Get the current progress of the research process."""
        return {
            "current_agent": self.current_agent,
            "current_task": self.current_task,
            "steps": self.steps,
            "overall_progress": self._calculate_overall_progress(),
        }

    def _calculate_overall_progress(self) -> int:
        """Calculate the overall progress percentage."""
        weights = {
            "planning": 10,
            "research": 30,
            "analysis": 20,
            "writing": 30,
            "review": 10,
        }

        total_progress = 0
        for step, weight in weights.items():
            step_progress = self.steps[step]["progress"] if step in self.steps else 0
            total_progress += (step_progress * weight) / 100

        return int(total_progress)


@CrewBase
class DeepResearchCrew:
    """
    Deep Research Crew for comprehensive research and report generation.
    This crew follows a structured research process:
    1. Planning: Analyze the topic and create a research plan
    2. Research: Execute searches and gather information (can be iterated multiple times)
    3. Analysis: Analyze and synthesize the information
    4. Writing: Generate a comprehensive report
    5. Review: Review and refine the report
    
    The research phase can be iterated multiple times (up to 10 iterations) to gather
    more comprehensive information. Each iteration builds upon the previous research results.
    """
    agents_config = "config/deep_research_agents.yaml"
    tasks_config = "config/deep_research_tasks.yaml"

    def __init__(self):
        super().__init__()

    @agent
    def research_planner(self) -> Agent:
        """Agent responsible for planning the research process."""
        return Agent(
            config=self.agents_config["research_planner"],
            verbose=True,
            max_iter=30,  # Increased iterations for more thorough planning
            respect_context_window=True,
            max_rpm=None,  # No limit on requests per minute for planning
            step_callback=step_callback,
            allow_delegation=False,  # Planner should focus on its own task
            llm='openai/gpt-4o',
        )

    @agent
    def expert_researcher(self) -> Agent:
        """Agent responsible for executing searches and gathering information."""
        return Agent(
            config=self.agents_config["expert_researcher"],
            verbose=True,
            tools=[
                SerperDevTool(),
                FirecrawlScrapeWebsiteTool(),
            ],
            max_iter=40,  # Higher iterations for thorough research
            max_rpm=10,  # Limit requests per minute to avoid rate limiting
            allow_delegation=False,  # Researcher should focus on its own tasks
            step_callback=step_callback,
            respect_context_window=True,  # Prevent token limit issues
            max_retry_limit=5,  # Increase retry limit for research tasks
            llm='openai/gpt-4o',
        )

    @agent
    def research_analyst(self) -> Agent:
        """Agent responsible for analyzing and synthesizing information."""
        return Agent(
            config=self.agents_config["research_analyst"],
            verbose=True,
            tools=[
                SerperDevTool(),
                FirecrawlScrapeWebsiteTool(),
            ],
            max_iter=35,  # Higher iterations for deeper analysis
            respect_context_window=True,
            allow_delegation=False,  # Analyst should focus on its own tasks
            step_callback=step_callback,
            llm='openai/gpt-4o',
        )

    @agent
    def research_writer(self) -> Agent:
        """Agent responsible for writing the report."""
        return Agent(
            config=self.agents_config["research_writer"],
            verbose=True,
            max_iter=25,  # Standard iterations for writing
            respect_context_window=True,
            allow_delegation=False,  # Writer should focus on its own tasks
            step_callback=step_callback,
            llm='openai/gpt-4o',
        )

    @agent
    def research_reviewer(self) -> Agent:
        """Agent responsible for reviewing and refining the report."""
        return Agent(
            config=self.agents_config["research_reviewer"],
            verbose=True,
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
            max_iter=30,  # Higher iterations for thorough review
            respect_context_window=True,
            allow_delegation=False,  # Reviewer should focus on its own tasks
            step_callback=step_callback,
            llm='openai/gpt-4o',
        )

    @task
    def deep_research_plan(self) -> Task:
        """Task for planning the research process."""
        return Task(
            config=self.tasks_config["deep_research_plan"],
        )

    @task
    def deep_research_task(self) -> Task:
        """
        Task for executing searches and gathering information.
        
        This task can be run multiple times in iterations, with each iteration
        building upon the results of previous iterations. The task will receive
        the following inputs:
        - research_plan: The research plan from the planning phase
        - iteration: The current iteration number
        - total_iterations: The total number of iterations
        - previous_research: Results from previous iterations (if any)
        """
        return Task(
            config=self.tasks_config["deep_research_task"],
            context=[self.deep_research_plan()],  # Use context from planning task
        )

    @task
    def deep_analysis_task(self) -> Task:
        """Task for analyzing and synthesizing information."""
        return Task(
            config=self.tasks_config["deep_analysis_task"],
            # Context will be provided directly in the inputs
        )

    @task
    def deep_writing_task(self) -> Task:
        """Task for writing the report."""
        return Task(
            config=self.tasks_config["deep_writing_task"],
            context=[self.deep_analysis_task()],  # Use context from analysis task
        )

    @task
    def deep_review_task(self) -> Task:
        """Task for reviewing and refining the report."""
        return Task(
            config=self.tasks_config["deep_review_task"],
            output_pydantic=DeepResearchOutput,
            context=[self.deep_writing_task()],  # Use context from writing task
        )

    @task
    def extract_learnings_directions_task(self) -> Task:
        """Task for extracting learnings and new research directions from research results."""
        return Task(
            config=self.tasks_config["extract_learnings_directions"],
        )

    @crew
    async def iterative_research_crew(
        self, 
        topic: str, 
        depth: int = 2, 
        breadth: int = 2,
    ) -> DeepResearchOutput:
        """
        Execute an iterative research process based on depth parameter.
        
        Args:
            topic: The research topic or question
            depth: How many levels of research to perform (1-5)
            breadth: How many search queries to perform at each level (1-5)
            
        Returns:
            DeepResearchOutput: The research output with report, sources, learnings, and directions
        """
        logger.info(f"Starting iterative research with max depth {depth} on topic: {topic}")
        
        # Initialize variables to track state across iterations
        current_topic = topic
        current_depth = 0
        all_learnings = []
        all_directions = []
        prior_context = ""
        
        # Store research results from all iterations
        all_research_results = []
        
        # Main research loop - continue until we reach max depth or have no new directions
        while current_depth < depth:
            logger.info(f"Starting research iteration at depth {current_depth}/{depth} on topic: {current_topic}")
            
            # Prepare inputs for the research
            research_inputs = {
                "topic": current_topic,
                "depth": depth,
                "breadth": breadth,
                "prior_context": prior_context,
                "prior_learnings": all_learnings,
                "prior_directions": all_directions,
                "iteration": 1,
                "total_iterations": breadth
            }
            
            # 1. Run the planner
            planner_crew = Crew(
                agents=[self.research_planner()],
                tasks=[self.deep_research_plan()],
                process=Process.sequential,
                verbose=True,
                step_callback=step_callback,
                memory=True,
            )
            
            logger.info(f"Starting planning phase for depth {current_depth}")
            plan_result = await planner_crew.kickoff_async(inputs=research_inputs)
            
            # Update inputs with plan result
            if hasattr(plan_result, "raw_output"):
                research_inputs["research_plan"] = plan_result.raw_output
            elif isinstance(plan_result, dict):
                research_inputs["research_plan"] = plan_result.get("content", "")
            
            # 2. Run the expert researcher with breadth iterations
            logger.info(f"Starting research phase with {breadth} breadth iterations")
            research_task = self.deep_research_task()
            researcher_agent = self.expert_researcher()
            
            combined_research_results = []
            for i in range(breadth):
                logger.info(f"Research breadth iteration {i+1}/{breadth}")
                
                # Update inputs with iteration information
                iteration_inputs = research_inputs.copy()
                iteration_inputs["iteration"] = i + 1
                iteration_inputs["total_iterations"] = breadth
                
                # Add previous research results to context if available
                if combined_research_results:
                    iteration_inputs["previous_research"] = "\n\n".join(combined_research_results)
                
                researcher_crew = Crew(
                    agents=[researcher_agent],
                    tasks=[research_task],
                    process=Process.sequential,
                    verbose=True,
                    step_callback=step_callback,
                    memory=True,
                )
                
                research_result = await researcher_crew.kickoff_async(inputs=iteration_inputs)
                
                # Extract and store the research result
                if hasattr(research_result, "raw_output"):
                    combined_research_results.append(research_result.raw_output)
                elif isinstance(research_result, dict):
                    combined_research_results.append(research_result.get("content", ""))
            
            # Add this iteration's research results to the overall collection
            all_research_results.extend(combined_research_results)
            
            # 3. Extract learnings and directions
            logger.info("Extracting learnings and directions from research results")
            extraction_inputs = {
                "topic": current_topic,
                "research_results": "\n\n".join(combined_research_results),
                "prior_learnings": all_learnings,
                "prior_directions": all_directions
            }
            
            extraction_crew = Crew(
                agents=[self.research_analyst()],
                tasks=[self.extract_learnings_directions_task()],
                process=Process.sequential,
                verbose=True,
                step_callback=step_callback,
                memory=True,
            )
            
            extraction_result = await extraction_crew.kickoff_async(inputs=extraction_inputs)
            
            # Parse the extraction result
            new_learnings = []
            new_directions = []
            
            if hasattr(extraction_result, "raw_output"):
                try:
                    import json
                    result_dict = json.loads(extraction_result.raw_output)
                    new_learnings = result_dict.get("learnings", [])
                    new_directions = result_dict.get("directions", [])
                except:
                    logger.warning("Failed to parse extraction result as JSON, using raw output")
                    new_learnings = ["Failed to extract structured learnings"]
                    new_directions = ["Failed to extract structured directions"]
            elif isinstance(extraction_result, dict):
                new_learnings = extraction_result.get("learnings", [])
                new_directions = extraction_result.get("directions", [])
            
            # Add new learnings and directions to our collections
            all_learnings.extend(new_learnings)
            all_directions.extend(new_directions)
            
            # Increment depth counter
            current_depth += 1
            
            # Check if we should continue to the next depth level
            if current_depth < depth and new_directions:
                # Select the next direction to explore
                next_direction = new_directions[0]
                logger.info(f"Moving to depth {current_depth} with direction: {next_direction}")
                
                # Update the topic for the next iteration
                current_topic = next_direction
                
                # Create context from current research
                prior_context = f"""
                Previous Research Topic: {current_topic}
                
                Key Learnings:
                {chr(10).join([f'- {learning}' for learning in new_learnings])}
                
                New Research Direction:
                {next_direction}
                """
            else:
                # No new directions or reached max depth, break the loop
                logger.info(f"Stopping at depth {current_depth}: {'reached max depth' if current_depth >= depth else 'no new directions'}")
                break
        
        # Generate the final report
        logger.info(f"Generating final report after {current_depth} depth iterations")
        
        # Prepare inputs for the final report
        final_inputs = {
            "topic": topic,  # Use the original topic for the report
            "depth": depth,
            "breadth": breadth,
            "research_results": "\n\n".join(all_research_results),
            "learnings": all_learnings,
            "directions": all_directions
        }
        
        final_crew = Crew(
            agents=[
                self.research_writer(),
                self.research_reviewer(),
            ],
            tasks=[
                self.deep_writing_task(),
                self.deep_review_task(),
            ],
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            memory=True,
        )
        
        final_result = await final_crew.kickoff_async(inputs=final_inputs)
        
        # Convert the result to DeepResearchOutput and add learnings/directions
        output = DeepResearchOutput.from_crew_output(final_result)
        output.learnings = all_learnings
        output.directions = all_directions
        
        return output

    @crew
    async def research_crew_async(self, **inputs) -> DeepResearchOutput:
        """
        Execute the full research process with the crew using CrewAI's standard crew pattern.
        Args:
            **inputs: Dictionary of inputs including topic, depth, outline, etc.
            research_iterations: Optional number of iterations for the researcher (default: 1, max: 10)
        Returns:
            DeepResearchOutput: The research output with report and sources
        """
        # Get the number of research iterations (default: 1, max: 10)
        research_iterations = min(inputs.get("research_iterations", 5), 15)
        logger.info(f"Running research with {research_iterations} iterations for the expert researcher")
        
        # Store intermediate results
        intermediate_results = {}
        
        # 1. Run the planner
        planner_crew = Crew(
            agents=[self.research_planner()],
            tasks=[self.deep_research_plan()],
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            memory=True,
        )
        
        logger.info("Starting planning phase...")
        plan_result = await planner_crew.kickoff_async(inputs=inputs)
        intermediate_results["plan"] = plan_result
        
        # Update inputs with plan result
        research_inputs = inputs.copy()
        if hasattr(plan_result, "raw_output"):
            research_inputs["research_plan"] = plan_result.raw_output
        elif isinstance(plan_result, dict):
            research_inputs["research_plan"] = plan_result.get("content", "")
        
        # 2. Run the expert researcher with iterations
        logger.info(f"Starting research phase with {research_iterations} iterations...")
        research_task = self.deep_research_task()
        researcher_agent = self.expert_researcher()
        
        combined_research_results = []
        for i in range(research_iterations):
            logger.info(f"Research iteration {i+1}/{research_iterations}")
            
            # Update inputs with iteration information
            iteration_inputs = research_inputs.copy()
            iteration_inputs["iteration"] = i + 1
            iteration_inputs["total_iterations"] = research_iterations
            
            # Add previous research results to context if available
            if combined_research_results:
                iteration_inputs["previous_research"] = "\n\n".join(combined_research_results)
            
            researcher_crew = Crew(
                agents=[researcher_agent],
                tasks=[research_task],
                verbose=True,
                step_callback=step_callback,
                memory=True,
            )
            
            research_result = await researcher_crew.kickoff_async(inputs=iteration_inputs)
            
            # Extract and store the research result
            if hasattr(research_result, "raw_output"):
                combined_research_results.append(research_result.raw_output)
            elif isinstance(research_result, dict):
                combined_research_results.append(research_result.get("content", ""))
        
        intermediate_results["research"] = combined_research_results
        
        # 3. Run the rest of the crew (analysis, writing, review)
        logger.info("Starting analysis, writing, and review phases...")
        
        # Update inputs with combined research results
        final_inputs = inputs.copy()
        if hasattr(plan_result, "raw_output"):
            final_inputs["research_plan"] = plan_result.raw_output
        elif isinstance(plan_result, dict):
            final_inputs["research_plan"] = plan_result.get("content", "")
            
        final_inputs["research_results"] = "\n\n".join(combined_research_results)
        
        final_crew = Crew(
            agents=[
                self.research_writer(),
                self.research_reviewer(),
            ],
            tasks=[
                self.deep_writing_task(),
                self.deep_review_task(),
            ],
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            memory=True,
        )
        
        final_result = await final_crew.kickoff_async(inputs=final_inputs)
        
        # Convert the result to DeepResearchOutput
        return DeepResearchOutput.from_crew_output(final_result)

    @crew
    async def crew(
        self,
        progress_callback: Optional[Callable] = None,
        topic: str = "",
        depth: Optional[int] = 2,
        breadth: Optional[int] = 2,
        research_iterations: Optional[int] = 1,
    ) -> DeepResearchOutput:
        """
        Execute the full research process with the crew.
        This method maintains backward compatibility with the original implementation
        while leveraging the new CrewAI structure.
        Args:
            progress_callback: Callback function for tracking progress
            topic: The research topic or question
            depth: The depth of research (1-5)
            breadth: The breadth of research at each depth (1-5)
            research_iterations: Number of iterations for the researcher (default: 1, max: 10)
            outline: Optional outline sections
            research_plan: Optional research plan (if already generated)
            feedback_provider: Optional function to get feedback for a specific iteration and stage
            plan_feedback: Optional feedback on the research plan
        Returns:
            DeepResearchOutput: The research output with report and sources
        """
        logger.info(f"Starting deep research process on topic: {topic}")

        # Set up a custom step callback that will also call the progress_callback if provided
        if progress_callback:
            global step_callback
            original_step_callback = step_callback

            def combined_callback(step):
                # Call the original step callback
                if original_step_callback:
                    original_step_callback(step)

                # Call the progress callback
                progress_callback(step)

            # Override the step callback
            step_callback = combined_callback

        try:
            # Use the iterative research crew for depth-based research
            if depth > 1:
                logger.info(f"Using iterative research with depth {depth} and breadth {breadth}")
                result = await self.iterative_research_crew(
                    topic=topic,
                    depth=depth,
                    breadth=breadth,
                )
                return result
            else:
                # For depth=1, use the standard research crew
                logger.info(f"Using standard research with {research_iterations} iterations")
                inputs = {
                    "topic": topic,
                    "depth": 1,
                    "breadth": breadth,
                    "research_iterations": min(research_iterations, 10),
                    "iteration": 1,
                    "total_iterations": 1
                }
                result = await self.research_crew_async(**inputs)
                return result
        except Exception as e:
            logger.error(f"Critical error in deep research process: {str(e)}")
            # Return a minimal output in case of critical failure
            return DeepResearchOutput(
                name=f"Research Report (Error): {topic}",
                content=f"An error occurred during the research process: {str(e)}",
                sources=[]
            )