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


def step_callback(step: Dict[str, Any]) -> None:
    """Simple callback function for monitoring agent steps."""
    if isinstance(step, dict):
        agent_name = step.get("agent_name", "Unknown")
        action = step.get("action", {})
        
        if isinstance(action, dict):
            if action.get("type") == "tool":
                logger.info(f"Agent {agent_name} using tool: {action.get('tool', 'unknown_tool')}")
            elif action.get("type") == "message":
                logger.info(f"Agent {agent_name} thinking...")


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
        elif isinstance(crew_output, dict):
            crew_output_dict = crew_output
        elif hasattr(crew_output, "raw_output"):
            return cls(name="Research Report", content=str(crew_output.raw_output))
        elif hasattr(crew_output, "return_values"):
            return_values = crew_output.return_values
            return cls(
                name="Research Report",
                content=str(return_values.get("output", "")),
                sources=return_values.get("sources", []),
                learnings=return_values.get("learnings", []),
                directions=return_values.get("directions", []),
            )
        else:
            return cls(name="Research Report", content=str(crew_output))

        return cls(
            name=str(crew_output_dict.get("name", "Research Report")),
            content=str(crew_output_dict.get("content", "")),
            sources=crew_output_dict.get("sources", []),
            learnings=crew_output_dict.get("learnings", []),
            directions=crew_output_dict.get("directions", []),
        )


class ProgressCallback:
    """Simplified callback handler for tracking agent progress."""

    def __init__(self, custom_callback=None):
        """Initialize the progress callback handler."""
        self.current_agent = None
        self.custom_callback = custom_callback

    def __call__(self, output: Any):
        """Process the callback output and show relevant text."""
        if isinstance(output, AgentAction):
            self.current_agent = self._get_agent_name(output)
            print(f"[{self.current_agent or 'Unknown Agent'}] Working on: {output.tool_input[:100]}...")

        elif isinstance(output, AgentFinish):
            if self.current_agent:
                return_values = getattr(output, "return_values", {})
                output_text = return_values.get("output", "") if isinstance(return_values, dict) else str(return_values)
                print(f"[{self.current_agent}] Completed: {output_text[:100]}...")

        elif isinstance(output, ToolResult):
            print(f"[{self.current_agent or 'Unknown Agent'}] Tool completed")

        # Call custom callback if provided
        if self.custom_callback:
            self.custom_callback(output)

    def _get_agent_name(self, output: AgentAction) -> Optional[str]:
        """Extract the agent name from the output."""
        agent_mapping = {
            "planner": ["planning", "plan", "outline"],
            "researcher": ["research", "search", "find", "gather"],
            "analyst": ["analysis", "analyze", "synthesize"],
            "writer": ["writing", "write", "draft", "compose"],
            "reviewer": ["review", "refine", "edit", "improve"],
        }

        tool_input = output.tool_input.lower() if hasattr(output, "tool_input") else ""

        for agent, keywords in agent_mapping.items():
            if any(keyword in tool_input for keyword in keywords):
                return agent

        return self.current_agent


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

    def _create_base_agent(self, config_key: str, **kwargs) -> Agent:
        base_config = {
            "verbose": True,
            "respect_context_window": True,
            "allow_delegation": False,
            "step_callback": step_callback,
            "llm": "openai/gpt-4o",
        }
        base_config.update(kwargs)
        return Agent(config=self.agents_config[config_key], **base_config)

    @agent
    def research_planner(self) -> Agent:
        """Agent responsible for planning the research process."""
        return self._create_base_agent("research_planner", max_iter=30)

    @agent
    def expert_researcher(self) -> Agent:
        """Agent responsible for executing searches and gathering information."""
        return self._create_base_agent(
            "expert_researcher",
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
            max_iter=40,
            max_rpm=10,
            max_retry_limit=5,
        )

    @agent
    def research_analyst(self) -> Agent:
        """Agent responsible for analyzing and synthesizing information."""
        return self._create_base_agent(
            "research_analyst",
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
            max_iter=35,
        )

    @agent
    def research_writer(self) -> Agent:
        """Agent responsible for writing the report."""
        return self._create_base_agent("research_writer", max_iter=25)

    @agent
    def research_reviewer(self) -> Agent:
        """Agent responsible for reviewing and refining the report."""
        return self._create_base_agent(
            "research_reviewer",
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
            max_iter=30,
        )

    def _create_task(self, task_key: str, **kwargs) -> Task:
        return Task(config=self.tasks_config[task_key], **kwargs)

    @task
    def deep_research_plan(self) -> Task:
        """Task for planning the research process."""
        return self._create_task("deep_research_plan")

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
        return self._create_task("deep_research_task", context=[self.deep_research_plan()])

    @task
    def deep_analysis_task(self) -> Task:
        """Task for analyzing and synthesizing information."""
        return self._create_task("deep_analysis_task")

    @task
    def deep_writing_task(self) -> Task:
        """Task for writing the report."""
        return self._create_task("deep_writing_task", context=[self.deep_analysis_task()])

    @task
    def deep_review_task(self) -> Task:
        """Task for reviewing and refining the report."""
        return self._create_task(
            "deep_review_task",
            output_pydantic=DeepResearchOutput,
            context=[self.deep_writing_task()],
        )

    @task
    def extract_learnings_directions_task(self) -> Task:
        """Task for extracting learnings and new research directions from research results."""
        return self._create_task("extract_learnings_directions")

    async def _run_crew(self, agents: List[Agent], tasks: List[Task], inputs: Dict[str, Any]) -> Any:
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            memory=True,
        )
        return await crew.kickoff_async(inputs=inputs)

    async def _summarize_research(self, results: List[str], topic: str) -> str:
        if not results:
            return ""

        summarization_result = await self._run_crew(
            agents=[self.research_analyst()],
            tasks=[
                Task(
                    description=f"""
                    Summarize the following research results on "{topic}" in a concise manner.
                    Focus on the most important findings and insights.
                    Keep your summary under 2000 words.
                    """,
                    expected_output="A concise summary of the research results",
                    agent=self.research_analyst()
                )
            ],
            inputs={"research_results": "\n\n".join(results)}
        )

        if hasattr(summarization_result, "raw_output"):
            return summarization_result.raw_output
        elif isinstance(summarization_result, dict):
            return summarization_result.get("content", "")
        return str(summarization_result)

    @crew
    async def iterative_research_crew(
        self, 
        topic: str, 
        depth: int = 2, 
        breadth: int = 2,
    ) -> DeepResearchOutput:
        current_topic = topic
        current_depth = 0
        all_learnings = []
        all_directions = []
        prior_context = ""
        all_research_results = []

        while current_depth < depth:
            research_inputs = {
                "topic": current_topic,
                "depth": depth,
                "breadth": breadth,
                "prior_context": prior_context,
                "iteration": 1,
                "total_iterations": breadth,
                "prior_learnings": all_learnings[-10:] if all_learnings else [],
                "prior_directions": all_directions[-5:] if all_directions else [],
                "research_iterations": breadth
            }
            
            plan_result = await self._run_crew(
                agents=[self.research_planner()],
                tasks=[self.deep_research_plan()],
                inputs=research_inputs
            )
            
            research_inputs["research_plan"] = (
                plan_result.raw_output if hasattr(plan_result, "raw_output")
                else plan_result.get("content", "") if isinstance(plan_result, dict)
                else ""
            )
            
            combined_research_results = []
            for i in range(breadth):
                iteration_inputs = {
                    **research_inputs,
                    "iteration": i + 1,
                    "total_iterations": breadth,
                }
                
                if combined_research_results:
                    previous_summary = await self._summarize_research(combined_research_results, current_topic)
                    iteration_inputs["previous_research"] = previous_summary
                
                research_result = await self._run_crew(
                    agents=[self.expert_researcher()],
                    tasks=[self.deep_research_task()],
                    inputs=iteration_inputs
                )
                
                result_content = (
                    research_result.raw_output if hasattr(research_result, "raw_output")
                    else research_result.get("content", "") if isinstance(research_result, dict)
                    else ""
                )
                if result_content:
                    combined_research_results.append(result_content)
            
            iteration_summary = await self._summarize_research(combined_research_results, current_topic)
            all_research_results.append(f"Depth {current_depth} - Topic: {current_topic}\n\n{iteration_summary}")
            extraction_result = await self._run_crew(
                agents=[self.research_analyst()],
                tasks=[self.extract_learnings_directions_task()],
                inputs={
                    "topic": current_topic,
                    "research_results": "\n\n".join(combined_research_results),
                    "prior_learnings": all_learnings[-10:] if all_learnings else [],
                    "prior_directions": all_directions[-5:] if all_directions else [],
                    "depth": depth,
                    "breadth": breadth,
                    "iteration": current_depth + 1,
                    "total_iterations": depth,
                    "prior_context": prior_context
                }
            )
            
            try:
                import json
                result_dict = json.loads(extraction_result.raw_output)
                new_learnings = result_dict.get("learnings", [])
                new_directions = result_dict.get("directions", [])
            except:
                new_learnings = ["Failed to extract structured learnings"]
                new_directions = ["Failed to extract structured directions"]
            
            all_learnings.extend(new_learnings)
            all_directions.extend(new_directions)
            
            current_depth += 1
            
            if current_depth < depth and new_directions:
                next_direction = new_directions[0]
                current_topic = next_direction
                prior_context = f"""
                Previous Research Topic: {current_topic}
                Key Learnings (Top 5):
                {chr(10).join([f'- {learning}' for learning in new_learnings[:5]])}
                New Research Direction:
                {next_direction}
                """
            else:
                break
        
        final_research_summary = await self._summarize_research(all_research_results, topic)
        
        final_inputs = {
            "topic": topic,
            "depth": depth,
            "breadth": breadth,
            "research_results": final_research_summary,
            "learnings": all_learnings[:20] if len(all_learnings) > 20 else all_learnings,
            "directions": all_directions[:10] if len(all_directions) > 10 else all_directions,
        }
        
        draft_result = await self._run_crew(
            agents=[self.research_writer()],
            tasks=[self.deep_writing_task()],
            inputs=final_inputs
        )
        
        draft_content = (
            draft_result.raw_output if hasattr(draft_result, "raw_output")
            else draft_result.get("content", "") if isinstance(draft_result, dict)
            else ""
        )
        
        final_result = await self._run_crew(
            agents=[self.research_reviewer()],
            tasks=[self.deep_review_task()],
            inputs={
                "topic": topic,
                "draft_report": draft_content,
                "learnings": final_inputs["learnings"],
                "directions": final_inputs["directions"]
            }
        )
        
        output = DeepResearchOutput.from_crew_output(final_result)
        output.learnings = all_learnings
        output.directions = all_directions
        
        return output

    @crew
    async def crew(
        self,
        progress_callback: Optional[Callable] = None,
        topic: str = "",
        depth: Optional[int] = 1,
        breadth: Optional[int] = 1,
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
        Returns:
            DeepResearchOutput: The research output with report and sources
        """
        logger.info(f"Starting deep research process on topic: {topic}")

        if progress_callback:
            global step_callback
            original_step_callback = step_callback

            def combined_callback(step):
                if original_step_callback:
                    original_step_callback(step)
                progress_callback(step)
            step_callback = combined_callback

        try:
            # Initialize required template variables
            depth = max(1, min(depth, 5))  # Ensure depth is between 1 and 5
            breadth = max(1, min(breadth, 5))  # Ensure breadth is between 1 and 5

            logger.info(f"Using iterative research with depth {depth} and breadth {breadth}")
            result = await self.iterative_research_crew(
                topic=topic,
                depth=depth,
                breadth=breadth,
            )
            return result
        except Exception as e:
            logger.error(f"Critical error in deep research process: {str(e)}")
            return DeepResearchOutput(
                name=f"Research Report (Error): {topic}",
                content=f"An error occurred during the research process: {str(e)}",
                sources=[]
            )