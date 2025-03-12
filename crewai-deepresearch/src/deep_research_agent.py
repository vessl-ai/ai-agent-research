from typing import Optional
from crewai import Agent, Task, Process, Crew
from crewai_tools import SerperDevTool, FirecrawlScrapeWebsiteTool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv
import logging
from queue import Queue
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DeepResearchAgent:
    # Class constants for reflection and review aspects
    REFLECTION_ASPECTS = ["Knowledge Synthesis", "Gap Analysis", "Research Direction"]
    
    REVIEW_ASPECTS = [
        {
            "name": "Content Analysis",
            "focus": """
- Review overall coverage and depth
- Check for gaps in research
- Verify key findings
- Evaluate comprehensiveness
- Assess depth of analysis""",
        },
        {
            "name": "Evidence Review",
            "focus": """
- Examine each source and citation
- Verify evidence quality
- Check fact accuracy
- Assess source credibility
- Validate key claims""",
        },
        {
            "name": "Structure Assessment",
            "focus": """
- Analyze organization and flow
- Check section transitions
- Evaluate argument progression
- Review logical coherence
- Assess information hierarchy""",
        },
        {
            "name": "Quality Enhancement",
            "focus": """
- Improve clarity and readability
- Strengthen weak sections
- Polish presentation
- Enhance formatting
- Refine language and style
- Attached the each sources to the research""",
        }
    ]

    def __init__(self):
        # Initialize Qdrant client for memory
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Initialize collection for research memory
        self._init_memory()
        
        # Initialize current result storage
        self.current_result = ""
        
        # Initialize human input handling
        self.human_input_queue = Queue()
        self.human_input_ready = Event()
        
        # Create the planner agent
        self.planner = Agent(
            role='Research Planner',
            goal='Create focused and actionable research plans',
            backstory="""You are a strategic research planner creating concise, focused research plans.

Your planning process:
1. Analyze the research topic thoroughly
2. Define clear research objectives and scope
3. Structure the research into logical sections
4. Specify required sources and methodologies

For each research plan, you will:
<Research Structure>
- Define main sections and subsections
- Specify key questions to answer
- Identify critical areas to investigate
</Research Structure>

<Research Sources>
- List specific types of sources needed
- Prioritize high-quality, authoritative sources
- Include both broad and specialized sources
</Research Sources>

<Research Methodology>
- Detail data collection approaches
- Specify analysis frameworks
- Define quality criteria
</Research Methodology>

You ensure all plans are:
- Well-organized with clear headers
- Specific and actionable
- Comprehensive yet focused
- Structured for efficient execution""",
            verbose=False,
            allow_delegation=False,
            process_callback=self._log_process
        )

        # Create the research agent with search tools
        self.researcher = Agent(
            role='Research Expert',
            goal='Execute research plans and gather comprehensive information',
            backstory="""You are an expert researcher conducting deep, thorough research.

Your research process:
1. Follow the approved research plan precisely
2. Use search tools strategically:
   - SerperDev for broad information gathering
   - FirecrawlScrapeWebsiteTool for deep analysis of specific sources
3. Evaluate and synthesize information

For each source you find:
<Source Evaluation>
- Assess credibility and authority
- Verify currency and relevance
- Cross-reference with other sources
</Source Evaluation>

<Information Gathering>
- Extract key findings and insights
- Collect supporting evidence
- Document methodologies used
</Information Gathering>

<Analysis>
- Synthesize information across sources
- Identify patterns and trends
- Draw well-supported conclusions
</Analysis>

You ensure all research:
- Is comprehensive and well-documented
- Uses high-quality, reliable sources
- Provides specific examples and evidence
- Addresses all aspects of the research plan""",
            verbose=False,
            allow_delegation=False,
            tools=[
                SerperDevTool(),
                FirecrawlScrapeWebsiteTool()
            ],
            process_callback=self._log_process
        )

        # Create the section writer agent
        self.section_writer = Agent(
            role='Section Writer',
            goal='Write comprehensive and well-structured research sections',
            backstory="""You are an expert writer specializing in creating detailed research sections.

Your writing process:
1. Analyze the research findings thoroughly
2. Structure content logically
3. Present information clearly
4. Support claims with evidence

For each section you write:
<Content Organization>
- Create clear topic sentences
- Develop logical paragraph flow
- Use appropriate transitions
- Maintain consistent focus
</Content Organization>

<Evidence Integration>
- Incorporate relevant research findings
- Cite sources appropriately
- Connect evidence to claims
- Provide context for findings
</Evidence Integration>

<Writing Quality>
- Use clear, academic language
- Maintain consistent tone
- Ensure technical accuracy
- Follow style guidelines
</Writing Quality>

You ensure all sections:
- Are comprehensive and well-organized
- Present information clearly
- Support claims with evidence
- Maintain academic standards""",
            verbose=False,
            allow_delegation=False,
            process_callback=self._log_process
        )

        # Create the section grader agent
        self.section_grader = Agent(
            role='Section Grader',
            goal='Evaluate and grade research section quality',
            backstory="""You are an expert evaluator assessing research section quality.

Your evaluation process:
1. Assess content completeness
2. Evaluate evidence quality
3. Check writing clarity
4. Grade technical accuracy

For each section you grade:
<Content Assessment>
- Evaluate topic coverage
- Check argument strength
- Assess logical flow
- Review evidence quality
</Content Assessment>

<Technical Review>
- Verify factual accuracy
- Check methodology usage
- Assess technical depth
- Evaluate source quality
</Technical Review>

<Writing Evaluation>
- Grade clarity and style
- Check organization
- Assess readability
- Review formatting
</Writing Evaluation>

You ensure all evaluations:
- Are thorough and objective
- Provide specific feedback
- Suggest improvements
- Use consistent criteria""",
            verbose=False,
            allow_delegation=False,
            process_callback=self._log_process
        )

        # Create the reviewer agent
        self.reviewer = Agent(
            role='Research Reviewer',
            goal='Review and refine research findings',
            backstory="""You are a meticulous reviewer ensuring research quality and completeness.

Your review process:
1. Evaluate research comprehensiveness
2. Assess logical flow and structure
3. Verify evidence and citations
4. Enhance clarity and presentation

For each review, you examine:
<Content Quality>
- Depth of research coverage
- Strength of evidence
- Logical consistency
- Completeness of analysis
</Content Quality>

<Structure and Organization>
- Clear and logical flow
- Effective section organization
- Proper transitions
- Balanced coverage
</Structure and Organization>

<Presentation>
- Professional tone
- Clear writing style
- Effective use of examples
- Proper formatting
</Presentation>

You ensure all final reports:
- Meet high academic standards
- Present clear, actionable insights
- Are well-supported by evidence
- Maintain professional quality""",
            verbose=False,
            allow_delegation=False,
            process_callback=self._log_process
        )

    def provide_human_input(self, input_text: str):
        """Provide human input to the agent"""
        self.human_input_queue.put(input_text)
        self.human_input_ready.set()

    def _handle_human_input(self, message: str) -> str:
        """Handle human input during agent execution"""
        # Log the request for human input
        logger.info(f"\n=== HUMAN INPUT REQUESTED ===\n{message}\n")
        
        # Add the request to the current result
        self.current_result += f"\n=====\n## HUMAN FEEDBACK: {message}\nPlease follow these guidelines:\n - If you are happy with the result, simply hit Enter without typing anything.\n - Otherwise, provide specific improvement requests.\n - You can provide multiple rounds of feedback until satisfied.\n=====\n"
        
        # Wait for human input
        self.human_input_ready.wait()
        self.human_input_ready.clear()
        
        # Get the input from the queue
        human_input = self.human_input_queue.get()
        
        return human_input

    def _init_memory(self):
        """Initialize Qdrant collection for storing research memory"""
        try:
            self.qdrant.create_collection(
                collection_name="research_memory",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except Exception as e:
            # Collection might already exist
            pass

    def _log_process(self, process: Process) -> None:
        """Log agent process details"""
        if process.type == "Agent Action":
            # Log detailed process for the process logs
            logger.info(f"\n{'='*50}")
            logger.info(f"🤖 Agent: {process.agent.role}")
            logger.info(f"🎯 Goal: {process.agent.goal}")
            logger.info(f"🛠️ Tool: {process.tool_name if process.tool_name else 'No tool used'}")
            logger.info(f"💭 Thought: {process.thought}")
            logger.info(f"🎬 Action: {process.action}")
            if process.tool_input:
                logger.info(f"📥 Tool Input: {process.tool_input}")
            if process.tool_output:
                logger.info(f"📤 Tool Output: {process.tool_output[:500]}...")  # Truncate long outputs
            logger.info(f"{'='*50}\n")

            # Add detailed research progress to the chat
            if process.agent.role == 'Research Expert':
                if process.tool_name:
                    self.current_result += f"\n📚 Researching: {process.thought}\n"
                    self.current_result += f"🔍 Using {process.tool_name} to gather information...\n"
                    if process.tool_output:
                        self.current_result += f"📋 Found: {process.tool_output[:300]}...\n"
                else:
                    self.current_result += f"\n💡 Analyzing: {process.thought}\n"
            elif process.agent.role == 'Research Planner':
                self.current_result += f"\n📋 Planning: {process.thought}\n"
                if hasattr(process, 'output') and process.output:
                    self.current_result += f"✍️ Plan details:\n{process.output}\n"
            elif process.agent.role == 'Section Writer':
                self.current_result += f"\n✍️ Writing: {process.thought}\n"
                if hasattr(process, 'output') and process.output:
                    self.current_result += f"📄 Section content:\n{process.output}\n"
            elif process.agent.role == 'Section Grader':
                self.current_result += f"\n📊 Grading: {process.thought}\n"
                if hasattr(process, 'output') and process.output:
                    self.current_result += f"🎯 Grade details:\n{process.output}\n"
            elif process.agent.role == 'Research Reviewer':
                self.current_result += f"\n✅ Reviewing: {process.thought}\n"
                if hasattr(process, 'output') and process.output:
                    self.current_result += f"📝 Review notes:\n{process.output}\n"

        elif process.type == "Task Action":
            logger.info(f"\n{'='*50}")
            logger.info(f"📋 Task: {process.task.description[:100]}...")
            logger.info(f"🤖 Assigned to: {process.agent.role}")
            logger.info(f"💭 Thought Process: {process.thought}")
            logger.info(f"{'='*50}\n")

    def create_research_plan(self, topic: str, feedback: Optional[str] = None) -> str:
        """
        Create or update a research plan
        
        Args:
            topic: The research topic
            feedback: Optional feedback for plan update
            
        Returns:
            str: The research plan with feedback request message
        """
        # Determine if this is an initial plan or an update
        is_update = feedback and feedback.strip()
        
        # Create the planning task with proper context formatting
        planning_task = Task(
            description=f"{'Update' if is_update else 'Create'} a comprehensive research plan for: {topic}",
            expected_output="A detailed research plan in markdown format with clear sections for structure, methodology, and deliverables",
            agent=self.planner,
            context=None  # Task context must be List[Task] or None
        )
        
        # Execute planning phase
        plan_crew = Crew(
            agents=[self.planner],
            tasks=[planning_task],
            verbose=False
        )
        plan_output = plan_crew.kickoff()
        plan = str(plan_output.raw)
        
        # Return plan with appropriate message, ensuring string type
        action_word = "updated" if is_update else "created"
        result = f"🤖 Research Planner has {action_word} the following plan. Please review and provide feedback:\n\n{plan}\n\nTo approve this plan, press Enter without typing anything.\nTo request changes, describe what should be modified."
        return str(result)

    def execute_research(self, topic: str, max_iterations: int = 3, queries_per_iteration: int = 2) -> str:
        """
        Execute iterative research process
        
        Args:
            topic: The research topic
            max_iterations: Maximum number of research iterations
            queries_per_iteration: Number of search queries per iteration
            
        Returns:
            str: The final research report
        """
        self.current_result = ""
        accumulated_research = ""
        
        # Create research iteration tasks
        iteration_tasks = []
        for iteration in range(max_iterations):
            iteration_tasks.append(Task(
                description=f"Execute research iteration {iteration + 1} for topic: {topic}",
                expected_output="Detailed research findings with citations and evidence",
                agent=self.researcher,
                context=None,
                tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()]
            ))

        # Execute research iterations sequentially
        for task in iteration_tasks:
            research_crew = Crew(
                agents=[self.researcher],
                tasks=[task],
                verbose=False
            )
            result = research_crew.kickoff()
            # Use the loop index for iteration number (adding 1 since index starts at 0)
            current_iteration = iteration_tasks.index(task) + 1
            self.current_result += f"\n📚 Research Iteration {current_iteration}/{max_iterations}\n"
            iteration_findings = str(result.raw)
            accumulated_research += f"\n\nFindings from Iteration {iteration}:\n{iteration_findings}"
            
            # Generate and execute search queries using for_each
            query_task = Task(
                description=f"Generate {queries_per_iteration} focused search queries for: {topic}",
                expected_output="List of specific, targeted search queries",
                agent=self.researcher,
                context=None  # Task context must be List[Task] or None
            )
            
            query_crew = Crew(
                agents=[self.researcher],
                tasks=[query_task],
                verbose=False
            )
            queries = str(query_crew.kickoff().raw).split('\n')
            
            # Create search tasks for each query
            search_tasks = []
            for i, query in enumerate(queries[:queries_per_iteration]):
                self.current_result += f"\n🔍 Search Query {i + 1}: {query}\n"
                
                search_tasks.append(Task(
                    description=f"Research and analyze: {query}",
                    expected_output="Detailed analysis with source citations and key findings",
                    agent=self.researcher,
                    context=None,
                    tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()]
                ))

            # Execute searches sequentially
            for search_task in search_tasks:
                search_crew = Crew(
                    agents=[self.researcher],
                    tasks=[search_task],
                    verbose=False
                )
                result = search_crew.kickoff()
                search_result = str(result.raw)
                # Extract query from task description
                query = search_task.description.replace("Research and analyze: ", "")
                accumulated_research += f"\n\nFindings from Query: {query}\n{search_result}"
            
            # Create reflection inputs using class constant
            reflection_inputs = [
                {
                    "aspect": aspect,
                    "research": accumulated_research
                }
                for aspect in self.REFLECTION_ASPECTS
            ]
            
            # Execute reflections sequentially
            reflection_results = []
            for reflection_input in reflection_inputs:
                reflection_task = Task(
                    description=f"Analyze research findings with focus on {reflection_input['aspect']}",
                    expected_output="Detailed reflection analysis with insights and recommendations",
                    agent=self.researcher,
                    context=None
                )
                
                reflection_crew = Crew(
                    agents=[self.researcher],
                    tasks=[reflection_task],
                    verbose=False
                )
                result = reflection_crew.kickoff()
                reflection_results.append(str(result.raw))
            
            # Combine reflection results
            combined_reflection = "\n\n".join(reflection_results)
            self.current_result += f"\n💭 Research Reflection:\n{combined_reflection}\n"
            
        # Phase 2: Section Writing and Grading
        sections = []
        section_topics = [
            "Introduction and Background",
            "Methodology and Approach",
            "Findings and Analysis",
            "Discussion and Implications",
            "Conclusions and Recommendations"
        ]
        
        for topic in section_topics:
            # Create section writing task
            writing_task = Task(
                description=f"Write the {topic} section of the research paper",
                expected_output="A well-structured, comprehensive section with proper citations and academic style",
                agent=self.section_writer,
                context=None
            )
            
            # Execute section writing
            writing_crew = Crew(
                agents=[self.section_writer],
                tasks=[writing_task],
                verbose=False
            )
            section_content = str(writing_crew.kickoff().raw)
            
            # Grade the section
            grading_task = Task(
                description=f"Evaluate the quality of the {topic} section",
                expected_output="Comprehensive evaluation with numerical grade and detailed feedback",
                agent=self.section_grader,
                context=None
            )
            
            # Execute section grading
            grading_crew = Crew(
                agents=[self.section_grader],
                tasks=[grading_task],
                verbose=False
            )
            grade_result = str(grading_crew.kickoff().raw)
            
            # Add section info to list
            sections.append({
                "topic": topic,
                "content": section_content,
                "evaluation": grade_result
            })
            
            # Update current result with progress
            self.current_result += f"\n📝 Section: {topic}\n"
            self.current_result += f"✍️ Content written\n"
            self.current_result += f"📊 Evaluation completed\n"
        
        # Combine all sections into final research result
        research_result = "\n\n".join([
            f"=== {section['topic']} ===\n"
            f"{section['content']}\n\n"
            f"--- Section Evaluation ---\n"
            f"{section['evaluation']}"
            for section in sections
        ])
        
        # Phase 3: Final Review using class constant
        # Create review inputs for parallel processing
        review_inputs = [
            {
                "aspect": aspect["name"],
                "focus": aspect["focus"],
                "content": research_result
            }
            for aspect in self.REVIEW_ASPECTS
        ]

        # Execute reviews sequentially
        final_review = []
        for review_input in review_inputs:
            review_task = Task(
                description=f"Review research content focusing on {review_input['aspect']}\nFocus areas:\n{review_input['focus']}",
                expected_output="Comprehensive review with analysis, recommendations, and implemented improvements",
                agent=self.reviewer,
                context=None
            )
            
            review_crew = Crew(
                agents=[self.reviewer],
                tasks=[review_task],
                verbose=False
            )
            result = review_crew.kickoff()
            review_content = str(result.raw)
            final_review.append(f"=== {review_input['aspect']} ===\n{review_content}")

        # Return the final synthesized review as string
        result = "\n\n".join(final_review)
        return str(result) if result is not None else ""

    def research(self, topic: str, feedback: Optional[str] = None) -> str:
        """
        Main research method that coordinates planning and execution
        
        Args:
            topic: The research topic
            feedback: Optional feedback from previous iteration
            
        Returns:
            str: Either a research plan or the final research report
        """
        if feedback is None:
            # Initial planning phase
            result = self.create_research_plan(topic)
        elif feedback.strip():
            # Update plan based on feedback
            result = self.create_research_plan(topic, feedback)
        else:
            # Execute research when plan is approved (empty feedback)
            result = self.execute_research(topic)
        # Ensure we always return a string
        return str(result) if result is not None else ""