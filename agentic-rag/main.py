from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()

def create_agents_with_pdfsearch():
    # Initialize the tools
    pdf_tool = PDFSearchTool(pdf="knowledge/costco_10k.pdf")
    search_tool = SerperDevTool()
    
    # Create a researcher agent with both tools
    researcher = Agent(
        role='Research Analyst',
        goal='Search through PDF documents and online sources to extract relevant information',
        backstory="""You are an expert research analyst with years of experience in 
        analyzing documents and extracting key information. You can effectively search
        both internal documents and online sources to gather comprehensive information.""",
        verbose=True,
        allow_delegation=True,
        tools=[pdf_tool, search_tool]
    )

    writer = Agent(
        role='Content Writer',
        goal='Synthesize information and create comprehensive responses',
        backstory="""You are a skilled writer who excels at organizing information 
        and creating clear, concise summaries.""",
        verbose=True,
        allow_delegation=True
    )

    return researcher, writer

def create_agents_with_docling():
    search_tool = SerperDevTool()
    
    # Create knowledge source using CrewDoclingSource
    content_source = CrewDoclingSource(
        file_paths=["costco_10k.pdf"],
    )
    
    # Create a researcher agent with knowledge source
    researcher = Agent(
        role='Research Analyst',
        goal='Search through PDF documents and online sources to extract relevant information',
        backstory="""You are an expert research analyst with years of experience in 
        analyzing documents and extracting key information. You can effectively search
        both internal documents and online sources to gather comprehensive information.""",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        knowledge_sources=[content_source]
    )

    writer = Agent(
        role='Content Writer',
        goal='Synthesize information and create comprehensive responses',
        backstory="""You are a skilled writer who excels at organizing information 
        and creating clear, concise summaries.""",
        verbose=True,
        allow_delegation=True
    )

    return researcher, writer

def create_tasks(researcher, writer, user_query):
    # Research task
    research_task = Task(
        description=f"""
        Conduct comprehensive research on: {user_query}
        1. Search through the PDF document for internal data
        2. Use the Serper search tool to find additional online information
        3. Compare and validate information from both sources
        4. Extract relevant financial data and market trends
        
        Make sure to:
        - Search thoroughly through both internal and external sources
        - Focus on reliable financial sources for market data
        - Validate information across multiple sources when possible
        """,
        agent=researcher,
        expected_output="A detailed analysis combining internal PDF data and external market research",
    )

    # Writing task
    writing_task = Task(
        description=f"""
        Using the research results, create a comprehensive response to: {user_query}
        
        Ensure the response:
        - Is well-structured and easy to understand
        - Includes relevant citations and sources
        - Provides clear analysis and insights
        - Includes specific data points and comparisons
        """,
        agent=writer,
        expected_output="A well-structured report that synthesizes the research findings",
    )

    return [research_task, writing_task]

def run_research(method, user_query):
    start_time = time.time()
    
    # Create agents based on method
    if method == "pdfsearch":
        researcher, writer = create_agents_with_pdfsearch()
    else:
        researcher, writer = create_agents_with_docling()
    
    # Create tasks
    tasks = create_tasks(researcher, writer, user_query)
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )
    
    # Get the result
    result = crew.kickoff()
    
    # Calculate duration
    duration = time.time() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_content = f"""
Research Method: {method}
Research Query: {user_query}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Duration: {duration:.2f} seconds

Results:
{result}
"""
    
    filename = f"research_result_{method}_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    return duration, filename, result

def main():
    user_query = "Compare the revenue growth of Costco and Walmart"
    
    # Run with PDFSearchTool
    print("\nRunning with PDFSearchTool...")
    pdf_duration, pdf_file, pdf_result = run_research("pdfsearch", user_query)
    
    # Run with CrewDoclingSource
    print("\nRunning with CrewDoclingSource...")
    docling_duration, docling_file, docling_result = run_research("docling", user_query)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"PDFSearchTool Duration: {pdf_duration:.2f} seconds")
    print(f"Results saved to: {pdf_file}")
    print(f"\nCrewDoclingSource Duration: {docling_duration:.2f} seconds")
    print(f"Results saved to: {docling_file}")

if __name__ == "__main__":
    main()
