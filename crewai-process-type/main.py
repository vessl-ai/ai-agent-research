import base64
from datetime import datetime
import os
import time

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

from langchain_openai import ChatOpenAI
import openlit

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LANGFUSE_AUTH=base64.b64encode(f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

openlit.init()

search_tool = SerperDevTool()

def create_agents():
    """Create and return a tuple of agents (researcher, writer, manager) for the crew.
    
    Returns:
        tuple: Contains researcher, writer, and manager agents
    """
    researcher = Agent(
        name="Research Analyst",
        role="Senior Research Analyst",
        goal="Conduct thorough research and gather comprehensive information",
        backstory="""You are an experienced research analyst with a keen eye for detail
        and ability to find relevant information from various sources.""",
        tools=[search_tool],
        verbose=True,
    )
    
    writer = Agent(
        name="Technical Writer",
        role="Senior Technical Writer",
        goal="Transform research into clear, well-structured reports",
        backstory="""You are a skilled technical writer who excels at organizing
        information and presenting it in a clear, readable format.""",
        verbose=True,
    )
    
    return researcher, writer

def create_tasks(researcher, writer, topic):
    """Create research and writing tasks for the crew.
    
    Args:
        researcher (Agent): The research analyst agent
        writer (Agent): The technical writer agent
        topic (str): The research topic to investigate
        
    Returns:
        tuple: Contains research_task and writing_task
    """
    research_task = Task(
        description=f"Conduct comprehensive research on {topic}. Focus on key aspects, "
                   "recent developments, and important trends.",
        expected_output="A detailed research document covering key aspects, recent "
                       "developments, and trends of the topic, with citations and sources.",
        agent=researcher
    )
    
    writing_task = Task(
        description=f"Create a detailed technical report about {topic} using the research provided. Include clear sections, examples, and explanations.",
        expected_output="A well-structured technical report with clear sections, examples, and explanations based on the research provided.",
        agent=writer
    )
    
    return research_task, writing_task

def run_sequential_process(topic):
    """Run the sequential process and measure performance.
    
    Args:
        topic (str): The research topic to investigate
        
    Returns:
        tuple: Contains (result, execution_time)
    """
    start_time = time.time()
    researcher, writer = create_agents() 
    research_task, writing_task = create_tasks(researcher, writer, topic)
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time

def run_hierarchical_process(topic):
    """Run the hierarchical process and measure performance"""
    
    start_time = time.time()
    researcher, writer = create_agents()
    research_task, writing_task = create_tasks(researcher, writer, topic)
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
        verbose=True
    )
    
    result = crew.kickoff()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time

def save_results(sequential_result, sequential_time, hierarchical_result, 
                hierarchical_time, topic):
    """Save the results and timing information to a file.
    
    Args:
        sequential_result (CrewOutput): Results from sequential process
        sequential_time (float): Execution time of sequential process
        hierarchical_result (CrewOutput): Results from hierarchical process
        hierarchical_time (float): Execution time of hierarchical process
        topic (str): The research topic investigated
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_results_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Research Topic: {topic}\n\n")
        f.write("=" * 50 + "\n")
        f.write("Sequential Process Results:\n")
        f.write(f"Execution Time: {sequential_time:.2f} seconds\n")
        f.write("-" * 30 + "\n")
        f.write(str(sequential_result))  # Convert CrewOutput to string
        f.write("\n" + "=" * 50 + "\n")
        f.write("Hierarchical Process Results:\n")
        f.write(f"Execution Time: {hierarchical_time:.2f} seconds\n")
        f.write("-" * 30 + "\n")
        f.write(str(hierarchical_result))  # Convert CrewOutput to string

def main():
    """Main function to run the comparison between sequential and hierarchical processes."""
    # Research topic
    research_topic = "Healthcare AI trends in 2025 Q1"
    
    # Run sequential process
    print("Running Sequential Process...")
    sequential_result, sequential_time = run_sequential_process(research_topic)
    
    # Run hierarchical process
    print("\nRunning Hierarchical Process...")
    hierarchical_result, hierarchical_time = run_hierarchical_process(research_topic)
    
    # Save results
    save_results(
        sequential_result,
        sequential_time,
        hierarchical_result,
        hierarchical_time,
        research_topic
    )
    
    # Print comparison
    print("\nExecution Time Comparison:")
    print(f"Sequential Process: {sequential_time:.2f} seconds")
    print(f"Hierarchical Process: {hierarchical_time:.2f} seconds")
    print(f"\nDetailed results have been saved to a file.")

if __name__ == "__main__":
    main()
