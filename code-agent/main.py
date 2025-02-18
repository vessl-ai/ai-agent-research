import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import CodeInterpreterTool, LlamaIndexTool
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from typing import Dict, Any
from datetime import datetime

# Load environment variables
load_dotenv()

class FinancialAnalysisVisualizationCrew:
    def __init__(self):
        # Initialize tools
        yahoo_tools = YahooFinanceToolSpec().to_tool_list()
        self.yahoo_finance = [LlamaIndexTool.from_tool(tool) for tool in yahoo_tools]
        self.code_interpreter = CodeInterpreterTool()
        
        # Create output directory if it doesn't exist
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_agents(self):
        # Research Agent
        researcher = Agent(
            role="Financial Research Analyst",
            goal="Research company revenue and net income trends",
            backstory="""You are an experienced financial analyst with expertise 
            in analyzing company income statements, focusing on revenue and profitability trends.""",
            tools=self.yahoo_finance,
            verbose=True
        )

        # Code Writer Agent
        coder = Agent(
            role="Python Visualization Expert",
            goal="Write efficient Python code for revenue and income visualization",
            backstory="""You are a Python expert specialized in financial data visualization.
            You excel at creating clear, informative charts for revenue and net income trends.
            You write clean, well-documented code that others can easily understand.""",
            verbose=True
        )

        # Code Interpreter Agent
        interpreter = Agent(
            role="Code Execution Specialist",
            goal="Execute and validate Python code for data visualization",
            backstory="""You are a code execution specialist who tests and validates
            Python code. You ensure the code runs correctly and produces the expected
            visualizations. You know how to use plotly to save interactive visualizations.""",
            tools=[self.code_interpreter],
            verbose=True
        )

        return researcher, coder, interpreter

    def create_tasks(self, researcher, coder, interpreter, symbol: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/{symbol}_analysis_{timestamp}.html"

        # Task 1: Research financial data
        research_task = Task(
            description=f"""Research the following information for {symbol} using Yahoo Finance tools:
            1. Get income statement data using income_statement
            2. Get company basic info using stock_basic_info
            
            Focus only on:
            - Total Revenue
            - Net Income
            For the most recent 3 years.
            
            Compile this information in a clear, structured format.
            Make sure to handle any API errors gracefully.
            """,
            expected_output="""A structured JSON with revenue and net income data""",
            agent=researcher
        )

        # Task 2: Write visualization code
        coding_task = Task(
            description=f"""Using the research results, create a visualization that shows:
            1. Revenue and Net Income Trends:
                - Create a dual-axis chart showing both metrics over 3 years
                - Revenue on primary Y-axis
                - Net Income on secondary Y-axis
                - Include a summary table below the chart
            
            The code should:
            - Import required libraries (plotly, pandas)
            - Process the income statement data
            - Create an interactive visualization using plotly.graph_objects
            - Add a summary table with the actual values
            - Use fig.write_html('{output_file}') to save the visualization
            
            Make sure to:
            - Include error handling
            - Add clear titles and labels
            - Format numbers appropriately (e.g., $1.2B)
            - Create a professional, easy-to-read layout
            """,
            expected_output="""Python code as a string that:
            1. Successfully processes revenue and net income data
            2. Creates a clear visualization with both metrics
            3. Includes error handling
            4. Saves the dashboard as an HTML file""",
            agent=coder
        )

        # Task 3: Execute and validate code
        execution_task = Task(
            description=f"""Execute the provided Python code and verify that:
            1. The code runs without errors
            2. The visualization is created successfully at {output_file}
            3. All data is displayed correctly
            4. The chart and summary table are properly formatted
            
            Make sure to:
            1. Execute the code using the code interpreter tool
            2. Verify that the HTML file exists at {output_file}
            3. Check that both the chart and table are included
            4. Ensure the file is properly saved and accessible
            
            If any issues are found, provide specific feedback for improvements.
            """,
            expected_output="""A report containing:
            1. Execution status (success/failure)
            2. Confirmation that {output_file} was created
            3. Validation of chart and table contents
            4. Any warnings or improvement suggestions""",
            agent=interpreter
        )

        return [research_task, coding_task, execution_task]

    def run_crew(self, symbol: str) -> Dict[str, Any]:
        """
        Run the crew to analyze and visualize stock data for a given symbol
        """
        # Create agents
        researcher, coder, interpreter = self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks(researcher, coder, interpreter, symbol)
        
        # Create crew
        crew = Crew(
            agents=[researcher, coder, interpreter],
            tasks=tasks,
            verbose=True,
        )
        
        # Start the crew
        result = crew.kickoff()
        
        return result

def main():
    # Initialize the crew
    stock_crew = FinancialAnalysisVisualizationCrew()
    
    # Run analysis for a stock symbol
    symbol = "NVDA"  # Example: Apple Inc.
    result = stock_crew.run_crew(symbol)
    
    print("\n=== Analysis Complete ===")
    print(result)

if __name__ == "__main__":
    main()
