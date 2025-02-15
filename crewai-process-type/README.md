# CrewAI Process Type Comparison

This project demonstrates and compares different process types (Sequential vs Hierarchical) in CrewAI for conducting research and generating reports. It uses a combination of AI agents to perform research analysis and technical writing tasks.

## Overview

The system consists of two main agents:
- **Research Analyst**: Conducts thorough research and gathers comprehensive information
- **Technical Writer**: Transforms research into clear, well-structured reports

## Features

- Supports both Sequential and Hierarchical processing
- Integrates with Langfuse for monitoring and tracking
- Uses SerperDev for web search capabilities
- Automatically saves results with timing comparisons
- Detailed performance metrics tracking

## Prerequisites

- CrewAI
- Langfuse
- SerperDev
- Python 3.12
- OpenAI API key

## Installation

1. Clone the repository
2. Install dependencies
3. Set up environment variables

```bash
pip install -r requirements.txt
```
    
## Usage

```bash
python main.py
```
The script will:
1. Run a sequential process
2. Run a hierarchical process
3. Compare execution times
4. Save detailed results to a timestamped file

## Process Types

### Sequential Process
- Tasks are executed one after another
- Simple and straightforward workflow
- Each agent completes their task before passing to the next

### Hierarchical Process
- Uses a manager LLM (GPT-4) to oversee the process
- More complex but potentially more efficient
- Allows for better task coordination

## Output

Results are saved to a file named `research_results_[timestamp].txt` containing:
- Research topic
- Execution times for both processes
- Detailed results from each process
- Performance comparisons
