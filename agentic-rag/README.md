# Agentic RAG System

This project implements and compares two different approaches to Agentic RAG (Retrieval-Augmented Generation) using CrewAI. It combines document analysis with web search capabilities to perform comprehensive research and generate detailed reports.

## Overview

The system utilizes two specialized AI agents:
- **Research Analyst**: Searches through documents and online sources to gather information
- **Content Writer**: Synthesizes the research into clear, comprehensive responses

### RAG Approaches
The system implements and compares two different methods:
1. **PDFSearchTool**: Direct PDF document analysis and search
2. **CrewDoclingSource**: Knowledge-based document processing

## Features

- Two different RAG implementation approaches
- PDF document analysis and search
- Web search integration via SerperDev
- Automated research and content generation
- Performance comparison between RAG methods
- Timestamp-based result tracking
- Automatic result saving with execution timing

## Prerequisites

- Python 3.12
- CrewAI
- SerperDev API key
- OpenAI API key
- Python packages:
  - crewai
  - crewai_tools
  - python-dotenv

## Installation

1. Clone the repository
2. Install required dependencies
3. Set up environment variables

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with the following:
```
# OpenAI API Key
OPENAI_API_KEY=
OPENAI_MODEL_NAME=

# Serper API Key
SERPER_API_KEY=
```

## Usage

```bash
python main.py
```

The script will:
1. Initialize research and writing agents
2. Process the user query using both RAG methods
3. Search through documents and online sources
4. Generate comprehensive reports
5. Save results with timing information
6. Compare performance between the two methods

## Output

Results are saved to files named `research_result_[method]_[timestamp].txt` containing:
- Research method used (pdfsearch or docling)
- Research query
- Timestamp
- Execution duration
- Detailed research findings and analysis

Performance metrics for both methods are displayed in the console for comparison.
