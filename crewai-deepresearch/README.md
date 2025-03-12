# CrewAI Deep Research

A powerful deep research implementation using CrewAI to perform iterative, depth-based research on any topic.

## Overview

This project implements a deep research system that follows an iterative approach to explore topics in depth. The system uses a crew of specialized AI agents to:

1. Plan the research
2. Gather information
3. Analyze findings
4. Extract key learnings and new research directions
5. Iteratively explore new directions based on depth parameter
6. Generate a comprehensive report

## How It Works

The deep research process follows this flow:

```
flowchart TB
    subgraph Input
        Q[User Query]
        B[Breadth Parameter]
        D[Depth Parameter]
    end

    DR[Deep Research] -->
    SQ[SERP Queries] -->
    PR[Process Results]

    subgraph Results[Results]
        direction TB
        NL((Learnings))
        ND((Directions))
    end

    PR --> NL
    PR --> ND

    DP{depth > 0?}

    RD["Next Direction:
    - Prior Goals
    - New Questions
    - Learnings"]

    MR[Markdown Report]

    %% Main Flow
    Q & B & D --> DR

    %% Results to Decision
    NL & ND --> DP

    %% Circular Flow
    DP -->|Yes| RD
    RD -->|New Context| DR

    %% Final Output
    DP -->|No| MR
```

### Parameters

- **Topic**: The research query or topic
- **Depth**: How many levels of iterative research to perform (1-5)
- **Breadth**: How many search queries to perform at each depth level (1-5)

### Process

1. **Planning Phase**: The research planner agent creates a structured research plan
2. **Research Phase**: The expert researcher agent executes searches and gathers information
3. **Analysis Phase**: The research analyst agent analyzes the findings and extracts key learnings and new directions
4. **Iterative Exploration**: If depth > 0, the system selects a new direction and starts a new research cycle
5. **Report Generation**: When depth = 0 or no new directions are found, the writer and reviewer agents generate a comprehensive report

## Implementation Details

The system uses an iterative approach rather than recursion to handle the depth-based research process. This provides several advantages:

- Avoids potential stack overflow issues with deep recursion
- Makes debugging easier
- Provides better control over the research flow
- Allows for easier tracking of progress across iterations

Each iteration builds upon the knowledge gained in previous iterations, creating a comprehensive research report that explores the topic in depth.

## Usage

### API Endpoints

- `POST /start_research`: Start a new research session
  ```json
  {
    "topic": "Quantum computing applications in medicine",
    "depth": 3,
    "breadth": 3
  }
  ```

- `GET /research/{research_id}`: Get the results of a research session

### Response Format

```json
{
  "research_id": "uuid",
  "result": "Markdown formatted research report",
  "sources": [
    {"url": "source_url", "title": "source_title", "snippet": "source_snippet"}
  ],
  "learnings": [
    "Key learning 1",
    "Key learning 2"
  ],
  "directions": [
    "Research direction 1",
    "Research direction 2"
  ],
  "process_logs": [
    "Log entry 1",
    "Log entry 2"
  ]
}
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Run the application: `python src/run.py`

## Configuration

The system uses YAML configuration files for agents and tasks:

- `src/config/deep_research_agents.yaml`: Agent configurations
- `src/config/deep_research_tasks.yaml`: Task configurations

## License

MIT