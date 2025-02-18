# Financial Analysis Visualization Crew

This project uses CrewAI to create an automated financial analysis and visualization system for stock data. It combines Yahoo Finance data retrieval with interactive Plotly visualizations to generate comprehensive financial reports.

## Overview

The system employs three specialized AI agents working together:
- **Financial Research Analyst**: Retrieves and analyzes company financial data
- **Python Visualization Expert**: Creates visualization code for financial metrics
- **Code Execution Specialist**: Validates and executes the visualization code

## Features

- Automated financial data retrieval using Yahoo Finance
- Interactive visualizations with Plotly
- Dual-axis charts showing Revenue and Net Income trends
- Summary tables with key financial metrics
- Automatic HTML report generation
- Timestamp-based result tracking

## Prerequisites

- Python 3.12
- CrewAI
- Required Python packages:
  - crewai
  - crewai_tools
  - llama_index
  - plotly
  - pandas
  - python-dotenv

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with necessary API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```python
from main import FinancialAnalysisVisualizationCrew

# Initialize the crew
stock_crew = FinancialAnalysisVisualizationCrew()

# Run analysis for a stock symbol
symbol = "NVDA"  # Example: NVIDIA Corporation
result = stock_crew.run_crew(symbol)
```

## Output

The system generates an HTML file in the `output` directory with the naming format:
`{symbol}_analysis_{timestamp}.html`

The output includes:
- Interactive dual-axis chart showing Revenue and Net Income trends
- Summary table with actual values
- Professional formatting with clear titles and labels
- Properly formatted numbers (e.g., $1.2B)

## Process Flow

1. **Research Phase**
   - Retrieves income statement data
   - Gets company basic information
   - Focuses on Total Revenue and Net Income for recent 3 years

2. **Visualization Phase**
   - Creates dual-axis chart using Plotly
   - Processes financial data
   - Generates summary table
   - Applies professional formatting

3. **Validation Phase**
   - Executes the visualization code
   - Verifies output generation
   - Validates data accuracy
   - Ensures proper file saving

## Example Analysis

For NVIDIA (NVDA), the system generates an interactive visualization showing:
- Revenue trends on the primary Y-axis
- Net Income trends on the secondary Y-axis
- A summary table with actual values
- Clear labels and professional formatting

The [output](./output/NVDA_analysis_20250217_143000.html) is saved as an interactive HTML file that can be viewed in any web browser.
