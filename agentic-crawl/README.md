# Agentic Web Crawler

A crew-based web crawler that extracts information from websites based on user prompts.

## Features

- Extracts URLs from a website's sitemap
- Intelligently filters URLs based on keywords from the user's prompt
- Crawls relevant URLs to extract content
- Formats the extracted content according to the user's requirements
- Uses a hierarchical agent structure with a manager for better coordination

## Installation

1. Clone the repository
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL_NAME=gemini-2.0-flash
```

## Usage

### Basic Usage

```bash
python main.py --url https://example.com --prompt "Find information about products with details and features" --model gemini-2.0-flash --debug
```

### Command Line Arguments

- `--url`: The base URL to crawl (required)
- `--prompt`: What information to look for (required)
- `--model`: Model to use (default: gemini-2.0-flash)
- `--debug`: Enable debug mode
- `--max-sitemap-urls`: Maximum number of URLs to process from sitemap (default: 50)
- `--max-crawl-urls`: Maximum number of URLs to crawl for content (default: 5)

## Handling Token Limit Issues

If you encounter MAX_TOKENS errors, you can try the following:

1. Reduce the number of URLs processed:
```bash
python main.py --url https://example.com --prompt "Your prompt" --max-sitemap-urls 30 --max-crawl-urls 3
```

2. Use a model with higher token limits:
```bash
python main.py --url https://example.com --prompt "Your prompt" --model gemini-3.0-pro
```

3. Make your prompt more specific to target specific pages/products.

4. If crawling large websites, consider using site-specific sections rather than the entire domain.

## Architecture

This crawler uses a hierarchical agent structure:

1. **Manager Agent**: Coordinates the overall process and maintains the user's requirements
2. **Sitemap Agent**: Discovers and extracts URLs from a website's sitemap, then filters them based on keywords from the user prompt
3. **Content Crawler Agent**: Crawls filtered URLs and extracts content
4. **Formatter Agent**: Formats the extracted content according to the user's requirements

The Manager Agent ensures that the user's prompt is consistently maintained across all tasks, avoiding issues with information loss between steps and token limits.

## Benefits of Hierarchical Processing

- Centralized prompt management through the Manager Agent
- Reduced token limit issues by avoiding repetitive passing of large data structures
- Better coordination between specialized agents
- More robust error handling and task delegation

## Troubleshooting

- If you encounter token limit issues, use the options described in the "Handling Token Limit Issues" section.
- For very large websites, you may need to target specific sections rather than the entire domain.
- If you encounter any issues with the hierarchical process, you can enable debug mode to see more detailed logs.