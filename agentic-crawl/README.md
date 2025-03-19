# Agentic Web Crawler

A crew-based web crawler that extracts information from websites based on user prompts.

## Features

- Extracts URLs from a website's sitemap
- Intelligently filters URLs based on keywords from the user's prompt
- Crawls relevant URLs to extract content in batches to avoid token limits
- Formats the extracted content according to the user's requirements
- Uses a sequential agent workflow for efficient web crawling

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

This crawler uses a sequential agent structure with batch processing:

1. **Sitemap Agent**: Discovers and extracts URLs from a website's sitemap, then filters them based on keywords from the user prompt
2. **Content Crawler Agent**: Crawls filtered URLs in small batches and extracts content, ensuring token limits aren't exceeded
3. **Formatter Agent**: Formats the combined batch results according to the user's requirements

The system maintains the user's prompt across all tasks and processes URLs in manageable batches to avoid token limit issues.

## Benefits of Batch Processing

- Processes large numbers of URLs in smaller chunks
- Avoids token limit issues by processing content incrementally
- Combines results from all batches for comprehensive output
- Preserves context and user requirements across processing steps

## Troubleshooting

- If you encounter token limit issues, use the options described in the "Handling Token Limit Issues" section.
- For very large websites, you may need to target specific sections rather than the entire domain.
- If you encounter any issues with the batch processing, you can enable debug mode to see more detailed logs.
- Adjust the batch size in the code if needed (default batch size: 2 URLs per batch)

## FastAPI Integration

The project has been refactored to use FastAPI, providing a web interface and API endpoints for the crawler functionality.

### Running the FastAPI server

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   python -m src.run
   ```

3. Open your browser and navigate to http://localhost:5000 to access the web interface.

### API Endpoints

- `GET /`: Web interface for the crawler
- `POST /start_crawl`: Start a new crawl session
  - Request body:
    ```json
    {
      "url": "https://example.com",
      "prompt": "Information to look for",
      "max_sitemap_urls": 50,
      "max_crawl_urls": 5,
      "model": "gemini-2.0-flash" // Optional
    }
    ```
- `GET /crawl/{crawl_id}`: Get results of a specific crawl session

### Command Line Usage

The original command line functionality is still available:

```
python main.py --url https://example.com --prompt "What information to look for"
```