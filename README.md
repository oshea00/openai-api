# OpenAI API Examples and Comparisons

This project provides comprehensive examples and comparisons for interacting with both the legacy OpenAI Chat Completions API and the newer Responses API. It demonstrates various approaches to structured responses, tool calling, reasoning capabilities, and performance characteristics.

## Project Overview

The repository contains example scripts that showcase different API patterns, response formats, and debugging capabilities. Each script includes detailed logging to help understand how the OpenAI library constructs API calls behind the scenes.

## Scripts

### Core API Demonstrations

- **`legacy_chat_text.py`** - Comprehensive examples using the legacy Chat Completions API
  - Basic text chat, structured responses with Pydantic models
  - JSON mode responses, strict JSON schema enforcement
  - Tool calling with two-phase pattern (tool_calls → tool results → final response)
  - Custom HTTP logging client for debugging API interactions

- **`responses_text.py`** - Modern examples using the Responses API
  - Native structured responses with text_format parameter
  - Reasoning capabilities with configurable effort levels
  - Built-in reasoning summary extraction
  - Enhanced schema validation and parsing

- **`completions.py`** - Comprehensive multimodal examples and model comparisons
  - Demonstrates "one shot" prompts on GPT-4.1-mini vs GPT-5-mini
  - Shows how to configure reasoning models for optimal speed
  - **PDF text analysis**: Extract and analyze text from PDF documents using pymupdf
  - **Image analysis**: Vision capabilities with detailed image descriptions
  - **Multimodal workflows**: Combined text, document, and image processing
  - Base64 encoding for image transmission and proper data URL formatting
  - Document summarization with content truncation for long texts
  - Explores trade-offs between model types and performance tuning

### Performance Comparisons

- **`completions_timed_compare.py`** - Quantitative performance analysis
  - Times identical prompts across different model types
  - Measures actual overhead of reasoning capabilities
  - Provides speed ratio calculations and performance insights

## Features

- **Structured Responses**: Multiple approaches using Pydantic models, JSON mode, and strict schemas
- **Reasoning Capabilities**: Native reasoning support with configurable effort levels
- **Multimodal Processing**: PDF text extraction, image analysis, and vision capabilities
- **Document Analysis**: Text extraction from PDFs using pymupdf with content summarization
- **Vision Analysis**: Image processing with base64 encoding and detailed visual descriptions
- **Tool Calling**: Complete examples of function calling patterns
- **Performance Analysis**: Timing comparisons between model types
- **Debugging Support**: Custom HTTP logging for API call inspection
- **Error Handling**: Robust error handling with graceful degradation

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd openai-api
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install openai pydantic python-dotenv httpx pymupdf
   ```

4. **Set up environment variables** (optional)
   Create a `.env` file with your OpenAI API credentials. These variables are optional but useful if you are running a local model that uses OpenAI conventions, in which case the base URL may be different as well as the API key:
   ```
   OPENAI_API_KEY=your_jwt_token_here
   OPENAI_BASE_URL=your_base_url_here
   ```

5. **Run example scripts**
   ```bash
   python legacy_chat_text.py      # Legacy API examples
   python responses_text.py        # Responses API examples
   python completions.py           # Model comparison + multimodal examples
   python completions_timed_compare.py  # Performance analysis
   ```

   **Note**: For multimodal examples in `completions.py`, ensure you have sample files in the `data/` directory:
   - `data/PyTorchCheatsheet.pdf` - For PDF text extraction demo
   - `data/claude_tester.png` - For image analysis demo

## Key Learnings

- **API Evolution**: Compare legacy chat completions vs modern responses API
- **Reasoning Models**: Understand performance trade-offs and optimization strategies
- **Multimodal AI**: Learn to integrate text, document, and image processing in a single workflow
- **Document Processing**: Extract and analyze content from PDF files for AI consumption
- **Vision Capabilities**: Leverage AI vision models for detailed image analysis and description
- **Structured Output**: Multiple approaches for getting structured data from models
- **Debugging**: Visibility into actual HTTP requests/responses for troubleshooting


