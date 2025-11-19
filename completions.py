"""
Completion Models Comparison: Non-Reasoning vs Reasoning

This script demonstrates and compares "one shot" prompts between:
1. A non-reasoning model (GPT-4.1-mini)
2. A reasoning model (GPT-5-mini) configured for expedient responses

The purpose is to show how to configure reasoning models for optimal speed while
still leveraging their enhanced capabilities. The reasoning model is set up with:
- verbosity="low" to minimize output overhead
- reasoning_effort="minimal" to reduce thinking time

This comparison helps understand the trade-offs between model types and how to
tune reasoning models when speed is a priority while maintaining quality.

Additionally, this script includes multimodal examples that demonstrate:
- PDF text analysis: Extract text from PDF documents using pymupdf (fitz)
- PDF visual analysis: Convert PDF pages to images for visual document analysis
- Image analysis: Include images in chat messages for visual analysis
- Document summarization and content analysis workflows
- Proper handling of long documents with content truncation
- Base64 encoding of images for API transmission
- Vision capabilities with detailed image descriptions
- PDF rasterization with configurable DPI and page limits
"""

from openai import OpenAI
import time
import base64
import io

try:
    import fitz  # pymupdf - install with: pip install pymupdf
except ImportError:
    print("❌ pymupdf not found. Install with: pip install pymupdf")
    fitz = None
from dotenv import load_dotenv

# Load environment variables from .env file
# OPENAI_API_KEY=jwt
# OPENAI_BASE_URL=
load_dotenv()

client = OpenAI()


def get_completion_4(messages: list[dict]) -> str:
    """
    Demonstrates basic chat completion using GPT-4.1-mini model.
    Uses temperature=0 for deterministic responses.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        str: The message object from the model's response
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0
    )
    return response.choices[0].message


def get_completion_5_oneshot(messages: list[dict]) -> str:
    """
    Demonstrates chat completion using GPT-5-mini model with reasoning capabilities.
    Uses low verbosity and minimal reasoning effort for faster responses.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        str: The message object from the model's response
    """
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        verbosity="low",
        reasoning_effort="minimal",
    )
    return response.choices[0].message


def get_completion_4_multimodal(messages: list[dict]) -> str:
    """
    Demonstrates multimodal chat completion using GPT-4.1-mini model.
    This function can handle messages with both text and document content.
    Uses temperature=0 for deterministic responses.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
                 Content can include text and document attachments.

    Returns:
        str: The message object from the model's response
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0
    )
    return response.choices[0].message


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using pymupdf (fitz).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Extracted text content from all pages
    """
    if fitz is None:
        print(
            "❌ pymupdf is required for PDF text extraction. Install with: pip install pymupdf"
        )
        return ""

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        text_content = ""

        # Extract text from each page
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()

        doc.close()
        return text_content.strip()

    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""


def rasterize_pdf_pages(pdf_path: str, dpi: int = 150, max_pages: int = 5) -> list[str]:
    """
    Convert PDF pages to PNG images and encode them as base64 data URLs.
    This approach captures visual elements like diagrams, charts, and complex layouts
    that might be lost during text extraction.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rasterization (higher = better quality, larger file)
        max_pages: Maximum number of pages to process (to avoid token limits)

    Returns:
        list[str]: List of base64-encoded data URLs for each page image
    """
    if fitz is None:
        print(
            "❌ pymupdf is required for PDF rasterization. Install with: pip install pymupdf"
        )
        return []

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page_images = []

        # Limit pages to avoid excessive token usage
        num_pages = min(doc.page_count, max_pages)

        for page_num in range(num_pages):
            try:
                page = doc[page_num]

                # Convert page to image (PNG format)
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # scaling matrix
                pix = page.get_pixmap(matrix=mat)

                # Convert to PNG bytes
                png_data = pix.tobytes("png")

                # Clean up the pixmap
                pix = None

                # Encode as base64 data URL
                png_base64 = base64.b64encode(png_data).decode("utf-8")
                data_url = f"data:image/png;base64,{png_base64}"

                page_images.append(data_url)

                print(f"📄 Rasterized page {page_num + 1}/{num_pages}")

            except Exception as page_error:
                print(f"❌ Error processing page {page_num + 1}: {page_error}")
                continue

        if num_pages < doc.page_count:
            print(f"📋 Limited to first {max_pages} pages to manage token usage")

        return page_images

    except Exception as e:
        print(f"❌ Error rasterizing PDF: {e}")
        return []
    finally:
        # Ensure document is properly closed
        if doc is not None:
            try:
                doc.close()
            except:
                pass  # Ignore errors when closing


def create_pdf_visual_summary_messages(pdf_path: str) -> list[dict]:
    """
    Creates messages that include PDF pages as images for visual analysis.
    This approach is excellent for PDFs with diagrams, charts, complex layouts,
    or visual elements that would be lost in text extraction.

    Args:
        pdf_path: Path to the PDF file to analyze

    Returns:
        list[dict]: Messages formatted for OpenAI API with page images
    """
    try:
        # Rasterize PDF pages as images
        page_images = rasterize_pdf_pages(pdf_path, dpi=150, max_pages=3)

        if not page_images:
            print(f"❌ Could not rasterize pages from {pdf_path}")
            return []

        # Build content array with text prompt and page images
        content = [
            {
                "type": "text",
                "text": f"""Please analyze this PDF document ({pdf_path}) by examining the visual content of its pages. 
                
Since you can see the actual page layouts, diagrams, charts, and visual elements, please provide:

1. **Document Overview**: What type of document is this and what is its main purpose?
2. **Visual Elements**: Describe any diagrams, charts, tables, or visual aids present
3. **Layout and Structure**: How is the information organized on the pages?
4. **Key Content**: What are the main topics, concepts, or information covered?
5. **Target Audience**: Who appears to be the intended audience?
6. **Notable Features**: Any unique formatting, highlighting, or special elements

Please be thorough in your visual analysis, as you can see details that text extraction might miss.""",
            }
        ]

        # Add each page image to the content
        for i, image_data_url in enumerate(page_images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "high"},
                }
            )

        # Create the complete message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with excellent visual analysis capabilities. You can examine document layouts, diagrams, charts, and visual elements to provide comprehensive document analysis.",
            },
            {"role": "user", "content": content},
        ]

        return messages

    except FileNotFoundError:
        print(f"❌ PDF file not found: {pdf_path}")
        return []
    except Exception as e:
        print(f"❌ Error processing PDF file for visual analysis: {e}")
        return []


def create_pdf_summary_messages(pdf_path: str) -> list[dict]:
    """
    Creates messages that include extracted PDF text content for analysis.
    Uses pymupdf to extract text from the PDF and includes it in the prompt.

    Args:
        pdf_path: Path to the PDF file to analyze

    Returns:
        list[dict]: Messages formatted for OpenAI API with extracted text content
    """
    try:
        # Extract text content from PDF
        pdf_text = extract_pdf_text(pdf_path)

        if not pdf_text:
            print(f"❌ No text could be extracted from {pdf_path}")
            return []

        # Truncate text if it's too long (GPT models have token limits)
        max_chars = 400000  # Adjust based on your needs and model limits
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[Content truncated due to length...]"

        # Create messages with extracted PDF text
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes document content and provides clear, concise summaries.",
            },
            {
                "role": "user",
                "content": f"""Please analyze the following PDF document content and provide a brief summary. 
Focus on the main topics, key concepts, and overall purpose of the document.

Document: {pdf_path}

Content:
{pdf_text}

Please provide:
1. A brief overview of the document's purpose
2. Main topics and sections covered
3. Key concepts or important points
4. Target audience (if apparent)""",
            },
        ]

        return messages

    except FileNotFoundError:
        print(f"❌ PDF file not found: {pdf_path}")
        return []
    except Exception as e:
        print(f"❌ Error processing PDF file: {e}")
        return []


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 for use with OpenAI vision models.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded image data with proper data URL format
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Determine the image format from file extension
        if image_path.lower().endswith(".png"):
            data_url = f"data:image/png;base64,{image_base64}"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            data_url = f"data:image/jpeg;base64,{image_base64}"
        elif image_path.lower().endswith(".gif"):
            data_url = f"data:image/gif;base64,{image_base64}"
        else:
            # Default to PNG if format is unclear
            data_url = f"data:image/png;base64,{image_base64}"

        return data_url

    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return ""


def create_image_analysis_messages(image_path: str) -> list[dict]:
    """
    Creates messages that include an image for visual analysis.
    Uses base64 encoding to include the image in the message content.

    Args:
        image_path: Path to the image file to analyze

    Returns:
        list[dict]: Messages formatted for OpenAI API with image content
    """
    try:
        # Encode the image to base64
        image_data_url = encode_image_to_base64(image_path)

        if not image_data_url:
            print(f"❌ Could not encode image: {image_path}")
            return []

        # Create messages with image content
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with vision capabilities that can analyze images and provide detailed descriptions.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Please analyze this image and provide a detailed description. Include:

1. What you see in the image (objects, people, text, etc.)
2. The overall composition and visual elements
3. Any text or writing visible in the image
4. The apparent purpose or context of the image
5. Notable colors, style, or artistic elements
6. Any technical or specific details that stand out

Please be thorough and descriptive in your analysis.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "high"},
                    },
                ],
            },
        ]

        return messages

    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return []
    except Exception as e:
        print(f"❌ Error processing image file: {e}")
        return []


def main():
    """
    Main function to execute completion demonstrations.
    Each demonstration is wrapped in a try-catch block to ensure
    that errors in one example don't stop the execution of others.
    """
    # Test messages for demonstrations
    test_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "say hello and comment on the weather.",
        },
    ]

    print("=== GPT-4.1-mini Completion ===")
    try:
        response = get_completion_4(messages=test_messages)
        print(response)
        print()
    except Exception as e:
        print(f"❌ Error in get_completion_4: {e}")
        print()

    print("=== GPT-5-mini One-shot Completion ===")
    try:
        response = get_completion_5_oneshot(messages=test_messages)
        print(response)
        print()
    except Exception as e:
        print(f"❌ Error in get_completion_5_oneshot: {e}")
        print()

    # Demonstrate PDF text extraction and analysis
    print("=== GPT-4.1-mini PDF Text Analysis ===")
    try:
        pdf_path = "data/PyTorchCheatsheet.pdf"
        print(f"Extracting text from: {pdf_path}")

        pdf_messages = create_pdf_summary_messages(pdf_path)

        if pdf_messages:  # Only proceed if PDF text was successfully extracted
            print("📄 Analyzing extracted PDF content...")
            response = get_completion_4_multimodal(messages=pdf_messages)
            print("� PDF Analysis Result:")
            print(response.content)
            print()
        else:
            print("❌ Could not extract text from PDF document")
            print()
    except Exception as e:
        print(f"❌ Error in PDF text analysis: {e}")
        print()

    # Demonstrate PDF visual analysis (pages as images)
    print("=== GPT-4.1-mini PDF Visual Analysis ===")
    try:
        pdf_path = "data/PyTorchCheatsheet.pdf"
        print(f"Converting PDF pages to images: {pdf_path}")

        pdf_visual_messages = create_pdf_visual_summary_messages(pdf_path)

        if pdf_visual_messages:  # Only proceed if PDF pages were successfully rasterized
            print("🖼️ Analyzing PDF pages as images...")
            response = get_completion_4_multimodal(messages=pdf_visual_messages)
            print("📊 PDF Visual Analysis Result:")
            print(response.content)
            print()
        else:
            print("❌ Could not rasterize PDF pages for visual analysis")
            print()
    except Exception as e:
        print(f"❌ Error in PDF visual analysis: {e}")
        print()

    # Demonstrate image analysis with vision capabilities
    print("=== GPT-4.1-mini Image Analysis ===")
    try:
        image_path = "data/claude_tester.png"
        print(f"Analyzing image: {image_path}")

        image_messages = create_image_analysis_messages(image_path)

        if image_messages:  # Only proceed if image was successfully encoded
            print("🖼️ Analyzing image content...")
            response = get_completion_4_multimodal(messages=image_messages)
            print("📸 Image Analysis Result:")
            print(response.content)
            print()
        else:
            print("❌ Could not load and encode image")
            print()
    except Exception as e:
        print(f"❌ Error in image analysis: {e}")
        print()


if __name__ == "__main__":
    main()
