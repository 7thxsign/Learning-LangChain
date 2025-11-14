import os
import requests
import base64
from io import BytesIO
from dotenv import load_dotenv

# Use the community package for local models like Ollama
from langchain_ollama import ChatOllama
# Use the core package for message types
from langchain_core.messages import HumanMessage

load_dotenv()

# --- Helper Function to Handle Image URL to Base64 ---

def url_to_base64(url: str) -> str:
    """Fetches an image from a URL and converts it to a base64 encoded string."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Get the binary content
        image_bytes = response.content
        
        # Encode to Base64
        return base64.b64encode(image_bytes).decode('utf-8')

    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        return ""


# --- Main Logic ---

# 1. Initialize ChatOllama
# NOTE: You MUST be running a multi-modal model in Ollama for this to work (e.g., 'llava', 'moondream').
llm = ChatOllama(model="llama3.1:8b", temperature=0.0) # This model needs to be pulled via 'ollama pull llava'

# 2. Process Image
image_url = 'https://www.sony.co.jp/en/Products/di_photo-gallery/images/extralarge/1887.JPG'
base64_image = url_to_base64(image_url)

if not base64_image:
    print("Could not process image, exiting.")
    exit()

# 3. Construct the Message (LangChain Core format)
message = HumanMessage(
    content=[
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        # Ollama expects the data URI for the image
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
    ]
)

print(f"Invoking Ollama model '{llm.model}' with image data...")

# 4. Invoke the Model
response = llm.invoke([message])

print("\n--- Model Response ---")
print(response.content)