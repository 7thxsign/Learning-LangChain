from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

message = {
    "role": "user",
    "content": [
        {'type': 'text', 'text': 'Describe the contents of this imagea.'},
        {'type': 'image', 'url': 'https://www.sony.co.jp/en/Products/di_photo-gallery/images/extralarge/1887.JPG'}
    ]
}

response = llm.invoke([message])
print(response.content)