import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    try:
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error getting weather: {e}"
    except requests.exceptions.JSONDecodeError:
        return "Error: Received invalid JSON from weather service."
    
llm = ChatGoogleGenerativeAI(model="gemini-pro")

agent = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful."
)
    
response = agent.invoke({
    'messages' : [
        {'role': 'user', 'content': 'What is the weather like in Mangaluru?'}
    ]
})

print("\n--- Full Agent Response ---")
if 'output' in response:
    print(response['output'])
else:
    print("Could not find 'output' key in the response.")