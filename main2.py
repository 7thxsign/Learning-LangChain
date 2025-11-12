import requests
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- THIS IS THE FIX ---

# 1. Remove all arguments from the @tool decorator.
# 2. Add a docstring (the """...""" block) right under the function definition.
#    LangChain will automatically use this as the description for the agent.
#    The function's name, "get_weather", will be used as the tool name (which is valid).

@tool
def get_weather(city: str):
    """Gets the weather for a given city using wttr.in"""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error getting weather: {e}"
    except requests.exceptions.JSONDecodeError:
        return "Error: Received invalid JSON from weather service."

# --- END OF FIX ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful."
)

print("Invoking agent for Mangaluru weather...")
response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in Mangaluru?'}
    ]
})

print("\n--- Full Agent Response ---")
print(response)

print("\n--- Agent's Final Answer ---")
print(response)
print(response['messages'][-1].content)