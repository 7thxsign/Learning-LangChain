from dataclasses import dataclass
import requests
import os
from dotenv import load_dotenv

# Corrected: Use the correct message imports
# Corrected: Use the correct message imports
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from pydantic import BaseModel, Field
load_dotenv()

# --- Schemas ---
class ResponseFormat(BaseModel):
    """The final structured response containing the weather information."""
    summary: str = Field(description="A humorous and sarcastic summary of the weather.")
    temperature_celcius: float = Field(description="The current temperature in Celsius.")
    temperature_fahrenheit: float = Field(description="The current temperature in Fahrenheit.")
    humidity: float = Field(description="The current humidity percentage.")

@dataclass
class Context:
    user_id: str

# --- Tools (Unchanged) ---
@tool
def get_weather(city: str) -> dict:
    """Gets the weather for a given city using wttr.in. Returns a JSON dictionary."""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error getting weather: {e}"}
    except requests.exceptions.JSONDecodeError:
        return {"error": "Received invalid JSON from weather service."}

@tool('locate_user', description="Get the location of the user based on the context.")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    user_id = runtime.context.user_id if runtime.context else 'Unknown'
    match user_id:
        case 'ABC123':
            return 'Mangaluru'
        case 'XYZ789':
            return 'Bengaluru'
        case 'LMN456':
            return 'Chennai'
        case _:
            return 'Unknown'

# --- Agent Setup ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 1. LLM for the FINAL response (with structured output enforced)
# This LLM will be used to generate the final, structured response 
# *after* the tools have run.
final_response_llm = llm.with_structured_output(ResponseFormat, method="json")

# 2. Tools
tools = [get_weather, locate_user]

# 3. Prompt Template
system_prompt = (
    "You are a helpful weather assistant, who always cracks jokes and is humorous "
    "while remaining helpful. Use the provided tools to find the user's location "
    "and the weather data. Once you have the data, summarize it for the user."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Agent Runnable Construction (The standard Tool-Calling Agent)
# This handles the tool-use logic.
agent_runnable = create_tool_calling_agent(
    llm=llm, # Use the standard LLM for tool calling
    tools=tools,
    prompt=prompt,
)

# 5. Agent Executor
agent_executor = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True 
)

# 6. Final Chain Construction
# The AgentExecutor runs the tools, and its final output (a Message object)
# is passed to the final_response_llm to be converted to the structured format.
chain = agent_executor | final_response_llm 


# --- Invocation ---

print("Invoking agent for Mangaluru weather...")

config = {
    'configurable': {
        'context': Context(user_id="ABC123"),
    }
}

# Invoke the full chain now
response = chain.invoke(
    {"input": "What is the weather like?", "chat_history": []}, # Agent needs "input"
    config=config
)


print("\n--- Agent's Final Answer ---")
# The final result is the Pydantic object directly (since it's the last step in the chain)
print(response)
print(response.summary)
print(response.temperature_celcius)