from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse,wrap_model_call
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

basic_model = init_chat_model('gemini-2.5-flash'),
advanced_model = init_chat_model('gemini-2.5-pro')

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model

    return handler(request)

agent = create_agent(model=basic_model, middleware=[dynamic_model_selection])

response = agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain quantum computing in simple terms.")
    ]
})

print(response["messages"][-1].content)