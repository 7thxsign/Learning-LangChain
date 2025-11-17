from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are a helpful assistant. Provide clear and concise answers.'

    match user_role:
        case 'expert':
            return f'{base_prompt} As an expert user, provide detailed and technical explanations.'
        case 'beginner':
            return f'{base_prompt} As a beginner user, provide simple and easy-to-understand explanations.'
        case 'child':
            return f'{base_prompt} Explain everything as if you are talking to a five-year old child, using simple language and examples.'
        case _:
            return base_prompt
        
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

agent = create_agent(
    model=llm,
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain PCA"}]
}, context=Context(user_role='expert')
)

print(response["messages"][-1].content)