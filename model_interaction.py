from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

conversation = [
    SystemMessage('You are a sarcastic chatbot that responds in a funny and unhinged tone.'),
    HumanMessage('Tell me a joke about programmers.'),
    AIMessage('Why do programmers prefer dark mode? Because light attracts bugs!'),
    HumanMessage('Haha, that was a good one! Can you tell me another joke?')
]


for chunk in llm.stream(conversation):
    print(chunk.content, end='', flush=True)
